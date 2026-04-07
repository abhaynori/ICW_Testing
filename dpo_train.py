#!/usr/bin/env python3
"""
DPO Training for ICW Watermarking

Generates preference pairs offline using detector scores,
then trains with TRL's DPOTrainer.  More stable than GRPO because
it requires no online generation during training.

Usage:
    python dpo_train.py --model full --method acrostics --samples 200
    python dpo_train.py --model full --method acrostics \\
        --warm-start-model sft_models/sft_acrostics_full_*/final_model \\
        --use-lora --n-candidates 8
"""

import argparse
import inspect
import json
import os
import re
import warnings
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from trl import DPOConfig, DPOTrainer
except Exception:
    DPOConfig = None
    DPOTrainer = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import (
    acrostics_detector, acrostics_embed_prompt, secret_sequence,
    get_acrostics_secret_sequence, set_acrostics_secret_sequence,
    unicode_detector, unicode_embed_prompt,
    initials_detector, initials_embed_prompt, green_letters,
    lexical_detector, lexical_embed_prompt, green_words,
    get_base_system_prompt,
)
from memory_config import get_model_config
from research_utils import load_causal_lm_with_adapter_support, patch_saved_model_config


# ---------------------------------------------------------------------------
# Shared utilities (duplicated from grpo_train.py for standalone operation)
# ---------------------------------------------------------------------------

def get_detector_and_args(method):
    current_secret_sequence = get_acrostics_secret_sequence()
    detector_map = {
        'unicode': (unicode_detector, ()),
        'initials': (initials_detector, (green_letters,)),
        'lexical': (lexical_detector, (green_words,)),
        'acrostics': (acrostics_detector, (current_secret_sequence,)),
    }
    if method not in detector_map:
        raise ValueError(f"Unknown method: {method}")
    return detector_map[method]


def get_prompt_function(method):
    prompt_map = {
        'unicode': unicode_embed_prompt,
        'initials': initials_embed_prompt,
        'lexical': lexical_embed_prompt,
        'acrostics': acrostics_embed_prompt,
    }
    if method not in prompt_map:
        raise ValueError(f"Unknown method: {method}")
    return prompt_map[method]


def build_messages(query, prompt_fn=None, include_instruction=True):
    if include_instruction and prompt_fn is not None:
        return prompt_fn(query)
    return [
        {"role": "system", "content": get_base_system_prompt()},
        {"role": "user", "content": query},
    ]


def _looks_like_local_path(source):
    if source is None:
        return False
    value = str(source).strip()
    if not value:
        return False
    if value.startswith((".", "/", "~")):
        return True
    if re.match(r"^[a-zA-Z]:[\\/]", value):
        return True
    normalized = value.replace("\\", "/")
    if "/" not in normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 3:
        return True
    if value.endswith(("/", "\\")):
        return True
    if len(parts) == 2:
        return os.path.exists(parts[0])
    return False


def resolve_pretrained_source(source, label):
    if source is None:
        return None
    raw = str(source).strip()
    if not raw:
        raise ValueError(f"{label} cannot be empty.")
    expanded = os.path.expanduser(raw)
    normalized = os.path.normpath(expanded)
    if os.path.exists(normalized):
        return normalized
    if _looks_like_local_path(raw):
        abs_candidate = os.path.abspath(normalized)
        raise FileNotFoundError(
            f"{label} path not found: '{raw}'. "
            f"Resolved as: '{abs_candidate}'. "
            f"Current working directory: '{os.getcwd()}'."
        )
    return raw


def load_causal_lm_with_dtype_fallback(model_name, model_kwargs, dtype_value=None):
    return load_causal_lm_with_adapter_support(
        model_name_or_path=model_name,
        model_kwargs=model_kwargs,
        dtype_value=dtype_value,
    )


def _slice_indices_for_split(size, split):
    train_end = int(size * 0.8)
    validation_end = int(size * 0.9)
    if split == "train":
        return 0, train_end
    if split == "validation":
        return train_end, validation_end
    if split == "test":
        return validation_end, size
    raise ValueError(f"Unsupported split: {split}")


def _format_alpaca_query(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    if input_text:
        return f"{instruction}\n\nContext: {input_text}"
    return instruction


def prepare_dataset(num_samples, split="train", dataset_name="eli5", seed=42):
    dataset_key = dataset_name.strip().lower()
    if dataset_key == "eli5":
        try:
            ds = load_dataset("sentence-transformers/eli5", "pair", split="train")
        except Exception as exc:
            warnings.warn(f"Could not load dataset 'eli5' split '{split}': {exc}")
            return None
        start, end = _slice_indices_for_split(len(ds), split)
        subset = ds.select(range(start, min(end, start + num_samples)))
        queries = subset["question"]
    elif dataset_key == "alpaca":
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        start, end = _slice_indices_for_split(len(ds), split)
        subset = ds.select(range(start, min(end, start + num_samples)))
        queries = [_format_alpaca_query(row) for row in subset]
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")
    return Dataset.from_dict({"query": queries})


# ---------------------------------------------------------------------------
# Preference pair generation
# ---------------------------------------------------------------------------

def generate_single_response(model, tokenizer, prompt_text, max_new_tokens, temperature, top_p):
    """Generate a single response given a formatted prompt string."""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


def generate_preference_pairs(
    model,
    tokenizer,
    dataset,
    method,
    prompt_fn,
    n_candidates=8,
    max_new_tokens=512,
    temperature=0.9,
    top_p=0.95,
    implicit_fraction=0.4,
    pair_min_gap=0.1,
    seed=42,
):
    """
    For each query, generate n_candidates completions, score with detector,
    and create (chosen, rejected) pairs from highest/lowest scoring.
    """
    detector, detector_args = get_detector_and_args(method)
    rng = np.random.default_rng(seed)
    pairs = []

    queries = dataset["query"]
    total = len(queries)

    print(f"\nGenerating preference pairs ({n_candidates} candidates/query)...")

    for idx, query in enumerate(queries):
        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  Progress: {idx + 1}/{total} (pairs so far: {len(pairs)})")

        # Decide instruction mode
        use_instruction = rng.random() >= implicit_fraction
        if use_instruction:
            messages = prompt_fn(query)
        else:
            messages = build_messages(query, include_instruction=False)

        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Generate n_candidates completions
        candidates = []
        for _ in range(n_candidates):
            response = generate_single_response(
                model, tokenizer, prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            score = detector(response, *detector_args)
            candidates.append((response, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_response, best_score = candidates[0]
        worst_response, worst_score = candidates[-1]

        # Only include if meaningful score gap
        if best_score - worst_score >= pair_min_gap:
            pairs.append({
                "prompt": prompt_text,
                "chosen": best_response,
                "rejected": worst_response,
                "chosen_score": float(best_score),
                "rejected_score": float(worst_score),
                "query": query,
                "include_instruction": use_instruction,
            })

    print(f"✓ Generated {len(pairs)} preference pairs from {total} queries")
    if len(pairs) < total:
        print(f"  ({total - len(pairs)} queries had score gap < {pair_min_gap})")

    return pairs


# ---------------------------------------------------------------------------
# DPO Trainer builder (version-compatible)
# ---------------------------------------------------------------------------

def build_dpo_config(base_args):
    """Build DPOConfig compatible with multiple TRL versions."""
    if DPOConfig is None:
        raise ImportError("TRL is required. Install with: pip install trl>=0.7.0")

    signature = inspect.signature(DPOConfig.__init__)
    accepted = {name for name in signature.parameters if name != "self"}

    config_kwargs = {}
    for key, value in base_args.items():
        if key in accepted:
            config_kwargs[key] = value

    return DPOConfig(**config_kwargs)


def build_dpo_trainer(model, training_args, train_dataset, tokenizer, ref_model=None):
    """Build DPOTrainer compatible with multiple TRL versions."""
    if DPOTrainer is None:
        raise ImportError("TRL is required. Install with: pip install trl>=0.7.0")

    signature = inspect.signature(DPOTrainer.__init__)
    accepted = {name for name in signature.parameters if name != "self"}

    trainer_kwargs = {"model": model, "args": training_args, "train_dataset": train_dataset}

    # Tokenizer arg name varies across versions
    if "tokenizer" in accepted:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in accepted:
        trainer_kwargs["processing_class"] = tokenizer

    # Reference model
    if ref_model is not None:
        if "ref_model" in accepted:
            trainer_kwargs["ref_model"] = ref_model
        elif "reference_model" in accepted:
            trainer_kwargs["reference_model"] = ref_model

    return DPOTrainer(**trainer_kwargs)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_dpo(
    model_strategy="full",
    method="acrostics",
    num_train_samples=200,
    num_epochs=3,
    batch_size=2,
    learning_rate=5e-6,
    output_dir="dpo_models",
    n_candidates=8,
    max_new_tokens=512,
    generation_temperature=0.9,
    generation_top_p=0.95,
    implicit_fraction=0.4,
    pair_min_gap=0.1,
    train_dataset_name="eli5",
    eval_splits="validation,test",
    eval_samples=50,
    eval_dataset_names="eli5,alpaca",
    eval_no_instruction=True,
    generation_batch_size=4,
    prompt_variant="paper",
    rules_variant="paper",
    base_system_prompt=None,
    system_prompt_prefix=None,
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    warm_start_model_path=None,
    beta=0.1,
    seed=42,
    save_pairs=False,
):
    """Train a model using DPO with watermarking preference pairs."""

    valid_prompt_variants = {"paper", "concise", "strict"}
    valid_rules_variants = {"paper", "minimal", "none"}
    if prompt_variant not in valid_prompt_variants:
        raise ValueError(f"Invalid prompt_variant '{prompt_variant}'.")
    if rules_variant not in valid_rules_variants:
        raise ValueError(f"Invalid rules_variant '{rules_variant}'.")

    os.environ["ICW_PROMPT_VARIANT"] = prompt_variant
    os.environ["ICW_RULES_VARIANT"] = rules_variant
    if base_system_prompt:
        os.environ["ICW_BASE_SYSTEM_PROMPT"] = base_system_prompt
    else:
        os.environ.pop("ICW_BASE_SYSTEM_PROMPT", None)
    if system_prompt_prefix:
        os.environ["ICW_SYSTEM_PROMPT_PREFIX"] = system_prompt_prefix
    else:
        os.environ.pop("ICW_SYSTEM_PROMPT_PREFIX", None)

    print("\n" + "=" * 80)
    print("DPO TRAINING FOR ICW WATERMARKING")
    print("=" * 80)
    print(f"Model Strategy: {model_strategy}")
    print(f"Method: {method}")
    print(f"Training Samples: {num_train_samples}")
    print(f"Training Dataset: {train_dataset_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"N Candidates per Prompt: {n_candidates}")
    print(f"Max New Tokens (generation): {max_new_tokens}")
    print(f"Generation Temperature: {generation_temperature}")
    print(f"Implicit Fraction: {implicit_fraction}")
    print(f"Pair Min Score Gap: {pair_min_gap}")
    print(f"DPO Beta: {beta}")
    print(f"Use LoRA: {use_lora}")
    if use_lora:
        print(f"LoRA Rank/Alpha/Dropout: {lora_rank}/{lora_alpha}/{lora_dropout}")
    if warm_start_model_path:
        print(f"Warm Start Model: {warm_start_model_path}")
    print(f"Eval Without Instructions: {eval_no_instruction}")
    print(f"Eval Splits: {eval_splits}")
    print(f"Eval Datasets: {eval_dataset_names}")
    print("=" * 80 + "\n")

    if DPOConfig is None or DPOTrainer is None:
        raise ImportError("TRL is required. Install with: pip install trl>=0.7.0")

    # Load model
    config = get_model_config(model_strategy)
    model_name = config["model_name"]

    print("Loading model...")
    model_kwargs = {
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if config.get("quantization"):
        model_kwargs["quantization_config"] = config["quantization"]
    dtype_value = config.get("dtype")

    # Resolve warm start
    resolved_source = model_name
    if warm_start_model_path:
        resolved_source = resolve_pretrained_source(warm_start_model_path, "Warm Start Model")
        print(f"Warm-starting from: {resolved_source}")

    model = load_causal_lm_with_adapter_support(
        model_name_or_path=resolved_source,
        model_kwargs=model_kwargs,
        dtype_value=dtype_value,
        is_trainable=bool(warm_start_model_path and use_lora),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_source if warm_start_model_path else model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "left":
        tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    # Handle LoRA from warm start or fresh
    if use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("PEFT is required for LoRA. Install with: pip install peft")

        # Check if warm start already has PEFT adapters
        has_peft = hasattr(model, "peft_config") or os.path.exists(
            os.path.join(str(resolved_source), "adapter_config.json")
        )
        if has_peft:
            print("✓ Existing PEFT adapters detected in warm-start checkpoint; reusing them.")
        else:
            target_modules = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            print("✓ Fresh LoRA adapters attached")

    print("✓ Model loaded successfully!\n")

    # Load reference model for DPO (the base instruct model)
    print(f"Loading reference model from: {model_name}")
    ref_model = load_causal_lm_with_dtype_fallback(
        model_name=model_name,
        model_kwargs=model_kwargs,
        dtype_value=dtype_value,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    print("✓ Reference model loaded\n")

    # Load dataset
    dataset = prepare_dataset(
        num_samples=num_train_samples,
        split="train",
        dataset_name=train_dataset_name,
        seed=seed,
    )
    if dataset is None:
        raise ValueError("Training dataset could not be loaded.")
    print(f"✓ Loaded {len(dataset)} samples\n")

    # Generate preference pairs
    prompt_fn = get_prompt_function(method)
    pairs = generate_preference_pairs(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        method=method,
        prompt_fn=prompt_fn,
        n_candidates=n_candidates,
        max_new_tokens=max_new_tokens,
        temperature=generation_temperature,
        top_p=generation_top_p,
        implicit_fraction=implicit_fraction,
        pair_min_gap=pair_min_gap,
        seed=seed,
    )

    if len(pairs) < 10:
        raise ValueError(
            f"Only {len(pairs)} preference pairs generated (need >= 10). "
            "Try increasing --n-candidates or decreasing --pair-min-gap."
        )

    # Build DPO dataset
    run_tag = f"{method}_{model_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = os.path.join(output_dir, run_tag)
    os.makedirs(run_output_dir, exist_ok=True)

    # Optionally save pairs for inspection
    if save_pairs:
        pairs_path = os.path.join(run_output_dir, "preference_pairs.json")
        with open(pairs_path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"✓ Saved {len(pairs)} preference pairs to: {pairs_path}")

    dpo_dataset = Dataset.from_dict({
        "prompt": [p["prompt"] for p in pairs],
        "chosen": [p["chosen"] for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    })

    # Compute score statistics for metadata
    chosen_scores = [p["chosen_score"] for p in pairs]
    rejected_scores = [p["rejected_score"] for p in pairs]
    mean_gap = np.mean([p["chosen_score"] - p["rejected_score"] for p in pairs])
    print(f"\nPreference pair statistics:")
    print(f"  Chosen mean score:  {np.mean(chosen_scores):.4f}")
    print(f"  Rejected mean score: {np.mean(rejected_scores):.4f}")
    print(f"  Mean score gap: {mean_gap:.4f}")
    print(f"  Total pairs: {len(pairs)}\n")

    # DPO configuration
    use_cuda = torch.cuda.is_available()
    supports_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())

    dpo_args = {
        "output_dir": run_output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "max_grad_norm": 1.0,
        "seed": seed,
        "beta": beta,
        "bf16": bool(use_cuda and supports_bf16 and config.get("quantization") is None),
        "fp16": bool(use_cuda and (not supports_bf16) and config.get("quantization") is None),
        "remove_unused_columns": False,
        "report_to": [],
    }

    # Add max_length/max_prompt_length if supported
    signature = inspect.signature(DPOConfig.__init__)
    accepted = {name for name in signature.parameters if name != "self"}
    if "max_length" in accepted:
        dpo_args["max_length"] = 2048
    if "max_prompt_length" in accepted:
        dpo_args["max_prompt_length"] = 1024

    training_args = build_dpo_config(dpo_args)

    # Initialize DPO Trainer
    print("Initializing DPO Trainer...")
    trainer = build_dpo_trainer(
        model=model,
        training_args=training_args,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        ref_model=ref_model,
    )
    print("✓ Trainer initialized\n")

    # Train
    print("Starting DPO training...")
    print("=" * 80 + "\n")
    trainer.train()

    # Save final model
    final_model_path = os.path.join(run_output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    patch_saved_model_config(final_model_path, model_name)

    metadata = {
        "base_model": model_name,
        "model_strategy": model_strategy,
        "method": method,
        "secret_sequence": secret_sequence if method == "acrostics" else None,
        "training_type": "dpo",
        "num_train_samples": num_train_samples,
        "num_preference_pairs": len(pairs),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_candidates": n_candidates,
        "max_new_tokens": max_new_tokens,
        "generation_temperature": generation_temperature,
        "implicit_fraction": implicit_fraction,
        "pair_min_gap": pair_min_gap,
        "mean_chosen_score": float(np.mean(chosen_scores)),
        "mean_rejected_score": float(np.mean(rejected_scores)),
        "mean_score_gap": float(mean_gap),
        "beta": beta,
        "use_lora": use_lora,
        "lora_rank": lora_rank if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "warm_start_model": warm_start_model_path,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    metadata_path = os.path.join(final_model_path, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Model saved to: {final_model_path}")
    print(f"✓ Metadata saved to: {metadata_path}")

    # Post-training evaluation
    # Import evaluate_model_on_split from grpo_train to reuse eval infrastructure
    try:
        from grpo_train import evaluate_model_on_split, compute_baseline_statistics
        _has_eval = True
    except ImportError:
        _has_eval = False
        warnings.warn("Could not import evaluation functions from grpo_train.py. Skipping post-training eval.")

    if _has_eval:
        print(f"\n{'=' * 80}")
        print("POST-TRAINING EVALUATION")

        # Compute baselines for eval normalization
        eval_baseline_mean, eval_baseline_std = compute_baseline_statistics(
            model, tokenizer, dataset, method,
            num_samples=min(50, len(dataset)),
            generation_batch_size=generation_batch_size,
        )

        eval_split_list = [s.strip() for s in eval_splits.split(",") if s.strip()]
        eval_dataset_list = [d.strip() for d in eval_dataset_names.split(",") if d.strip()]

        for eval_ds in eval_dataset_list:
            for split in eval_split_list:
                eval_data = prepare_dataset(
                    num_samples=eval_samples,
                    split=split,
                    dataset_name=eval_ds,
                    seed=seed,
                )
                if eval_data is None:
                    print(f"⚠️  Skipping evaluation for {eval_ds}:{split}")
                    continue

                evaluate_model_on_split(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=eval_data,
                    method=method,
                    prompt_fn=prompt_fn,
                    include_instruction=not eval_no_instruction,
                    max_samples=eval_samples,
                    output_dir=run_output_dir,
                    split_name=split,
                    dataset_name=eval_ds,
                    generation_batch_size=generation_batch_size,
                    baseline_mean=eval_baseline_mean,
                    baseline_std=eval_baseline_std,
                )

    print(f"\n{'=' * 80}")
    print("DPO TRAINING COMPLETED")
    print("=" * 80)
    print(f"\nModel: {final_model_path}")
    print(f"\nTo test: python cli.py --model-path {final_model_path} --samples 50")
    print(f"To compare: python compare_models.py --base {model_strategy} "
          f"--trained {final_model_path} --method {method}")

    return final_model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global secret_sequence
    parser = argparse.ArgumentParser(
        description="DPO Training for ICW Watermarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dpo_train.py --model full --method acrostics --samples 200
  python dpo_train.py --model full --method acrostics \\
      --warm-start-model sft_models/sft_acrostics_full_*/final_model \\
      --use-lora --n-candidates 8
        """,
    )

    parser.add_argument("--model", "-m", type=str, default="full",
                        choices=["small", "4bit", "8bit", "full", "cpu"],
                        help="Model strategy (default: full)")
    parser.add_argument("--method", type=str, required=True,
                        choices=["unicode", "initials", "lexical", "acrostics"],
                        help="Watermarking method")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of training samples (default: 200)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size (default: 2)")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    parser.add_argument("--n-candidates", type=int, default=8,
                        help="Completions per prompt for pair generation (default: 8)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens per generated completion (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature for pair generation (default: 0.9)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p for pair generation (default: 0.95)")
    parser.add_argument("--implicit-fraction", type=float, default=0.4,
                        help="Fraction of prompts WITHOUT watermark instructions (default: 0.4)")
    parser.add_argument("--pair-min-gap", type=float, default=0.1,
                        help="Minimum score gap to include a preference pair (default: 0.1)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (KL penalty strength, default: 0.1)")
    parser.add_argument("--train-dataset", type=str, default="eli5",
                        choices=["eli5", "alpaca"],
                        help="Training dataset (default: eli5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="dpo_models",
                        help="Output directory (default: dpo_models)")
    parser.add_argument("--save-pairs", action="store_true",
                        help="Save generated preference pairs to JSON")

    # Model / LoRA
    parser.add_argument("--warm-start-model", type=str, default=None,
                        help="Path to SFT checkpoint for warm starting")
    parser.add_argument("--use-lora", action="store_true", default=True,
                        help="Enable LoRA (default)")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false",
                        help="Disable LoRA")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument("--lora-target-modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
                        help="Comma-separated LoRA target modules")

    # Prompt variants
    parser.add_argument("--prompt-variant", type=str, default="paper",
                        choices=["paper", "concise", "strict"])
    parser.add_argument("--rules-variant", type=str, default="paper",
                        choices=["paper", "minimal", "none"])
    parser.add_argument(
        "--secret-sequence",
        type=str,
        default=secret_sequence,
        help="Acrostics secret string to realize (default: current configured secret)",
    )
    parser.add_argument("--base-system-prompt", type=str, default=None)
    parser.add_argument("--system-prompt-prefix", type=str, default=None)

    # Eval
    parser.add_argument("--eval-splits", type=str, default="validation,test")
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--eval-datasets", type=str, default="eli5,alpaca")
    parser.add_argument("--eval-no-instruction", dest="eval_no_instruction",
                        action="store_true", default=True)
    parser.add_argument("--eval-with-instruction", dest="eval_no_instruction",
                        action="store_false")
    parser.add_argument("--gen-batch-size", type=int, default=4)

    args = parser.parse_args()

    try:
        secret_sequence = set_acrostics_secret_sequence(args.secret_sequence)
    except ValueError as exc:
        parser.error(str(exc))

    # Validation
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.n_candidates < 2:
        parser.error("--n-candidates must be >= 2")
    if not (0.0 <= args.implicit_fraction <= 1.0):
        parser.error("--implicit-fraction must be in [0.0, 1.0]")
    if args.pair_min_gap < 0:
        parser.error("--pair-min-gap must be >= 0")
    if args.beta <= 0:
        parser.error("--beta must be > 0")

    train_dpo(
        model_strategy=args.model,
        method=args.method,
        num_train_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        n_candidates=args.n_candidates,
        max_new_tokens=args.max_new_tokens,
        generation_temperature=args.temperature,
        generation_top_p=args.top_p,
        implicit_fraction=args.implicit_fraction,
        pair_min_gap=args.pair_min_gap,
        train_dataset_name=args.train_dataset,
        eval_splits=args.eval_splits,
        eval_samples=args.eval_samples,
        eval_dataset_names=args.eval_datasets,
        eval_no_instruction=args.eval_no_instruction,
        generation_batch_size=args.gen_batch_size,
        prompt_variant=args.prompt_variant,
        rules_variant=args.rules_variant,
        base_system_prompt=args.base_system_prompt,
        system_prompt_prefix=args.system_prompt_prefix,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        warm_start_model_path=args.warm_start_model,
        beta=args.beta,
        seed=args.seed,
        save_pairs=args.save_pairs,
    )


if __name__ == "__main__":
    main()
