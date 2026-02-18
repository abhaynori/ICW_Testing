#!/usr/bin/env python3
"""
GRPO Training for ICW Watermarking

This script uses Group Relative Policy Optimization (GRPO) from TRL to fine-tune
models to better follow watermarking instructions. The existing detector functions
are used as reward signals.

Usage:
    python grpo_train.py --model small --method unicode --epochs 3
    python grpo_train.py --model 4bit --method acrostics --samples 100
"""

import torch
import argparse
import os
import json
import inspect
import warnings
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import numpy as np

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:
    GRPOConfig = None
    GRPOTrainer = None

try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None

# Import existing detector functions and prompts from main.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import detector functions
from main import (
    unicode_detector, unicode_embed_prompt,
    initials_detector, initials_embed_prompt, green_letters,
    lexical_detector, lexical_embed_prompt, green_words,
    acrostics_detector, acrostics_embed_prompt, secret_sequence,
    get_base_system_prompt
)
from memory_config import get_model_config


def get_detector_and_args(method):
    detector_map = {
        'unicode': (unicode_detector, ()),
        'initials': (initials_detector, (green_letters,)),
        'lexical': (lexical_detector, (green_words,)),
        'acrostics': (acrostics_detector, (secret_sequence,))
    }

    if method not in detector_map:
        raise ValueError(f"Unknown method: {method}")

    return detector_map[method]


class WatermarkRewardFunction:
    """Reward function that uses existing detectors."""

    def __init__(self, method, baseline_mean=0.0, baseline_std=1.0):
        """
        Args:
            method: One of 'unicode', 'initials', 'lexical', 'acrostics'
            baseline_mean: Mean score from non-watermarked baseline
            baseline_std: Std of non-watermarked baseline
        """
        self.method = method
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std

        # Set up detector and arguments
        self.detector, self.detector_args = get_detector_and_args(method)

    @staticmethod
    def _to_text(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("text", "content", "completion"):
                if key in item and isinstance(item[key], str):
                    return item[key]
        if isinstance(item, (list, tuple)):
            parts = []
            for entry in item:
                if isinstance(entry, dict):
                    if "content" in entry and isinstance(entry["content"], str):
                        parts.append(entry["content"])
                    elif "text" in entry and isinstance(entry["text"], str):
                        parts.append(entry["text"])
                elif isinstance(entry, str):
                    parts.append(entry)
            if parts:
                return " ".join(parts)
        return str(item)

    def _extract_texts(self, args, kwargs):
        for key in ("completions", "texts", "responses", "outputs"):
            value = kwargs.get(key)
            if value is not None:
                return [self._to_text(v) for v in value]
        if args:
            first = args[0]
            if isinstance(first, list):
                return [self._to_text(v) for v in first]
            return [self._to_text(first)]
        return []

    def __call__(self, *args, **kwargs):
        """
        Compute rewards for a batch of generated texts.

        Args:
            *args/**kwargs: TRL passes completions in varying formats by version.

        Returns:
            rewards: Tensor of reward values
        """
        texts = self._extract_texts(args, kwargs)
        rewards = []
        for text in texts:
            # Get detector score
            score = self.detector(text, *self.detector_args)

            # Normalize by baseline (z-score style)
            # Higher scores = better watermarking
            if self.baseline_std > 0:
                normalized_score = (score - self.baseline_mean) / self.baseline_std
            else:
                normalized_score = score - self.baseline_mean

            rewards.append(normalized_score)

        return torch.tensor(rewards, dtype=torch.float32)


def get_prompt_function(method):
    """Get the prompt function for a given method."""
    if method == 'unicode':
        return unicode_embed_prompt
    elif method == 'initials':
        return initials_embed_prompt
    elif method == 'lexical':
        return lexical_embed_prompt
    elif method == 'acrostics':
        return acrostics_embed_prompt
    else:
        raise ValueError(f"Unknown method: {method}")


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


def build_grpo_config(base_training_args, generation_args, num_generations=4):
    """
    Build GRPOConfig in a way that is compatible across TRL versions.
    """
    signature = inspect.signature(GRPOConfig.__init__)
    accepted = {name for name in signature.parameters if name != "self"}

    config_kwargs = {}
    for key, value in base_training_args.items():
        if key in accepted:
            config_kwargs[key] = value

    # Generation controls vary heavily across TRL versions.
    if "generation_kwargs" in accepted:
        config_kwargs["generation_kwargs"] = dict(generation_args)
    else:
        for key, value in generation_args.items():
            if key in accepted:
                config_kwargs[key] = value
        if (
            "max_completion_length" in accepted
            and "max_new_tokens" in generation_args
            and "max_completion_length" not in config_kwargs
        ):
            config_kwargs["max_completion_length"] = generation_args["max_new_tokens"]

    # Number of completions field name changed across versions.
    if "num_generations" in accepted:
        config_kwargs["num_generations"] = num_generations
    elif "num_generation_per_prompt" in accepted:
        config_kwargs["num_generation_per_prompt"] = num_generations
    elif "num_return_sequences" in accepted:
        config_kwargs["num_return_sequences"] = num_generations

    dropped = sorted(
        set(base_training_args).union(generation_args) - set(config_kwargs)
    )
    if dropped:
        warnings.warn(
            "Ignoring unsupported GRPOConfig args for this TRL version: "
            + ", ".join(dropped)
        )

    return GRPOConfig(**config_kwargs)


def prepare_dataset(num_samples=100, split="train", dataset_name="eli5"):
    """Prepare training/eval dataset from ELI5 or Alpaca."""
    dataset_key = dataset_name.strip().lower()
    print(f"Loading {num_samples} samples from {dataset_key} ({split})...")
    try:
        if dataset_key == "eli5":
            dataset = load_dataset("sentence-transformers/eli5", "pair", split=split)
            queries = dataset["question"][:num_samples]
        elif dataset_key == "alpaca":
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            start, end = _slice_indices_for_split(len(dataset), split)
            sampled = dataset.select(range(start, min(end, start + num_samples)))
            queries = [_format_alpaca_query(row) for row in sampled]
        else:
            raise ValueError("dataset_name must be 'eli5' or 'alpaca'")
    except Exception as exc:
        print(f"⚠️  Could not load dataset '{dataset_key}' split '{split}': {exc}")
        return None

    # Create dataset with queries
    dataset_dict = {"query": queries}
    dataset = Dataset.from_dict(dataset_dict)

    print(f"✓ Loaded {len(dataset)} samples")
    return dataset


def build_messages(query, prompt_fn=None, include_instruction=True):
    """Build chat messages with optional watermark instruction."""
    if include_instruction and prompt_fn is not None:
        return prompt_fn(query)

    return [
        {"role": "system", "content": get_base_system_prompt()},
        {"role": "user", "content": query}
    ]


def generate_responses_batch(
    model,
    tokenizer,
    messages_batch,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9
):
    """Generate a batch of responses from chat messages."""
    prompt_texts = [
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        for messages in messages_batch
    ]

    try:
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id
            )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(messages_batch) == 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"⚠️  OOM at generation batch size {len(messages_batch)}. Falling back to single-sample generation.")
        fallback_responses = []
        for messages in messages_batch:
            fallback_responses.extend(
                generate_responses_batch(
                    model=model,
                    tokenizer=tokenizer,
                    messages_batch=[messages],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            )
        return fallback_responses

    responses = []
    attention_mask = encoded["attention_mask"]
    for idx in range(outputs.shape[0]):
        prompt_len = int(attention_mask[idx].sum().item())
        responses.append(
            tokenizer.decode(outputs[idx][prompt_len:], skip_special_tokens=True)
        )

    return responses


def compute_baseline_statistics(
    model,
    tokenizer,
    dataset,
    method,
    num_samples=50,
    generation_batch_size=4
):
    """
    Compute baseline statistics from non-watermarked generations.
    These are used to normalize rewards.
    """
    print(f"\nComputing baseline statistics for {method}...")

    detector, detector_args = get_detector_and_args(method)

    scores = []

    samples = dataset.select(range(min(num_samples, len(dataset))))
    queries = samples["query"]
    total = len(queries)

    for start in range(0, total, generation_batch_size):
        end = min(start + generation_batch_size, total)
        batch_queries = queries[start:end]
        messages_batch = [build_messages(query, include_instruction=False) for query in batch_queries]
        responses = generate_responses_batch(
            model=model,
            tokenizer=tokenizer,
            messages_batch=messages_batch,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )

        for response in responses:
            score = detector(response, *detector_args)
            scores.append(score)

        processed = end
        if processed % 10 == 0 or processed == total:
            print(f"  Progress: {processed}/{total}")

    mean = np.mean(scores)
    std = np.std(scores)

    print(f"✓ Baseline computed: mean={mean:.3f}, std={std:.3f}")
    print(f"  This will be used to normalize rewards during training")

    return mean, std


def evaluate_model_on_split(
    model,
    tokenizer,
    dataset,
    method,
    prompt_fn,
    include_instruction,
    max_samples,
    output_dir,
    split_name,
    dataset_name="eli5",
    generation_batch_size=4,
    baseline_mean=None,
    baseline_std=None
):
    """Evaluate detector scores on a dataset split."""
    detector, detector_args = get_detector_and_args(method)
    scores = []
    records = []

    samples = dataset.select(range(min(max_samples, len(dataset))))

    model.eval()
    queries = samples["query"]
    total = len(queries)

    for start in range(0, total, generation_batch_size):
        end = min(start + generation_batch_size, total)
        batch_queries = queries[start:end]
        messages_batch = [
            build_messages(query, prompt_fn=prompt_fn, include_instruction=include_instruction)
            for query in batch_queries
        ]
        responses = generate_responses_batch(
            model=model,
            tokenizer=tokenizer,
            messages_batch=messages_batch,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )

        for query, response in zip(batch_queries, responses):
            score = detector(response, *detector_args)
            scores.append(score)
            records.append({
                "query": query,
                "response": response,
                "score": score
            })

        processed = end
        if processed % 10 == 0 or processed == total:
            print(f"  Eval progress ({split_name}): {processed}/{total}")

    mean = float(np.mean(scores)) if scores else 0.0
    std = float(np.std(scores)) if scores else 0.0
    normalized_mean = None
    if baseline_mean is not None:
        if baseline_std is not None and baseline_std > 0:
            normalized_mean = float((mean - baseline_mean) / baseline_std)
        else:
            normalized_mean = float(mean - baseline_mean)

    summary = {
        "dataset": dataset_name,
        "split": split_name,
        "method": method,
        "include_instruction": include_instruction,
        "num_samples": len(scores),
        "mean_score": mean,
        "std_score": std
    }
    if normalized_mean is not None:
        summary["mean_score_vs_baseline_z"] = normalized_mean

    output = {
        "summary": summary,
        "samples": records
    }

    eval_path = os.path.join(output_dir, f"eval_{dataset_name}_{split_name}.json")
    with open(eval_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✓ Saved {split_name} evaluation to: {eval_path}")
    print(f"  Mean score: {mean:.4f}, Std: {std:.4f}")
    if normalized_mean is not None:
        print(f"  Mean score vs training baseline (z): {normalized_mean:.4f}")

    return summary


def train_grpo(
    model_strategy="small",
    method="unicode",
    num_train_samples=100,
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-5,
    output_dir="grpo_models",
    eval_splits="validation,test",
    eval_samples=50,
    train_dataset_name="eli5",
    eval_dataset_names="eli5,alpaca",
    eval_no_instruction=True,
    generation_batch_size=4,
    prompt_variant="paper",
    rules_variant="paper",
    base_system_prompt=None,
    system_prompt_prefix=None,
    use_lora=False,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"
):
    """
    Train a model using GRPO with watermarking rewards.

    Args:
        model_strategy: Model configuration (small, 4bit, etc.)
        method: Watermarking method (unicode, initials, lexical, acrostics)
        num_train_samples: Number of training samples
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        output_dir: Directory to save trained models
        eval_splits: Comma-separated list of dataset splits to evaluate
        eval_samples: Number of samples for each evaluation split
        train_dataset_name: Dataset for GRPO training
        eval_dataset_names: Comma-separated list of datasets for post-training eval
        eval_no_instruction: Disable watermark instruction during eval
    """

    valid_prompt_variants = {"paper", "concise", "strict"}
    valid_rules_variants = {"paper", "minimal", "none"}
    if prompt_variant not in valid_prompt_variants:
        raise ValueError(
            f"Invalid prompt_variant '{prompt_variant}'. "
            f"Choose from: {', '.join(sorted(valid_prompt_variants))}"
        )
    if rules_variant not in valid_rules_variants:
        raise ValueError(
            f"Invalid rules_variant '{rules_variant}'. "
            f"Choose from: {', '.join(sorted(valid_rules_variants))}"
        )

    # Configure prompt/rule variants consumed by main.py prompt builders.
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

    print("\n" + "="*80)
    print("GRPO TRAINING FOR ICW WATERMARKING")
    print("="*80)
    print(f"Model Strategy: {model_strategy}")
    print(f"Method: {method}")
    print(f"Training Samples: {num_train_samples}")
    print(f"Training Dataset: {train_dataset_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Generation Batch Size: {generation_batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Prompt Variant: {prompt_variant}")
    print(f"Rules Variant: {rules_variant}")
    print(f"Use LoRA: {use_lora}")
    if use_lora:
        print(f"LoRA Rank/Alpha/Dropout: {lora_rank}/{lora_alpha}/{lora_dropout}")
        print(f"LoRA Target Modules: {lora_target_modules}")
    print(f"Eval Without Instructions: {eval_no_instruction}")
    print(f"Eval Splits: {eval_splits}")
    print(f"Eval Datasets: {eval_dataset_names}")
    print("="*80 + "\n")

    if GRPOConfig is None or GRPOTrainer is None:
        raise ImportError(
            "TRL with GRPO support is required for training. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    # Load model and tokenizer
    print("Loading base model...")
    config = get_model_config(model_strategy)
    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Decoder-only models should be left-padded when batching generations.
    if getattr(tokenizer, "padding_side", "right") != "left":
        tokenizer.padding_side = "left"

    model_kwargs = {
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True
    }

    # Note: GRPO training works best with full precision or 8-bit
    # 4-bit quantization may not work well with gradient updates
    if model_strategy == "4bit":
        print("⚠️  Warning: 4-bit models may not train well. Consider using 'small' or '8bit'")

    if config.get("quantization"):
        model_kwargs["quantization_config"] = config["quantization"]
    elif config.get("dtype"):
        model_kwargs["torch_dtype"] = config["dtype"]

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if use_lora:
        if LoraConfig is None or get_peft_model is None or TaskType is None:
            raise ImportError(
                "PEFT is required for LoRA training. Install dependencies with: pip install peft"
            )
        target_modules = [item.strip() for item in lora_target_modules.split(",") if item.strip()]
        if not target_modules:
            raise ValueError("LoRA target modules cannot be empty.")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        print("✓ LoRA adapters attached")

    print("✓ Model loaded successfully!\n")

    # Prepare dataset
    dataset = prepare_dataset(
        num_samples=num_train_samples,
        split="train",
        dataset_name=train_dataset_name
    )
    if dataset is None:
        raise ValueError("Training dataset could not be loaded.")

    # Compute baseline statistics
    baseline_mean, baseline_std = compute_baseline_statistics(
        model,
        tokenizer,
        dataset,
        method,
        num_samples=min(50, num_train_samples),
        generation_batch_size=generation_batch_size
    )

    # Create reward function
    reward_fn = WatermarkRewardFunction(method, baseline_mean, baseline_std)

    # Get prompt function
    prompt_fn = get_prompt_function(method)

    # Prepare prompts for training
    def tokenize_function(examples):
        """Convert queries to prompts with watermarking instructions."""
        prompts = []
        for query in examples["query"]:
            messages = prompt_fn(query)
            # Format as chat template
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            prompts.append(prompt_text)

        return {"prompt": prompts}

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # GRPO Configuration
    base_training_args = {
        "output_dir": os.path.join(output_dir, f"{method}_{model_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "max_grad_norm": 1.0,
        "seed": 42,
    }
    generation_args = {
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    training_args = build_grpo_config(
        base_training_args=base_training_args,
        generation_args=generation_args,
        num_generations=4,
    )

    # Initialize GRPO Trainer
    print("\nInitializing GRPO Trainer...")

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset,
        "tokenizer": tokenizer,
    }

    try:
        trainer = GRPOTrainer(
            **trainer_kwargs,
            reward_function=reward_fn,
        )
    except TypeError as exc:
        # Compatibility fallback for TRL versions using reward_funcs instead.
        if "reward_function" not in str(exc):
            raise
        try:
            trainer = GRPOTrainer(
                **trainer_kwargs,
                reward_funcs=reward_fn,
            )
        except TypeError:
            trainer = GRPOTrainer(
                **trainer_kwargs,
                reward_funcs=[reward_fn],
            )

    print("✓ Trainer initialized\n")

    # Train
    print("Starting training...")
    print("="*80 + "\n")

    trainer.train()

    print("\n" + "="*80)
    print("Training completed!")
    print("="*80 + "\n")

    # Save final model
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    print(f"Saving final model to {final_model_path}...")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Save training metadata
    metadata = {
        "base_model": model_name,
        "model_strategy": model_strategy,
        "method": method,
        "num_train_samples": num_train_samples,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_dataset_name": train_dataset_name,
        "eval_dataset_names": eval_dataset_names,
        "prompt_variant": prompt_variant,
        "rules_variant": rules_variant,
        "base_system_prompt": get_base_system_prompt(),
        "system_prompt_prefix": system_prompt_prefix or "",
        "use_lora": use_lora,
        "lora_rank": lora_rank if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "lora_dropout": lora_dropout if use_lora else None,
        "lora_target_modules": lora_target_modules if use_lora else "",
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "timestamp": datetime.now().isoformat()
    }

    metadata_path = os.path.join(final_model_path, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Model saved to: {final_model_path}")
    print(f"✓ Metadata saved to: {metadata_path}")

    # Optional evaluation on validation/test splits without watermark instructions
    if eval_splits:
        split_list = [s.strip() for s in eval_splits.split(",") if s.strip()]
        dataset_list = [d.strip().lower() for d in eval_dataset_names.split(",") if d.strip()]
        valid_datasets = {"eli5", "alpaca"}
        for dataset_name in dataset_list:
            if dataset_name not in valid_datasets:
                warnings.warn(
                    f"Unknown eval dataset '{dataset_name}', skipping. "
                    f"Supported: {', '.join(sorted(valid_datasets))}"
                )
        if split_list:
            print("\n" + "="*80)
            print("POST-TRAINING EVALUATION")
            print("="*80)

            for dataset_name in dataset_list:
                if dataset_name not in valid_datasets:
                    continue
                for split_name in split_list:
                    eval_dataset = prepare_dataset(
                        num_samples=eval_samples,
                        split=split_name,
                        dataset_name=dataset_name
                    )
                    if eval_dataset is None:
                        print(f"⚠️  Skipping evaluation for {dataset_name}:{split_name}")
                        continue

                    evaluate_model_on_split(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=eval_dataset,
                        method=method,
                        prompt_fn=prompt_fn,
                        include_instruction=not eval_no_instruction,
                        max_samples=eval_samples,
                        output_dir=training_args.output_dir,
                        split_name=split_name,
                        dataset_name=dataset_name,
                        generation_batch_size=generation_batch_size,
                        baseline_mean=baseline_mean,
                        baseline_std=baseline_std
                    )

    return final_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train ICW watermarking models using GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train small model on Unicode watermarking
  python grpo_train.py --model small --method unicode --epochs 3

  # Train with more samples for better results
  python grpo_train.py --model small --method acrostics --samples 200 --epochs 5

  # Train on all methods (run sequentially)
  for method in unicode initials lexical acrostics; do
    python grpo_train.py --model small --method $method --epochs 3
  done
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='small',
        choices=['small', '4bit', '8bit', 'full', 'cpu'],
        help='Model strategy (4bit is allowed but generally unstable for training)'
    )

    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['unicode', 'initials', 'lexical', 'acrostics'],
        help='Watermarking method to train on'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of training samples (default: 100)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )

    parser.add_argument(
        '--eval-splits',
        type=str,
        default='validation,test',
        help='Comma-separated dataset splits to evaluate (default: validation,test)'
    )

    parser.add_argument(
        '--eval-samples',
        type=int,
        default=50,
        help='Number of evaluation samples per split (default: 50)'
    )

    parser.add_argument(
        '--train-dataset',
        type=str,
        default='eli5',
        choices=['eli5', 'alpaca'],
        help='Dataset used for GRPO training (default: eli5)'
    )

    parser.add_argument(
        '--eval-datasets',
        type=str,
        default='eli5,alpaca',
        help='Comma-separated datasets for post-training evaluation (default: eli5,alpaca)'
    )

    parser.add_argument(
        '--eval-no-instruction',
        dest='eval_no_instruction',
        action='store_true',
        default=True,
        help='Disable watermarking instructions during eval (default behavior)'
    )

    parser.add_argument(
        '--eval-with-instruction',
        dest='eval_no_instruction',
        action='store_false',
        help='Enable watermarking instructions during eval'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training (default: 4)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate (default: 1e-5)'
    )

    parser.add_argument(
        '--prompt-variant',
        type=str,
        default='paper',
        choices=['paper', 'concise', 'strict'],
        help='Instruction prompt style variant (default: paper)'
    )

    parser.add_argument(
        '--rules-variant',
        type=str,
        default='paper',
        choices=['paper', 'minimal', 'none'],
        help='Rules variant for system prompts (default: paper)'
    )

    parser.add_argument(
        '--base-system-prompt',
        type=str,
        default=None,
        help='Override baseline non-watermarked system prompt'
    )

    parser.add_argument(
        '--system-prompt-prefix',
        type=str,
        default=None,
        help='Prefix injected at the top of watermarking system prompts'
    )

    parser.add_argument(
        '--use-lora',
        action='store_true',
        help='Enable LoRA adapters during GRPO training'
    )

    parser.add_argument(
        '--lora-rank',
        type=int,
        default=16,
        help='LoRA rank r (default: 16)'
    )

    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha (default: 32)'
    )

    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=0.05,
        help='LoRA dropout (default: 0.05)'
    )

    parser.add_argument(
        '--lora-target-modules',
        type=str,
        default='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj',
        help='Comma-separated LoRA target modules'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='grpo_models',
        help='Output directory for trained models (default: grpo_models)'
    )

    parser.add_argument(
        '--gen-batch-size',
        type=int,
        default=4,
        help='Batch size for generation during baseline/eval (default: 4)'
    )

    args = parser.parse_args()

    if args.gen_batch_size < 1:
        parser.error("--gen-batch-size must be >= 1")
    if args.lora_rank < 1:
        parser.error("--lora-rank must be >= 1")
    if args.lora_alpha < 1:
        parser.error("--lora-alpha must be >= 1")
    if args.lora_dropout < 0 or args.lora_dropout >= 1:
        parser.error("--lora-dropout must be in [0, 1)")

    # Train
    model_path = train_grpo(
        model_strategy=args.model,
        method=args.method,
        num_train_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        eval_splits=args.eval_splits,
        eval_samples=args.eval_samples,
        train_dataset_name=args.train_dataset,
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
        lora_target_modules=args.lora_target_modules
    )

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Test the trained model:")
    print(f"   python cli.py --model-path {model_path} --samples 50")
    print(f"   # Validation/test-style evaluation without watermark instructions")
    print(f"   python cli.py --model-path {model_path} --samples 50 --split test --no-wm-instruction")
    print(f"\n2. Compare with base model:")
    print(f"   python compare_models.py --base {args.model} --trained {model_path} --method {args.method} --split test --no-wm-instruction")
    print(f"\n3. Train on other methods:")
    print(f"   python grpo_train.py --model {args.model} --method <other_method> --epochs {args.epochs}")
    print()


if __name__ == "__main__":
    main()
