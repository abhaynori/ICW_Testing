#!/usr/bin/env python3
"""
SFT training script for generating warm-start checkpoints for GRPO.

Usage examples:
  python sft_train.py --model small --method lexical --samples 500 --epochs 1
  python sft_train.py --model small --method acrostics --train-dataset alpaca --samples 1000 --use-lora
"""

import argparse
import json
import os
import re
import warnings
from datetime import datetime

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

from main import (
    acrostics_embed_prompt,
    get_base_system_prompt,
    initials_embed_prompt,
    lexical_embed_prompt,
    unicode_embed_prompt,
)
from memory_config import get_model_config


def get_prompt_function(method):
    if method == "unicode":
        return unicode_embed_prompt
    if method == "initials":
        return initials_embed_prompt
    if method == "lexical":
        return lexical_embed_prompt
    if method == "acrostics":
        return acrostics_embed_prompt
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


def _first_nonempty_text(value):
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, list):
        for item in value:
            text = _first_nonempty_text(item)
            if text:
                return text
    if isinstance(value, dict):
        for key in ("text", "answer", "output", "response", "content"):
            if key in value:
                text = _first_nonempty_text(value[key])
                if text:
                    return text
        for nested_value in value.values():
            text = _first_nonempty_text(nested_value)
            if text:
                return text
    return None


def _extract_eli5_answer(row):
    for key in ("answer", "answer1", "answer2", "answers", "output", "response", "text"):
        if key in row:
            text = _first_nonempty_text(row[key])
            if text:
                return text
    return None


def _clean_target(text):
    if text is None:
        return None
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def load_sft_pairs(dataset_name="eli5", split="train", num_samples=500):
    dataset_key = dataset_name.strip().lower()
    pairs = []

    if dataset_key == "alpaca":
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        start, end = _slice_indices_for_split(len(dataset), split)
        subset = dataset.select(range(start, end))
        for row in subset:
            query = _format_alpaca_query(row)
            target = _clean_target(row.get("output"))
            if query and target:
                pairs.append({"query": query, "target": target})
            if len(pairs) >= num_samples:
                break
        return pairs

    if dataset_key == "eli5":
        dataset = load_dataset("sentence-transformers/eli5", "pair", split=split)
        for row in dataset:
            query = (row.get("question") or "").strip()
            target = _clean_target(_extract_eli5_answer(row))
            if query and target:
                pairs.append({"query": query, "target": target})
            if len(pairs) >= num_samples:
                break
        return pairs

    raise ValueError(f"Unsupported dataset '{dataset_name}'. Use one of: eli5, alpaca")


def build_messages(query, prompt_fn=None, include_instruction=True):
    if include_instruction and prompt_fn is not None:
        return prompt_fn(query)
    return [
        {"role": "system", "content": get_base_system_prompt()},
        {"role": "user", "content": query},
    ]


def render_chat(tokenizer, messages, add_generation_prompt=False):
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    except Exception:
        lines = []
        for message in messages:
            role = message.get("role", "user").strip().lower()
            content = (message.get("content") or "").strip()
            if role == "system":
                lines.append(f"System: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            else:
                lines.append(f"User: {content}")
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n".join(lines)


def prepare_sft_dataset(
    records,
    tokenizer,
    prompt_fn,
    include_instruction,
    max_length=1024,
):
    features = []
    skipped = 0

    for row in records:
        query = row["query"]
        target = row["target"]

        prompt_messages = build_messages(
            query=query,
            prompt_fn=prompt_fn,
            include_instruction=include_instruction,
        )
        full_messages = prompt_messages + [{"role": "assistant", "content": target}]

        prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = render_chat(tokenizer, full_messages, add_generation_prompt=False)

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        prompt_len = len(prompt_tokens["input_ids"])

        if prompt_len >= len(input_ids):
            skipped += 1
            continue

        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    if skipped:
        print(f"⚠️  Skipped {skipped} samples with no trainable assistant tokens after truncation.")

    if not features:
        raise ValueError("No valid SFT training examples were created.")

    return Dataset.from_list(features)


class SFTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer pad_token_id is required for batching.")

        input_ids = []
        attention_masks = []
        labels = []
        for item in features:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_masks.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch


def train_sft(
    model_strategy="small",
    method="unicode",
    num_train_samples=500,
    num_epochs=1,
    batch_size=2,
    learning_rate=2e-5,
    output_dir="sft_models",
    train_dataset_name="eli5",
    train_split="train",
    include_instruction=True,
    max_length=1024,
    save_steps=100,
    logging_steps=10,
    prompt_variant="paper",
    rules_variant="paper",
    base_system_prompt=None,
    system_prompt_prefix=None,
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
):
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
    print("SFT TRAINING FOR GRPO WARM START")
    print("=" * 80)
    print(f"Model Strategy: {model_strategy}")
    print(f"Method: {method}")
    print(f"Training Samples: {num_train_samples}")
    print(f"Training Dataset: {train_dataset_name}:{train_split}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Max Sequence Length: {max_length}")
    print(f"Include WM Instruction: {include_instruction}")
    print(f"Use LoRA: {use_lora}")
    print("=" * 80 + "\n")

    config = get_model_config(model_strategy)
    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if getattr(tokenizer, "padding_side", "right") != "right":
        tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if config.get("quantization"):
        model_kwargs["quantization_config"] = config["quantization"]
    elif config.get("dtype"):
        model_kwargs["torch_dtype"] = config["dtype"]

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    quantized = bool(config.get("quantization"))
    if quantized and not use_lora:
        raise ValueError(
            "Quantized strategies ('4bit'/'8bit') require --use-lora for SFT."
        )

    if use_lora:
        if LoraConfig is None or get_peft_model is None or TaskType is None:
            raise ImportError(
                "PEFT is required for LoRA SFT. Install with: pip install peft"
            )
        if quantized and prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(model)

        target_modules = [item.strip() for item in lora_target_modules.split(",") if item.strip()]
        if not target_modules:
            raise ValueError("LoRA target modules cannot be empty.")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        print("✓ LoRA adapters attached")

    prompt_fn = get_prompt_function(method)
    raw_records = load_sft_pairs(
        dataset_name=train_dataset_name,
        split=train_split,
        num_samples=num_train_samples,
    )
    if len(raw_records) < num_train_samples:
        warnings.warn(
            f"Requested {num_train_samples} samples, but only found {len(raw_records)} usable samples."
        )
    print(f"✓ Loaded {len(raw_records)} SFT pairs")

    train_dataset = prepare_sft_dataset(
        records=raw_records,
        tokenizer=tokenizer,
        prompt_fn=prompt_fn,
        include_instruction=include_instruction,
        max_length=max_length,
    )
    print(f"✓ Built {len(train_dataset)} tokenized training examples")

    run_output_dir = os.path.join(
        output_dir,
        f"sft_{method}_{model_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    use_cuda = torch.cuda.is_available()
    supports_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=logging_steps,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=bool(use_cuda and not supports_bf16),
        bf16=supports_bf16,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=SFTDataCollator(tokenizer),
    )

    print("\nStarting SFT training...")
    print("=" * 80 + "\n")
    trainer.train()

    final_model_path = os.path.join(run_output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    metadata = {
        "base_model": model_name,
        "model_strategy": model_strategy,
        "method": method,
        "num_train_samples": num_train_samples,
        "num_train_examples": len(train_dataset),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "train_dataset_name": train_dataset_name,
        "train_split": train_split,
        "include_instruction": include_instruction,
        "prompt_variant": prompt_variant,
        "rules_variant": rules_variant,
        "base_system_prompt": get_base_system_prompt(),
        "system_prompt_prefix": system_prompt_prefix or "",
        "use_lora": use_lora,
        "lora_rank": lora_rank if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "lora_dropout": lora_dropout if use_lora else None,
        "lora_target_modules": lora_target_modules if use_lora else "",
        "timestamp": datetime.now().isoformat(),
    }
    metadata_path = os.path.join(final_model_path, "sft_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("SFT TRAINING COMPLETED")
    print("=" * 80)
    print(f"Model saved to: {final_model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("\nUse this path for GRPO warm start:")
    print(f"  --warm-start-model {final_model_path}")
    print()

    return final_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train SFT checkpoints for GRPO warm starts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sft_train.py --model small --method unicode --samples 500 --epochs 1
  python sft_train.py --model small --method lexical --train-dataset alpaca --samples 1000 --use-lora
  python sft_train.py --model small --method acrostics --no-wm-instruction
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="small",
        choices=["small", "4bit", "8bit", "full", "cpu"],
        help="Model strategy (default: small)",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["unicode", "initials", "lexical", "acrostics"],
        help="Watermarking method used for SFT prompts",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of SFT training pairs (default: 500)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size (default: 2)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="eli5",
        choices=["eli5", "alpaca"],
        help="Dataset used for SFT training (default: eli5)",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length for tokenization (default: 1024)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Checkpoint save frequency in steps (default: 100)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency in steps (default: 10)",
    )
    parser.add_argument(
        "--wm-instruction",
        dest="include_instruction",
        action="store_true",
        default=True,
        help="Include watermarking instruction in SFT prompts (default behavior)",
    )
    parser.add_argument(
        "--no-wm-instruction",
        dest="include_instruction",
        action="store_false",
        help="Train on baseline assistant prompts without watermarking instruction",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default="paper",
        choices=["paper", "concise", "strict"],
        help="Instruction prompt style variant (default: paper)",
    )
    parser.add_argument(
        "--rules-variant",
        type=str,
        default="paper",
        choices=["paper", "minimal", "none"],
        help="Rules variant for system prompts (default: paper)",
    )
    parser.add_argument(
        "--base-system-prompt",
        type=str,
        default=None,
        help="Override baseline non-watermarked system prompt",
    )
    parser.add_argument(
        "--system-prompt-prefix",
        type=str,
        default=None,
        help="Prefix injected at the top of watermarking system prompts",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Enable LoRA adapters during SFT (default behavior)",
    )
    parser.add_argument(
        "--no-lora",
        dest="use_lora",
        action="store_false",
        help="Disable LoRA and full-fine-tune all model parameters",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank r (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma-separated LoRA target modules",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sft_models",
        help="Output directory for SFT checkpoints (default: sft_models)",
    )

    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_length < 128:
        parser.error("--max-length must be >= 128")
    if args.save_steps < 1:
        parser.error("--save-steps must be >= 1")
    if args.logging_steps < 1:
        parser.error("--logging-steps must be >= 1")
    if args.lora_rank < 1:
        parser.error("--lora-rank must be >= 1")
    if args.lora_alpha < 1:
        parser.error("--lora-alpha must be >= 1")
    if args.lora_dropout < 0 or args.lora_dropout >= 1:
        parser.error("--lora-dropout must be in [0, 1)")

    train_sft(
        model_strategy=args.model,
        method=args.method,
        num_train_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        train_dataset_name=args.train_dataset,
        train_split=args.train_split,
        include_instruction=args.include_instruction,
        max_length=args.max_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        prompt_variant=args.prompt_variant,
        rules_variant=args.rules_variant,
        base_system_prompt=args.base_system_prompt,
        system_prompt_prefix=args.system_prompt_prefix,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )


if __name__ == "__main__":
    main()
