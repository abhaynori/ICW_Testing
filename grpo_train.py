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
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import numpy as np

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:
    GRPOConfig = None
    GRPOTrainer = None

# Import existing detector functions and prompts from main.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import detector functions
from main import (
    unicode_detector, unicode_embed_prompt,
    initials_detector, initials_embed_prompt, green_letters,
    lexical_detector, lexical_embed_prompt, green_words,
    acrostics_detector, acrostics_embed_prompt, secret_sequence
)
from memory_config import get_model_config

BASE_SYSTEM_PROMPT = "You are a helpful assistant. Provide clear, informative answers."


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


def prepare_dataset(num_samples=100, split="train"):
    """Prepare training dataset from ELI5."""
    print(f"Loading {num_samples} samples from ELI5 dataset...")
    try:
        eli5 = load_dataset("sentence-transformers/eli5", "pair", split=split)
    except Exception as exc:
        print(f"⚠️  Could not load split '{split}': {exc}")
        return None
    questions = eli5["question"][:num_samples]

    # Create dataset with queries
    dataset_dict = {"query": questions}
    dataset = Dataset.from_dict(dataset_dict)

    print(f"✓ Loaded {len(dataset)} samples")
    return dataset


def build_messages(query, prompt_fn=None, include_instruction=True):
    """Build chat messages with optional watermark instruction."""
    if include_instruction and prompt_fn is not None:
        return prompt_fn(query)

    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
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

    eval_path = os.path.join(output_dir, f"eval_{split_name}.json")
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
    eval_no_instruction=True,
    generation_batch_size=4
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
        eval_no_instruction: Disable watermark instruction during eval
    """

    print("\n" + "="*80)
    print("GRPO TRAINING FOR ICW WATERMARKING")
    print("="*80)
    print(f"Model Strategy: {model_strategy}")
    print(f"Method: {method}")
    print(f"Training Samples: {num_train_samples}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Generation Batch Size: {generation_batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Eval Without Instructions: {eval_no_instruction}")
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

    print("✓ Model loaded successfully!\n")

    # Prepare dataset
    dataset = prepare_dataset(num_train_samples)
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
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    try:
        training_args = GRPOConfig(
            **base_training_args,
            num_generation_per_prompt=4,  # Older naming in some TRL releases
        )
    except TypeError:
        training_args = GRPOConfig(
            **base_training_args,
            num_generations=4,  # Newer naming in some TRL releases
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
        if split_list:
            print("\n" + "="*80)
            print("POST-TRAINING EVALUATION")
            print("="*80)

            for split_name in split_list:
                eval_dataset = prepare_dataset(eval_samples, split=split_name)
                if eval_dataset is None:
                    print(f"⚠️  Skipping evaluation for split '{split_name}'")
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
        eval_no_instruction=args.eval_no_instruction,
        generation_batch_size=args.gen_batch_size
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
