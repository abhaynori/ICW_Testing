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
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
import numpy as np

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
        if method == 'unicode':
            self.detector = unicode_detector
            self.detector_args = ()
        elif method == 'initials':
            self.detector = initials_detector
            self.detector_args = (green_letters,)
        elif method == 'lexical':
            self.detector = lexical_detector
            self.detector_args = (green_words,)
        elif method == 'acrostics':
            self.detector = acrostics_detector
            self.detector_args = (secret_sequence,)
        else:
            raise ValueError(f"Unknown method: {method}")

    def __call__(self, texts):
        """
        Compute rewards for a batch of generated texts.

        Args:
            texts: List of generated text strings

        Returns:
            rewards: Tensor of reward values
        """
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
    eli5 = load_dataset("sentence-transformers/eli5", "pair", split=split)
    questions = eli5["question"][:num_samples]

    # Create dataset with queries
    dataset_dict = {"query": questions}
    dataset = Dataset.from_dict(dataset_dict)

    print(f"✓ Loaded {len(dataset)} samples")
    return dataset


def compute_baseline_statistics(model, tokenizer, dataset, method, num_samples=50):
    """
    Compute baseline statistics from non-watermarked generations.
    These are used to normalize rewards.
    """
    print(f"\nComputing baseline statistics for {method}...")

    detector_map = {
        'unicode': (unicode_detector, ()),
        'initials': (initials_detector, (green_letters,)),
        'lexical': (lexical_detector, (green_words,)),
        'acrostics': (acrostics_detector, (secret_sequence,))
    }

    detector, detector_args = detector_map[method]

    scores = []

    # Generate non-watermarked samples
    for i, example in enumerate(dataset.select(range(min(num_samples, len(dataset))))):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_samples}")

        query = example["query"]

        # Non-watermarked prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Provide clear, informative answers."},
            {"role": "user", "content": query}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        score = detector(response, *detector_args)
        scores.append(score)

    mean = np.mean(scores)
    std = np.std(scores)

    print(f"✓ Baseline computed: mean={mean:.3f}, std={std:.3f}")
    print(f"  This will be used to normalize rewards during training")

    return mean, std


def train_grpo(
    model_strategy="small",
    method="unicode",
    num_train_samples=100,
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-5,
    output_dir="grpo_models"
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
    """

    print("\n" + "="*80)
    print("GRPO TRAINING FOR ICW WATERMARKING")
    print("="*80)
    print(f"Model Strategy: {model_strategy}")
    print(f"Method: {method}")
    print(f"Training Samples: {num_train_samples}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("="*80 + "\n")

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

    # Compute baseline statistics
    baseline_mean, baseline_std = compute_baseline_statistics(
        model, tokenizer, dataset, method, num_samples=min(50, num_train_samples)
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
    training_args = GRPOConfig(
        output_dir=os.path.join(output_dir, f"{method}_{model_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_grad_norm=1.0,
        seed=42,
        # GRPO specific
        num_generation_per_prompt=4,  # Generate multiple completions per prompt
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
    )

    # Initialize GRPO Trainer
    print("\nInitializing GRPO Trainer...")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        reward_function=reward_fn,
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
        choices=['small', '8bit', 'cpu'],
        help='Model strategy (Note: 4bit not recommended for training)'
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

    args = parser.parse_args()

    # Train
    model_path = train_grpo(
        model_strategy=args.model,
        method=args.method,
        num_train_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Test the trained model:")
    print(f"   python cli.py --model-path {model_path} --method {args.method} --samples 50")
    print(f"\n2. Compare with base model:")
    print(f"   python compare_models.py --base {args.model} --trained {model_path} --method {args.method}")
    print(f"\n3. Train on other methods:")
    print(f"   python grpo_train.py --model {args.model} --method <other_method> --epochs {args.epochs}")
    print()


if __name__ == "__main__":
    main()
