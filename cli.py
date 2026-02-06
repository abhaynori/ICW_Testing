#!/usr/bin/env python3
"""
Command-line interface for ICW Testing.

Usage:
    python cli.py --model 4bit --temperature 0.7 --samples 50
    python cli.py --list-models
    python cli.py --model qwen-1.5b --temperature 0.5
"""

import argparse
import sys
import os

# Import the main experiment code
from memory_config import get_model_config, MODEL_PROFILES

def list_models():
    """Display available model configurations."""
    print("\n" + "="*80)
    print("AVAILABLE MODEL CONFIGURATIONS")
    print("="*80)

    configs = {
        "small": "Qwen 1.5B (fast, low memory)",
        "4bit": "Qwen 7B with 4-bit quantization (~4GB VRAM)",
        "8bit": "Qwen 7B with 8-bit quantization (~7GB VRAM)",
        "full": "Qwen 7B full precision (~13GB VRAM)",
        "cpu": "Qwen 1.5B CPU-only mode (slow)",
        "auto": "Automatically detect best configuration"
    }

    print("\nQuick configurations:")
    for key, desc in configs.items():
        print(f"  {key:10} - {desc}")

    print("\n" + "="*80)
    print("MODEL DETAILS")
    print("="*80)

    for key, profile in MODEL_PROFILES.items():
        print(f"\n{key}:")
        print(f"  Model: {profile['name']}")
        print(f"  Full precision: {profile['full_memory']}")
        print(f"  4-bit quantized: {profile['4bit_memory']}")
        print(f"  8-bit quantized: {profile['8bit_memory']}")

    print("\n" + "="*80)
    print("EXAMPLES")
    print("="*80)
    print("\n  # Run with 4-bit quantized model at temperature 0.7")
    print("  python cli.py --model 4bit --temperature 0.7 --samples 50")
    print("\n  # Run with small model, low temperature, quick test")
    print("  python cli.py --model small --temperature 0.3 --samples 20")
    print("\n  # Run with auto-detection")
    print("  python cli.py --model auto --samples 100")
    print("\n  # List available models")
    print("  python cli.py --list-models")
    print()

def run_experiment(
    model_strategy,
    temperature,
    num_samples,
    output_dir,
    model_path=None,
    disable_wm_instruction=False,
    data_split="train",
    generation_batch_size=4
):
    """Run the ICW experiment with specified configuration."""

    # Validate inputs
    valid_strategies = ["small", "4bit", "8bit", "full", "cpu", "auto"]
    if model_strategy not in valid_strategies and model_path is None:
        print(f"Error: Invalid model strategy '{model_strategy}'")
        print(f"Valid options: {', '.join(valid_strategies)}")
        sys.exit(1)

    if not (0 < temperature <= 2.0):
        print(f"Error: Temperature must be between 0 and 2.0 (got {temperature})")
        sys.exit(1)

    if not (1 <= num_samples <= 1000):
        print(f"Error: Number of samples must be between 1 and 1000 (got {num_samples})")
        sys.exit(1)

    valid_splits = {"train", "validation", "test"}
    if data_split not in valid_splits:
        print(f"Error: Invalid split '{data_split}'")
        print(f"Valid options: {', '.join(sorted(valid_splits))}")
        sys.exit(1)

    if generation_batch_size < 1:
        print(f"Error: Generation batch size must be >= 1 (got {generation_batch_size})")
        sys.exit(1)

    # Set environment variables to pass configuration to main.py
    os.environ['ICW_MEMORY_STRATEGY'] = model_strategy
    os.environ['ICW_TEMPERATURE'] = str(temperature)
    os.environ['ICW_NUM_SAMPLES'] = str(num_samples)
    os.environ['ICW_DATASET_SPLIT'] = data_split
    os.environ['ICW_GENERATION_BATCH_SIZE'] = str(generation_batch_size)
    os.environ['ICW_OUTPUT_DIR'] = output_dir
    os.environ['ICW_DISABLE_WM_INSTRUCTION'] = '1' if disable_wm_instruction else '0'

    # If using a custom trained model path
    if model_path:
        os.environ['ICW_MODEL_PATH'] = model_path
    else:
        os.environ.pop('ICW_MODEL_PATH', None)

    # Display configuration
    if model_path:
        config = {"model_name": model_path, "description": "Custom trained model"}
    else:
        config = get_model_config(model_strategy)

    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Model Strategy:   {model_strategy}")
    print(f"Model Name:       {config['model_name']}")
    print(f"Description:      {config.get('description', 'N/A')}")
    print(f"Temperature:      {temperature}")
    print(f"Num Samples:      {num_samples}")
    print(f"Dataset Split:    {data_split}")
    print(f"Gen Batch Size:   {generation_batch_size}")
    print(f"Output Dir:       {output_dir}")
    print("="*80 + "\n")

    # Import and run main
    print("Starting experiment...\n")

    # Modify sys.argv to prevent argparse conflicts
    original_argv = sys.argv.copy()
    sys.argv = ['main.py']

    try:
        # Import and run main pipeline
        from main import run_pipeline
        run_pipeline()
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Experiment failed")
        print(f"{'='*80}")
        print(f"{e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv

def main():
    parser = argparse.ArgumentParser(
        description="ICW Testing CLI - Easy model configuration and testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-models
  %(prog)s --model 4bit --temperature 0.7 --samples 50
  %(prog)s --model small --temperature 0.3 --samples 20
  %(prog)s --model auto --samples 100
        """
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available model configurations and exit'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='auto',
        help='Model strategy: small, 4bit, 8bit, full, cpu, auto (default: auto)'
    )

    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.7,
        help='Sampling temperature (0-2.0, default: 0.7)'
    )

    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=50,
        help='Number of samples to generate (1-1000, default: 50)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to custom trained model (e.g., GRPO-trained model)'
    )

    parser.add_argument(
        '--no-wm-instruction',
        action='store_true',
        help='Disable watermarking instructions for evaluation (useful for validation/test)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'validation', 'test'],
        help='ELI5 dataset split to evaluate on (default: train)'
    )

    parser.add_argument(
        '--gen-batch-size',
        type=int,
        default=4,
        help='Batch size for generation throughput (default: 4)'
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    run_experiment(
        model_strategy=args.model,
        temperature=args.temperature,
        num_samples=args.samples,
        output_dir=args.output,
        model_path=args.model_path,
        disable_wm_instruction=args.no_wm_instruction,
        data_split=args.split,
        generation_batch_size=args.gen_batch_size
    )

if __name__ == "__main__":
    main()
