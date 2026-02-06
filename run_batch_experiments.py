#!/usr/bin/env python3
"""
Run batch experiments across multiple configurations.

This script runs the ICW testing framework across different model configurations
and temperatures, using the CLI interface for clean configuration.

Usage:
    python run_batch_experiments.py
    python run_batch_experiments.py --quick  # Quick test with fewer samples
    python run_batch_experiments.py --full   # Full test with all configurations
"""

import subprocess
import os
import json
import argparse
from datetime import datetime
import pandas as pd

# Experiment configurations
QUICK_CONFIG = {
    "models": ["small"],
    "temperatures": [0.7],
    "samples": 20
}

DEFAULT_CONFIG = {
    "models": ["small", "4bit"],
    "temperatures": [0.3, 0.7],
    "samples": 50
}

FULL_CONFIG = {
    "models": ["small", "4bit", "8bit"],
    "temperatures": [0.3, 0.7, 1.0],
    "samples": 100
}

def run_experiment(model_strategy, temperature, num_samples, base_output_dir="batch_outputs"):
    """Run a single experiment using the CLI interface."""

    # Create unique output directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model_strategy}_T{temperature}_N{num_samples}_{timestamp}"
    output_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Running experiment: {exp_name}")
    print(f"{'='*80}\n")

    # Run the CLI command
    cmd = [
        "python", "cli.py",
        "--model", model_strategy,
        "--temperature", str(temperature),
        "--samples", str(num_samples),
        "--output", output_dir
    ]

    try:
        log_file = os.path.join(output_dir, "experiment_log.txt")
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 1 hour timeout
            )

        if result.returncode == 0:
            print(f"✓ Experiment completed successfully")
            print(f"  Results saved to: {output_dir}")
            return True, exp_name, output_dir
        else:
            print(f"✗ Experiment failed with return code {result.returncode}")
            print(f"  Check log: {log_file}")
            return False, exp_name, output_dir

    except subprocess.TimeoutExpired:
        print(f"✗ Experiment timed out after 1 hour")
        return False, exp_name, output_dir
    except Exception as e:
        print(f"✗ Experiment failed with error: {e}")
        return False, exp_name, output_dir

def extract_results(output_dir):
    """Extract results from an experiment directory."""
    results_file = os.path.join(output_dir, "results.csv")

    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            return df.to_dict('records')
        except Exception as e:
            print(f"Warning: Could not read results from {results_file}: {e}")
            return None
    return None

def summarize_all_results(base_output_dir="batch_outputs"):
    """Create a summary of all experiment results."""
    print("\n" + "="*80)
    print("BATCH EXPERIMENT SUMMARY")
    print("="*80)

    # Find all experiment directories
    if not os.path.exists(base_output_dir):
        print("No experiments found.")
        return

    exp_dirs = [d for d in os.listdir(base_output_dir)
                if os.path.isdir(os.path.join(base_output_dir, d))]

    if not exp_dirs:
        print("No experiment directories found.")
        return

    all_results = []

    for exp_dir in sorted(exp_dirs):
        full_path = os.path.join(base_output_dir, exp_dir)

        # Parse experiment name
        parts = exp_dir.split('_')
        if len(parts) >= 3:
            model_strategy = parts[0]
            temperature = parts[1].replace('T', '')

            # Extract results
            results = extract_results(full_path)
            if results:
                for result in results:
                    all_results.append({
                        'Experiment': exp_dir,
                        'Model': model_strategy,
                        'Temperature': temperature,
                        'Method': result.get('Method', 'Unknown'),
                        'ROC-AUC': result.get('ROC-AUC', 0),
                        'TPR@1%FPR': result.get('TPR@1%FPR', 0),
                        'TPR@10%FPR': result.get('TPR@10%FPR', 0)
                    })

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + df.to_string(index=False))

        summary_file = os.path.join(base_output_dir, "batch_summary.csv")
        df.to_csv(summary_file, index=False)
        print(f"\n✓ Summary saved to: {summary_file}")

        # Create pivot tables for easier comparison
        print("\n" + "="*80)
        print("ROC-AUC BY MODEL AND METHOD")
        print("="*80)
        pivot = df.pivot_table(
            values='ROC-AUC',
            index='Method',
            columns='Model',
            aggfunc='mean'
        )
        print("\n" + pivot.to_string())

    else:
        print("No results could be extracted from experiments.")

def main():
    parser = argparse.ArgumentParser(
        description="Run batch ICW experiments with multiple configurations"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--quick',
        action='store_true',
        help='Quick test: 1 model, 1 temperature, 20 samples'
    )
    group.add_argument(
        '--full',
        action='store_true',
        help='Full test: 3 models, 3 temperatures, 100 samples'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='batch_outputs',
        help='Base directory for batch outputs (default: batch_outputs)'
    )

    args = parser.parse_args()

    # Select configuration
    if args.quick:
        config = QUICK_CONFIG
        print("Running QUICK batch experiments")
    elif args.full:
        config = FULL_CONFIG
        print("Running FULL batch experiments")
    else:
        config = DEFAULT_CONFIG
        print("Running DEFAULT batch experiments")

    print(f"Models: {config['models']}")
    print(f"Temperatures: {config['temperatures']}")
    print(f"Samples per experiment: {config['samples']}")

    total_experiments = len(config['models']) * len(config['temperatures'])
    print(f"Total experiments: {total_experiments}")

    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Track results
    successful = 0
    failed = 0
    experiment_log = []

    # Run all experiments
    for model in config['models']:
        for temp in config['temperatures']:
            success, exp_name, exp_dir = run_experiment(
                model_strategy=model,
                temperature=temp,
                num_samples=config['samples'],
                base_output_dir=args.output_dir
            )

            experiment_log.append({
                'experiment': exp_name,
                'model': model,
                'temperature': temp,
                'samples': config['samples'],
                'success': success,
                'output_dir': exp_dir
            })

            if success:
                successful += 1
            else:
                failed += 1

    # Save experiment log
    log_file = os.path.join(args.output_dir, "experiment_log.json")
    with open(log_file, 'w') as f:
        json.dump(experiment_log, f, indent=2)

    print(f"\n{'='*80}")
    print(f"BATCH EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{total_experiments}")
    print(f"Failed: {failed}/{total_experiments}")
    print(f"Log saved to: {log_file}")
    print(f"{'='*80}\n")

    # Summarize results
    summarize_all_results(args.output_dir)

if __name__ == "__main__":
    main()
