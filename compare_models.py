#!/usr/bin/env python3
"""
Compare base model vs GRPO-trained model performance.

This script runs experiments with both the base model and a GRPO-trained model
to quantify the improvement in watermarking performance.

Usage:
    python compare_models.py --base small --trained grpo_models/unicode_small_*/final_model --method unicode
    python compare_models.py --base small --trained grpo_models/acrostics_small_*/final_model --method acrostics --samples 100
"""

import argparse
import subprocess
import os
import json
import pandas as pd
from datetime import datetime
import glob


def run_comparison(base_model, trained_model_path, method, num_samples=50):
    """
    Run comparison between base and trained models.

    Args:
        base_model: Base model strategy (small, 4bit, etc.)
        trained_model_path: Path to GRPO-trained model
        method: Watermarking method
        num_samples: Number of test samples
    """

    print("\n" + "="*80)
    print("MODEL COMPARISON: BASE vs GRPO-TRAINED")
    print("="*80)
    print(f"Base Model:     {base_model}")
    print(f"Trained Model:  {trained_model_path}")
    print(f"Method:         {method}")
    print(f"Test Samples:   {num_samples}")
    print("="*80 + "\n")

    # Create output directory for comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = f"comparisons/comparison_{method}_{timestamp}"
    os.makedirs(comparison_dir, exist_ok=True)

    # Run base model
    print("\n" + "="*80)
    print("RUNNING BASE MODEL")
    print("="*80 + "\n")

    base_output_dir = os.path.join(comparison_dir, "base_model")
    base_cmd = [
        "python", "cli.py",
        "--model", base_model,
        "--samples", str(num_samples),
        "--output", base_output_dir
    ]

    try:
        subprocess.run(base_cmd, check=True)
        print("\n✓ Base model evaluation complete")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Base model evaluation failed: {e}")
        return None

    # Run trained model
    print("\n" + "="*80)
    print("RUNNING GRPO-TRAINED MODEL")
    print("="*80 + "\n")

    trained_output_dir = os.path.join(comparison_dir, "trained_model")
    trained_cmd = [
        "python", "cli.py",
        "--model-path", trained_model_path,
        "--samples", str(num_samples),
        "--output", trained_output_dir
    ]

    try:
        subprocess.run(trained_cmd, check=True)
        print("\n✓ Trained model evaluation complete")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Trained model evaluation failed: {e}")
        return None

    # Load and compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    base_results_file = os.path.join(base_output_dir, "results.csv")
    trained_results_file = os.path.join(trained_output_dir, "results.csv")

    if not os.path.exists(base_results_file) or not os.path.exists(trained_results_file):
        print("\n✗ Could not find results files")
        return None

    base_results = pd.read_csv(base_results_file)
    trained_results = pd.read_csv(trained_results_file)

    # Create comparison DataFrame
    comparison_data = []

    for method_name in base_results["Method"].unique():
        base_row = base_results[base_results["Method"] == method_name].iloc[0]
        trained_row = trained_results[trained_results["Method"] == method_name].iloc[0]

        comparison_data.append({
            "Method": method_name,
            "Base ROC-AUC": base_row["ROC-AUC"],
            "Trained ROC-AUC": trained_row["ROC-AUC"],
            "ROC-AUC Improvement": trained_row["ROC-AUC"] - base_row["ROC-AUC"],
            "Base TPR@1%": base_row["TPR@1%FPR"],
            "Trained TPR@1%": trained_row["TPR@1%FPR"],
            "TPR@1% Improvement": trained_row["TPR@1%FPR"] - base_row["TPR@1%FPR"],
            "Base TPR@10%": base_row["TPR@10%FPR"],
            "Trained TPR@10%": trained_row["TPR@10%FPR"],
            "TPR@10% Improvement": trained_row["TPR@10%FPR"] - base_row["TPR@10%FPR"]
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Display results
    print("\n" + comparison_df.to_string(index=False))

    # Save comparison
    comparison_file = os.path.join(comparison_dir, "comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✓ Comparison saved to: {comparison_file}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for _, row in comparison_df.iterrows():
        print(f"\n{row['Method']}:")
        print(f"  ROC-AUC:    {row['Base ROC-AUC']:.4f} → {row['Trained ROC-AUC']:.4f} ({row['ROC-AUC Improvement']:+.4f})")
        print(f"  TPR@1%FPR:  {row['Base TPR@1%']:.4f} → {row['Trained TPR@1%']:.4f} ({row['TPR@1% Improvement']:+.4f})")
        print(f"  TPR@10%FPR: {row['Base TPR@10%']:.4f} → {row['Trained TPR@10%']:.4f} ({row['TPR@10% Improvement']:+.4f})")

        # Interpretation
        if row['ROC-AUC Improvement'] > 0.1:
            print(f"  ✓✓ Significant improvement!")
        elif row['ROC-AUC Improvement'] > 0.05:
            print(f"  ✓ Moderate improvement")
        elif row['ROC-AUC Improvement'] > 0:
            print(f"  ✓ Slight improvement")
        else:
            print(f"  ⚠️  No improvement (may need more training)")

    # Save metadata
    metadata = {
        "base_model": base_model,
        "trained_model_path": trained_model_path,
        "method": method,
        "num_samples": num_samples,
        "timestamp": timestamp,
        "comparison_dir": comparison_dir
    }

    metadata_file = os.path.join(comparison_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to: {metadata_file}")
    print(f"\nAll results saved to: {comparison_dir}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare base and GRPO-trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare small base model with trained Unicode model
  python compare_models.py --base small --trained grpo_models/unicode_small_*/final_model --method unicode

  # Compare with more samples for better statistics
  python compare_models.py --base small --trained grpo_models/acrostics_small_*/final_model --method acrostics --samples 100

  # Use glob pattern to find latest trained model
  python compare_models.py --base small --trained "grpo_models/unicode_small_*/final_model" --method unicode
        """
    )

    parser.add_argument(
        '--base',
        type=str,
        required=True,
        help='Base model strategy (small, 4bit, 8bit, etc.)'
    )

    parser.add_argument(
        '--trained',
        type=str,
        required=True,
        help='Path to GRPO-trained model (supports glob patterns)'
    )

    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['unicode', 'initials', 'lexical', 'acrostics'],
        help='Watermarking method to evaluate'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=50,
        help='Number of test samples (default: 50)'
    )

    args = parser.parse_args()

    # Resolve glob pattern if needed
    trained_model_path = args.trained
    if '*' in trained_model_path:
        matches = glob.glob(trained_model_path)
        if not matches:
            print(f"Error: No models found matching pattern: {trained_model_path}")
            return
        # Use the most recent match
        trained_model_path = sorted(matches)[-1]
        print(f"Found trained model: {trained_model_path}")

    # Check if trained model exists
    if not os.path.exists(trained_model_path):
        print(f"Error: Trained model not found at: {trained_model_path}")
        return

    # Run comparison
    result = run_comparison(
        base_model=args.base,
        trained_model_path=trained_model_path,
        method=args.method,
        num_samples=args.samples
    )

    if result is not None:
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)


if __name__ == "__main__":
    main()
