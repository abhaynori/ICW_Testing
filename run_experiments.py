"""
Script to run experiments across multiple models and temperatures.
Usage: python run_experiments.py
"""

import subprocess
import os
import json
from datetime import datetime

# Experiment configurations
MODELS_TO_TEST = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # "microsoft/Phi-4",  # Uncomment when available/accessible
]

TEMPERATURES = [0.3, 0.7, 1.0]

def update_config(model_name, temperature):
    """Update main.py with specified model and temperature."""
    with open("main.py", "r") as f:
        content = f.read()
    
    # Replace MODEL_NAME
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("MODEL_NAME = "):
            lines[i] = f'MODEL_NAME = "{model_name}"'
        elif line.startswith("TEMPERATURE = "):
            lines[i] = f"TEMPERATURE = {temperature}"
    
    with open("main.py", "w") as f:
        f.write("\n".join(lines))

def run_experiment(model_name, temperature):
    """Run experiment with given configuration."""
    print(f"\n{'='*80}")
    print(f"Running experiment: {model_name} @ T={temperature}")
    print(f"{'='*80}\n")
    
    # Update config
    update_config(model_name, temperature)
    
    # Create unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    exp_dir = f"outputs/exp_{model_short}_T{temperature}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Run main.py
    try:
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        # Save stdout/stderr
        with open(f"{exp_dir}/stdout.txt", "w") as f:
            f.write(result.stdout)
        with open(f"{exp_dir}/stderr.txt", "w") as f:
            f.write(result.stderr)
        
        # Move generated files to experiment directory
        for file in ["generation_log.jsonl", "icw_roc_auc.png", "icw_t1fpr.png"]:
            src = f"outputs/{file}"
            if os.path.exists(src):
                dst = f"{exp_dir}/{file}"
                os.rename(src, dst)
        
        print(f"✓ Experiment completed successfully")
        print(f"  Results saved to: {exp_dir}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"✗ Experiment timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"✗ Experiment failed with error: {e}")
        return False

def summarize_results():
    """Summarize all experiment results."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    exp_dirs = [d for d in os.listdir("outputs") if d.startswith("exp_")]
    
    results = []
    for exp_dir in sorted(exp_dirs):
        log_file = f"outputs/{exp_dir}/generation_log.jsonl"
        if os.path.exists(log_file):
            with open(log_file) as f:
                first_log = json.loads(f.readline())
                model = first_log['model'].split('/')[-1]
                temp = first_log['temperature']
                
                # Count entries
                with open(log_file) as f:
                    num_generations = sum(1 for _ in f)
                
                results.append({
                    'Model': model,
                    'Temperature': temp,
                    'Generations': num_generations,
                    'Directory': exp_dir
                })
    
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n", df.to_string(index=False))
        df.to_csv("outputs/experiment_summary.csv", index=False)
        print("\nSummary saved to outputs/experiment_summary.csv")
    else:
        print("\nNo experiment results found.")

if __name__ == "__main__":
    print("Starting ICW experiments across multiple models and temperatures...")
    print(f"Models: {len(MODELS_TO_TEST)}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Total experiments: {len(MODELS_TO_TEST) * len(TEMPERATURES)}")
    
    successful = 0
    failed = 0
    
    for model in MODELS_TO_TEST:
        for temp in TEMPERATURES:
            if run_experiment(model, temp):
                successful += 1
            else:
                failed += 1
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"{'='*80}\n")
    
    summarize_results()
