"""
Script to run experiments across multiple models and temperatures.
Usage: python run_experiments.py
"""

import subprocess
import os
import json
import time
from datetime import datetime

# Experiment configurations
MODELS_TO_TEST = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # "microsoft/Phi-4",  # Uncomment when available/accessible
]

TEMPERATURES = [0.3, 0.7, 1.0]

def run_experiment(model_name, temperature):
    """Run experiment with given configuration."""
    print(f"\n{'='*80}")
    print(f"Running experiment: {model_name} @ T={temperature}")
    print(f"{'='*80}\n")
    
    # Create unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    exp_dir = f"outputs/exp_{model_short}_T{temperature}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Run main.py with live, line-by-line streaming
    cmd = ["python", "main.py"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # ensure immediate flushing from child
    
    # Pass configuration via environment variables
    env["ICW_MODEL_PATH"] = model_name
    env["ICW_TEMPERATURE"] = str(temperature)
    env["ICW_MEMORY_STRATEGY"] = "full"  # Use full precision for H200
    # Optional: set sample count if needed, e.g. env["ICW_NUM_SAMPLES"] = "50"

    start_time = time.time()
    timeout_seconds = 1800  # 30 minutes
    print(f"Starting main.py (logs streaming below). Timeout: {timeout_seconds//60} min")
    print(f"Logs will also be saved to: {exp_dir}/stdout.txt and stderr.txt")

    try:
        with open(f"{exp_dir}/stdout.txt", "w") as f_out, open(f"{exp_dir}/stderr.txt", "w") as f_err:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            # Stream stdout/stderr line by line
            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()

                if stdout_line:
                    print(stdout_line, end="", flush=True)
                    f_out.write(stdout_line)

                if stderr_line:
                    print(f"[stderr] {stderr_line}", end="", flush=True)
                    f_err.write(stderr_line)

                # Exit condition: process ended and no more output
                if stdout_line == "" and stderr_line == "" and process.poll() is not None:
                    break

                # Timeout guard
                if time.time() - start_time > timeout_seconds:
                    process.kill()
                    print(f"\n✗ Experiment timed out after {timeout_seconds//60} minutes")
                    return False

            return_code = process.wait()

        duration = time.time() - start_time
        print(f"\nRuntime: {duration/60:.1f} minutes")

        if return_code != 0:
            print(f"✗ Experiment failed with return code {return_code}")
            return False

        # Move generated files to experiment directory
        for file in ["generation_log.jsonl", "icw_roc_auc.png", "icw_t1fpr.png", "results.csv"]:
            src = f"outputs/{file}"
            if os.path.exists(src):
                dst = f"{exp_dir}/{file}"
                os.rename(src, dst)

        print(f"✓ Experiment completed successfully")
        print(f"  Results saved to: {exp_dir}")
        return True

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
