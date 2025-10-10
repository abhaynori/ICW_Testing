"""
Visualization script to compare ICW results across different models and temperatures.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sns.set_style("whitegrid")

def load_all_experiments():
    """Load logs from all experiment directories."""
    experiments = []
    
    # Check for experiment directories
    if os.path.exists("outputs"):
        for exp_dir in os.listdir("outputs"):
            if exp_dir.startswith("exp_"):
                log_path = f"outputs/{exp_dir}/generation_log.jsonl"
                if os.path.exists(log_path):
                    exp_data = {
                        'directory': exp_dir,
                        'logs': []
                    }
                    with open(log_path) as f:
                        for line in f:
                            exp_data['logs'].append(json.loads(line))
                    
                    if exp_data['logs']:
                        first_log = exp_data['logs'][0]
                        exp_data['model'] = first_log['model'].split('/')[-1]
                        exp_data['temperature'] = first_log['temperature']
                        experiments.append(exp_data)
    
    # Also check main outputs folder
    main_log = "outputs/generation_log.jsonl"
    if os.path.exists(main_log):
        exp_data = {'directory': 'main', 'logs': []}
        with open(main_log) as f:
            for line in f:
                exp_data['logs'].append(json.loads(line))
        
        if exp_data['logs']:
            first_log = exp_data['logs'][0]
            exp_data['model'] = first_log['model'].split('/')[-1]
            exp_data['temperature'] = first_log['temperature']
            experiments.append(exp_data)
    
    return experiments

def plot_output_lengths(experiments):
    """Plot distribution of output lengths by model and method."""
    data = []
    for exp in experiments:
        for log in exp['logs']:
            data.append({
                'Model': exp['model'],
                'Temperature': exp['temperature'],
                'Method': log['method'],
                'Output Length': len(log['output']),
                'Query Length': len(log['query'])
            })
    
    df = pd.DataFrame(data)
    
    # Plot output lengths by method and model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # By method
    df.groupby('Method')['Output Length'].mean().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Average Output Length by ICW Method')
    axes[0].set_ylabel('Characters')
    axes[0].tick_params(axis='x', rotation=30)
    
    # By model
    if 'Model' in df.columns and df['Model'].nunique() > 1:
        df.groupby('Model')['Output Length'].mean().plot(kind='bar', ax=axes[1], color='lightcoral')
        axes[1].set_title('Average Output Length by Model')
        axes[1].set_ylabel('Characters')
        axes[1].tick_params(axis='x', rotation=30)
    else:
        axes[1].text(0.5, 0.5, 'Run multiple models\nto compare', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Average Output Length by Model')
    
    plt.tight_layout()
    plt.savefig('outputs/output_length_comparison.png', dpi=300)
    print("Saved: outputs/output_length_comparison.png")
    plt.close()

def plot_temperature_effects(experiments):
    """Plot how temperature affects output characteristics."""
    data = []
    for exp in experiments:
        for log in exp['logs']:
            output = log['output']
            data.append({
                'Model': exp['model'],
                'Temperature': exp['temperature'],
                'Method': log['method'],
                'Output Length': len(output),
                'Unique Words': len(set(output.lower().split())),
                'Total Words': len(output.split())
            })
    
    df = pd.DataFrame(data)
    if df['Temperature'].nunique() <= 1:
        print("Skipping temperature plot (need multiple temperatures)")
        return
    
    df['Lexical Diversity'] = df['Unique Words'] / df['Total Words']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Temperature vs Length
    temp_stats = df.groupby('Temperature')['Output Length'].agg(['mean', 'std']).reset_index()
    axes[0].errorbar(temp_stats['Temperature'], temp_stats['mean'], 
                     yerr=temp_stats['std'], marker='o', capsize=5)
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Average Output Length (chars)')
    axes[0].set_title('Temperature Effect on Output Length')
    axes[0].grid(True, alpha=0.3)
    
    # Temperature vs Lexical Diversity
    temp_div = df.groupby('Temperature')['Lexical Diversity'].agg(['mean', 'std']).reset_index()
    axes[1].errorbar(temp_div['Temperature'], temp_div['mean'], 
                     yerr=temp_div['std'], marker='o', capsize=5, color='coral')
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Lexical Diversity (unique/total words)')
    axes[1].set_title('Temperature Effect on Vocabulary')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/temperature_effects.png', dpi=300)
    print("Saved: outputs/temperature_effects.png")
    plt.close()

def plot_method_comparison_heatmap(experiments):
    """Create heatmap comparing methods across models."""
    data = []
    for exp in experiments:
        method_counts = defaultdict(int)
        for log in exp['logs']:
            method_counts[log['method']] += 1
        
        for method, count in method_counts.items():
            data.append({
                'Model': exp['model'],
                'Temperature': f"T={exp['temperature']}",
                'Method': method,
                'Count': count
            })
    
    df = pd.DataFrame(data)
    
    if len(df) > 0 and df['Model'].nunique() > 1:
        # Pivot for heatmap
        pivot = df.pivot_table(values='Count', 
                               index='Method', 
                               columns=['Model', 'Temperature'],
                               fill_value=0)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt='g', cmap='YlOrRd', cbar_kws={'label': 'Samples'})
        plt.title('Sample Counts: Methods × Models × Temperature')
        plt.xlabel('Model @ Temperature')
        plt.ylabel('ICW Method')
        plt.tight_layout()
        plt.savefig('outputs/method_model_heatmap.png', dpi=300)
        print("Saved: outputs/method_model_heatmap.png")
        plt.close()
    else:
        print("Skipping heatmap (need multiple models)")

def generate_summary_report(experiments):
    """Generate a comprehensive text summary."""
    report = []
    report.append("="*80)
    report.append("ICW EXPERIMENT SUMMARY REPORT")
    report.append("="*80)
    report.append(f"\nTotal Experiments: {len(experiments)}")
    
    for i, exp in enumerate(experiments, 1):
        report.append(f"\n--- Experiment {i}: {exp['directory']} ---")
        report.append(f"Model: {exp['model']}")
        report.append(f"Temperature: {exp['temperature']}")
        report.append(f"Total Generations: {len(exp['logs'])}")
        
        # Method breakdown
        method_counts = defaultdict(int)
        method_lengths = defaultdict(list)
        for log in exp['logs']:
            method_counts[log['method']] += 1
            method_lengths[log['method']].append(len(log['output']))
        
        report.append("\nGenerations by Method:")
        for method in sorted(method_counts.keys()):
            avg_len = sum(method_lengths[method]) / len(method_lengths[method])
            report.append(f"  - {method}: {method_counts[method]} samples, avg {avg_len:.0f} chars")
    
    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open("outputs/experiment_report.txt", "w") as f:
        f.write(report_text)
    print("\nReport saved to: outputs/experiment_report.txt")

def main():
    """Run all visualizations."""
    print("Loading experiments...")
    experiments = load_all_experiments()
    
    if not experiments:
        print("\nNo experiments found!")
        print("Run main.py first to generate data.")
        return
    
    print(f"Found {len(experiments)} experiment(s)\n")
    
    print("Generating visualizations...")
    plot_output_lengths(experiments)
    plot_temperature_effects(experiments)
    plot_method_comparison_heatmap(experiments)
    generate_summary_report(experiments)
    
    print("\n" + "="*80)
    print("Visualization complete! Check the outputs/ directory for plots.")
    print("="*80)

if __name__ == "__main__":
    main()
