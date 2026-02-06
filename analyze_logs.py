"""
Analyze only the most recent complete run from the log file.
This handles mixed data from multiple runs.
"""

import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import pandas as pd

DEFAULT_LOG_PATH = "outputs/generation_log.jsonl"


def load_logs(filepath=DEFAULT_LOG_PATH):
    """Load all log entries from JSONL file."""
    logs = []
    try:
        with open(filepath) as f:
            for line in f:
                logs.append(json.loads(line))
    except FileNotFoundError:
        print(f"Log file not found: {filepath}")
        return []
    return logs

def analyze_mixed_logs(logs):
    if not logs:
        print("No logs found. Run main.py first to generate data.")
        return

    print(f"Total log entries: {len(logs)}\n")

    # Group by method and prompt type
    runs = defaultdict(list)
    for log in logs:
        method = log['method']
        prompt_key = log['prompt'].split('User:')[0][:100]  # First 100 chars of system prompt
        key = (method, prompt_key)
        runs[key].append(log)

    print("="*80)
    print("DETECTED RUNS:")
    print("="*80)
    for (method, prompt), entries in sorted(runs.items()):
        timestamp = entries[0]['timestamp']
        print(f"\n{method}:")
        print(f"  Entries: {len(entries)}")
        print(f"  First timestamp: {timestamp}")
        print(f"  Prompt: {prompt}...")
        if len(entries) > 0:
            sample = entries[0]['output'][:150]
            print(f"  Sample output: {sample}...")

    # Find most recent timestamp
    all_timestamps = [datetime.fromisoformat(log['timestamp']) for log in logs]
    latest_time = max(all_timestamps)
    print(f"\n{'='*80}")
    print(f"Latest timestamp: {latest_time}")
    print(f"{'='*80}\n")

    # Get logs from last hour (recent run)
    recent_cutoff = latest_time.timestamp() - 3600  # 1 hour ago
    recent_logs = [
        log for log in logs 
        if datetime.fromisoformat(log['timestamp']).timestamp() > recent_cutoff
    ]

    print(f"Recent entries (last hour): {len(recent_logs)}")

    if len(recent_logs) >= 40:  # Need at least 10 per method
        print("\n" + "="*80)
        print("ANALYZING MOST RECENT RUN ONLY")
        print("="*80)
        
        # Group by method
        by_method = defaultdict(list)
        for log in recent_logs:
            by_method[log['method']].append(log)
        
        for method, entries in sorted(by_method.items()):
            print(f"\n{method}: {len(entries)} samples")
            if entries:
                print(f"  Sample: {entries[0]['output'][:200]}...")
                
                # Check output quality
                outputs = [e['output'] for e in entries]
                avg_length = np.mean([len(o) for o in outputs])
                avg_words = np.mean([len(o.split()) for o in outputs])
                print(f"  Avg length: {avg_length:.0f} chars, {avg_words:.0f} words")
                
                # Check for broken outputs
                broken = sum(1 for o in outputs if len(o.split()) < 5 or ' ' not in o)
                if broken > 0:
                    print(f"  ⚠️  WARNING: {broken} outputs appear broken!")
    else:
        print("\n⚠️  Not enough recent entries for a complete run.")
        print("   Run main.py fresh to generate new data.")
        
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("1. Delete old log file: rm outputs/generation_log.jsonl")
    print("2. Run fresh: python3 main.py")
    print("3. This will generate clean, consistent data")

def analyze_by_method(logs):
    """Group and analyze logs by ICW method."""
    by_method = defaultdict(list)
    for log in logs:
        by_method[log['method']].append(log)
    
    print("\n=== Analysis by Method ===")
    for method, entries in by_method.items():
        print(f"\n{method}: {len(entries)} generations")
        print(f"Model: {entries[0]['model']}")
        print(f"Temperature: {entries[0]['temperature']}")
        
        # Show first example
        if entries:
            ex = entries[0]
            print(f"\nExample Query: {ex['query'][:100]}...")
            print(f"Example Output: {ex['output'][:150]}...")

def compare_outputs(logs, query_idx=0):
    """Compare outputs for the same query across different methods."""
    by_query = defaultdict(list)
    for log in logs:
        by_query[log['query']].append(log)
    
    queries = list(by_query.keys())
    if query_idx >= len(queries):
        print(f"Query index {query_idx} out of range")
        return
    
    query = queries[query_idx]
    entries = by_query[query]
    
    print(f"\n=== Comparing outputs for query: {query[:100]}... ===\n")
    for entry in entries:
        print(f"Method: {entry['method']}")
        print(f"Output: {entry['output'][:200]}...")
        print("-" * 80)

def export_to_csv(logs, output_file="outputs/generation_summary.csv"):
    """Export logs to CSV for further analysis."""
    df = pd.DataFrame(logs)
    df['output_length'] = df['output'].str.len()
    df['query_length'] = df['query'].str.len()
    df.to_csv(output_file, index=False)
    print(f"\nExported {len(df)} entries to {output_file}")
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    print(df.groupby('method')[['output_length', 'query_length']].describe())

def search_logs(logs, keyword, field='output'):
    """Search logs for entries containing a keyword."""
    results = [log for log in logs if keyword.lower() in log[field].lower()]
    print(f"\n=== Found {len(results)} entries containing '{keyword}' in {field} ===\n")
    for i, log in enumerate(results[:5]):  # Show first 5
        print(f"{i+1}. Method: {log['method']}")
        print(f"   {field.capitalize()}: {log[field][:150]}...")
        print()

if __name__ == "__main__":
    logs = load_logs()
    if not logs:
        print("No logs found. Run main.py first to generate data.")
    else:
        analyze_mixed_logs(logs)
