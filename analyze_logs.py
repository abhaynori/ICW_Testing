"""
Utility script to analyze generation logs from ICW experiments.
"""

import json
import pandas as pd
from collections import defaultdict

def load_logs(filepath="outputs/generation_log.jsonl"):
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
    # Load logs
    logs = load_logs()
    
    if not logs:
        print("No logs found. Run main.py first to generate data.")
    else:
        print(f"Loaded {len(logs)} log entries")
        
        # Run analyses
        analyze_by_method(logs)
        compare_outputs(logs, query_idx=0)
        export_to_csv(logs)
        
        # Example search
        # search_logs(logs, "example", field='output')
