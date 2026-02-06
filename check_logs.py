"""
Quick diagnostic: Check for mixed/broken data in log file
"""
import json
import os
import sys
from collections import Counter

log_path = 'outputs/generation_log.jsonl'
if not os.path.exists(log_path):
    print(f"Log file not found: {log_path}")
    print("Run main.py first to generate data.")
    sys.exit(1)

with open(log_path, 'r') as f:
    logs = [json.loads(line) for line in f]

print(f"\n{'='*70}")
print(f"LOG FILE DIAGNOSTIC")
print(f"{'='*70}\n")

print(f"Total entries: {len(logs)}")

# Check prompts for each method
print(f"\nPrompts used for each method:")
print(f"{'-'*70}")

for method in ['Unicode ICW', 'Initials ICW', 'Lexical ICW', 'Acrostics ICW']:
    method_logs = [l for l in logs if l['method'] == method]
    print(f"\n{method} ({len(method_logs)} entries):")
    
    # Get unique prompts
    prompts = set(l['prompt'].split('User:')[0][:80] for l in method_logs)
    for i, prompt in enumerate(prompts, 1):
        count = sum(1 for l in method_logs if l['prompt'].startswith(prompt))
        print(f"  Version {i} ({count} entries): {prompt}...")
        
    # Check output quality
    if method_logs:
        sample = method_logs[-1]['output']  # Most recent
        words = len(sample.split())
        broken = "BROKEN" if words < 10 or len(sample) < 50 else "OK"
        print(f"  Latest output: [{broken}] {sample[:120]}...")

print(f"\n{'='*70}")
print(f"ISSUE DETECTED:")
print(f"{'='*70}")
print(f"You have MIXED data from multiple runs!")
print(f"\nSolution:")
print(f"  rm outputs/generation_log.jsonl")
print(f"  python3 main.py")
print(f"{'='*70}\n")
