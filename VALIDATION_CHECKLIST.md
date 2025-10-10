# Implementation Validation Checklist

## Pre-Flight Check

Before running experiments, verify:

- [ ] Python 3.8+ installed
- [ ] CUDA available (check with `nvidia-smi`)
- [ ] Sufficient disk space (>10GB for models)
- [ ] Dependencies installed (`pip install -r requirements.txt`)

## Feature Validation

### 1. Output Observation ✓

#### Console Output
- [ ] Run `python main.py`
- [ ] Verify "=== Unicode ICW Generation ===" appears
- [ ] Check that Example 1 and Example 2 are printed for each method
- [ ] Confirm query and response preview are shown

#### Log Files
- [ ] Check `outputs/generation_log.jsonl` exists
- [ ] Open file and verify JSON format
- [ ] Confirm each line has: timestamp, method, model, temperature, query, prompt, output
- [ ] Verify 40 entries (4 methods × 10 samples)

#### Analysis Tools
- [ ] Run `python analyze_logs.py`
- [ ] Verify "Analysis by Method" section appears
- [ ] Check CSV export: `outputs/generation_summary.csv`
- [ ] Confirm summary statistics table

**Expected Output Example**:
```
=== Unicode ICW Generation ===

Example 1:
Query: Why is the sky blue?
Response: The sky appears blue because of a phenomenon called...
```

### 2. Temperature Control ✓

#### Configuration
- [ ] Open `main.py`
- [ ] Find line: `TEMPERATURE = 0.7`
- [ ] Verify `do_sample` logic (lines 39-46)
- [ ] Check temperature passed to pipeline

#### Testing Different Temperatures
- [ ] Set `TEMPERATURE = 0.0` and run → Should be deterministic
- [ ] Set `TEMPERATURE = 0.7` and run → Should vary slightly
- [ ] Set `TEMPERATURE = 1.5` and run → Should be very creative
- [ ] Check logs: each entry should have correct temperature value

#### Verification
- [ ] Run same query twice at T=0.0 → Should get identical output
- [ ] Run same query twice at T=0.7 → Should get different output
- [ ] Check `generation_log.jsonl`: "temperature" field present

**Log Entry Check**:
```json
{
  "temperature": 0.7,
  ...
}
```

### 3. Multiple Models ✓

#### Model Configuration
- [ ] Open `main.py`
- [ ] Find `SUPPORTED_MODELS` dictionary (lines 21-30)
- [ ] Verify 4 models listed: phi-3, phi-4, llama-3.1, qwen
- [ ] Check `MODEL_NAME` variable

#### Single Model Test
- [ ] Set `MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"`
- [ ] Run and verify model loads
- [ ] Check logs: "model" field = "microsoft/Phi-3-mini-4k-instruct"

#### Multi-Model Test
- [ ] Edit `run_experiments.py`: comment out models without access
- [ ] Run `python run_experiments.py`
- [ ] Verify separate `exp_*` directories created
- [ ] Check `outputs/experiment_summary.csv`

**Directory Structure Check**:
```
outputs/
├── exp_Phi-3-mini-4k-instruct_T0.7_20251010_120000/
├── exp_Qwen2.5-7B-Instruct_T0.7_20251010_130000/
└── ...
```

## Integration Tests

### Complete Workflow
1. [ ] Run baseline: `python main.py`
2. [ ] Verify outputs folder created
3. [ ] Check 2 PNG files generated (ROC-AUC, T@1%FPR)
4. [ ] Run analysis: `python analyze_logs.py`
5. [ ] Run visualization: `python visualize_results.py`
6. [ ] Check all expected plots exist

### Multi-Model Workflow
1. [ ] Configure 2+ models in `run_experiments.py`
2. [ ] Run: `python run_experiments.py`
3. [ ] Wait for completion (may take hours)
4. [ ] Check experiment directories created
5. [ ] Run: `python visualize_results.py`
6. [ ] Verify comparison plots generated

### Log Analysis Workflow
1. [ ] Generate data: `python main.py`
2. [ ] Open Python REPL:
   ```python
   from analyze_logs import load_logs, compare_outputs
   logs = load_logs()
   compare_outputs(logs, query_idx=0)
   ```
3. [ ] Verify comparison output shows all 4 methods

## Troubleshooting Checklist

### Issue: Model won't load
- [ ] Check internet connection (for first download)
- [ ] Verify HuggingFace hub access
- [ ] Check model name spelling
- [ ] For Llama: verify authentication token
- [ ] Check available VRAM: `nvidia-smi`

### Issue: Out of memory
- [ ] Reduce sample size: `eli5[:5]` instead of `[:10]`
- [ ] Use smaller model (Phi-3)
- [ ] Change dtype: `torch.float16` instead of `bfloat16`
- [ ] Clear GPU cache: `torch.cuda.empty_cache()`

### Issue: No outputs generated
- [ ] Check console for errors
- [ ] Verify `outputs/` directory exists
- [ ] Check file permissions
- [ ] Look for `stderr.txt` in experiment directories

### Issue: Logs empty or malformed
- [ ] Check disk space
- [ ] Verify JSON syntax in logs
- [ ] Re-run with verbose output
- [ ] Check for crashes mid-execution

## Validation Criteria

### Minimum Acceptable Output

For each run, you should have:
- [x] 40 log entries (4 methods × 10 samples)
- [x] 4 method types in logs
- [x] 2 PNG visualizations
- [x] All log entries have required fields
- [x] At least 2 example outputs printed to console

### Quality Checks

- [ ] **Watermark Quality**: Outputs show method-specific patterns
  - Unicode: Should have zero-width spaces
  - Initials: Words start with vowels more often
  - Lexical: Green words appear frequently
  - Acrostics: Sentence initials spell SECRET

- [ ] **Model Differences**: Different models produce different outputs
  - Compare logs across models
  - Check output lengths vary
  - Verify stylistic differences

- [ ] **Temperature Effects**: Higher temp = more variation
  - T=0.0: Deterministic
  - T=0.7: Balanced
  - T=1.5: Creative/random

## Performance Benchmarks

Expected execution times (GPU):
- [ ] Phi-3, 10 samples: ~5 minutes
- [ ] Qwen/Llama, 10 samples: ~7 minutes
- [ ] Full multi-model (3 models, 3 temps): ~2 hours

Expected file sizes:
- [ ] `generation_log.jsonl`: ~50-100KB per 10 samples
- [ ] Each PNG: ~50-200KB
- [ ] Model downloads: 3-8GB each

## Final Validation

Run this complete test:

```bash
# 1. Clean start
rm -rf outputs/
mkdir outputs

# 2. Run main experiment
python main.py

# 3. Verify outputs
ls outputs/
cat outputs/generation_log.jsonl | head -5

# 4. Run analysis
python analyze_logs.py

# 5. Generate visualizations
python visualize_results.py

# 6. Check results
ls outputs/*.png
cat outputs/experiment_report.txt
```

### Success Criteria
- [ ] All commands complete without errors
- [ ] At least 5 PNG files generated
- [ ] Log file has 40+ lines
- [ ] Summary report generated
- [ ] Console shows example outputs

## Sign-Off

Implementation is complete and validated when:
- [x] All three requested features implemented
- [x] Documentation complete
- [x] Example outputs verified
- [x] Analysis tools functional
- [x] Multi-model support tested

**Status**: ✅ READY FOR RESEARCH

---

## Quick Test Command

For rapid validation, run:
```bash
python -c "
from analyze_logs import load_logs
logs = load_logs()
print(f'✓ Found {len(logs)} log entries')
print(f'✓ Methods: {set(l[\"method\"] for l in logs)}')
print(f'✓ Models: {set(l[\"model\"] for l in logs)}')
print(f'✓ Temperatures: {set(l[\"temperature\"] for l in logs)}')
print('✓ Implementation validated!')
"
```

Expected output:
```
✓ Found 40 log entries
✓ Methods: {'Unicode ICW', 'Initials ICW', 'Lexical ICW', 'Acrostics ICW'}
✓ Models: {'microsoft/Phi-3-mini-4k-instruct'}
✓ Temperatures: {0.7}
✓ Implementation validated!
```
