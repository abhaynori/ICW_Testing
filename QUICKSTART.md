# Quick Start Guide - ICW Improvements

## What's New

### ✅ 1. Output Observation
- **Generation Logs**: All prompts and outputs are now saved to `outputs/generation_log.jsonl`
- **Console Display**: First 2 examples from each method printed during execution
- **Analysis Tools**: Use `analyze_logs.py` to explore and compare outputs

### ✅ 2. Configurable Temperature
- **Parameter**: `TEMPERATURE` variable at top of `main.py`
- **Range**: 0.0 (deterministic) to 2.0 (highly random)
- **Default**: 0.7 (balanced)

### ✅ 3. Multiple Model Support
- **Supported Models**:
  - Phi-3: `microsoft/Phi-3-mini-4k-instruct`
  - Phi-4: `microsoft/Phi-4`
  - Llama-3.1: `meta-llama/Meta-Llama-3.1-8B-Instruct`
  - Qwen: `Qwen/Qwen2.5-7B-Instruct`

## Quick Usage

### 1. Run with default settings (Phi-3, T=0.7)
```bash
python main.py
```

### 2. Change model and temperature
Edit `main.py`:
```python
TEMPERATURE = 0.5
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
```

### 3. Analyze generated outputs
```bash
python analyze_logs.py
```

### 4. Run experiments across multiple configurations
```bash
python run_experiments.py
```

## Output Files

```
outputs/
├── generation_log.jsonl          # All prompts & outputs
├── icw_roc_auc.png               # ROC-AUC comparison chart
├── icw_t1fpr.png                 # T@1%FPR comparison chart
├── generation_summary.csv        # Exported analysis
└── exp_*/                        # Individual experiment results
    ├── generation_log.jsonl
    ├── icw_roc_auc.png
    ├── icw_t1fpr.png
    ├── stdout.txt
    └── stderr.txt
```

## Example Log Entry

```json
{
  "timestamp": "2025-10-10T12:34:56.789",
  "method": "Unicode ICW",
  "model": "microsoft/Phi-3-mini-4k-instruct",
  "temperature": 0.7,
  "query": "Why is the sky blue?",
  "prompt": "<|system|>Insert zero-width space...<|end|>\n<|user|>Why is the sky blue?<|end|>\n<|assistant|>",
  "output": "The sky appears blue because..."
}
```

## Analyzing Results

### View outputs by method
```python
from analyze_logs import load_logs, analyze_by_method

logs = load_logs()
analyze_by_method(logs)
```

### Compare same query across methods
```python
from analyze_logs import compare_outputs

compare_outputs(logs, query_idx=0)
```

### Search for specific content
```python
from analyze_logs import search_logs

search_logs(logs, "example", field='output')
```

## Temperature Effects

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.0 | Deterministic, always same output | Reproducibility |
| 0.3-0.5 | Low randomness, consistent | Production systems |
| 0.6-0.8 | Balanced creativity | Research (recommended) |
| 0.9-1.2 | High creativity | Creative tasks |
| 1.3+ | Very random | Exploration |

## Model Differences

### Phi-3 (microsoft/Phi-3-mini-4k-instruct)
- **Size**: 3.8B parameters
- **Context**: 4K tokens
- **Strengths**: Fast, efficient, good instruction following

### Phi-4 (microsoft/Phi-4)
- **Size**: TBD
- **Strengths**: Enhanced reasoning capabilities

### Llama-3.1 (meta-llama/Meta-Llama-3.1-8B-Instruct)
- **Size**: 8B parameters
- **Context**: 8K tokens
- **Strengths**: Strong general performance, good instruction following

### Qwen (Qwen/Qwen2.5-7B-Instruct)
- **Size**: 7B parameters
- **Strengths**: Multilingual, strong reasoning

## Tips

1. **Start small**: Use default 10 samples to test before scaling up
2. **Monitor GPU**: Larger models need more VRAM
3. **Log everything**: Logs are your friend for analysis
4. **Compare systematically**: Use `run_experiments.py` for fair comparisons
5. **Adjust sample size**: Edit `eli5[:10]` to `eli5[:100]` for more data

## Troubleshooting

### Out of memory?
- Use smaller model (Phi-3)
- Reduce batch size
- Use `torch_dtype=torch.float16` instead of `bfloat16`

### Model not found?
- Check HuggingFace model hub
- Some models require authentication (Llama)
- Update model names if newer versions available

### Slow generation?
- Normal for first run (model download)
- Use smaller dataset sample
- Consider GPU acceleration

## Next Steps

1. Review `outputs/generation_log.jsonl` to see actual prompts/outputs
2. Run `analyze_logs.py` for summary statistics
3. Try different temperatures and models
4. Compare watermark detectability across configurations
5. Analyze robustness to attacks
