# ICW Testing - CLI Usage Guide

This guide explains how to use the command-line interface for easy model configuration and testing.

## Quick Start

### List Available Models

```bash
python cli.py --list-models
```

This will show all available model configurations with memory requirements.

### Run a Single Experiment

```bash
# Quick test with small model
python cli.py --model small --temperature 0.7 --samples 20

# Production run with 4-bit quantized model
python cli.py --model 4bit --temperature 0.7 --samples 100

# Auto-detect best model for your hardware
python cli.py --model auto --samples 50
```

## CLI Options

### Main Script: `cli.py`

```bash
python cli.py [OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--list-models` | - | List available model configurations and exit | - |
| `--model` | `-m` | Model strategy (small, 4bit, 8bit, full, cpu, auto) | auto |
| `--temperature` | `-t` | Sampling temperature (0-2.0) | 0.7 |
| `--samples` | `-n` | Number of samples to generate (1-1000) | 50 |
| `--output` | `-o` | Output directory | outputs |

#### Model Strategies

| Strategy | Model | Memory | Description |
|----------|-------|--------|-------------|
| `small` | Qwen 1.5B | ~1-3GB | Fast, low memory, good for testing |
| `4bit` | Qwen 7B | ~4-5GB | Best balance of quality and memory |
| `8bit` | Qwen 7B | ~7GB | Better quality than 4bit |
| `full` | Qwen 7B | ~13GB | Full precision, best quality |
| `cpu` | Qwen 1.5B | Any | CPU-only mode (slow) |
| `auto` | - | - | Auto-detect best option |

### Examples

#### Basic Usage

```bash
# Run with 4-bit model at temperature 0.7
python cli.py --model 4bit --temperature 0.7 --samples 50

# Quick test with small model
python cli.py --model small --temperature 0.3 --samples 20

# High creativity with full model
python cli.py --model full --temperature 1.0 --samples 100
```

#### Custom Output Directory

```bash
# Save results to custom directory
python cli.py --model 4bit --output my_experiment_results
```

#### Testing Different Temperatures

```bash
# Low temperature (more focused)
python cli.py --model 4bit --temperature 0.3 --samples 50

# Medium temperature (balanced)
python cli.py --model 4bit --temperature 0.7 --samples 50

# High temperature (more creative)
python cli.py --model 4bit --temperature 1.0 --samples 50
```

## Batch Experiments

For running multiple experiments with different configurations, use the batch experiment script.

### Batch Script: `run_batch_experiments.py`

```bash
python run_batch_experiments.py [OPTIONS]
```

#### Options

| Option | Description |
|--------|-------------|
| `--quick` | Quick test: 1 model, 1 temperature, 20 samples |
| `--full` | Full test: 3 models, 3 temperatures, 100 samples |
| `--output-dir DIR` | Base directory for batch outputs (default: batch_outputs) |

### Examples

#### Quick Batch Test

```bash
# Run a quick batch test (1 model, 1 temperature, 20 samples)
python run_batch_experiments.py --quick
```

This runs:
- Model: small
- Temperature: 0.7
- Samples: 20

#### Default Batch Test

```bash
# Run default batch experiments
python run_batch_experiments.py
```

This runs:
- Models: small, 4bit
- Temperatures: 0.3, 0.7
- Samples: 50
- Total: 4 experiments

#### Full Batch Test

```bash
# Run comprehensive batch experiments
python run_batch_experiments.py --full
```

This runs:
- Models: small, 4bit, 8bit
- Temperatures: 0.3, 0.7, 1.0
- Samples: 100
- Total: 9 experiments

#### Custom Output Directory

```bash
# Save batch results to custom directory
python run_batch_experiments.py --output-dir my_batch_results
```

## Output Structure

### Single Experiment Output

When you run `cli.py`, results are saved to the specified output directory:

```
outputs/
├── generation_log.jsonl      # Detailed generation logs
├── results.csv                # Summary metrics
└── icw_evaluation.png         # Visualization plots
```

### Batch Experiment Output

When you run `run_batch_experiments.py`, results are organized by experiment:

```
batch_outputs/
├── small_T0.7_N50_20231210_143022/
│   ├── generation_log.jsonl
│   ├── results.csv
│   ├── icw_evaluation.png
│   └── experiment_log.txt
├── 4bit_T0.7_N50_20231210_144531/
│   └── ...
├── batch_summary.csv          # Combined results from all experiments
└── experiment_log.json        # Log of all experiments run
```

## Understanding Results

### Results CSV

Each experiment produces a `results.csv` file with these columns:

- `Method`: Watermarking method (Unicode ICW, Initials ICW, Lexical ICW, Acrostics ICW)
- `ROC-AUC`: Detection accuracy (0.5 = random, 1.0 = perfect)
- `TPR@1%FPR`: True positive rate at 1% false positive rate
- `TPR@10%FPR`: True positive rate at 10% false positive rate

### Interpreting ROC-AUC

- **0.90-1.00**: Strong watermark, easily detectable
- **0.70-0.90**: Moderate watermark, detectable
- **0.55-0.70**: Weak watermark
- **0.50-0.55**: Very weak/no watermark (near random)

### Batch Summary

The `batch_summary.csv` combines all results and includes:

- `Experiment`: Experiment name
- `Model`: Model strategy used
- `Temperature`: Temperature parameter
- `Method`: Watermarking method
- All metrics from individual experiments

## Hardware Recommendations

### MacBook (M1/M2/M3)

```bash
python cli.py --model small --samples 50
```

### GPU with 8GB VRAM

```bash
python cli.py --model 4bit --samples 100
```

### GPU with 16GB VRAM

```bash
python cli.py --model 8bit --samples 100
# or
python cli.py --model full --samples 100
```

### GPU with 24GB+ VRAM

```bash
python cli.py --model full --samples 200
```

### CPU Only (No GPU)

```bash
python cli.py --model cpu --samples 20
```

Note: CPU mode is very slow. Use small sample sizes.

## Advanced Usage

### Custom Model Configuration

If you need to use a different model, you can modify `memory_config.py` to add new model profiles.

### Old Method (Manual Editing)

You can still use the original method by editing `main.py` directly:

```python
# In main.py, lines 21-23
MEMORY_STRATEGY = "4bit"
TEMPERATURE = 0.7
NUM_SAMPLES = 50
```

Then run:

```bash
python main.py
```

However, the CLI method is recommended for easier configuration management.

## Troubleshooting

### Out of Memory Errors

If you get OOM errors:

1. Try a smaller model: `--model small`
2. Reduce samples: `--samples 20`
3. Use CPU mode: `--model cpu` (slow but works)

### CUDA/GPU Not Detected

The script will automatically fall back to CPU mode. To force a specific configuration:

```bash
# Force small model on CPU
python cli.py --model cpu --samples 20
```

### Invalid Temperature

Temperature must be between 0 and 2.0:

```bash
# Valid
python cli.py --temperature 0.7

# Invalid (will error)
python cli.py --temperature 3.0
```

### Invalid Sample Count

Samples must be between 1 and 1000:

```bash
# Valid
python cli.py --samples 50

# Invalid (will error)
python cli.py --samples 5000
```

## Tips

1. **Start small**: Test with `--model small --samples 20` first
2. **Use batch mode**: For systematic comparisons, use `run_batch_experiments.py`
3. **Check hardware**: Run `python cli.py --list-models` to see recommendations
4. **Save results**: Use `--output` to organize experiments in different directories
5. **Monitor memory**: Watch your GPU/RAM usage during experiments

## Migration from Old Workflow

### Before (Manual Editing)

```python
# Edit main.py
MEMORY_STRATEGY = "4bit"
TEMPERATURE = 0.7
NUM_SAMPLES = 50

# Run
python main.py
```

### After (CLI)

```bash
# Single command
python cli.py --model 4bit --temperature 0.7 --samples 50
```

The CLI approach is cleaner and allows for easier automation and experimentation.
