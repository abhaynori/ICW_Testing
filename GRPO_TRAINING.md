## GRPO Training for ICW Watermarking

This guide explains how to use Group Relative Policy Optimization (GRPO) to train models that produce better watermarks.

## Overview

GRPO training fine-tunes language models to better follow watermarking instructions by:
1. **Using existing detectors as rewards**: Your detector functions (unicode_detector, initials_detector, etc.) provide the training signal
2. **Reinforcement learning**: Models learn to maximize detector scores through policy optimization
3. **No manual labeling needed**: The detectors automatically score generated text

## Why Use GRPO?

Base models often struggle with watermarking instructions:
- **Unicode ICW**: Small models can't insert actual Unicode characters
- **Initials ICW**: Models don't consistently bias word choice
- **Lexical ICW**: Models can't track and use the green word list
- **Acrostics ICW**: Models may produce weak acrostic patterns

**GRPO training teaches models** to better follow these instructions by directly optimizing for detector scores.

## Installation

First, install the TRL library for GRPO:

```bash
pip install trl accelerate peft
```

Update your full requirements:

```bash
pip install torch transformers datasets scikit-learn nltk python-Levenshtein sentence-transformers pandas matplotlib trl accelerate peft
```

## Quick Start

### 1. Train a Model

```bash
# Train on Unicode watermarking (recommended for first try)
python grpo_train.py --model small --method unicode --epochs 3 --samples 100

# Train on Acrostics (usually shows best results)
python grpo_train.py --model small --method acrostics --epochs 3 --samples 100
```

This will:
- Load the base model
- Generate baseline statistics
- Train for 3 epochs using detector scores as rewards
- Save the trained model to `grpo_models/`

### 2. Test the Trained Model

```bash
# Find the trained model path (it has a timestamp)
ls grpo_models/

# Test it (replace with your actual path)
python cli.py --model-path grpo_models/unicode_small_20231210_143022/final_model --samples 50
```

### 3. Compare Base vs Trained

```bash
# Compare performance improvement
python compare_models.py \
    --base small \
    --trained "grpo_models/unicode_small_*/final_model" \
    --method unicode \
    --samples 100
```

## Training Options

### Basic Training Script: `grpo_train.py`

```bash
python grpo_train.py [OPTIONS]
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | small | Model to train: small, 8bit, cpu (NOT 4bit) |
| `--method` | required | Watermarking method: unicode, initials, lexical, acrostics |
| `--samples` | 100 | Number of training samples |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 4 | Training batch size |
| `--learning-rate` | 1e-5 | Learning rate |
| `--output-dir` | grpo_models | Output directory |

### Important Notes

**Model Selection:**
- ✓ **small** (Qwen 1.5B): Best for training, fast, works on most hardware
- ✓ **8bit**: Better quality, needs 8GB+ VRAM
- ✓ **cpu**: Slow but works everywhere
- ✗ **4bit**: NOT recommended - quantized weights don't train well

**Method Selection:**
- **unicode**: Fast to train, clear signal
- **acrostics**: Usually shows best improvement
- **lexical**: More complex, may need more epochs
- **initials**: Similar to lexical

## Training Process

### What Happens During Training

1. **Baseline Computation** (before training):
   ```
   Computing baseline statistics for unicode...
     Progress: 10/50
     Progress: 20/50
     ...
   ✓ Baseline computed: mean=0.123, std=0.456
   ```

2. **GRPO Training** (during training):
   ```
   Epoch 1/3
   Step 10: reward=0.45, loss=2.34
   Step 20: reward=0.78, loss=1.89
   ...
   ```

   The reward increases as the model learns to produce better watermarks.

3. **Model Saving** (after training):
   ```
   ✓ Model saved to: grpo_models/unicode_small_20231210_143022/final_model
   ✓ Metadata saved to: grpo_models/unicode_small_20231210_143022/final_model/training_metadata.json
   ```

### Training Time Estimates

| Model | Samples | Epochs | Time (GPU) | Time (CPU) |
|-------|---------|--------|------------|------------|
| small | 100 | 3 | ~30 min | ~3 hours |
| small | 200 | 5 | ~1.5 hours | ~8 hours |
| 8bit | 100 | 3 | ~1 hour | Not recommended |

## Examples

### Example 1: Quick Test

Train and test Unicode watermarking quickly:

```bash
# Train (takes ~15 min on GPU)
python grpo_train.py --model small --method unicode --epochs 2 --samples 50

# Test the trained model
python cli.py --model-path "grpo_models/unicode_small_*/final_model" --samples 30

# Compare with base
python compare_models.py --base small --trained "grpo_models/unicode_small_*/final_model" --method unicode
```

### Example 2: Production Training

Train with more data for better results:

```bash
# Train thoroughly (takes ~1 hour on GPU)
python grpo_train.py --model small --method acrostics --epochs 5 --samples 200

# Evaluate with more samples
python cli.py --model-path "grpo_models/acrostics_small_*/final_model" --samples 100

# Detailed comparison
python compare_models.py --base small --trained "grpo_models/acrostics_small_*/final_model" --method acrostics --samples 100
```

### Example 3: Train on All Methods

Train separate models for each watermarking method:

```bash
# Train all methods (sequential)
for method in unicode initials lexical acrostics; do
    echo "Training $method..."
    python grpo_train.py --model small --method $method --epochs 3 --samples 100
done

# Compare all methods
for method in unicode initials lexical acrostics; do
    echo "Comparing $method..."
    python compare_models.py --base small --trained "grpo_models/${method}_small_*/final_model" --method $method --samples 50
done
```

### Example 4: Larger Model Training

Train a larger model for better quality (requires 8GB+ VRAM):

```bash
# Train 8-bit model
python grpo_train.py --model 8bit --method acrostics --epochs 3 --samples 150 --batch-size 2

# Test it
python cli.py --model-path "grpo_models/acrostics_8bit_*/final_model" --samples 50
```

## Understanding Results

### Reward Function

The reward function uses your existing detectors:

```python
# For Unicode ICW
reward = (unicode_count / num_words - baseline_mean) / baseline_std

# For Initials ICW
reward = ((green_count - gamma * num_words) / sqrt(...) - baseline_mean) / baseline_std

# For Lexical ICW
reward = ((green_count - gamma * candidates) / sqrt(...) - baseline_mean) / baseline_std

# For Acrostics ICW
reward = ((mu - levenshtein_distance) / sigma - baseline_mean) / baseline_std
```

Higher rewards = better watermarks.

### Interpreting Improvements

After running `compare_models.py`, you'll see:

```
Unicode ICW:
  ROC-AUC:    0.5234 → 0.8756 (+0.3522)
  TPR@1%FPR:  0.0120 → 0.4580 (+0.4460)
  TPR@10%FPR: 0.0890 → 0.7234 (+0.6344)
  ✓✓ Significant improvement!
```

**Interpretation:**
- **ROC-AUC improvement > 0.1**: Significant (excellent)
- **ROC-AUC improvement > 0.05**: Moderate (good)
- **ROC-AUC improvement > 0**: Slight (some benefit)
- **ROC-AUC improvement ≤ 0**: No improvement (need more training)

### Expected Improvements by Method

Based on our design, here's what to expect:

| Method | Expected Improvement | Notes |
|--------|---------------------|-------|
| **Acrostics** | High (0.1-0.3 ROC-AUC) | Sentence-level task, easiest to learn |
| **Unicode** | Moderate (0.05-0.15) | Clear signal but hard for small models |
| **Lexical** | Moderate (0.05-0.15) | Needs to learn word selection |
| **Initials** | Low-Moderate (0.03-0.10) | Similar to lexical but harder |

## Troubleshooting

### Issue: Training is very slow

**Solution:**
- Use GPU if available
- Reduce `--samples` (try 50 instead of 100)
- Reduce `--batch-size` (try 2 instead of 4)
- Use `--model small` instead of larger models

### Issue: Out of memory during training

**Solution:**
```bash
# Reduce batch size
python grpo_train.py --model small --method unicode --batch-size 2

# Use CPU mode (slow but works)
python grpo_train.py --model cpu --method unicode --samples 50
```

### Issue: No improvement in watermark quality

**Possible causes:**
1. **Not enough training**: Try more epochs (`--epochs 5`)
2. **Not enough data**: Try more samples (`--samples 200`)
3. **Method too hard**: Try acrostics first (easiest method)
4. **Model too small**: Models struggle with complex instructions

**Solutions:**
```bash
# More thorough training
python grpo_train.py --model small --method acrostics --epochs 5 --samples 200

# Try different learning rate
python grpo_train.py --model small --method unicode --learning-rate 5e-6 --epochs 5
```

### Issue: Training crashes or errors

**Check:**
1. TRL installed: `pip install trl accelerate peft`
2. Enough disk space for model checkpoints
3. Using compatible model (not 4bit)

## Advanced Usage

### Custom Hyperparameters

```bash
# Lower learning rate for stability
python grpo_train.py --model small --method unicode --learning-rate 5e-6

# More epochs for better convergence
python grpo_train.py --model small --method acrostics --epochs 10

# Smaller batches for memory
python grpo_train.py --model small --method lexical --batch-size 1 --samples 200
```

### Accessing Trained Models Programmatically

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load trained model
model_path = "grpo_models/unicode_small_20231210_143022/final_model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Use for generation
messages = [{"role": "user", "content": "Explain photosynthesis"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
```

### Training Metadata

Each trained model saves metadata:

```bash
cat grpo_models/unicode_small_20231210_143022/final_model/training_metadata.json
```

```json
{
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "model_strategy": "small",
  "method": "unicode",
  "num_train_samples": 100,
  "num_epochs": 3,
  "batch_size": 4,
  "learning_rate": 1e-05,
  "baseline_mean": 0.123,
  "baseline_std": 0.456,
  "timestamp": "2023-12-10T14:30:22"
}
```

## Best Practices

1. **Start small**: Begin with `--model small --samples 50 --epochs 2` to test
2. **Check baseline**: Make sure baseline statistics are computed correctly
3. **Monitor rewards**: Watch if rewards increase during training
4. **Compare thoroughly**: Always run `compare_models.py` to quantify improvement
5. **Save checkpoints**: Models are saved automatically, keep good ones
6. **Method-specific training**: Train separate models for each method
7. **Use validation**: Test on different samples than training

## Integration with Experiments

### Use Trained Models in Batch Experiments

Modify `run_batch_experiments.py` to use trained models:

```python
# Add to experiment configurations
TRAINED_MODELS_CONFIG = {
    "models": [
        "grpo_models/unicode_small_*/final_model",
        "grpo_models/acrostics_small_*/final_model"
    ],
    "temperatures": [0.7],
    "samples": 100
}
```

### Compare Multiple Training Runs

```bash
# Train with different hyperparameters
python grpo_train.py --model small --method unicode --epochs 3 --learning-rate 1e-5 --samples 100
python grpo_train.py --model small --method unicode --epochs 3 --learning-rate 5e-6 --samples 100

# Compare them
python compare_models.py --base small --trained "grpo_models/unicode_small_<run1>/final_model" --method unicode
python compare_models.py --base small --trained "grpo_models/unicode_small_<run2>/final_model" --method unicode
```

## FAQ

### Q: Which method should I train first?
**A:** Start with **acrostics** - it usually shows the clearest improvement because it's a sentence-level task that's easier for models to learn.

### Q: How long does training take?
**A:** On a GPU, 100 samples for 3 epochs takes ~30 minutes for the small model. On CPU, expect 3-5x longer.

### Q: Can I train the 4-bit quantized model?
**A:** No, 4-bit quantization freezes weights in a way that prevents proper gradient updates. Use `small`, `8bit`, or `cpu` instead.

### Q: Will this work for any model?
**A:** Yes, as long as it's a chat-based model that supports `apply_chat_template()`. The code is designed to be model-agnostic.

### Q: How much improvement should I expect?
**A:** Typically 0.05-0.3 ROC-AUC improvement. Acrostics shows the most improvement, Unicode/Lexical show moderate, Initials shows less.

### Q: Can I use my own reward function?
**A:** Yes! The existing detectors are used as rewards. You can modify `WatermarkRewardFunction` in `grpo_train.py` to use custom scoring.

## Next Steps

After training models:

1. **Quantify improvements**: Run comprehensive comparisons
2. **Test robustness**: Use trained models with paraphrase attacks
3. **Analyze failures**: Check which samples still fail detection
4. **Iterate**: Retrain with more data or different hyperparameters
5. **Document**: Save metadata about successful training runs

## References

- GRPO Paper: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- TRL Library: [https://github.com/huggingface/trl](https://github.com/huggingface/trl)
- ICW Paper: [Instruction-based Content Watermarking](https://arxiv.org/abs/2505.16934)
