# GRPO Training Quick Start

## Installation

```bash
pip install trl accelerate peft
```

## Train a Model (3 steps)

### Step 1: Train
```bash
python grpo_train.py --model small --method acrostics --epochs 3 --samples 100
```

### Step 2: Test
```bash
python cli.py --model-path "grpo_models/acrostics_small_*/final_model" --samples 50
```

### Step 3: Compare
```bash
python compare_models.py --base small --trained "grpo_models/acrostics_small_*/final_model" --method acrostics
```

## Common Commands

### Train on Each Method

```bash
# Unicode (moderate improvement expected)
python grpo_train.py --model small --method unicode --epochs 3

# Initials (lower improvement expected)
python grpo_train.py --model small --method initials --epochs 3

# Lexical (moderate improvement expected)
python grpo_train.py --model small --method lexical --epochs 3

# Acrostics (best improvement expected)
python grpo_train.py --model small --method acrostics --epochs 3
```

### Quick Test (Fast)

```bash
python grpo_train.py --model small --method unicode --epochs 2 --samples 50
```

### Production Training (Better Results)

```bash
python grpo_train.py --model small --method acrostics --epochs 5 --samples 200
```

## Tips

- ✓ Start with **acrostics** (easiest, shows best improvement)
- ✓ Use **small** model (fast, works on most hardware)
- ✓ Use **50-100 samples** for quick tests, **200+** for production
- ✓ Run **compare_models.py** to quantify improvement
- ✗ Don't use 4bit (doesn't train well)

## Expected Results

| Method | Expected ROC-AUC Improvement |
|--------|------------------------------|
| Acrostics | +0.1 to +0.3 |
| Unicode | +0.05 to +0.15 |
| Lexical | +0.05 to +0.15 |
| Initials | +0.03 to +0.10 |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | `--batch-size 2` or `--model cpu` |
| Too slow | `--samples 50` or use GPU |
| No improvement | `--epochs 5 --samples 200` |

## Full Documentation

See [GRPO_TRAINING.md](GRPO_TRAINING.md) for complete guide.
