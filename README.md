# Instruction-based Content Watermarking (ICW) Research

This repository implements and evaluates various Instruction-based Content Watermarking (ICW) techniques for LLM-generated text.

## Features

- **Multiple ICW Strategies**: Unicode, Initials, Lexical, and Acrostics watermarking
- **Multi-Model Support**: Qwen-2.5, Llama-3.1, Phi-4 (all with reasoning capabilities)
- **Flexible Architecture**: Uses `tokenizer.apply_chat_template()` for easy model switching
- **Configurable Temperature**: Control generation randomness
- **Comprehensive Logging**: All prompts and outputs saved to JSONL files
- **Evaluation Metrics**: ROC-AUC and T@1%FPR
- **Robustness Testing**: Paraphrase attacks and IPI simulation

## Installation

```bash
pip install torch transformers datasets scikit-learn nltk python-Levenshtein sentence-transformers pandas matplotlib
```

## Configuration

Edit the configuration section at the top of `main.py`:

```python
# Configuration
TEMPERATURE = 0.7  # Range: 0.0 (deterministic) to 2.0 (very random)
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Choose from SUPPORTED_MODELS
```

### Supported Models

**All models have reasoning capabilities:**

```python
SUPPORTED_MODELS = {
    "qwen-2.5": "Qwen/Qwen2.5-7B-Instruct",      # 7B, multilingual, strong reasoning
    "llama-3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 8B, excellent instruction following
    "phi-4": "microsoft/Phi-4",                   # Enhanced reasoning (when available)
}
```

To change models, simply update `MODEL_NAME`:
```python
MODEL_NAME = SUPPORTED_MODELS["llama-3.1"]  # Use Llama-3.1 model
# Or directly:
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
```

**Note**: The code uses `tokenizer.apply_chat_template()` which makes it easy to switch between any chat-based models without changing prompt formatting.

### Temperature Settings

- **0.0**: Deterministic, most predictable outputs
- **0.3-0.7**: Balanced creativity and consistency (recommended)
- **1.0+**: Highly creative but less predictable

## Usage

Run the main script:
```bash
python main.py
```

## Output Files

### 1. Generation Logs (`outputs/generation_log.jsonl`)

Each line contains:
```json
{
  "timestamp": "2025-10-10T12:34:56",
  "method": "Unicode ICW",
  "model": "microsoft/Phi-3-mini-4k-instruct",
  "temperature": 0.7,
  "query": "Original question",
  "prompt": "Full prompt with instructions",
  "output": "Generated response"
}
```

### 2. Plots
- `icw_roc_auc.png`: ROC-AUC scores for each method
- `icw_t1fpr.png`: True positive rate at 1% false positive rate

## ICW Methods

### 1. Unicode ICW
Embeds zero-width spaces (U+200B) after words.
- **Detector**: Counts frequency of zero-width spaces

### 2. Initials ICW
Biases word initial letters toward a "green" set (vowels).
- **Detector**: Z-score test for green letter frequency

### 3. Lexical ICW
Encourages specific "green" words while avoiding "red" words.
- **Detector**: Z-score test on adjectives/adverbs/verbs

### 4. Acrostics ICW
Encodes a secret sequence in sentence-initial letters.
- **Detector**: Levenshtein distance from expected pattern

## Evaluation

The script generates:
1. **Console output**: First 2 examples from each method
2. **Summary table**: ROC-AUC and T@1%FPR for all methods
3. **Bar charts**: Visual comparison of metrics

## Robustness Testing

- **Paraphrase Attack**: Uses semantic similarity to simulate text rewriting
- **IPI (Indirect Prompt Injection)**: Tests watermarking via hidden instructions in documents

## Example Commands

### Run with different models:
```python
# Edit main.py:
MODEL_NAME = SUPPORTED_MODELS["llama-3.1"]
```

### Adjust temperature:
```python
# Edit main.py:
TEMPERATURE = 0.3  # More deterministic
```

### Analyze logs:
```python
import json

with open("outputs/generation_log.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        print(f"Method: {entry['method']}")
        print(f"Query: {entry['query']}")
        print(f"Output: {entry['output'][:100]}...")
        print()
```

## Research Questions

1. **Model Comparison**: How do different LLMs respond to watermarking instructions?
2. **Temperature Effects**: How does randomness affect watermark detectability?
3. **Robustness**: Which methods survive paraphrasing attacks?
4. **IPI Vulnerability**: Can watermarks be injected via document content?

## Citation

If you use this code, please cite the original ICW paper:
```
[Add appropriate citation here]
```

## License

MIT License


This script replicates In-Context Watermarking (ICW) from arXiv:2505.16934 on Microsoft's Phi-3-mini-4k-instruct.

## Installation
Install the necessary libraries:

pip install nltk transformers datasets torch scikit-learn numpy python-Levenshtein sentence-transformers accelerate

## Usage
- Run `main.py` to generate watermarked texts, evaluate detection metrics, and test robustness.
- Adjust dataset size (e.g., [:100]) for full runs; requires GPU for efficiency.
- Potential issues: Phi-3 may ignore complex instructions—add few-shot examples if needed.
- For PDFs in IPI, install pymupdf separately if extending.

## Notes
- Uses deterministic generation (temperature=0.0) for reproducibility.
- Expand vocabularies (e.g., green_words) for better realism.