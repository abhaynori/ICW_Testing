# ICW Replication on Phi-3

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