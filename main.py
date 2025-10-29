import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import nltk
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
import json
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ===== CONFIGURATION =====
MEMORY_STRATEGY = "small"  # Optimized for Apple Silicon
TEMPERATURE = 0.7  # Higher temperature for more natural variation
NUM_SAMPLES = 50  # Increased from 10 for more reliable statistics
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-configure based on memory strategy
from memory_config import get_model_config
config = get_model_config(MEMORY_STRATEGY)
MODEL_NAME = config["model_name"]

print(f"\n{'='*80}")
print(f"ICW WATERMARKING EVALUATION")
print(f"{'='*80}")
print(f"Memory Strategy: {MEMORY_STRATEGY}")
print(f"Model: {MODEL_NAME}")
print(f"Temperature: {TEMPERATURE}")
print(f"Samples: {NUM_SAMPLES}")
print(f"{'='*80}\n")

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model_kwargs = {
    "device_map": config.get("device_map", "auto"),
    "trust_remote_code": True,
    "low_cpu_mem_usage": True
}

if config.get("quantization"):
    model_kwargs["quantization_config"] = config["quantization"]
elif config.get("dtype"):
    model_kwargs["torch_dtype"] = config["dtype"]

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded successfully!\n")

# Generation configuration
generation_config = {
    "max_new_tokens": 500,
    "min_new_tokens": 100,
    "do_sample": True,
    "temperature": TEMPERATURE,
    "top_p": 0.9,
    "pad_token_id": tokenizer.pad_token_id,
}

def generate_response(messages):
    """Generate response using chat template."""
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(input_ids, **generation_config)
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

def log_generation(query, prompt, output, method, filename="generation_log.jsonl"):
    """Log generation to file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "query": query,
        "prompt": prompt,
        "output": output
    }
    with open(os.path.join(OUTPUT_DIR, filename), "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Load dataset
print("Loading dataset...")
eli5 = load_dataset("sentence-transformers/eli5", "pair", split="train")["question"][:NUM_SAMPLES]
print(f"✓ Loaded {len(eli5)} questions\n")

# ============================================================================
# WATERMARKING STRATEGIES
# ============================================================================

# --- 1. Unicode ICW (Exclamation Points) ---
def unicode_embed_prompt(query):
    """Prompt for Unicode watermarking using exclamation points."""
    system_msg = (
        "Follow this formatting rule STRICTLY: Place an exclamation mark (!) after EVERY single word. "
        "Example: 'The! sky! is! blue!' "
        "Write a detailed explanation with 5-8 sentences. Every word must end with !."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query}
    ]

def unicode_detector(text):
    """Detect exclamation point pattern (FIXED VERSION)."""
    tokens = text.split()
    if len(tokens) < 5:
        return 0
    
    tokens_with_exclamation = sum(1 for token in tokens if token.endswith('!'))
    n = len(tokens)
    
    # Natural text: ~1.5% of words end with !
    p_natural = 0.015
    
    # Binomial test
    observed = tokens_with_exclamation
    expected = n * p_natural
    std = np.sqrt(n * p_natural * (1 - p_natural))
    
    if std == 0:
        return 0
    
    z_score = (observed - expected) / std
    return z_score

# --- 2. Initials ICW (Vowel Preference) ---
green_letters = set('aeiou')

def initials_embed_prompt(query):
    """Prompt for Initials watermarking."""
    system_msg = (
        f"Writing guideline: Try to start as many words as possible with these letters: {', '.join(sorted(green_letters))}. "
        f"Be natural and informative. Write 5-8 sentences. Don't mention this guideline."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query}
    ]

def initials_detector(text, green_letters, gamma=None):
    """Detect initials watermark (FIXED VERSION)."""
    words = [w for w in text.lower().split() if w and w[0].isalpha()]
    
    if len(words) < 5:
        return 0
    
    green_count = sum(w[0] in green_letters for w in words)
    n = len(words)
    
    # Empirical vowel frequency in English word initials (~38%)
    if gamma is None:
        gamma = 0.38
    
    # Binomial test
    expected = n * gamma
    std = np.sqrt(n * gamma * (1 - gamma))
    
    if std == 0:
        return 0
    
    z_score = (green_count - expected) / std
    return z_score

# --- 3. Lexical ICW (Preferred Words) ---
green_words = set([
    'amazing', 'beautiful', 'bright', 'brilliant', 'excellent', 'fantastic',
    'great', 'happy', 'incredible', 'joyful', 'magnificent', 'outstanding',
    'perfect', 'quick', 'radiant', 'splendid', 'superb', 'terrific',
    'unique', 'vibrant', 'wonderful', 'enable', 'enhance', 'improve',
    'increase', 'optimize', 'strengthen', 'utilize', 'achieve', 'create'
])

def lexical_embed_prompt(query):
    """Prompt for Lexical watermarking."""
    word_list = ', '.join(sorted(list(green_words))[:15])  # Show subset
    system_msg = (
        f"Writing style: Prefer using descriptive words like: {word_list}, and similar positive adjectives/verbs. "
        f"Write naturally and informatively with 5-8 sentences."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query}
    ]

def lexical_detector(text, green_words, gamma=None):
    """Detect lexical watermark (FIXED VERSION)."""
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    
    # Filter to adjectives, adverbs, verbs
    candidates = [word for word, tag in tagged 
                  if (tag.startswith('JJ') or tag.startswith('RB') or tag.startswith('VB'))
                  and word.isalpha()]
    
    if len(candidates) < 5:
        return 0
    
    green_count = sum(word in green_words for word in candidates)
    n = len(candidates)
    
    # Estimate gamma: green words as fraction of all descriptive words
    if gamma is None:
        # Conservative: assume green words are 2% of descriptive vocabulary
        gamma = 0.02
    
    # Binomial test
    expected = n * gamma
    std = np.sqrt(n * gamma * (1 - gamma))
    
    if std == 0:
        return 0
    
    z_score = (green_count - expected) / std
    return z_score

# --- 4. Acrostics ICW (SECRET Pattern) ---
secret_sequence = "SECRET"

def acrostics_embed_prompt(query):
    """Prompt for Acrostics watermarking."""
    system_msg = (
        f"Structure your response as an acrostic: Start each sentence with letters spelling '{secret_sequence}' in order. "
        f"Repeat the pattern if you write more than {len(secret_sequence)} sentences. "
        f"Write naturally with 6-12 sentences. Be informative and don't mention this pattern."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query}
    ]

def acrostics_detector(text, secret_sequence):
    """Detect acrostic pattern (FIXED VERSION)."""
    sentences = sent_tokenize(text)
    
    if len(sentences) < 3:
        return 0
    
    initials = ''.join(sent[0].upper() for sent in sentences if sent and sent[0].isalpha())
    
    if len(initials) < 3:
        return 0
    
    # Compare against repeating secret sequence
    n = len(initials)
    expected_seq = (secret_sequence.upper() * (n // len(secret_sequence) + 1))[:n]
    
    # Count matches
    matches = sum(a == b for a, b in zip(initials, expected_seq))
    
    # Random baseline: 1/26 chance per letter
    p_random = 1.0 / 26.0
    expected_matches = n * p_random
    std = np.sqrt(n * p_random * (1 - p_random))
    
    if std == 0:
        return 0
    
    z_score = (matches - expected_matches) / std
    return z_score

# ============================================================================
# TEXT GENERATION
# ============================================================================

print("="*80)
print("GENERATING WATERMARKED TEXT")
print("="*80)

# Generate all watermarked texts
unicode_watermarked = []
initials_watermarked = []
lexical_watermarked = []
acrostics_watermarked = []
non_wm_texts = []

for i, query in enumerate(eli5):
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{NUM_SAMPLES}")
    
    # Unicode ICW
    messages = unicode_embed_prompt(query)
    response = generate_response(messages)
    unicode_watermarked.append(response)
    prompt_str = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}"
    log_generation(query, prompt_str, response, "Unicode ICW")
    
    # Initials ICW
    messages = initials_embed_prompt(query)
    response = generate_response(messages)
    initials_watermarked.append(response)
    prompt_str = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}"
    log_generation(query, prompt_str, response, "Initials ICW")
    
    # Lexical ICW
    messages = lexical_embed_prompt(query)
    response = generate_response(messages)
    lexical_watermarked.append(response)
    prompt_str = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}"
    log_generation(query, prompt_str, response, "Lexical ICW")
    
    # Acrostics ICW
    messages = acrostics_embed_prompt(query)
    response = generate_response(messages)
    acrostics_watermarked.append(response)
    prompt_str = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}"
    log_generation(query, prompt_str, response, "Acrostics ICW")
    
    # Non-watermarked baseline
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Provide clear, informative answers."},
        {"role": "user", "content": query}
    ]
    response = generate_response(messages)
    non_wm_texts.append(response)
    prompt_str = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}"
    log_generation(query, prompt_str, response, "Non-watermarked")

print(f"✓ Generated {NUM_SAMPLES} samples for each method\n")

# ============================================================================
# WATERMARK DETECTION & ANALYSIS
# ============================================================================

def analyze_watermark_compliance(texts, detector, detector_args, method_name):
    """Analyze how well watermarks were embedded."""
    print(f"\n{'='*80}")
    print(f"{method_name} - Compliance Analysis")
    print(f"{'='*80}")
    
    # Show first 3 examples
    for i in range(min(3, len(texts))):
        print(f"\nExample {i+1}:")
        print(f"Text: {texts[i][:200]}...")
        
        # Method-specific analysis
        if "Unicode" in method_name:
            tokens = texts[i].split()
            excl_count = sum(1 for t in tokens if t.endswith('!'))
            ratio = excl_count / len(tokens) if tokens else 0
            print(f"  Tokens: {len(tokens)}, With !: {excl_count}, Ratio: {ratio:.1%}")
            print(f"  Expected: ~95-100% for perfect watermark")
        
        elif "Initials" in method_name:
            words = [w for w in texts[i].lower().split() if w and w[0].isalpha()]
            vowel_count = sum(1 for w in words if w[0] in green_letters)
            ratio = vowel_count / len(words) if words else 0
            print(f"  Words: {len(words)}, Vowel starts: {vowel_count}, Ratio: {ratio:.1%}")
            print(f"  Expected: >50% for watermark, ~38% for natural")
        
        elif "Lexical" in method_name:
            tokens = word_tokenize(texts[i].lower())
            green_found = [t for t in tokens if t in green_words]
            print(f"  Green words used: {green_found[:5]}")
            print(f"  Count: {len(green_found)}")
        
        elif "Acrostics" in method_name:
            sentences = sent_tokenize(texts[i])
            initials = ''.join(s[0].upper() for s in sentences if s and s[0].isalpha())
            expected = (secret_sequence.upper() * (len(initials) // len(secret_sequence) + 1))[:len(initials)]
            matches = sum(a == b for a, b in zip(initials, expected))
            print(f"  Sentences: {len(sentences)}")
            print(f"  Initials: {initials}")
            print(f"  Expected: {expected}")
            print(f"  Matches: {matches}/{len(initials)} ({matches/len(initials)*100:.0f}%)")

def evaluate_strategy(wm_texts, detector, detector_args, non_wm_texts, method_name):
    """Evaluate watermarking strategy with detailed statistics."""
    
    # Calculate scores
    wm_scores = [detector(text, *detector_args) for text in wm_texts]
    non_wm_scores = [detector(text, *detector_args) for text in non_wm_texts]
    
    # Statistics
    print(f"\n{'='*80}")
    print(f"{method_name} - Detection Scores")
    print(f"{'='*80}")
    
    print(f"\nWatermarked texts (n={len(wm_scores)}):")
    print(f"  Mean:   {np.mean(wm_scores):7.3f}")
    print(f"  Median: {np.median(wm_scores):7.3f}")
    print(f"  Std:    {np.std(wm_scores):7.3f}")
    print(f"  Range:  [{np.min(wm_scores):.3f}, {np.max(wm_scores):.3f}]")
    print(f"  Q1-Q3:  [{np.percentile(wm_scores, 25):.3f}, {np.percentile(wm_scores, 75):.3f}]")
    
    print(f"\nNon-watermarked texts (n={len(non_wm_scores)}):")
    print(f"  Mean:   {np.mean(non_wm_scores):7.3f}")
    print(f"  Median: {np.median(non_wm_scores):7.3f}")
    print(f"  Std:    {np.std(non_wm_scores):7.3f}")
    print(f"  Range:  [{np.min(non_wm_scores):.3f}, {np.max(non_wm_scores):.3f}]")
    print(f"  Q1-Q3:  [{np.percentile(non_wm_scores, 25):.3f}, {np.percentile(non_wm_scores, 75):.3f}]")
    
    separation = np.mean(wm_scores) - np.mean(non_wm_scores)
    pooled_std = np.sqrt((np.var(wm_scores) + np.var(non_wm_scores)) / 2)
    cohens_d = separation / pooled_std if pooled_std > 0 else 0
    
    print(f"\nSeparation:")
    print(f"  Mean difference: {separation:7.3f}")
    print(f"  Cohen's d:       {cohens_d:7.3f} ", end="")
    if abs(cohens_d) < 0.2:
        print("(negligible)")
    elif abs(cohens_d) < 0.5:
        print("(small)")
    elif abs(cohens_d) < 0.8:
        print("(medium)")
    else:
        print("(large)")
    
    # ROC-AUC
    labels = [1] * len(wm_scores) + [0] * len(non_wm_scores)
    scores = wm_scores + non_wm_scores
    
    if len(set(scores)) < 2:
        print(f"\n⚠️  WARNING: All scores identical - ROC-AUC undefined")
        return 0.5, 0.0
    
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # TPR at 1% FPR
    tpr_at_1fpr = tpr[np.where(fpr <= 0.01)[0][-1]] if any(fpr <= 0.01) else 0
    
    # TPR at 5% FPR
    tpr_at_5fpr = tpr[np.where(fpr <= 0.05)[0][-1]] if any(fpr <= 0.05) else 0
    
    print(f"\nMetrics:")
    print(f"  ROC-AUC:    {auc:.4f}")
    print(f"  TPR@1%FPR:  {tpr_at_1fpr:.4f}")
    print(f"  TPR@5%FPR:  {tpr_at_5fpr:.4f}")
    
    return auc, tpr_at_1fpr, tpr_at_5fpr

# ============================================================================
# RUN EVALUATIONS
# ============================================================================

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

results = []

# Unicode ICW
analyze_watermark_compliance(unicode_watermarked, unicode_detector, (), "Unicode ICW")
unicode_auc, unicode_t1, unicode_t5 = evaluate_strategy(
    unicode_watermarked, unicode_detector, (), non_wm_texts, "Unicode ICW"
)
results.append({"Method": "Unicode ICW", "ROC-AUC": unicode_auc, "TPR@1%FPR": unicode_t1, "TPR@5%FPR": unicode_t5})

# Initials ICW
analyze_watermark_compliance(initials_watermarked, initials_detector, (green_letters,), "Initials ICW")
initials_auc, initials_t1, initials_t5 = evaluate_strategy(
    initials_watermarked, initials_detector, (green_letters,), non_wm_texts, "Initials ICW"
)
results.append({"Method": "Initials ICW", "ROC-AUC": initials_auc, "TPR@1%FPR": initials_t1, "TPR@5%FPR": initials_t5})

# Lexical ICW
analyze_watermark_compliance(lexical_watermarked, lexical_detector, (green_words,), "Lexical ICW")
lexical_auc, lexical_t1, lexical_t5 = evaluate_strategy(
    lexical_watermarked, lexical_detector, (green_words,), non_wm_texts, "Lexical ICW"
)
results.append({"Method": "Lexical ICW", "ROC-AUC": lexical_auc, "TPR@1%FPR": lexical_t1, "TPR@5%FPR": lexical_t5})

# Acrostics ICW
analyze_watermark_compliance(acrostics_watermarked, acrostics_detector, (secret_sequence,), "Acrostics ICW")
acrostics_auc, acrostics_t1, acrostics_t5 = evaluate_strategy(
    acrostics_watermarked, acrostics_detector, (secret_sequence,), non_wm_texts, "Acrostics ICW"
)
results.append({"Method": "Acrostics ICW", "ROC-AUC": acrostics_auc, "TPR@1%FPR": acrostics_t1, "TPR@5%FPR": acrostics_t5})

# ============================================================================
# SUMMARY & VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))

# Save results
results_file = os.path.join(OUTPUT_DIR, "results.csv")
df.to_csv(results_file, index=False)
print(f"\n✓ Results saved to {results_file}")

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ROC-AUC
axes[0].bar(df["Method"], df["ROC-AUC"], color="skyblue", edgecolor="navy")
axes[0].set_ylabel("ROC-AUC", fontsize=11)
axes[0].set_title("ROC-AUC Score", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, 1)
axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# TPR@1%FPR
axes[1].bar(df["Method"], df["TPR@1%FPR"], color="lightcoral", edgecolor="darkred")
axes[1].set_ylabel("TPR @ 1% FPR", fontsize=11)
axes[1].set_title("True Positive Rate at 1% FPR", fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

# TPR@5%FPR
axes[2].bar(df["Method"], df["TPR@5%FPR"], color="lightgreen", edgecolor="darkgreen")
axes[2].set_ylabel("TPR @ 5% FPR", fontsize=11)
axes[2].set_title("True Positive Rate at 5% FPR", fontsize=12, fontweight="bold")
axes[2].set_ylim(0, 1)
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_file = os.path.join(OUTPUT_DIR, "icw_evaluation.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Plots saved to {plot_file}")
plt.show()

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print(f"  - generation_log.jsonl (detailed logs)")
print(f"  - results.csv (summary table)")
print(f"  - icw_evaluation.png (visualizations)")
print("\n")