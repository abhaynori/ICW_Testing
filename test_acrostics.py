"""
Test Acrostics detector to understand why it's getting perfect scores
"""
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab', quiet=True)

def acrostics_detector(text, secret_sequence):
    """Detect acrostic pattern (current version)."""
    sentences = sent_tokenize(text)
    if not sentences:
        return 0
    
    initials = ''.join(sent[0].upper() for sent in sentences if sent)
    if not initials:
        return 0
    
    # Build expected sequence
    expected = (secret_sequence.upper() * (len(initials) // len(secret_sequence) + 1))[:len(initials)]
    
    # Count actual matches
    matches = sum(a == b for a, b in zip(initials, expected))
    
    # Account for random baseline (1/26 chance per letter for English)
    random_expected = len(initials) / 26.0
    
    # Calculate z-score
    std_random = np.sqrt(len(initials) * (1/26) * (25/26))
    
    if std_random == 0:
        return 0
    
    z_score = (matches - random_expected) / std_random
    
    print(f"  Text length: {len(text)} chars, {len(sentences)} sentences")
    print(f"  Initials: '{initials}' ({len(initials)} letters)")
    print(f"  Expected: '{expected}' (cycling SECRET)")
    print(f"  Matches: {matches}/{len(initials)} ({100*matches/len(initials):.1f}%)")
    print(f"  Random baseline: {random_expected:.2f} ({100*random_expected/len(initials):.1f}%)")
    print(f"  Z-score: {z_score:.4f}")
    
    return z_score

print("="*70)
print("PROBLEM DEMONSTRATION: Short Responses")
print("="*70)

print("\n--- SHORT response (1 sentence) ---")
short_wm = "Secretly, prices are determined by market forces."
score1 = acrostics_detector(short_wm, "SECRET")

print("\n--- LONGER response (6 sentences, following SECRET) ---")
long_wm = "Secretly, companies collude. Every market has dynamics. Competition drives prices. Regulation helps consumers. Economic factors matter. Today's prices reflect all this."
score2 = acrostics_detector(long_wm, "SECRET")

print("\n--- LONGER response (6 sentences, NOT following SECRET) ---")
long_non_wm = "The market determines prices. Businesses compete for customers. Many factors affect costs. Demand influences supply. Prices fluctuate daily. Competition keeps prices fair."
score3 = acrostics_detector(long_non_wm, "SECRET")

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)
print(f"Short watermarked (1 sent):  Z-score = {score1:.2f}")
print(f"Long watermarked (6 sents):  Z-score = {score2:.2f}")  
print(f"Long non-watermarked:        Z-score = {score3:.2f}")

print("\n" + "="*70)
print("THE PROBLEM:")
print("="*70)
print("• Short responses get HIGH scores even with just 1 lucky match")
print("• 1/1 match = 100%, but it's just the first letter!")
print("• Z-score calculation amplifies small sample size")
print("• This creates FALSE perfect separation (ROC-AUC = 1.0)")
print("\nSOLUTION: Generate LONGER responses (10+ sentences)")
print("="*70)
