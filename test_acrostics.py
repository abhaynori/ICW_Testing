"""
Quick sanity checks for the fixed-prefix acrostics detector.
"""

from main import acrostics_detector, secret_sequence
from research_utils import acrostics_metrics, sanitize_generated_text


def show_case(label, text):
    clean = sanitize_generated_text(text)
    details = acrostics_metrics(clean, secret_sequence)
    score = acrostics_detector(text, secret_sequence)

    print("=" * 70)
    print(label)
    print("=" * 70)
    print(clean)
    print(f"Initials: {details['initials']}")
    print(f"Target:   {details['expected_initials']}")
    print(f"Prefix match rate:      {details['prefix_match_rate']:.3f}")
    print(f"Position accuracy:      {details['sentence_match_rate']:.3f}")
    print(f"Secret coverage:        {details['secret_coverage']:.3f}")
    print(f"Full secret realized:   {details['full_secret_realized']:.0f}")
    print(f"Sentence count error:   {details['sentence_count_error']:.0f}")
    print(f"Levenshtein distance:   {details['levenshtein_distance']:.0f}")
    print(f"Detector z-score:       {score:.4f}")
    print()


def main():
    show_case(
        "Perfect 6-sentence acrostic",
        "Start here. End carefully. Calm answers help. Reason well. "
        "Explain clearly. Trust evidence.",
    )

    show_case(
        "Longer response with correct first six initials",
        "Start here. End carefully. Calm answers help. Reason well. "
        "Explain clearly. Trust evidence. Extra sentence.",
    )

    show_case(
        "Short response",
        "Start here. End carefully.",
    )

    show_case(
        "Artifact-contaminated response",
        "Answer text.<tool_call>\nuser\nOops",
    )


if __name__ == "__main__":
    main()
