#!/usr/bin/env python3
"""
Apply post-processing attacks to evaluation artifacts and summarize retention.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from difflib import SequenceMatcher

import numpy as np

from main import (
    acrostics_detector,
    green_letters,
    green_words,
    initials_detector,
    lexical_detector,
    secret_sequence,
    unicode_detector,
)
from research_utils import acrostics_metrics, response_stats, safe_sentence_tokenize, tokenize_words


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by", "for",
    "from", "had", "has", "have", "if", "in", "into", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "there", "these", "they", "this", "to", "was", "were",
    "will", "with", "would", "you", "your",
}


def get_detector_and_args(method):
    detector_map = {
        "unicode": (unicode_detector, ()),
        "initials": (initials_detector, (green_letters,)),
        "lexical": (lexical_detector, (green_words,)),
        "acrostics": (acrostics_detector, (secret_sequence,)),
    }
    if method not in detector_map:
        raise ValueError(f"Unsupported method '{method}'")
    return detector_map[method]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def attack_format_cleanup(text: str) -> str:
    cleaned = text.replace("\u200b", "")
    cleaned = cleaned.replace("\n", " ").replace("\t", " ")
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return _normalize_whitespace(cleaned)


def attack_truncate_sentence_50(text: str) -> str:
    sentences = safe_sentence_tokenize(text)
    keep = max(1, int(np.ceil(len(sentences) * 0.5)))
    return " ".join(sentences[:keep]).strip()


def attack_truncate_word_50(text: str) -> str:
    words = text.split()
    keep = max(1, int(np.ceil(len(words) * 0.5)))
    return " ".join(words[:keep]).strip()


def attack_sentence_merge(text: str) -> str:
    sentences = safe_sentence_tokenize(text)
    if not sentences:
        return text
    merged = ", ".join(sentence.rstrip(".!?") for sentence in sentences)
    return merged + "."


def attack_sentence_split(text: str) -> str:
    parts = []
    for sentence in safe_sentence_tokenize(text):
        fragments = re.split(r"[;,]|(?:\s+and\s+)|(?:\s+but\s+)", sentence)
        fragments = [fragment.strip(" .") for fragment in fragments if fragment.strip(" .")]
        if len(fragments) <= 1:
            parts.append(sentence.strip())
        else:
            parts.extend(fragment + "." for fragment in fragments)
    return _normalize_whitespace(" ".join(parts))


def attack_word_dropout(text: str) -> str:
    words = text.split()
    kept = [word for idx, word in enumerate(words) if (idx + 1) % 5 != 0]
    return " ".join(kept).strip() or text


def attack_compression(text: str) -> str:
    compressed = []
    for sentence in safe_sentence_tokenize(text):
        tokens = sentence.split()
        keep = []
        for idx, token in enumerate(tokens):
            normalized = re.sub(r"[^A-Za-z']", "", token).lower()
            if idx == 0 or normalized not in STOPWORDS:
                keep.append(token)
        if keep:
            compressed.append(" ".join(keep).rstrip(".!?") + ".")
    return _normalize_whitespace(" ".join(compressed))


ATTACKS = {
    "format_cleanup": attack_format_cleanup,
    "truncate_sentence_50": attack_truncate_sentence_50,
    "truncate_word_50": attack_truncate_word_50,
    "sentence_merge": attack_sentence_merge,
    "sentence_split": attack_sentence_split,
    "word_dropout": attack_word_dropout,
    "compression": attack_compression,
}


def similarity_proxy(left: str, right: str) -> float:
    seq_score = SequenceMatcher(None, left, right).ratio()
    left_words = set(tokenize_words(left.lower()))
    right_words = set(tokenize_words(right.lower()))
    if not left_words and not right_words:
        jaccard = 1.0
    elif not left_words or not right_words:
        jaccard = 0.0
    else:
        jaccard = len(left_words & right_words) / float(len(left_words | right_words))
    return 0.5 * seq_score + 0.5 * jaccard


def summarize_attack(method: str, attack_name: str, records: list[dict]) -> dict:
    if not records:
        return {
            "attack": attack_name,
            "num_samples": 0,
            "mean_attacked_score": 0.0,
            "mean_delta": 0.0,
            "mean_similarity_proxy": 0.0,
        }

    attacked_scores = [record["attacked_score"] for record in records]
    deltas = [record["score_delta"] for record in records]
    similarities = [record["similarity_proxy"] for record in records]
    attacked_words = [record["attacked_n_words"] for record in records]
    attacked_sentences = [record["attacked_n_sentences"] for record in records]

    summary = {
        "attack": attack_name,
        "num_samples": len(records),
        "mean_attacked_score": float(np.mean(attacked_scores)),
        "std_attacked_score": float(np.std(attacked_scores)),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas)),
        "mean_similarity_proxy": float(np.mean(similarities)),
        "std_similarity_proxy": float(np.std(similarities)),
        "mean_attacked_words": float(np.mean(attacked_words)),
        "mean_attacked_sentences": float(np.mean(attacked_sentences)),
        "retention_rate_nonnegative_delta": float(np.mean([delta >= 0 for delta in deltas])),
    }

    if method == "acrostics":
        prefix_rates = [record["acrostics_prefix_match_rate"] for record in records]
        cycle_rates = [record["acrostics_sentence_match_rate"] for record in records]
        coverages = [record["acrostics_secret_coverage"] for record in records]
        summary.update(
            {
                "acrostics_mean_prefix_match_rate": float(np.mean(prefix_rates)),
                "acrostics_mean_sentence_match_rate": float(np.mean(cycle_rates)),
                "acrostics_mean_secret_coverage": float(np.mean(coverages)),
            }
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run deterministic robustness attacks over eval_*.json artifacts."
    )
    parser.add_argument("--eval-json", required=True, help="Path to an eval_*.json artifact")
    parser.add_argument(
        "--method",
        required=True,
        choices=["unicode", "initials", "lexical", "acrostics"],
        help="Watermark method used for the eval artifact",
    )
    parser.add_argument(
        "--attacks",
        default="format_cleanup,truncate_sentence_50,truncate_word_50,sentence_merge,sentence_split,word_dropout,compression",
        help="Comma-separated attack names",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults next to eval artifact.",
    )
    parser.add_argument(
        "--min-similarity-proxy",
        type=float,
        default=0.0,
        help="Filter attacked samples below this similarity proxy threshold",
    )
    args = parser.parse_args()

    detector, detector_args = get_detector_and_args(args.method)

    with open(args.eval_json, "r") as handle:
        payload = json.load(handle)

    samples = payload.get("samples", [])
    requested_attacks = [attack.strip() for attack in args.attacks.split(",") if attack.strip()]
    invalid = [attack for attack in requested_attacks if attack not in ATTACKS]
    if invalid:
        raise ValueError(f"Unsupported attacks: {', '.join(invalid)}")

    attacked_payload = {
        "source_eval_json": args.eval_json,
        "source_summary": payload.get("summary", {}),
        "method": args.method,
        "attacks": {},
    }

    for attack_name in requested_attacks:
        attack_fn = ATTACKS[attack_name]
        attack_records = []
        for record in samples:
            original_text = record["response"]
            attacked_text = attack_fn(original_text)
            attacked_score = float(detector(attacked_text, *detector_args))
            sim_proxy = similarity_proxy(original_text, attacked_text)
            if sim_proxy < args.min_similarity_proxy:
                continue

            attacked_stats = response_stats(attacked_text)
            attacked_record = {
                "query": record.get("query"),
                "original_response": original_text,
                "attacked_response": attacked_text,
                "original_score": float(record.get("detector_score", record.get("score", 0.0))),
                "attacked_score": attacked_score,
                "score_delta": attacked_score - float(record.get("detector_score", record.get("score", 0.0))),
                "similarity_proxy": sim_proxy,
                "attack": attack_name,
                "attacked_n_words": attacked_stats["n_words"],
                "attacked_n_chars": attacked_stats["n_chars"],
                "attacked_n_sentences": attacked_stats["n_sentences"],
            }

            if args.method == "acrostics":
                details = acrostics_metrics(attacked_text, secret_sequence)
                attacked_record.update(
                    {
                        "acrostics_prefix_match_rate": details["prefix_match_rate"],
                        "acrostics_sentence_match_rate": details["sentence_match_rate"],
                        "acrostics_secret_coverage": details["secret_coverage"],
                    }
                )

            attack_records.append(attacked_record)

        attacked_payload["attacks"][attack_name] = {
            "summary": summarize_attack(args.method, attack_name, attack_records),
            "samples": attack_records,
        }

    output_path = args.output
    if output_path is None:
        stem, _ = os.path.splitext(args.eval_json)
        output_path = stem + "_robustness.json"

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(output_path, "w") as handle:
        json.dump(attacked_payload, handle, indent=2)

    print(f"✓ Saved robustness results to: {output_path}")
    for attack_name in requested_attacks:
        summary = attacked_payload["attacks"][attack_name]["summary"]
        print(
            f"  {attack_name}: "
            f"delta={summary['mean_delta']:.4f}, "
            f"score={summary['mean_attacked_score']:.4f}, "
            f"similarity={summary['mean_similarity_proxy']:.4f}"
        )


if __name__ == "__main__":
    main()
