#!/usr/bin/env python3
"""
Shared utilities for research-grade evaluation and model packaging.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    import Levenshtein
except Exception:
    Levenshtein = None

try:
    from nltk.tokenize import sent_tokenize
except Exception:
    sent_tokenize = None

try:
    from peft import AutoPeftModelForCausalLM, PeftConfig
except Exception:
    AutoPeftModelForCausalLM = None
    PeftConfig = None

from transformers import AutoConfig, AutoModelForCausalLM


def fallback_sent_tokenize(text: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def safe_sentence_tokenize(text: str) -> list[str]:
    if sent_tokenize is None:
        return fallback_sent_tokenize(text)
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        sentences = fallback_sent_tokenize(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def extract_sentence_initials(sentences: list[str]) -> str:
    initials = []
    for sentence in sentences:
        for char in sentence.strip():
            if char.isalpha():
                initials.append(char.upper())
                break
    return "".join(initials)


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text)


def response_stats(text: str) -> dict[str, Any]:
    sentences = safe_sentence_tokenize(text)
    words = tokenize_words(text)
    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences]
    return {
        "n_chars": len(text),
        "n_words": len(words),
        "n_sentences": len(sentences),
        "mean_sentence_words": (
            sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        ),
        "sentence_lengths": sentence_lengths,
    }


def sentence_count_bin(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 6:
        return "4-6"
    if count <= 10:
        return "7-10"
    return "11+"


def _levenshtein_distance(left: str, right: str) -> int:
    if Levenshtein is not None:
        return int(Levenshtein.distance(left, right))

    # Fallback dynamic programming implementation.
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    prev = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        cur = [i]
        for j, right_char in enumerate(right, start=1):
            substitution = prev[j - 1] + (left_char != right_char)
            insertion = cur[j - 1] + 1
            deletion = prev[j] + 1
            cur.append(min(substitution, insertion, deletion))
        prev = cur
    return prev[-1]


def acrostics_metrics(text: str, secret_sequence: str) -> dict[str, Any]:
    sentences = safe_sentence_tokenize(text)
    initials = extract_sentence_initials(sentences)
    secret = (secret_sequence or "").strip().upper()

    if not secret:
        raise ValueError("secret_sequence must be non-empty")

    n = len(initials)
    expected = (secret * (n // len(secret) + 1))[:n] if n else ""
    cycle_matches = sum(
        1 for actual, target in zip(initials, expected) if actual == target
    )
    prefix_len = min(n, len(secret))
    prefix_matches = sum(
        1 for idx in range(prefix_len) if initials[idx] == secret[idx]
    )
    distance = _levenshtein_distance(initials, expected)

    return {
        "initials": initials,
        "expected_initials": expected,
        "prefix_match_rate": prefix_matches / prefix_len if prefix_len else 0.0,
        "sentence_match_rate": cycle_matches / n if n else 0.0,
        "secret_coverage": min(n, len(secret)) / float(len(secret)),
        "full_secret_realized": 1.0 if n >= len(secret) else 0.0,
        "levenshtein_distance": float(distance),
        "normalized_levenshtein_distance": distance / float(max(1, n)),
        "sentence_count_bin": sentence_count_bin(len(sentences)),
        "n_sentences": len(sentences),
    }


def is_local_peft_checkpoint(model_name_or_path: str) -> bool:
    path = os.path.normpath(os.path.expanduser(str(model_name_or_path)))
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def get_peft_base_model_name(model_name_or_path: str) -> str | None:
    if not is_local_peft_checkpoint(model_name_or_path):
        return None
    if PeftConfig is None:
        raise ImportError(
            "PEFT is required to inspect local adapter checkpoints. Install with: pip install peft"
        )
    config = PeftConfig.from_pretrained(model_name_or_path)
    return getattr(config, "base_model_name_or_path", None)


def load_causal_lm_with_adapter_support(
    model_name_or_path: str,
    model_kwargs: dict[str, Any],
    dtype_value=None,
    is_trainable: bool = False,
):
    """
    Load either a regular CausalLM or a local PEFT adapter checkpoint.
    """
    normalized = os.path.normpath(os.path.expanduser(str(model_name_or_path)))

    def _load_regular(kwargs: dict[str, Any]):
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    if is_local_peft_checkpoint(normalized):
        if AutoPeftModelForCausalLM is None:
            raise ImportError(
                "PEFT is required to load local adapter checkpoints. Install with: pip install peft"
            )

        def _load_peft(kwargs: dict[str, Any]):
            return AutoPeftModelForCausalLM.from_pretrained(
                normalized,
                is_trainable=is_trainable,
                **kwargs,
            )

        if dtype_value is None:
            return _load_peft(dict(model_kwargs))

        modern_kwargs = dict(model_kwargs)
        modern_kwargs["dtype"] = dtype_value
        try:
            return _load_peft(modern_kwargs)
        except TypeError:
            legacy_kwargs = dict(model_kwargs)
            legacy_kwargs["torch_dtype"] = dtype_value
            return _load_peft(legacy_kwargs)

    if dtype_value is None:
        return _load_regular(dict(model_kwargs))

    modern_kwargs = dict(model_kwargs)
    modern_kwargs["dtype"] = dtype_value
    try:
        return _load_regular(modern_kwargs)
    except TypeError:
        legacy_kwargs = dict(model_kwargs)
        legacy_kwargs["torch_dtype"] = dtype_value
        return _load_regular(legacy_kwargs)


def build_lm_eval_model_args(model_name_or_path: str, trust_remote_code: bool = True) -> str:
    """
    Construct lm-eval model_args, using `peft=` when pointing at a local adapter.
    """
    normalized = os.path.normpath(os.path.expanduser(str(model_name_or_path)))
    if is_local_peft_checkpoint(normalized):
        base_model = get_peft_base_model_name(normalized)
        if not base_model:
            raise ValueError(
                f"Could not determine base model for adapter checkpoint: {model_name_or_path}"
            )
        return (
            f"pretrained={base_model},peft={normalized},"
            f"trust_remote_code={'True' if trust_remote_code else 'False'}"
        )
    return (
        f"pretrained={model_name_or_path},"
        f"trust_remote_code={'True' if trust_remote_code else 'False'}"
    )


def patch_saved_model_config(model_dir: str, base_model_name: str) -> str | None:
    """
    Ensure saved model directories contain a config.json with model_type so
    downstream tools like lm-eval can load adapter-based checkpoints.
    """
    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, "config.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as handle:
            payload = json.load(handle)
    else:
        payload = {}

    if payload.get("model_type"):
        return config_path

    base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    base_payload = base_config.to_dict()

    if "architectures" in base_payload and "architectures" not in payload:
        payload["architectures"] = base_payload["architectures"]
    payload["model_type"] = base_payload["model_type"]

    # Preserve any adapter-specific metadata already saved.
    for key, value in base_payload.items():
        if key in {"model_type", "architectures"}:
            continue
        payload.setdefault(key, value)

    with open(config_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return config_path
