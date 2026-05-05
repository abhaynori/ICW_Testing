#!/usr/bin/env python3
"""
Rejection Sampling SFT Data Generator for ICW Watermarking

Generates high-quality watermarked training data by:
1. Loading a model (base or SFT warm-start)
2. Generating N candidate responses per query WITH watermark instructions
3. Scoring each with the detector
4. Keeping the top-K responses above a score threshold
5. Saving as a JSON dataset for SFT training

Supports curriculum training by varying the fraction of prompts
that include watermark instructions (--implicit-fraction).

Usage:
    # Round 1: Generate data with explicit instructions from base model
    python generate_sft_data.py --model full --method acrostics \
        --samples 500 --n-candidates 16 --top-k 2 --min-score 1.0

    # Round 1 from SFT warm-start (better quality)
    python generate_sft_data.py --model full --method acrostics \
        --warm-start-model sft_models/sft_acrostics_full_*/final_model \
        --samples 500 --n-candidates 16 --top-k 2 --min-score 1.0

    # Round 2+: Curriculum (30% implicit)
    python generate_sft_data.py --model full --method acrostics \
        --warm-start-model <round1_model>/final_model \
        --samples 500 --n-candidates 16 --top-k 2 --min-score 0.5 \
        --implicit-fraction 0.3
"""

import argparse
import json
import os
import re
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import (
    acrostics_detector, acrostics_embed_prompt, secret_sequence,
    get_acrostics_secret_sequence, set_acrostics_secret_sequence,
    unicode_detector, unicode_embed_prompt,
    initials_detector, initials_embed_prompt, green_letters,
    lexical_detector, lexical_embed_prompt, green_words,
    get_base_system_prompt,
)
from memory_config import get_model_config
from research_utils import acrostics_metrics, sanitize_generated_text


# ---------------------------------------------------------------------------
# Utilities (shared with other scripts)
# ---------------------------------------------------------------------------

def get_detector_and_args(method):
    current_secret_sequence = get_acrostics_secret_sequence()
    detector_map = {
        'unicode': (unicode_detector, ()),
        'initials': (initials_detector, (green_letters,)),
        'lexical': (lexical_detector, (green_words,)),
        'acrostics': (acrostics_detector, (current_secret_sequence,)),
    }
    if method not in detector_map:
        raise ValueError(f"Unknown method: {method}")
    return detector_map[method]


def get_prompt_function(method):
    prompt_map = {
        'unicode': unicode_embed_prompt,
        'initials': initials_embed_prompt,
        'lexical': lexical_embed_prompt,
        'acrostics': acrostics_embed_prompt,
    }
    if method not in prompt_map:
        raise ValueError(f"Unknown method: {method}")
    return prompt_map[method]


def build_messages(query, prompt_fn=None, include_instruction=True, target_sentence_count=None):
    if include_instruction and prompt_fn is not None:
        messages = [dict(message) for message in prompt_fn(query)]
    else:
        messages = [
        {"role": "system", "content": get_base_system_prompt()},
        {"role": "user", "content": query},
        ]

    if target_sentence_count is not None:
        constraint = (
            f"Respond in exactly {target_sentence_count} sentences. "
            f"Stop after sentence {target_sentence_count}. "
            "Do not use bullets, numbering, headings, labels, prefaces, or closing notes."
        )
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = messages[0]["content"].rstrip() + "\n\nAdditional requirement:\n" + constraint
        else:
            messages.insert(0, {"role": "system", "content": constraint})

    return messages


def _has_generation_artifacts(text):
    if "<tool_call>" in text:
        return True
    return bool(
        re.search(r"(?:^|\n)\s*(system|user|assistant)\s*\n", text, flags=re.IGNORECASE)
    )


def _strict_acrostics_record(response):
    clean = sanitize_generated_text(response)
    details = acrostics_metrics(clean, secret_sequence)
    valid = (
        not _has_generation_artifacts(response)
        and clean == response.strip()
        and details["full_secret_realized"] >= 1.0
        and details["sentence_count_error"] == 0.0
    )
    return valid, clean, details


def _format_alpaca_query(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    if input_text:
        return f"{instruction}\n\nContext: {input_text}"
    return instruction


def _slice_indices_for_split(size, split):
    train_end = int(size * 0.8)
    validation_end = int(size * 0.9)
    if split == "train":
        return 0, train_end
    if split == "validation":
        return train_end, validation_end
    if split == "test":
        return validation_end, size
    raise ValueError(f"Unsupported split: {split}")


def load_queries(dataset_name="eli5", split="train", num_samples=500):
    dataset_key = dataset_name.strip().lower()
    queries = []

    if dataset_key == "eli5":
        ds = load_dataset("sentence-transformers/eli5", "pair", split="train")
        start, end = _slice_indices_for_split(len(ds), split)
        subset = ds.select(range(start, min(end, start + num_samples)))
        for row in subset:
            q = (row.get("question") or "").strip()
            if q:
                queries.append(q)
        return queries

    if dataset_key == "alpaca":
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        start, end = _slice_indices_for_split(len(ds), split)
        subset = ds.select(range(start, min(end, start + num_samples)))
        for row in subset:
            q = _format_alpaca_query(row)
            if q:
                queries.append(q)
        return queries

    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def load_causal_lm_with_dtype_fallback(model_name, model_kwargs, dtype_value=None):
    if dtype_value is None:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    modern_kwargs = dict(model_kwargs)
    modern_kwargs["dtype"] = dtype_value
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **modern_kwargs)
    except TypeError:
        legacy_kwargs = dict(model_kwargs)
        legacy_kwargs["torch_dtype"] = dtype_value
        return AutoModelForCausalLM.from_pretrained(model_name, **legacy_kwargs)


def _looks_like_local_path(source):
    if source is None:
        return False
    value = str(source).strip()
    if not value:
        return False
    if value.startswith((".", "/", "~")):
        return True
    if re.match(r"^[a-zA-Z]:[\\/]", value):
        return True
    normalized = value.replace("\\", "/")
    if "/" not in normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 3:
        return True
    if value.endswith(("/", "\\")):
        return True
    if len(parts) == 2:
        # Two-part strings can be either local relative paths or HF repo ids.
        # Treat as local only if the full path exists in cwd (not just the
        # top-level segment, which can false-positive on cache/work dirs).
        return os.path.exists(normalized)
    return False


def resolve_pretrained_source(source, label):
    if source is None:
        return None
    raw = str(source).strip()
    if not raw:
        raise ValueError(f"{label} cannot be empty.")
    expanded = os.path.expanduser(raw)
    normalized = os.path.normpath(expanded)
    if os.path.exists(normalized):
        return normalized
    if _looks_like_local_path(raw):
        abs_candidate = os.path.abspath(normalized)
        raise FileNotFoundError(
            f"{label} path not found: '{raw}'. "
            f"Resolved as: '{abs_candidate}'. "
            f"Current working directory: '{os.getcwd()}'."
        )
    return raw


def generate_responses_batch(model, tokenizer, messages_batch,
                              max_new_tokens=512, temperature=0.9, top_p=0.95):
    """Generate a batch of responses from chat messages."""
    prompt_texts = [
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        for messages in messages_batch
    ]

    try:
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(messages_batch) == 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  ⚠️  OOM at batch size {len(messages_batch)}, falling back to single.")
        results = []
        for messages in messages_batch:
            results.extend(
                generate_responses_batch(
                    model, tokenizer, [messages],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        return results

    responses = []
    # Trim the full padded prompt prefix so left-padded batches do not retain
    # the tail of the prompt in the decoded continuation.
    prompt_len = encoded["input_ids"].shape[1]
    for idx in range(outputs.shape[0]):
        decoded = tokenizer.decode(outputs[idx][prompt_len:], skip_special_tokens=True)
        responses.append(
            sanitize_generated_text(decoded)
        )
    return responses


# ---------------------------------------------------------------------------
# Core: rejection sampling
# ---------------------------------------------------------------------------

def generate_rejection_sampled_data(
    model,
    tokenizer,
    queries,
    method,
    prompt_fn,
    n_candidates=16,
    top_k=2,
    min_score=1.0,
    max_new_tokens=512,
    temperature=0.9,
    top_p=0.95,
    implicit_fraction=0.0,
    gen_batch_size=4,
    seed=42,
    strict_acrostics=False,
):
    """
    For each query, generate n_candidates responses, score with detector,
    and keep top-k responses above min_score threshold.

    Args:
        implicit_fraction: Fraction of queries where watermark instructions
            are NOT included in the prompt (for curriculum training).
            The model is still expected to produce watermarked output.
    """
    detector, detector_args = get_detector_and_args(method)
    rng = np.random.default_rng(seed)

    records = []
    total_candidates = 0
    total_kept = 0
    skipped_queries = 0

    print(f"\nGenerating rejection-sampled SFT data...")
    print(f"  Queries: {len(queries)}")
    print(f"  Candidates per query: {n_candidates}")
    print(f"  Top-K to keep: {top_k}")
    print(f"  Min detector score: {min_score}")
    print(f"  Implicit fraction: {implicit_fraction}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Strict acrostics filtering: {strict_acrostics if method == 'acrostics' else False}")
    print()

    for idx, query in enumerate(queries):
        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(queries) - 1:
            print(f"  Query {idx + 1}/{len(queries)} | "
                  f"kept so far: {total_kept}/{total_candidates} candidates")

        # Decide whether to include watermark instructions
        use_instruction = rng.random() >= implicit_fraction
        target_sentence_count = len(secret_sequence) if method == "acrostics" else None
        messages = build_messages(
            query,
            prompt_fn=prompt_fn,
            include_instruction=use_instruction,
            target_sentence_count=target_sentence_count,
        )

        # Generate n_candidates responses in batches
        candidates = []
        for batch_start in range(0, n_candidates, gen_batch_size):
            batch_end = min(batch_start + gen_batch_size, n_candidates)
            batch_size = batch_end - batch_start
            # Repeat the same messages for each candidate in the batch
            messages_batch = [messages] * batch_size
            responses = generate_responses_batch(
                model, tokenizer, messages_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for response in responses:
                score = detector(response, *detector_args)
                candidates.append((response, float(score)))

        total_candidates += len(candidates)

        if strict_acrostics and method == "acrostics":
            strict_candidates = []
            for response, score in candidates:
                valid, clean_response, details = _strict_acrostics_record(response)
                if not valid or score < min_score:
                    continue
                strict_candidates.append((response, score, clean_response, details))

            strict_candidates.sort(key=lambda item: item[1], reverse=True)
            kept = 0
            for response, score, clean_response, details in strict_candidates[:top_k]:
                records.append(
                    {
                        "query": query,
                        "target": clean_response,
                        "detector_score": score,
                        "include_instruction": use_instruction,
                        "acrostics_prefix_match_rate": details["prefix_match_rate"],
                        "acrostics_sentence_match_rate": details["sentence_match_rate"],
                        "acrostics_secret_coverage": details["secret_coverage"],
                        "acrostics_full_secret_realized": details["full_secret_realized"],
                        "acrostics_sentence_count_error": details["sentence_count_error"],
                    }
                )
                kept += 1

            total_kept += kept
            if kept == 0:
                skipped_queries += 1
            continue

        # Sort by score descending, keep top-k above threshold
        candidates.sort(key=lambda x: x[1], reverse=True)
        kept = 0
        for response, score in candidates[:top_k]:
            if score < min_score:
                continue

            record = {
                    "query": query,
                    "target": response,
                    "detector_score": score,
                    "include_instruction": use_instruction,
                }
            if method == "acrostics":
                valid, clean_response, details = _strict_acrostics_record(response)
                record["target"] = clean_response
                record.update(
                    {
                        "acrostics_prefix_match_rate": details["prefix_match_rate"],
                        "acrostics_sentence_match_rate": details["sentence_match_rate"],
                        "acrostics_secret_coverage": details["secret_coverage"],
                        "acrostics_full_secret_realized": details["full_secret_realized"],
                        "acrostics_sentence_count_error": details["sentence_count_error"],
                    }
                )
                if strict_acrostics and not valid:
                    continue

            records.append(record)
            kept += 1

        total_kept += kept
        if kept == 0:
            skipped_queries += 1

    print(f"\n{'=' * 60}")
    print(f"Rejection Sampling Summary")
    print(f"{'=' * 60}")
    print(f"  Total queries: {len(queries)}")
    print(f"  Queries with ≥1 valid response: {len(queries) - skipped_queries}")
    print(f"  Queries with 0 valid responses: {skipped_queries}")
    print(f"  Total candidates generated: {total_candidates}")
    print(f"  Total kept (score ≥ {min_score}): {total_kept}")
    print(f"  Yield rate: {total_kept / total_candidates * 100:.1f}%")
    if records:
        scores = [r["detector_score"] for r in records]
        print(f"  Kept scores: mean={np.mean(scores):.3f}, "
              f"min={np.min(scores):.3f}, max={np.max(scores):.3f}")
    print(f"{'=' * 60}")

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global secret_sequence
    parser = argparse.ArgumentParser(
        description="Generate rejection-sampled SFT data for ICW watermarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from base model
  python generate_sft_data.py --model full --method acrostics \\
      --samples 500 --n-candidates 16 --top-k 2 --min-score 1.0

  # Generate from SFT warm-start (better quality)
  python generate_sft_data.py --model full --method acrostics \\
      --warm-start-model sft_models/sft_acrostics_full_*/final_model \\
      --samples 500 --n-candidates 16 --top-k 2 --min-score 1.0

  # Curriculum round (30% implicit)
  python generate_sft_data.py --model full --method acrostics \\
      --warm-start-model <prev_model>/final_model \\
      --samples 500 --implicit-fraction 0.3 --min-score 0.5
        """,
    )

    parser.add_argument("--model", "-m", type=str, default="full",
                        choices=["small", "4bit", "8bit", "full", "cpu"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["unicode", "initials", "lexical", "acrostics"])
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of queries to process (default: 500)")
    parser.add_argument("--n-candidates", type=int, default=16,
                        help="Candidates to generate per query (default: 16)")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Top-K responses to keep per query (default: 2)")
    parser.add_argument("--min-score", type=float, default=1.0,
                        help="Minimum detector z-score to keep (default: 1.0)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens per response (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature (default: 0.9)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p sampling (default: 0.95)")
    parser.add_argument("--implicit-fraction", type=float, default=0.0,
                        help="Fraction of queries WITHOUT watermark instructions (default: 0.0)")
    parser.add_argument("--gen-batch-size", type=int, default=4,
                        help="Generation batch size (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-variant", choices=["paper", "concise", "strict"], default="paper")
    parser.add_argument("--rules-variant", choices=["paper", "minimal", "none"], default="paper")
    parser.add_argument(
        "--secret-sequence",
        type=str,
        default=secret_sequence,
        help="Acrostics secret string to realize (default: current configured secret)",
    )
    parser.add_argument("--base-system-prompt", default=None)
    parser.add_argument("--system-prompt-prefix", default=None)
    parser.add_argument(
        "--strict-acrostics",
        action="store_true",
        help="For acrostics, keep only exact |X|-sentence secret realizations with no generation artifacts.",
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="eli5",
                        choices=["eli5", "alpaca"])
    parser.add_argument("--split", type=str, default="train")

    # Model loading
    parser.add_argument("--warm-start-model", type=str, default=None,
                        help="Path to existing model checkpoint")

    # Output
    parser.add_argument("--output-dir", type=str, default="sft_data",
                        help="Output directory (default: sft_data)")

    args = parser.parse_args()

    try:
        secret_sequence = set_acrostics_secret_sequence(args.secret_sequence)
    except ValueError as exc:
        parser.error(str(exc))

    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.n_candidates < 1:
        parser.error("--n-candidates must be >= 1")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if not (0.0 <= args.implicit_fraction <= 1.0):
        parser.error("--implicit-fraction must be in [0.0, 1.0]")

    # Print config
    print("\n" + "=" * 80)
    print("REJECTION SAMPLING SFT DATA GENERATION")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    if args.method == "acrostics":
        print(f"Secret Sequence: {secret_sequence}")
    print(f"Dataset: {args.dataset}:{args.split}")
    print(f"Queries: {args.samples}")
    print(f"Candidates/query: {args.n_candidates}")
    print(f"Top-K: {args.top_k}")
    print(f"Min score: {args.min_score}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Implicit fraction: {args.implicit_fraction}")
    print(f"Prompt variant: {args.prompt_variant}")
    print(f"Rules variant: {args.rules_variant}")
    if args.warm_start_model:
        print(f"Warm-start model: {args.warm_start_model}")
    print("=" * 80)

    os.environ["ICW_PROMPT_VARIANT"] = args.prompt_variant
    os.environ["ICW_RULES_VARIANT"] = args.rules_variant
    if args.base_system_prompt:
        os.environ["ICW_BASE_SYSTEM_PROMPT"] = args.base_system_prompt
    else:
        os.environ.pop("ICW_BASE_SYSTEM_PROMPT", None)
    if args.system_prompt_prefix:
        os.environ["ICW_SYSTEM_PROMPT_PREFIX"] = args.system_prompt_prefix
    else:
        os.environ.pop("ICW_SYSTEM_PROMPT_PREFIX", None)

    # Load model
    config = get_model_config(args.model)
    model_name = config["model_name"]

    model_kwargs = {
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if config.get("quantization"):
        model_kwargs["quantization_config"] = config["quantization"]
    dtype_value = config.get("dtype")

    resolved_source = model_name
    if args.warm_start_model:
        resolved_source = resolve_pretrained_source(
            args.warm_start_model, "Warm Start Model"
        )
        print(f"\nLoading model from: {resolved_source}")
    else:
        print(f"\nLoading base model: {model_name}")

    model = load_causal_lm_with_dtype_fallback(
        model_name=resolved_source,
        model_kwargs=model_kwargs,
        dtype_value=dtype_value,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_source if args.warm_start_model else model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "left":
        tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    print("✓ Model loaded\n")

    # Load queries
    queries = load_queries(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.samples,
    )
    print(f"✓ Loaded {len(queries)} queries\n")

    # Generate data
    prompt_fn = get_prompt_function(args.method)
    records = generate_rejection_sampled_data(
        model=model,
        tokenizer=tokenizer,
        queries=queries,
        method=args.method,
        prompt_fn=prompt_fn,
        n_candidates=args.n_candidates,
        top_k=args.top_k,
        min_score=args.min_score,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        implicit_fraction=args.implicit_fraction,
        gen_batch_size=args.gen_batch_size,
        seed=args.seed,
        strict_acrostics=args.strict_acrostics,
    )

    if len(records) == 0:
        print("\n⚠️  No records passed the score threshold! Try:")
        print("  - Lowering --min-score")
        print("  - Increasing --n-candidates")
        print("  - Using a warm-start model")
        return

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.method}_{args.model}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, filename)

    output = {
        "metadata": {
            "method": args.method,
            "model_strategy": args.model,
            "warm_start_model": args.warm_start_model,
            "secret_sequence": secret_sequence if args.method == "acrostics" else None,
            "dataset": args.dataset,
            "split": args.split,
            "num_queries": len(queries),
            "n_candidates": args.n_candidates,
            "top_k": args.top_k,
            "min_score": args.min_score,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "implicit_fraction": args.implicit_fraction,
            "prompt_variant": args.prompt_variant,
            "rules_variant": args.rules_variant,
            "strict_acrostics": args.strict_acrostics,
            "seed": args.seed,
            "num_records": len(records),
            "timestamp": datetime.now().isoformat(),
        },
        "records": records,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved {len(records)} training records to: {output_path}")
    print(f"\nNext step — train SFT on this data:")
    print(f"  python sft_train.py --model {args.model} --method {args.method} \\")
    print(f"      --use-lora --sft-data {output_path} \\")
    print(f"      --samples {len(records)} --epochs 2 --max-length 1024")


if __name__ == "__main__":
    main()
