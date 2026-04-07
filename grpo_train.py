#!/usr/bin/env python3
"""
GRPO Training for ICW Watermarking

This script uses Group Relative Policy Optimization (GRPO) from TRL to fine-tune
models to better follow watermarking instructions. The existing detector functions
are used as reward signals.

Usage:
    python grpo_train.py --model small --method unicode --epochs 3
    python grpo_train.py --model 4bit --method acrostics --samples 100
"""

import torch
import argparse
import os
import json
import inspect
import warnings
import re
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import numpy as np

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:
    GRPOConfig = None
    GRPOTrainer = None

try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None

# Import existing detector functions and prompts from main.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import detector functions
from main import (
    unicode_detector, unicode_embed_prompt,
    initials_detector, initials_embed_prompt, green_letters,
    lexical_detector, lexical_embed_prompt, green_words,
    acrostics_detector, acrostics_embed_prompt, secret_sequence,
    get_acrostics_secret_sequence, get_base_system_prompt,
    set_acrostics_secret_sequence,
)
from memory_config import get_model_config
from research_utils import (
    acrostics_metrics,
    load_causal_lm_with_adapter_support,
    patch_saved_model_config,
    response_stats,
    sanitize_generated_text,
)


def get_detector_and_args(method):
    current_secret_sequence = get_acrostics_secret_sequence()
    detector_map = {
        'unicode': (unicode_detector, ()),
        'initials': (initials_detector, (green_letters,)),
        'lexical': (lexical_detector, (green_words,)),
        'acrostics': (acrostics_detector, (current_secret_sequence,))
    }

    if method not in detector_map:
        raise ValueError(f"Unknown method: {method}")

    return detector_map[method]


def resolve_eval_profiles(
    profile_names,
    max_new_tokens,
    min_new_tokens=None,
    natural_max_new_tokens=None,
    natural_min_new_tokens=None,
    controlled_max_new_tokens=None,
    controlled_min_new_tokens=None,
):
    """
    Resolve named evaluation profiles into concrete generation settings.
    """
    names = [name.strip().lower() for name in str(profile_names).split(",") if name.strip()]
    if not names:
        names = ["natural"]

    valid = {"natural", "controlled"}
    invalid = [name for name in names if name not in valid]
    if invalid:
        raise ValueError(
            f"Unsupported eval profile(s): {', '.join(invalid)}. "
            f"Choose from: {', '.join(sorted(valid))}"
        )

    profiles = []
    for name in names:
        if name == "natural":
            profiles.append(
                {
                    "name": "natural",
                    "max_new_tokens": natural_max_new_tokens or max_new_tokens,
                    "min_new_tokens": natural_min_new_tokens,
                }
            )
            continue

        controlled_max = controlled_max_new_tokens or max_new_tokens
        controlled_min = controlled_min_new_tokens
        if controlled_min is None:
            if min_new_tokens is not None:
                controlled_min = min_new_tokens
            else:
                controlled_min = max(64, min(128, controlled_max // 2))
        controlled_min = min(controlled_min, controlled_max)
        profiles.append(
            {
                "name": "controlled",
                "max_new_tokens": controlled_max,
                "min_new_tokens": controlled_min,
            }
        )

    return profiles


def resolve_eval_modes(mode_names):
    """
    Resolve named evaluation modes into concrete instruction settings.
    """
    names = [name.strip().lower() for name in str(mode_names).split(",") if name.strip()]
    if not names:
        names = ["implicit", "explicit"]

    valid = {"implicit", "explicit"}
    invalid = [name for name in names if name not in valid]
    if invalid:
        raise ValueError(
            f"Unsupported eval mode(s): {', '.join(invalid)}. "
            f"Choose from: {', '.join(sorted(valid))}"
        )

    resolved = []
    seen = set()
    mode_map = {"implicit": False, "explicit": True}
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        resolved.append((name, mode_map[name]))
    return resolved


def build_method_record_metrics(method, text):
    """
    Per-response, method-specific metrics used in richer analysis.
    """
    metrics = {}
    if method == "acrostics":
        details = acrostics_metrics(text, secret_sequence)
        metrics.update(
            {
                "acrostics_initials": details["initials"],
                "acrostics_expected_initials": details["expected_initials"],
                "acrostics_prefix_match_rate": details["prefix_match_rate"],
                "acrostics_sentence_match_rate": details["sentence_match_rate"],
                "acrostics_secret_coverage": details["secret_coverage"],
                "acrostics_full_secret_realized": details["full_secret_realized"],
                "acrostics_levenshtein_distance": details["levenshtein_distance"],
                "acrostics_normalized_levenshtein_distance": details["normalized_levenshtein_distance"],
                "acrostics_sentence_count_bin": details["sentence_count_bin"],
                "acrostics_sentence_count_error": details["sentence_count_error"],
                "acrostics_extra_sentence_count": details["extra_sentence_count"],
                "acrostics_missing_sentence_count": details["missing_sentence_count"],
            }
        )
    return metrics


class WatermarkRewardFunction:
    """Reward function that uses existing detectors."""

    def __init__(
        self,
        method,
        baseline_mean=0.0,
        baseline_std=1.0,
        detector_weight=1.0,
        reward_shaping=True,
        shaping_format_weight=0.10,
        shaping_partial_weight=0.40,
        shaping_length_weight=0.10,
        shaping_target_words=120,
        max_abs_reward=10.0
    ):
        """
        Args:
            method: One of 'unicode', 'initials', 'lexical', 'acrostics'
            baseline_mean: Mean score from non-watermarked baseline
            baseline_std: Std of non-watermarked baseline
        """
        self.method = method
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std
        self.detector_weight = detector_weight
        self.reward_shaping = reward_shaping
        self.shaping_format_weight = shaping_format_weight
        self.shaping_partial_weight = shaping_partial_weight
        self.shaping_length_weight = shaping_length_weight
        self.shaping_target_words = max(1, int(shaping_target_words))
        self.max_abs_reward = max_abs_reward
        # Some TRL versions require reward callables to expose __name__.
        self.__name__ = f"watermark_reward_{method}"

        # Set up detector and arguments
        self.detector, self.detector_args = get_detector_and_args(method)

    @staticmethod
    def _to_text(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("text", "content", "completion"):
                if key in item and isinstance(item[key], str):
                    return item[key]
        if isinstance(item, (list, tuple)):
            parts = []
            for entry in item:
                if isinstance(entry, dict):
                    if "content" in entry and isinstance(entry["content"], str):
                        parts.append(entry["content"])
                    elif "text" in entry and isinstance(entry["text"], str):
                        parts.append(entry["text"])
                elif isinstance(entry, str):
                    parts.append(entry)
            if parts:
                return " ".join(parts)
        return str(item)

    def _extract_texts(self, args, kwargs):
        for key in ("completions", "texts", "responses", "outputs"):
            value = kwargs.get(key)
            if value is not None:
                return [self._to_text(v) for v in value]
        if args:
            first = args[0]
            if isinstance(first, list):
                return [self._to_text(v) for v in first]
            return [self._to_text(first)]
        return []

    @staticmethod
    def _safe_words(text):
        return re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text)

    @staticmethod
    def _split_sentences(text):
        pieces = re.split(r"[.!?\n]+", text)
        return [piece.strip() for piece in pieces if piece.strip()]

    def _format_score(self, text):
        words = self._safe_words(text)
        word_count = len(words)
        sentences = self._split_sentences(text)

        has_content = 1.0 if word_count >= 8 else 0.0
        has_structure = 1.0 if len(sentences) >= 2 else 0.0
        if self.method == "acrostics":
            return 0.50 * has_content + 0.50 * has_structure
        has_reasonable_length = min(word_count / float(self.shaping_target_words), 1.0)

        # In [0, 1], rewards outputs that are non-empty, structured, and long enough.
        return 0.35 * has_content + 0.25 * has_structure + 0.40 * has_reasonable_length

    def _partial_progress_score(self, text):
        words = self._safe_words(text)
        if not words:
            return 0.0

        if self.method == "unicode":
            zero_width_count = text.count("\u200b")
            return min(zero_width_count / max(1, len(words) * 0.20), 1.0)

        if self.method == "initials":
            green_hits = sum(1 for token in words if token[0].lower() in green_letters)
            return green_hits / max(1, len(words))

        if self.method == "lexical":
            word_set = {token.lower() for token in words}
            green_hits = sum(1 for token in green_words if token.lower() in word_set)
            # Saturate once a reasonable number of green words appears.
            return min(green_hits / 8.0, 1.0)

        if self.method == "acrostics":
            details = acrostics_metrics(sanitize_generated_text(text), secret_sequence)
            return details["sentence_match_rate"] * details["secret_coverage"]

        return 0.0

    @staticmethod
    def _artifact_penalty(text):
        penalty = 0.0
        if "<tool_call>" in text:
            penalty += 1.0
        if re.search(r"(?:^|\n)\s*(system|user|assistant)\s*\n", text, flags=re.IGNORECASE):
            penalty += 1.0
        return min(penalty, 1.0)

    def _acrostics_training_score(self, text):
        """Train toward a single-pass fixed-secret acrostic over exactly |X| sentences."""
        details = acrostics_metrics(sanitize_generated_text(text), secret_sequence)
        sentence_count_penalty = min(
            details["sentence_count_error"] / float(len(secret_sequence)),
            1.0,
        )
        return (
            0.70 * details["prefix_match_rate"]
            + 0.20 * details["full_secret_realized"]
            - 0.10 * sentence_count_penalty
        )

    def _length_score(self, text):
        words = self._safe_words(text)
        return min(len(words) / float(self.shaping_target_words), 1.0)

    def __call__(self, *args, **kwargs):
        """
        Compute rewards for a batch of generated texts.

        Args:
            *args/**kwargs: TRL passes completions in varying formats by version.

        Returns:
            rewards: Tensor of reward values
        """
        texts = self._extract_texts(args, kwargs)
        rewards = []
        for text in texts:
            clean_text = sanitize_generated_text(text)
            if self.method == "acrostics":
                score = self._acrostics_training_score(clean_text)
            else:
                score = self.detector(clean_text, *self.detector_args)

            # Normalize by baseline (z-score style)
            # Higher scores = better watermarking
            if self.baseline_std > 0:
                normalized_score = (score - self.baseline_mean) / self.baseline_std
            else:
                normalized_score = score - self.baseline_mean

            reward_value = self.detector_weight * normalized_score

            if self.reward_shaping:
                if self.method == "acrostics":
                    reward_value += self.shaping_format_weight * self._format_score(clean_text)
                    reward_value -= 0.25 * self._artifact_penalty(text)
                else:
                    reward_value += self.shaping_format_weight * self._format_score(clean_text)
                    reward_value += self.shaping_partial_weight * self._partial_progress_score(clean_text)
                    reward_value += self.shaping_length_weight * self._length_score(clean_text)

            if self.max_abs_reward is not None and self.max_abs_reward > 0:
                reward_value = float(np.clip(reward_value, -self.max_abs_reward, self.max_abs_reward))

            rewards.append(reward_value)

        return torch.tensor(rewards, dtype=torch.float32)


def get_prompt_function(method):
    """Get the prompt function for a given method."""
    if method == 'unicode':
        return unicode_embed_prompt
    elif method == 'initials':
        return initials_embed_prompt
    elif method == 'lexical':
        return lexical_embed_prompt
    elif method == 'acrostics':
        return acrostics_embed_prompt
    else:
        raise ValueError(f"Unknown method: {method}")


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


def _format_alpaca_query(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    if input_text:
        return f"{instruction}\n\nContext: {input_text}"
    return instruction


def load_causal_lm_with_dtype_fallback(model_name, model_kwargs, dtype_value=None):
    """
    Load CausalLM model while handling dtype argument differences across
    transformers versions (`dtype` vs `torch_dtype`).
    """
    return load_causal_lm_with_adapter_support(
        model_name_or_path=model_name,
        model_kwargs=model_kwargs,
        dtype_value=dtype_value,
    )


def _looks_like_local_path(source):
    if source is None:
        return False
    value = str(source).strip()
    if not value:
        return False

    if value.startswith((".", "/", "~")):
        return True

    # Windows absolute paths (e.g., C:\foo\bar)
    if re.match(r"^[a-zA-Z]:[\\/]", value):
        return True

    normalized = value.replace("\\", "/")
    if "/" not in normalized:
        return False

    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 3:
        return True

    # Trailing slash generally indicates a filesystem path intent.
    if value.endswith(("/", "\\")):
        return True

    # Two-part strings can be either local relative paths or HF repo ids.
    # Treat as local only if the full path exists in cwd (not just the
    # top-level segment, which can false-positive on HF cache dirs).
    if len(parts) == 2:
        return os.path.exists(normalized)

    return False


def resolve_pretrained_source(source, label):
    """
    Resolve local model paths safely and keep HF repo ids untouched.
    """
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


def resolve_eval_output_dir(model_source, explicit_output_dir=None):
    """
    Choose a deterministic eval output location for local checkpoints and base model ids.
    """
    if explicit_output_dir:
        return explicit_output_dir

    if _looks_like_local_path(model_source):
        normalized = os.path.normpath(os.path.expanduser(str(model_source)))
        parent_dir = os.path.dirname(normalized) or "."
        return os.path.join(parent_dir, "eval_results")

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_source)).strip("_")
    if not safe_name:
        safe_name = "model"
    return os.path.join("eval_results", safe_name)


def build_grpo_config(base_training_args, generation_args, num_generations=4):
    """
    Build GRPOConfig in a way that is compatible across TRL versions.
    """
    signature = inspect.signature(GRPOConfig.__init__)
    accepted = {name for name in signature.parameters if name != "self"}

    config_kwargs = {}
    consumed_keys = set()
    for key, value in base_training_args.items():
        if key in accepted:
            config_kwargs[key] = value
            consumed_keys.add(key)

    # Generation controls vary heavily across TRL versions.
    if "generation_kwargs" in accepted:
        config_kwargs["generation_kwargs"] = dict(generation_args)
        consumed_keys.update(generation_args.keys())
    else:
        for key, value in generation_args.items():
            if key in accepted:
                config_kwargs[key] = value
                consumed_keys.add(key)
        if (
            "max_completion_length" in accepted
            and "max_new_tokens" in generation_args
            and "max_completion_length" not in config_kwargs
        ):
            config_kwargs["max_completion_length"] = generation_args["max_new_tokens"]
            consumed_keys.add("max_new_tokens")

    # Number of completions field name changed across versions.
    if "num_generations" in accepted:
        config_kwargs["num_generations"] = num_generations
    elif "num_generation_per_prompt" in accepted:
        config_kwargs["num_generation_per_prompt"] = num_generations
    elif "num_return_sequences" in accepted:
        config_kwargs["num_return_sequences"] = num_generations

    dropped = sorted(set(base_training_args).union(generation_args) - consumed_keys)
    if dropped:
        warnings.warn(
            "Ignoring unsupported GRPOConfig args for this TRL version: "
            + ", ".join(dropped)
        )

    return GRPOConfig(**config_kwargs)


def build_grpo_trainer(
    model,
    training_args,
    train_dataset,
    tokenizer,
    reward_fn,
    reference_model=None,
    require_explicit_reference=False
):
    """
    Build GRPOTrainer in a way that is compatible across TRL versions.
    """
    signature = inspect.signature(GRPOTrainer.__init__)
    accepted = {name for name in signature.parameters if name != "self"}

    trainer_kwargs = {}
    trainer_accepts_reference_arg = (
        "ref_model" in accepted or "reference_model" in accepted
    )
    explicit_reference_arg_used = False

    if "model" in accepted:
        trainer_kwargs["model"] = model
    elif "policy" in accepted:
        trainer_kwargs["policy"] = model
    else:
        raise TypeError("Unsupported GRPOTrainer signature: missing model/policy argument.")

    if "args" in accepted:
        trainer_kwargs["args"] = training_args
    elif "config" in accepted:
        trainer_kwargs["config"] = training_args

    if "train_dataset" in accepted:
        trainer_kwargs["train_dataset"] = train_dataset
    elif "dataset" in accepted:
        trainer_kwargs["dataset"] = train_dataset

    if "tokenizer" in accepted:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in accepted:
        trainer_kwargs["processing_class"] = tokenizer
    elif "processor" in accepted:
        trainer_kwargs["processor"] = tokenizer

    if reference_model is not None:
        if "ref_model" in accepted:
            trainer_kwargs["ref_model"] = reference_model
            explicit_reference_arg_used = True
        elif "reference_model" in accepted:
            trainer_kwargs["reference_model"] = reference_model
            explicit_reference_arg_used = True
        else:
            message = (
                "Reference model was requested, but this TRL version does not expose "
                "a ref_model/reference_model argument."
            )
            if require_explicit_reference:
                raise RuntimeError(
                    message
                    + " Explicit reference usage is required, so aborting. "
                    + "Upgrade/downgrade TRL to a version whose GRPOTrainer accepts "
                    + "ref_model/reference_model. "
                    + "If you can tolerate implicit reference handling, rerun with "
                    + "--allow-implicit-reference --beta 0.04."
                )
            warnings.warn(
                message
                + " Falling back to default KL reference behavior."
            )

    reward_attempts = []
    if "reward_function" in accepted:
        reward_attempts.append(("reward_function", reward_fn))
    if "reward_funcs" in accepted:
        reward_attempts.append(("reward_funcs", reward_fn))
        reward_attempts.append(("reward_funcs", [reward_fn]))
    if "reward_fn" in accepted:
        reward_attempts.append(("reward_fn", reward_fn))

    if not reward_attempts:
        raise TypeError(
            "Unsupported GRPOTrainer signature: no reward_function/reward_funcs/reward_fn argument."
        )

    last_error = None
    for reward_key, reward_value in reward_attempts:
        try:
            trainer = GRPOTrainer(
                **trainer_kwargs,
                **{reward_key: reward_value},
            )
            # Persist compatibility details for downstream logging/metadata.
            trainer._icw_trainer_accepts_reference_arg = trainer_accepts_reference_arg
            trainer._icw_explicit_reference_arg_used = bool(
                reference_model is not None and explicit_reference_arg_used
            )
            return trainer
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to initialize GRPOTrainer for unknown reasons.")


def prepare_dataset(num_samples=100, split="train", dataset_name="eli5", seed=42):
    """Prepare training/eval dataset from ELI5, Alpaca, or a mixed pool."""
    dataset_key = dataset_name.strip().lower()
    print(f"Loading {num_samples} samples from {dataset_key} ({split})...")
    try:
        if dataset_key == "eli5":
            # Some environments expose only the train split for this dataset config.
            # Mirror Alpaca behavior by slicing train into train/validation/test.
            dataset = load_dataset("sentence-transformers/eli5", "pair", split="train")
            start, end = _slice_indices_for_split(len(dataset), split)
            sampled = dataset.select(range(start, min(end, start + num_samples)))
            queries = sampled["question"]
        elif dataset_key == "alpaca":
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            start, end = _slice_indices_for_split(len(dataset), split)
            sampled = dataset.select(range(start, min(end, start + num_samples)))
            queries = [_format_alpaca_query(row) for row in sampled]
        elif dataset_key == "mixed":
            eli5_target = max(1, num_samples // 2)
            alpaca_target = max(1, num_samples - eli5_target)

            eli5_dataset = load_dataset("sentence-transformers/eli5", "pair", split="train")
            eli5_start, eli5_end = _slice_indices_for_split(len(eli5_dataset), split)
            eli5_subset = eli5_dataset.select(range(eli5_start, min(eli5_end, eli5_start + eli5_target)))
            eli5_queries = list(eli5_subset["question"])

            alpaca_dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            alpaca_start, alpaca_end = _slice_indices_for_split(len(alpaca_dataset), split)
            alpaca_subset = alpaca_dataset.select(range(alpaca_start, min(alpaca_end, alpaca_start + alpaca_target)))
            alpaca_queries = [_format_alpaca_query(row) for row in alpaca_subset]

            queries = eli5_queries + alpaca_queries
            if len(queries) < num_samples:
                warnings.warn(
                    f"Requested {num_samples} mixed samples but only loaded {len(queries)} "
                    f"(eli5={len(eli5_queries)}, alpaca={len(alpaca_queries)})."
                )
            rng = np.random.default_rng(seed)
            rng.shuffle(queries)
        else:
            raise ValueError("dataset_name must be 'eli5', 'alpaca', or 'mixed'")
    except Exception as exc:
        print(f"⚠️  Could not load dataset '{dataset_key}' split '{split}': {exc}")
        return None

    # Create dataset with queries
    dataset_dict = {"query": queries}
    dataset = Dataset.from_dict(dataset_dict)

    print(f"✓ Loaded {len(dataset)} samples")
    return dataset


def build_messages(query, prompt_fn=None, include_instruction=True, target_sentence_count=None):
    """Build chat messages with optional watermark instruction."""
    if include_instruction and prompt_fn is not None:
        messages = [dict(message) for message in prompt_fn(query)]
    else:
        messages = [
            {"role": "system", "content": get_base_system_prompt()},
            {"role": "user", "content": query}
        ]

    if target_sentence_count is not None:
        constraint = (
            f"Respond in exactly {target_sentence_count} sentences. "
            f"Stop after sentence {target_sentence_count}."
        )
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = messages[0]["content"].rstrip() + "\n\nAdditional requirement:\n" + constraint
        else:
            messages.insert(0, {"role": "system", "content": constraint})

    return messages


def generate_responses_batch(
    model,
    tokenizer,
    messages_batch,
    max_new_tokens=200,
    min_new_tokens=None,
    temperature=0.7,
    top_p=0.9
):
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
            truncation=True
        ).to(model.device)

        gen_kwargs = dict(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        if min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = min_new_tokens

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(messages_batch) == 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"⚠️  OOM at generation batch size {len(messages_batch)}. Falling back to single-sample generation.")
        fallback_responses = []
        for messages in messages_batch:
            fallback_responses.extend(
                generate_responses_batch(
                    model=model,
                    tokenizer=tokenizer,
                    messages_batch=[messages],
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            )
        return fallback_responses

    responses = []
    # `generate()` returns the full padded prompt prefix followed by new tokens.
    # Slice by the shared encoded prompt width, not the non-pad token count, or
    # left-padded batches will leak prompt fragments into the decoded response.
    prompt_len = encoded["input_ids"].shape[1]
    for idx in range(outputs.shape[0]):
        decoded = tokenizer.decode(outputs[idx][prompt_len:], skip_special_tokens=True)
        responses.append(
            sanitize_generated_text(decoded)
        )

    return responses


def compute_baseline_statistics(
    model,
    tokenizer,
    dataset,
    method,
    num_samples=50,
    generation_batch_size=4,
    reward_override_fn=None,
    max_new_tokens=512,
    min_new_tokens=None,
):
    """
    Compute baseline statistics from non-watermarked generations.
    These are used to normalize rewards.

    Args:
        reward_override_fn: Optional callable(text) -> float.  When provided,
            this is used instead of the standard detector to compute scores.
            Used for acrostics to align baselines with the per-sentence reward.
    """
    print(f"\nComputing baseline statistics for {method}...")

    detector, detector_args = get_detector_and_args(method)

    scores = []

    samples = dataset.select(range(min(num_samples, len(dataset))))
    queries = samples["query"]
    total = len(queries)

    for start in range(0, total, generation_batch_size):
        end = min(start + generation_batch_size, total)
        batch_queries = queries[start:end]
        messages_batch = [build_messages(query, include_instruction=False) for query in batch_queries]
        responses = generate_responses_batch(
            model=model,
            tokenizer=tokenizer,
            messages_batch=messages_batch,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=0.7,
            top_p=0.9
        )

        for response in responses:
            if reward_override_fn is not None:
                score = reward_override_fn(response)
            else:
                score = detector(response, *detector_args)
            scores.append(score)

        processed = end
        if processed % 10 == 0 or processed == total:
            print(f"  Progress: {processed}/{total}")

    mean = np.mean(scores)
    std = np.std(scores)

    print(f"✓ Baseline computed: mean={mean:.3f}, std={std:.3f}")
    print(f"  This will be used to normalize rewards during training")

    return mean, std


def evaluate_model_on_split(
    model,
    tokenizer,
    dataset,
    method,
    prompt_fn,
    include_instruction,
    max_samples,
    output_dir,
    split_name,
    dataset_name="eli5",
    generation_batch_size=4,
    baseline_mean=None,
    baseline_std=None,
    reward_override_fn=None,
    eval_profile_name="natural",
    eval_temperature=0.7,
    eval_top_p=0.9,
    max_new_tokens=512,
    min_new_tokens=None,
):
    """Evaluate detector scores on a dataset split.

    Args:
        reward_override_fn: Optional callable(text) -> float.  When provided,
            this is used as the *primary* scoring metric (for z-score
            computation).  The standard detector score is still recorded
            alongside it for paper-comparable ROC-AUC reporting.
        min_new_tokens: Optional minimum tokens to generate per response.
            Helps ensure responses are long enough for reliable detector scores.
    """
    detector, detector_args = get_detector_and_args(method)
    scores = []
    detector_scores = []
    records = []

    samples = dataset.select(range(min(max_samples, len(dataset))))

    model.eval()
    queries = samples["query"]
    total = len(queries)
    target_sentence_count = (
        len(secret_sequence)
        if method == "acrostics" and eval_profile_name == "controlled"
        else None
    )

    for start in range(0, total, generation_batch_size):
        end = min(start + generation_batch_size, total)
        batch_queries = queries[start:end]
        messages_batch = [
            build_messages(
                query,
                prompt_fn=prompt_fn,
                include_instruction=include_instruction,
                target_sentence_count=target_sentence_count,
            )
            for query in batch_queries
        ]
        responses = generate_responses_batch(
            model=model,
            tokenizer=tokenizer,
            messages_batch=messages_batch,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=eval_temperature,
            top_p=eval_top_p
        )

        for query, response in zip(batch_queries, responses):
            response = sanitize_generated_text(response)
            det_score = detector(response, *detector_args)
            detector_scores.append(det_score)

            if reward_override_fn is not None:
                score = reward_override_fn(response)
            else:
                score = det_score
            scores.append(score)

            response_metrics = response_stats(response)
            method_metrics = build_method_record_metrics(method, response)

            record = {
                "query": query,
                "response": response,
                "score": score,
                "detector_score": det_score,
                **response_metrics,
                **method_metrics,
            }
            records.append(record)

        processed = end
        if processed % 10 == 0 or processed == total:
            print(f"  Eval progress ({split_name}): {processed}/{total}")

    mean = float(np.mean(scores)) if scores else 0.0
    std = float(np.std(scores)) if scores else 0.0
    det_mean = float(np.mean(detector_scores)) if detector_scores else 0.0
    det_std = float(np.std(detector_scores)) if detector_scores else 0.0
    word_counts = [r["n_words"] for r in records]
    char_counts = [r["n_chars"] for r in records]
    sentence_counts = [r["n_sentences"] for r in records]
    mean_words = float(np.mean(word_counts)) if word_counts else 0.0
    std_words = float(np.std(word_counts)) if word_counts else 0.0
    mean_chars = float(np.mean(char_counts)) if char_counts else 0.0
    std_chars = float(np.std(char_counts)) if char_counts else 0.0
    mean_sentences = float(np.mean(sentence_counts)) if sentence_counts else 0.0
    std_sentences = float(np.std(sentence_counts)) if sentence_counts else 0.0
    normalized_mean = None
    if baseline_mean is not None:
        if baseline_std is not None and baseline_std > 0:
            normalized_mean = float((mean - baseline_mean) / baseline_std)
        else:
            normalized_mean = float(mean - baseline_mean)

    summary = {
        "dataset": dataset_name,
        "split": split_name,
        "eval_profile": eval_profile_name,
        "method": method,
        "include_instruction": include_instruction,
        "num_samples": len(scores),
        "mean_score": mean,
        "std_score": std,
        "mean_words": mean_words,
        "std_words": std_words,
        "mean_chars": mean_chars,
        "std_chars": std_chars,
        "mean_sentences": mean_sentences,
        "std_sentences": std_sentences,
        "eval_temperature": eval_temperature,
        "eval_top_p": eval_top_p,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
    }
    if reward_override_fn is not None:
        summary["detector_mean_score"] = det_mean
        summary["detector_std_score"] = det_std
    if normalized_mean is not None:
        summary["mean_score_vs_baseline_z"] = normalized_mean
    else:
        summary["mean_score_vs_baseline_z"] = None

    if method == "acrostics" and records:
        prefix_rates = [record["acrostics_prefix_match_rate"] for record in records]
        sentence_rates = [record["acrostics_sentence_match_rate"] for record in records]
        coverages = [record["acrostics_secret_coverage"] for record in records]
        full_secret = [record["acrostics_full_secret_realized"] for record in records]
        normalized_distances = [
            record["acrostics_normalized_levenshtein_distance"] for record in records
        ]
        sentence_count_errors = [
            record["acrostics_sentence_count_error"] for record in records
        ]
        extra_sentence_counts = [
            record["acrostics_extra_sentence_count"] for record in records
        ]
        missing_sentence_counts = [
            record["acrostics_missing_sentence_count"] for record in records
        ]
        bin_hist = {}
        for record in records:
            bin_name = record["acrostics_sentence_count_bin"]
            bin_hist[bin_name] = bin_hist.get(bin_name, 0) + 1

        summary.update(
            {
                "acrostics_mean_prefix_match_rate": float(np.mean(prefix_rates)),
                "acrostics_mean_sentence_match_rate": float(np.mean(sentence_rates)),
                "acrostics_mean_secret_coverage": float(np.mean(coverages)),
                "acrostics_full_secret_realization_rate": float(np.mean(full_secret)),
                "acrostics_mean_sentence_count_error": float(np.mean(sentence_count_errors)),
                "acrostics_mean_extra_sentence_count": float(np.mean(extra_sentence_counts)),
                "acrostics_mean_missing_sentence_count": float(np.mean(missing_sentence_counts)),
                "acrostics_mean_normalized_levenshtein_distance": float(
                    np.mean(normalized_distances)
                ),
                "acrostics_sentence_count_bins": bin_hist,
            }
        )

    output = {
        "summary": summary,
        "samples": records
    }

    eval_path = os.path.join(output_dir, f"eval_{dataset_name}_{split_name}_{eval_profile_name}.json")
    with open(eval_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✓ Saved {split_name} evaluation to: {eval_path}")
    print(f"  Mean score: {mean:.4f}, Std: {std:.4f}")
    print(
        f"  Response length: {mean_words:.0f} ± {std_words:.0f} words | "
        f"{mean_sentences:.1f} ± {std_sentences:.1f} sentences"
    )
    if reward_override_fn is not None:
        print(f"  Detector score: {det_mean:.4f}, Std: {det_std:.4f}")
    if normalized_mean is not None:
        print(f"  Mean score vs training baseline (z): {normalized_mean:.4f}")
    if method == "acrostics" and records:
        print(
            "  Acrostics metrics: "
            f"prefix={summary['acrostics_mean_prefix_match_rate']:.4f}, "
            f"position={summary['acrostics_mean_sentence_match_rate']:.4f}, "
            f"coverage={summary['acrostics_mean_secret_coverage']:.4f}, "
            f"count_error={summary['acrostics_mean_sentence_count_error']:.2f}"
        )

    return summary


def train_grpo(
    model_strategy="small",
    method="unicode",
    num_train_samples=100,
    num_epochs=3,
    batch_size=4,
    learning_rate=1e-5,
    output_dir="grpo_models",
    eval_splits="validation,test",
    eval_samples=50,
    train_dataset_name="eli5",
    eval_dataset_names="eli5,alpaca",
    eval_profiles="natural",
    eval_modes=None,
    eval_no_instruction=True,
    generation_batch_size=4,
    prompt_variant="paper",
    rules_variant="paper",
    base_system_prompt=None,
    system_prompt_prefix=None,
    use_lora=False,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    warm_start_model_path=None,
    reference_model_path=None,
    disable_reference_model=False,
    reward_shaping=True,
    shaping_format_weight=0.10,
    shaping_partial_weight=0.40,
    shaping_length_weight=0.10,
    shaping_target_words=120,
    max_abs_reward=10.0,
    num_generations=4,
    max_new_tokens=200,
    min_new_tokens=None,
    generation_temperature=0.7,
    generation_top_p=0.9,
    natural_max_new_tokens=None,
    natural_min_new_tokens=None,
    controlled_max_new_tokens=None,
    controlled_min_new_tokens=None,
    remove_invalid_values=True,
    require_explicit_reference=True,
    beta=0.04,
    seed=42,
    implicit_fraction=0.4
):
    """
    Train a model using GRPO with watermarking rewards.

    Args:
        model_strategy: Model configuration (small, 4bit, etc.)
        method: Watermarking method (unicode, initials, lexical, acrostics)
        num_train_samples: Number of training samples
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        output_dir: Directory to save trained models
        eval_splits: Comma-separated list of dataset splits to evaluate
        eval_samples: Number of samples for each evaluation split
        train_dataset_name: Dataset for GRPO training
        eval_dataset_names: Comma-separated list of datasets for post-training eval
        eval_profiles: Comma-separated evaluation generation profiles
        eval_modes: Comma-separated evaluation modes (implicit, explicit)
        eval_no_instruction: Disable watermark instruction during eval
        warm_start_model_path: Optional SFT checkpoint path used to initialize policy
        reference_model_path: Optional override for KL reference policy
        disable_reference_model: Disable explicit KL reference model loading
        reward_shaping: Add explicit shaping terms for partial progress/format/length
        num_generations: Number of sampled completions per prompt
        max_new_tokens: Max tokens generated per completion
        min_new_tokens: Optional minimum tokens generated per response during controlled eval
        generation_temperature: Sampling temperature used during GRPO rollouts
        generation_top_p: Nucleus sampling top-p used during GRPO rollouts
        remove_invalid_values: Filter inf/nan logits during generation for stability
        require_explicit_reference: Fail if TRL cannot consume explicit reference model args
        beta: KL coefficient. Must be >0 to ensure reference policy contributes to loss.
        seed: Global experiment seed used for trainer + dataset sampling.
        implicit_fraction: Fraction of training prompts WITHOUT watermark instructions.
            These prompts still get rewarded for watermark quality, training the model
            to produce watermarks without explicit instruction (internalization).
    """

    valid_prompt_variants = {"paper", "concise", "strict"}
    valid_rules_variants = {"paper", "minimal", "none"}
    if prompt_variant not in valid_prompt_variants:
        raise ValueError(
            f"Invalid prompt_variant '{prompt_variant}'. "
            f"Choose from: {', '.join(sorted(valid_prompt_variants))}"
        )
    if rules_variant not in valid_rules_variants:
        raise ValueError(
            f"Invalid rules_variant '{rules_variant}'. "
            f"Choose from: {', '.join(sorted(valid_rules_variants))}"
        )
    if eval_modes is None:
        eval_modes = "implicit,explicit" if eval_no_instruction else "explicit"

    # Configure prompt/rule variants consumed by main.py prompt builders.
    os.environ["ICW_PROMPT_VARIANT"] = prompt_variant
    os.environ["ICW_RULES_VARIANT"] = rules_variant
    if base_system_prompt:
        os.environ["ICW_BASE_SYSTEM_PROMPT"] = base_system_prompt
    else:
        os.environ.pop("ICW_BASE_SYSTEM_PROMPT", None)
    if system_prompt_prefix:
        os.environ["ICW_SYSTEM_PROMPT_PREFIX"] = system_prompt_prefix
    else:
        os.environ.pop("ICW_SYSTEM_PROMPT_PREFIX", None)

    effective_shaping_format_weight = shaping_format_weight
    effective_shaping_partial_weight = shaping_partial_weight
    effective_shaping_length_weight = shaping_length_weight
    effective_shaping_target_words = shaping_target_words
    if method == "acrostics":
        effective_shaping_partial_weight = 0.0
        effective_shaping_length_weight = 0.0
        effective_shaping_target_words = None

    print("\n" + "="*80)
    print("GRPO TRAINING FOR ICW WATERMARKING")
    print("="*80)
    print(f"Model Strategy: {model_strategy}")
    print(f"Method: {method}")
    if method == "acrostics":
        print(f"Secret Sequence: {secret_sequence}")
    print(f"Training Samples: {num_train_samples}")
    print(f"Training Dataset: {train_dataset_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Generation Batch Size: {generation_batch_size}")
    print(f"Num Generations / Prompt: {num_generations}")
    print(f"Seed: {seed}")
    print(f"Learning Rate: {learning_rate}")
    print(
        "Generation Params (max_new_tokens/temperature/top_p/remove_invalid_values): "
        f"{max_new_tokens}/{generation_temperature}/{generation_top_p}/{remove_invalid_values}"
    )
    print(f"Prompt Variant: {prompt_variant}")
    print(f"Rules Variant: {rules_variant}")
    print(f"Use LoRA: {use_lora}")
    if use_lora:
        print(f"LoRA Rank/Alpha/Dropout: {lora_rank}/{lora_alpha}/{lora_dropout}")
        print(f"LoRA Target Modules: {lora_target_modules}")
    print(f"Warm Start Model: {warm_start_model_path or 'None'}")
    if disable_reference_model:
        kl_reference_text = "Disabled"
    else:
        kl_reference_text = reference_model_path or "Auto"
    print(f"KL Reference Override: {kl_reference_text}")
    print(f"Require Explicit Reference: {require_explicit_reference}")
    print(f"KL Beta: {beta}")
    print(f"Reward Shaping: {reward_shaping}")
    if reward_shaping:
        print(
            "Reward Shaping Weights (format/partial/length): "
            f"{effective_shaping_format_weight}/"
            f"{effective_shaping_partial_weight}/"
            f"{effective_shaping_length_weight}"
        )
        if effective_shaping_target_words is not None:
            print(f"Reward Shaping Target Words: {effective_shaping_target_words}")
    print(f"Reward Clip |r|<= {max_abs_reward}")
    print(f"Implicit Fraction: {implicit_fraction}")
    print(f"Eval Without Instructions: {eval_no_instruction}")
    print(f"Eval Splits: {eval_splits}")
    print(f"Eval Datasets: {eval_dataset_names}")
    print(f"Eval Profiles: {eval_profiles}")
    print(f"Eval Modes: {eval_modes}")
    print("="*80 + "\n")

    if GRPOConfig is None or GRPOTrainer is None:
        raise ImportError(
            "TRL with GRPO support is required for training. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    # Load model and tokenizer
    print("Loading base model...")
    config = get_model_config(model_strategy)
    config = dict(config)
    model_name = config["model_name"]
    policy_model_path = warm_start_model_path or model_name
    resolved_policy_source = resolve_pretrained_source(policy_model_path, "Policy model")
    if warm_start_model_path:
        print(f"Warm-starting policy from: {resolved_policy_source}")
        print(f"KL default reference will remain base instruct model: {model_name}")

    tokenizer_source = resolved_policy_source
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except Exception:
        if tokenizer_source != model_name:
            warnings.warn(
                f"Could not load tokenizer from '{tokenizer_source}'. "
                f"Falling back to base model tokenizer '{model_name}'."
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        else:
            raise
    # Decoder-only models should be left-padded when batching generations.
    if getattr(tokenizer, "padding_side", "right") != "left":
        tokenizer.padding_side = "left"

    model_kwargs = {
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True
    }

    use_cuda = torch.cuda.is_available()
    supports_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())

    # Note: GRPO training works best with full precision or 8-bit
    # 4-bit quantization may not work well with gradient updates
    if model_strategy == "4bit":
        print("⚠️  Warning: 4-bit models may not train well. Consider using 'small' or '8bit'")
    if model_strategy == "full" and not use_lora and learning_rate > 3e-6:
        warnings.warn(
            "Full-parameter GRPO with learning_rate > 3e-6 is often unstable "
            "(NaN logits / CUDA asserts). Consider --use-lora or lowering "
            "--learning-rate to 1e-6..3e-6."
        )

    dtype_value = config.get("dtype")
    if model_strategy == "full" and supports_bf16:
        # Prefer bf16 on modern GPUs; it is materially more stable than fp16 for RLHF-style updates.
        dtype_value = torch.bfloat16
        config["dtype"] = dtype_value
    if config.get("quantization"):
        model_kwargs["quantization_config"] = config["quantization"]
        dtype_value = None

    model = load_causal_lm_with_adapter_support(
        model_name_or_path=resolved_policy_source,
        model_kwargs=model_kwargs,
        dtype_value=dtype_value,
        is_trainable=bool(warm_start_model_path and use_lora),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    existing_peft_adapters = bool(getattr(model, "peft_config", None))

    if use_lora:
        if LoraConfig is None or get_peft_model is None or TaskType is None:
            raise ImportError(
                "PEFT is required for LoRA training. Install dependencies with: pip install peft"
            )
        if existing_peft_adapters:
            print("✓ Existing PEFT adapters detected in warm-start checkpoint; reusing them.")
        else:
            target_modules = [item.strip() for item in lora_target_modules.split(",") if item.strip()]
            if not target_modules:
                raise ValueError("LoRA target modules cannot be empty.")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            model = get_peft_model(model, lora_config)
            print("✓ LoRA adapters attached")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    print("✓ Policy model loaded successfully!\n")

    # For warm starts, default KL reference to the original instruct model.
    reference_source = None if disable_reference_model else reference_model_path
    if reference_source is None and warm_start_model_path:
        reference_source = model_name
    resolved_reference_source = resolve_pretrained_source(reference_source, "Reference model")

    reference_model = None
    if resolved_reference_source:
        print(f"Loading KL reference model from: {resolved_reference_source}")
        reference_kwargs = dict(model_kwargs)
        reference_model = load_causal_lm_with_dtype_fallback(
            model_name=resolved_reference_source,
            model_kwargs=reference_kwargs,
            dtype_value=dtype_value,
        )
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        print("✓ Reference model loaded for KL regularization\n")

    # Prepare dataset
    dataset = prepare_dataset(
        num_samples=num_train_samples,
        split="train",
        dataset_name=train_dataset_name,
        seed=seed
    )
    if dataset is None:
        raise ValueError("Training dataset could not be loaded.")

    # Compute baseline statistics for the training reward.
    _baseline_reward_override = None
    if method == "acrostics":
        _baseline_reward_override = WatermarkRewardFunction(method)._acrostics_training_score

    baseline_mean, baseline_std = compute_baseline_statistics(
        model,
        tokenizer,
        dataset,
        method,
        num_samples=min(50, num_train_samples),
        generation_batch_size=generation_batch_size,
        reward_override_fn=_baseline_reward_override,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )
    eval_baseline_mean, eval_baseline_std = compute_baseline_statistics(
        model,
        tokenizer,
        dataset,
        method,
        num_samples=min(50, num_train_samples),
        generation_batch_size=generation_batch_size,
        reward_override_fn=None,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )

    # Create reward function
    reward_fn = WatermarkRewardFunction(
        method,
        baseline_mean,
        baseline_std,
        reward_shaping=reward_shaping,
        shaping_format_weight=effective_shaping_format_weight,
        shaping_partial_weight=effective_shaping_partial_weight,
        shaping_length_weight=effective_shaping_length_weight,
        shaping_target_words=effective_shaping_target_words or shaping_target_words,
        max_abs_reward=max_abs_reward
    )

    # Get prompt function
    prompt_fn = get_prompt_function(method)

    # Prepare prompts for training (with mixed supervision support)
    _implicit_rng = np.random.default_rng(seed + 1000)

    def tokenize_function(examples):
        """Convert queries to prompts, mixing instructed and implicit modes."""
        prompts = []
        for query in examples["query"]:
            if implicit_fraction > 0 and _implicit_rng.random() < implicit_fraction:
                # No watermark instruction — base system prompt only.
                # The reward still scores watermark quality, training the
                # model to produce watermarks without explicit instruction.
                messages = build_messages(query, include_instruction=False)
            else:
                messages = prompt_fn(query)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            prompts.append(prompt_text)

        return {"prompt": prompts}

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # GRPO Configuration
    base_training_args = {
        "output_dir": os.path.join(output_dir, f"{method}_{model_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "max_grad_norm": 1.0,
        "seed": seed,
        "beta": beta,
        "bf16": bool(use_cuda and supports_bf16 and config.get("quantization") is None),
        "fp16": bool(use_cuda and (not supports_bf16) and config.get("quantization") is None),
    }
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": generation_temperature,
        "top_p": generation_top_p,
        "remove_invalid_values": remove_invalid_values,
    }
    training_args = build_grpo_config(
        base_training_args=base_training_args,
        generation_args=generation_args,
        num_generations=num_generations,
    )

    # Initialize GRPO Trainer
    print("\nInitializing GRPO Trainer...")
    trainer = build_grpo_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        reference_model=reference_model,
        require_explicit_reference=require_explicit_reference,
    )

    trainer_accepts_reference_arg = bool(
        getattr(trainer, "_icw_trainer_accepts_reference_arg", False)
    )
    explicit_reference_arg_used = bool(
        getattr(trainer, "_icw_explicit_reference_arg_used", False)
    )

    if reference_model is not None:
        if explicit_reference_arg_used:
            print("✓ Explicit KL reference model is being used by this TRL version.")
        elif not trainer_accepts_reference_arg:
            if beta <= 0:
                raise RuntimeError(
                    "TRL cannot accept explicit ref_model/reference_model in this version, "
                    "and beta <= 0 disables KL/reference contribution. "
                    "Set --beta > 0 (e.g., 0.04) or use a TRL version with explicit "
                    "reference-model support."
                )
            print(
                "⚠️  Explicit KL reference model is NOT being passed to GRPOTrainer "
                "(TRL lacks ref_model/reference_model in this version)."
            )
            print(
                "✓ Using TRL internal reference mechanism because beta > 0."
            )
        else:
            print(
                "⚠️  Explicit KL reference model may not be active "
                "(trainer accepted reference args but none were attached)."
            )
    else:
        print("ℹ️  No explicit KL reference model requested.")

    print("✓ Trainer initialized\n")

    # Train
    print("Starting training...")
    print("="*80 + "\n")

    trainer.train()

    print("\n" + "="*80)
    print("Training completed!")
    print("="*80 + "\n")

    # Save final model
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    print(f"Saving final model to {final_model_path}...")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    patch_saved_model_config(final_model_path, model_name)

    # Save training metadata
    metadata = {
        "base_model": model_name,
        "policy_model_init": policy_model_path,
        "policy_model_init_resolved": resolved_policy_source,
        "warm_start_model_path": warm_start_model_path,
        "kl_reference_model_path": reference_source,
        "kl_reference_model_path_resolved": resolved_reference_source,
        "disable_reference_model": disable_reference_model,
        "trainer_accepts_reference_arg": trainer_accepts_reference_arg,
        "explicit_reference_arg_used": explicit_reference_arg_used,
        "require_explicit_reference": require_explicit_reference,
        "model_strategy": model_strategy,
        "method": method,
        "secret_sequence": secret_sequence if method == "acrostics" else None,
        "num_train_samples": num_train_samples,
        "num_epochs": num_epochs,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_dataset_name": train_dataset_name,
        "eval_dataset_names": eval_dataset_names,
        "eval_profiles": eval_profiles,
        "eval_modes": eval_modes,
        "prompt_variant": prompt_variant,
        "rules_variant": rules_variant,
        "base_system_prompt": get_base_system_prompt(),
        "system_prompt_prefix": system_prompt_prefix or "",
        "use_lora": use_lora,
        "existing_peft_adapters_in_policy_init": existing_peft_adapters,
        "lora_rank": lora_rank if use_lora else None,
        "lora_alpha": lora_alpha if use_lora else None,
        "lora_dropout": lora_dropout if use_lora else None,
        "lora_target_modules": lora_target_modules if use_lora else "",
        "reward_shaping": reward_shaping,
        "shaping_format_weight": effective_shaping_format_weight if reward_shaping else None,
        "shaping_partial_weight": effective_shaping_partial_weight if reward_shaping else None,
        "shaping_length_weight": effective_shaping_length_weight if reward_shaping else None,
        "shaping_target_words": effective_shaping_target_words if reward_shaping else None,
        "max_abs_reward": max_abs_reward,
        "beta": beta,
        "num_generations": num_generations,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "natural_max_new_tokens": natural_max_new_tokens,
        "natural_min_new_tokens": natural_min_new_tokens,
        "controlled_max_new_tokens": controlled_max_new_tokens,
        "controlled_min_new_tokens": controlled_min_new_tokens,
        "generation_temperature": generation_temperature,
        "generation_top_p": generation_top_p,
        "remove_invalid_values": remove_invalid_values,
        "policy_dtype": str(dtype_value) if dtype_value is not None else None,
        "implicit_fraction": implicit_fraction,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "timestamp": datetime.now().isoformat()
    }

    metadata_path = os.path.join(final_model_path, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Model saved to: {final_model_path}")
    print(f"✓ Metadata saved to: {metadata_path}")

    # Post-training evaluation: dual mode (explicit + implicit)
    if eval_splits:
        split_list = [s.strip() for s in eval_splits.split(",") if s.strip()]
        dataset_list = [d.strip().lower() for d in eval_dataset_names.split(",") if d.strip()]
        profile_specs = resolve_eval_profiles(
            profile_names=eval_profiles,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            natural_max_new_tokens=natural_max_new_tokens,
            natural_min_new_tokens=natural_min_new_tokens,
            controlled_max_new_tokens=controlled_max_new_tokens,
            controlled_min_new_tokens=controlled_min_new_tokens,
        )
        eval_mode_specs = resolve_eval_modes(eval_modes)
        valid_datasets = {"eli5", "alpaca"}
        for dataset_name in dataset_list:
            if dataset_name not in valid_datasets:
                warnings.warn(
                    f"Unknown eval dataset '{dataset_name}', skipping. "
                    f"Supported: {', '.join(sorted(valid_datasets))}"
                )
        if split_list:
            print("\n" + "="*80)
            print("POST-TRAINING EVALUATION")
            print("="*80)

            for dataset_name in dataset_list:
                if dataset_name not in valid_datasets:
                    continue
                for split_name in split_list:
                    eval_dataset = prepare_dataset(
                        num_samples=eval_samples,
                        split=split_name,
                        dataset_name=dataset_name,
                        seed=seed
                    )
                    if eval_dataset is None:
                        print(f"⚠️  Skipping evaluation for {dataset_name}:{split_name}")
                        continue

                    for profile in profile_specs:
                        for mode_label, use_instruction in eval_mode_specs:
                            eval_split_label = f"{split_name}_{mode_label}"
                            print(
                                f"\n--- {dataset_name}:{split_name} [{mode_label}] "
                                f"[{profile['name']}] "
                                f"(instruction={'yes' if use_instruction else 'no'}) ---"
                            )
                            evaluate_model_on_split(
                                model=model,
                                tokenizer=tokenizer,
                                dataset=eval_dataset,
                                method=method,
                                prompt_fn=prompt_fn,
                                include_instruction=use_instruction,
                                max_samples=eval_samples,
                                output_dir=training_args.output_dir,
                                split_name=eval_split_label,
                                dataset_name=dataset_name,
                                generation_batch_size=generation_batch_size,
                                baseline_mean=eval_baseline_mean,
                                baseline_std=eval_baseline_std,
                                reward_override_fn=None,
                                eval_profile_name=profile["name"],
                                max_new_tokens=profile["max_new_tokens"],
                                min_new_tokens=profile["min_new_tokens"],
                            )

    return final_model_path


def main():
    global secret_sequence
    parser = argparse.ArgumentParser(
        description="Train ICW watermarking models using GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train small model on Unicode watermarking
  python grpo_train.py --model small --method unicode --epochs 3

  # Train with more samples for better results
  python grpo_train.py --model small --method acrostics --samples 200 --epochs 5

  # Train on all methods (run sequentially)
  for method in unicode initials lexical acrostics; do
    python grpo_train.py --model small --method $method --epochs 3
  done

  # Warm-start GRPO from SFT checkpoint, keep KL reference on instruct model
  python grpo_train.py --model small --method lexical --warm-start-model path/to/sft_model
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='small',
        choices=['small', '4bit', '8bit', 'full', 'cpu'],
        help='Model strategy (4bit is allowed but generally unstable for training)'
    )

    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['unicode', 'initials', 'lexical', 'acrostics'],
        help='Watermarking method to train on'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of training samples (default: 100)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )

    parser.add_argument(
        '--eval-splits',
        type=str,
        default='validation,test',
        help='Comma-separated dataset splits to evaluate (default: validation,test)'
    )

    parser.add_argument(
        '--eval-samples',
        type=int,
        default=50,
        help='Number of evaluation samples per split (default: 50)'
    )

    parser.add_argument(
        '--train-dataset',
        type=str,
        default='eli5',
        choices=['eli5', 'alpaca', 'mixed'],
        help='Dataset used for GRPO training (default: eli5)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for trainer and dataset sampling (default: 42)'
    )

    parser.add_argument(
        '--eval-datasets',
        type=str,
        default='eli5,alpaca',
        help='Comma-separated datasets for post-training evaluation (default: eli5,alpaca)'
    )
    parser.add_argument(
        '--eval-profiles',
        type=str,
        default='natural',
        help="Comma-separated eval generation profiles: natural, controlled (default: natural)"
    )
    parser.add_argument(
        '--eval-modes',
        type=str,
        default=None,
        help="Comma-separated eval modes: implicit, explicit. "
             "Defaults to implicit,explicit unless legacy --eval-with-instruction is used."
    )

    parser.add_argument(
        '--eval-no-instruction',
        dest='eval_no_instruction',
        action='store_true',
        default=True,
        help='Disable watermarking instructions during eval (default behavior)'
    )

    parser.add_argument(
        '--eval-with-instruction',
        dest='eval_no_instruction',
        action='store_false',
        help='Enable watermarking instructions during eval'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training (default: 4)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate (default: 1e-5)'
    )

    parser.add_argument(
        '--beta',
        type=float,
        default=0.04,
        help='KL coefficient for GRPO reference policy (default: 0.04)'
    )

    parser.add_argument(
        '--num-generations',
        type=int,
        default=4,
        help='Number of sampled completions per prompt during GRPO (default: 4)'
    )

    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=200,
        help='Max tokens generated per completion during GRPO (default: 200)'
    )

    parser.add_argument(
        '--min-new-tokens',
        type=int,
        default=None,
        help='Default min tokens generated per response for controlled eval (default: None). '
             'If unset, the controlled profile chooses a conservative floor automatically.'
    )
    parser.add_argument(
        '--natural-max-new-tokens',
        type=int,
        default=None,
        help='Override max_new_tokens for the natural eval profile'
    )
    parser.add_argument(
        '--natural-min-new-tokens',
        type=int,
        default=None,
        help='Override min_new_tokens for the natural eval profile'
    )
    parser.add_argument(
        '--controlled-max-new-tokens',
        type=int,
        default=None,
        help='Override max_new_tokens for the controlled eval profile'
    )
    parser.add_argument(
        '--controlled-min-new-tokens',
        type=int,
        default=None,
        help='Override min_new_tokens for the controlled eval profile'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature for GRPO rollouts (default: 0.7)'
    )

    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p for GRPO rollout sampling (default: 0.9)'
    )

    parser.add_argument(
        '--allow-invalid-logits',
        dest='remove_invalid_values',
        action='store_false',
        default=True,
        help='Disable invalid-logit filtering (not recommended; default keeps filtering on)'
    )

    parser.add_argument(
        '--prompt-variant',
        type=str,
        default='paper',
        choices=['paper', 'concise', 'strict'],
        help='Instruction prompt style variant (default: paper)'
    )

    parser.add_argument(
        '--rules-variant',
        type=str,
        default='paper',
        choices=['paper', 'minimal', 'none'],
        help='Rules variant for system prompts (default: paper)'
    )
    parser.add_argument(
        '--secret-sequence',
        type=str,
        default=secret_sequence,
        help='Acrostics secret string to realize (default: current configured secret)'
    )

    parser.add_argument(
        '--base-system-prompt',
        type=str,
        default=None,
        help='Override baseline non-watermarked system prompt'
    )

    parser.add_argument(
        '--system-prompt-prefix',
        type=str,
        default=None,
        help='Prefix injected at the top of watermarking system prompts'
    )

    parser.add_argument(
        '--use-lora',
        action='store_true',
        help='Enable LoRA adapters during GRPO training'
    )

    parser.add_argument(
        '--lora-rank',
        type=int,
        default=16,
        help='LoRA rank r (default: 16)'
    )

    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha (default: 32)'
    )

    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=0.05,
        help='LoRA dropout (default: 0.05)'
    )

    parser.add_argument(
        '--lora-target-modules',
        type=str,
        default='q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj',
        help='Comma-separated LoRA target modules'
    )

    parser.add_argument(
        '--warm-start-model',
        type=str,
        default=None,
        help='Path/name of SFT checkpoint used to initialize GRPO policy model'
    )

    parser.add_argument(
        '--reference-model',
        type=str,
        default=None,
        help='Path/name for KL reference model (default: base instruct model when warm-starting)'
    )

    parser.add_argument(
        '--no-reference-model',
        action='store_true',
        help='Disable explicit KL reference model loading'
    )

    parser.add_argument(
        '--allow-implicit-reference',
        dest='require_explicit_reference',
        action='store_false',
        default=True,
        help='Allow training when TRL cannot accept explicit ref_model/reference_model args'
    )

    parser.add_argument(
        '--reward-shaping',
        action='store_true',
        default=True,
        help='Enable reward shaping terms (default: enabled)'
    )
    parser.add_argument(
        '--no-reward-shaping',
        dest='reward_shaping',
        action='store_false',
        help='Disable reward shaping'
    )

    parser.add_argument(
        '--shaping-format-weight',
        type=float,
        default=0.10,
        help='Weight for format/structure shaping reward (default: 0.10)'
    )

    parser.add_argument(
        '--shaping-partial-weight',
        type=float,
        default=0.40,
        help='Weight for partial-progress shaping reward (default: 0.40)'
    )

    parser.add_argument(
        '--shaping-length-weight',
        type=float,
        default=0.10,
        help='Weight for length shaping reward (default: 0.10)'
    )

    parser.add_argument(
        '--shaping-target-words',
        type=int,
        default=120,
        help='Target word count for shaping terms (default: 120)'
    )

    parser.add_argument(
        '--max-abs-reward',
        type=float,
        default=10.0,
        help='Clip reward magnitude to this absolute value (default: 10.0)'
    )

    parser.add_argument(
        '--implicit-fraction',
        type=float,
        default=0.4,
        help='Fraction of training prompts WITHOUT watermark instructions (default: 0.4). '
             'These prompts still get rewarded for watermark quality, training the model '
             'to internalize watermarking behavior.'
    )

    parser.add_argument(
        '--eval-only',
        type=str,
        default=None,
        metavar='MODEL_PATH',
        help='Skip training; only run dual eval (explicit + implicit) on an existing model. '
             'Pass the path to a saved model directory (e.g. sft_models/.../final_model).'
    )
    parser.add_argument(
        '--eval-output-dir',
        type=str,
        default=None,
        help='Directory for eval-only artifacts. Defaults next to local checkpoints or under eval_results/<model_id>.'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='grpo_models',
        help='Output directory for trained models (default: grpo_models)'
    )

    parser.add_argument(
        '--gen-batch-size',
        type=int,
        default=4,
        help='Batch size for generation during baseline/eval (default: 4)'
    )

    args = parser.parse_args()

    try:
        secret_sequence = set_acrostics_secret_sequence(args.secret_sequence)
    except ValueError as exc:
        parser.error(str(exc))

    if args.gen_batch_size < 1:
        parser.error("--gen-batch-size must be >= 1")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.beta < 0:
        parser.error("--beta must be >= 0")
    if args.num_generations < 1:
        parser.error("--num-generations must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.min_new_tokens is not None and args.min_new_tokens < 1:
        parser.error("--min-new-tokens must be >= 1 when set")
    for arg_name in (
        "natural_max_new_tokens",
        "natural_min_new_tokens",
        "controlled_max_new_tokens",
        "controlled_min_new_tokens",
    ):
        arg_value = getattr(args, arg_name)
        if arg_value is not None and arg_value < 1:
            parser.error(f"--{arg_name.replace('_', '-')} must be >= 1 when set")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if not (0 < args.top_p <= 1):
        parser.error("--top-p must be in (0, 1]")
    if args.lora_rank < 1:
        parser.error("--lora-rank must be >= 1")
    if args.lora_alpha < 1:
        parser.error("--lora-alpha must be >= 1")
    if args.lora_dropout < 0 or args.lora_dropout >= 1:
        parser.error("--lora-dropout must be in [0, 1)")
    if args.shaping_target_words < 1:
        parser.error("--shaping-target-words must be >= 1")
    if args.max_abs_reward <= 0:
        parser.error("--max-abs-reward must be > 0")
    if args.no_reference_model and args.reference_model is not None:
        parser.error("--reference-model cannot be set together with --no-reference-model")
    if not (0.0 <= args.implicit_fraction <= 1.0):
        parser.error("--implicit-fraction must be in [0.0, 1.0]")
    if args.eval_modes is None:
        args.eval_modes = "implicit,explicit" if args.eval_no_instruction else "explicit"
    try:
        resolve_eval_modes(args.eval_modes)
    except ValueError as exc:
        parser.error(str(exc))

    # Eval-only mode: load an existing model and run profile-aware eval
    if args.eval_only is not None:
        print("\n" + "="*80)
        print("EVAL-ONLY MODE")
        print("="*80)
        print(f"Model: {args.eval_only}")
        print(f"Method: {args.method}")
        if args.method == "acrostics":
            print(f"Secret Sequence: {secret_sequence}")
        print("="*80 + "\n")

        config = get_model_config(args.model)
        model_name = config["model_name"]
        tokenizer_source = args.eval_only
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        except Exception:
            if tokenizer_source != model_name:
                warnings.warn(
                    f"Could not load tokenizer from '{tokenizer_source}'. "
                    f"Falling back to base model tokenizer '{model_name}'."
                )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "padding_side", "right") != "left":
            tokenizer.padding_side = "left"

        model = load_causal_lm_with_dtype_fallback(
            model_name=args.eval_only,
            model_kwargs={
                **(
                    {"quantization_config": config["quantization"]}
                    if config.get("quantization")
                    else {}
                ),
                "device_map": config.get("device_map", "auto"),
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            },
            dtype_value=config.get("dtype"),
        )

        prompt_fn = get_prompt_function(args.method)
        dataset = prepare_dataset(
            num_samples=min(50, args.samples),
            split="train",
            dataset_name=args.train_dataset,
            seed=args.seed
        )

        baseline_mean, baseline_std = compute_baseline_statistics(
            model, tokenizer, dataset, args.method,
            num_samples=min(50, args.samples),
            generation_batch_size=args.gen_batch_size,
            reward_override_fn=None,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
        )

        eval_output_dir = resolve_eval_output_dir(args.eval_only, args.eval_output_dir)
        os.makedirs(eval_output_dir, exist_ok=True)

        split_list = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
        dataset_list = [d.strip().lower() for d in args.eval_datasets.split(",") if d.strip()]
        profile_specs = resolve_eval_profiles(
            profile_names=args.eval_profiles,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            natural_max_new_tokens=args.natural_max_new_tokens,
            natural_min_new_tokens=args.natural_min_new_tokens,
            controlled_max_new_tokens=args.controlled_max_new_tokens,
            controlled_min_new_tokens=args.controlled_min_new_tokens,
        )
        eval_mode_specs = resolve_eval_modes(args.eval_modes)
        eval_manifest = {
            "created_at": datetime.now().isoformat(),
            "model_source": args.eval_only,
            "method": args.method,
            "model_strategy": args.model,
            "train_dataset": args.train_dataset,
            "eval_datasets": dataset_list,
            "eval_splits": split_list,
            "eval_profiles": args.eval_profiles,
            "eval_modes": args.eval_modes,
            "eval_samples": args.eval_samples,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
        }
        with open(os.path.join(eval_output_dir, "eval_manifest.json"), "w") as handle:
            json.dump(eval_manifest, handle, indent=2)

        print("\n" + "="*80)
        print("PROFILE-AWARE EVALUATION")
        print("="*80)

        for dataset_name in dataset_list:
            for split_name in split_list:
                eval_dataset = prepare_dataset(
                    num_samples=args.eval_samples,
                    split=split_name,
                    dataset_name=dataset_name,
                    seed=args.seed
                )
                if eval_dataset is None:
                    print(f"⚠️  Skipping {dataset_name}:{split_name}")
                    continue
                for profile in profile_specs:
                    for mode_label, use_instruction in eval_mode_specs:
                        eval_split_label = f"{split_name}_{mode_label}"
                        print(
                            f"\n--- {dataset_name}:{split_name} [{mode_label}] "
                            f"[{profile['name']}] "
                            f"(instruction={'yes' if use_instruction else 'no'}) ---"
                        )
                        evaluate_model_on_split(
                            model=model,
                            tokenizer=tokenizer,
                            dataset=eval_dataset,
                            method=args.method,
                            prompt_fn=prompt_fn,
                            include_instruction=use_instruction,
                            max_samples=args.eval_samples,
                            output_dir=eval_output_dir,
                            split_name=eval_split_label,
                            dataset_name=dataset_name,
                            generation_batch_size=args.gen_batch_size,
                            baseline_mean=baseline_mean,
                            baseline_std=baseline_std,
                            reward_override_fn=None,
                            eval_profile_name=profile["name"],
                            max_new_tokens=profile["max_new_tokens"],
                            min_new_tokens=profile["min_new_tokens"],
                        )

        print(f"\n✓ Eval results saved to: {eval_output_dir}")
        return

    # Train
    model_path = train_grpo(
        model_strategy=args.model,
        method=args.method,
        num_train_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        eval_splits=args.eval_splits,
        eval_samples=args.eval_samples,
        train_dataset_name=args.train_dataset,
        eval_dataset_names=args.eval_datasets,
        eval_profiles=args.eval_profiles,
        eval_modes=args.eval_modes,
        eval_no_instruction=args.eval_no_instruction,
        generation_batch_size=args.gen_batch_size,
        prompt_variant=args.prompt_variant,
        rules_variant=args.rules_variant,
        base_system_prompt=args.base_system_prompt,
        system_prompt_prefix=args.system_prompt_prefix,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        warm_start_model_path=args.warm_start_model,
        reference_model_path=args.reference_model,
        disable_reference_model=args.no_reference_model,
        reward_shaping=args.reward_shaping,
        shaping_format_weight=args.shaping_format_weight,
        shaping_partial_weight=args.shaping_partial_weight,
        shaping_length_weight=args.shaping_length_weight,
        shaping_target_words=args.shaping_target_words,
        max_abs_reward=args.max_abs_reward,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        generation_temperature=args.temperature,
        generation_top_p=args.top_p,
        natural_max_new_tokens=args.natural_max_new_tokens,
        natural_min_new_tokens=args.natural_min_new_tokens,
        controlled_max_new_tokens=args.controlled_max_new_tokens,
        controlled_min_new_tokens=args.controlled_min_new_tokens,
        remove_invalid_values=args.remove_invalid_values,
        require_explicit_reference=args.require_explicit_reference,
        beta=args.beta,
        seed=args.seed,
        implicit_fraction=args.implicit_fraction
    )

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. (Optional) Create an SFT warm-start checkpoint:")
    print(f"   python sft_train.py --model {args.model} --method {args.method} --samples 500 --epochs 1")
    print(f"   # Then use it in GRPO:")
    print(f"   python grpo_train.py --model {args.model} --method {args.method} --warm-start-model <path_to_sft_final_model>")
    print(f"\n2. Test the trained model:")
    print(f"   python cli.py --model-path {model_path} --samples 50")
    print(f"   # Validation/test-style evaluation without watermark instructions")
    print(f"   python cli.py --model-path {model_path} --samples 50 --split test --no-wm-instruction")
    print(f"\n3. Compare with base model:")
    print(f"   python compare_models.py --base {args.model} --trained {model_path} --method {args.method} --split test --no-wm-instruction")
    print(f"\n4. Train on other methods:")
    print(f"   python grpo_train.py --model {args.model} --method <other_method> --epochs {args.epochs}")
    print()


if __name__ == "__main__":
    main()
