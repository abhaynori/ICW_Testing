#!/usr/bin/env python3
"""
FRPO-GRPO: Fine-tuning-Robust Policy Optimization for Watermark Training.

Implements FRPO (Sabbaghi et al. 2026, arXiv:2602.08813) as a GRPOTrainer
subclass that overrides _compute_loss.  This is the correct implementation
matching the paper's official GitHub (https://github.com/Helloworld10011/FRPO).

The FRPO objective replaces the GRPO advantage-weighted gradient with a
log-partition function over groups:

    L_FRPO = lambda * log(mean_g(exp(-A_g/lambda) * min(r, r_clip)))

where A_g are the per-sample advantages within group g.  This finds
"reward-flat" basins in policy space so subsequent fine-tuning cannot
easily collapse the watermark reward.

Key differences from GRPO:
  - advantages = exp(-A / lambda)           (all positive, no sign flip)
  - per_token_loss = min(r, r_clip) * adv  (NO negation)
  - loss = lambda * log(mean(loss_seq per group))  with jackknife bias correction
  - Offset baseline subtracted for variance reduction

Usage:
  python frpo_grpo_train.py \\
      --warm-start-model <path> \\
      --lambda-frpo 0.5 \\
      --samples 1000 --epochs 3 \\
      --output-dir frpo_grpo_logs/run_$(date +%Y%m%d_%H%M%S)

  Set --lambda-frpo to a large value (e.g. 1e6) to approximate standard GRPO.
  Rule of thumb from paper: lambda ~ std(reward) / sqrt(2 * 0.3) ~ std/0.77.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
from main import (
    acrostics_detector, secret_sequence,
    get_base_system_prompt, set_acrostics_secret_sequence,
)
from grpo_train import (
    WatermarkRewardFunction,
    prepare_dataset,
    build_messages,
    get_prompt_function,
    build_grpo_config,
    compute_baseline_statistics,
    generate_responses_batch,
)
from research_utils import (
    load_causal_lm_with_adapter_support,
    patch_saved_model_config,
    sanitize_generated_text,
)
from finetune_robustness import load_eval_queries, score_queries, summarize


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ── FRPO helpers ───────────────────────────────────────────────────────────────

def _nanmin(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def _nanmax(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


# ── FRPOTrainer ────────────────────────────────────────────────────────────────

class FRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass implementing the FRPO loss from Sabbaghi et al. 2026.

    Overrides _compute_loss to replace the GRPO objective with the FRPO
    log-partition objective, which explicitly seeks reward-flat policy regions
    robust to downstream fine-tuning.

    Parameters
    ----------
    lamb : float
        Risk sensitivity (lambda in the paper).  Small = strong robustness
        (focus on worst completions). Large -> approximate standard GRPO.
    do_offset : bool
        Subtract a grouped offset baseline for variance reduction (default True).
    jackknife : bool
        Apply jackknife LOO bias correction to the log-partition estimate (default True).
    delta : float | None
        Optional hard cap on importance weights coef_1 (default None = no cap).
    """

    def __init__(
        self,
        *args,
        lamb: float = 1.0,
        do_offset: bool = True,
        jackknife: bool = True,
        delta: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lamb = lamb
        self.do_offset = do_offset
        self.jackknife = jackknife
        self.frpo_delta = delta  # avoid shadowing self.args.delta

    def _compute_loss(self, model, inputs):
        # ── per-token log probs ────────────────────────────────────────────────
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = (
            completion_mask
            if "tool_mask" not in inputs
            else completion_mask * inputs["tool_mask"]
        )

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        # ── KL divergence ─────────────────────────────────────────────────────
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # ── FRPO advantage transformation ─────────────────────────────────────
        # TRL passes advantages as [B_local] 1D tensor, already group-normalised.
        advantages = inputs["advantages"]  # [B_local]
        if advantages.dim() > 1:
            advantages = advantages.squeeze(-1)  # flatten if TRL pre-unsqueezed

        # exp(-A/lambda): upweights bad completions, all values > 0
        advantages = torch.exp(-advantages / self.lamb)  # [B_local]

        # ── importance sampling weights ────────────────────────────────────────
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio                                    # [B, T]
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (
                (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)                                                       # [B, 1]
        else:
            raise ValueError(
                f"Unknown importance_sampling_level: {self.importance_sampling_level}"
            )

        coef_1 = torch.exp(log_importance_weights)                                # r
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high) # r_clip

        if self.frpo_delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.frpo_delta)

        # ── FRPO per-token loss: min(r, r_clip) * exp(-A/λ)  (NO negation) ────
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        # ── sequence-level aggregation ────────────────────────────────────────
        T = mask.sum(-1).clamp(min=1.0)                              # [B_local]
        loss_seq = (per_token_loss * mask).sum(-1) / T               # [B_local]

        # ── group by prompt hash ──────────────────────────────────────────────
        pi = inputs["prompt_ids"]
        pm = inputs["prompt_mask"]
        hash_local = (
            (pi * pm).sum(-1).float() / pm.sum(-1).clamp(min=1.0).float()
        ).round().to(torch.long)                                      # [B_local]

        # Single-GPU path (no distributed all_gather needed)
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world = dist.get_world_size()
            rank = dist.get_rank()
            B_loc = loss_seq.size(0)

            loss_list = [torch.zeros_like(loss_seq) for _ in range(world)]
            hash_list = [torch.zeros_like(hash_local) for _ in range(world)]
            dist.all_gather(loss_list, loss_seq.detach())
            dist.all_gather(hash_list, hash_local)

            loss_all_ng = torch.cat(loss_list, dim=0)
            hash_all = torch.cat(hash_list, dim=0)

            base = loss_all_ng.detach()
            idx = torch.arange(rank * B_loc, (rank + 1) * B_loc, device=base.device)
            delta = torch.zeros_like(base).scatter(
                0, idx, (loss_seq - base[idx]).to(base.dtype)
            )
            loss_all = base + delta
        else:
            loss_all = loss_seq
            hash_all = hash_local
            B_loc = loss_seq.size(0)

        # ── group averages ────────────────────────────────────────────────────
        uniq, inv = torch.unique(hash_all, return_inverse=True)
        sums = torch.zeros(
            uniq.numel(), device=loss_all.device, dtype=loss_all.dtype
        ).scatter_add_(0, inv, loss_all)
        cnts = torch.bincount(inv, minlength=uniq.numel()).clamp_min(1)
        loss_group = sums / cnts.to(loss_all.dtype)                  # [#groups]
        G = loss_group.numel()

        # ── log-partition with optional jackknife bias correction ─────────────
        if not self.jackknife:
            loss_reward = torch.log(loss_group.clamp_min(1e-12)).mean()
        else:
            jackknife_terms = []
            for g in range(G):
                group_indices = (inv == g).nonzero(as_tuple=True)[0]
                n_g = group_indices.size(0)
                group_losses = loss_all[group_indices]
                original_avg = loss_group[g]

                if n_g == 1:
                    jackknife_terms.append(torch.log(original_avg.clamp_min(1e-12)))
                else:
                    group_sum = group_losses.sum()
                    loo_avgs = (group_sum.unsqueeze(0) - group_losses) / (n_g - 1)
                    jk_term = (
                        n_g * torch.log(original_avg.clamp_min(1e-12))
                        - (n_g - 1) * torch.log(loo_avgs.clamp_min(1e-12)).mean()
                    )
                    jackknife_terms.append(jk_term)
            loss_reward = torch.stack(jackknife_terms).mean()

        # ── grouped offset baseline ───────────────────────────────────────────
        if self.do_offset:
            aux_tok = torch.min(coef_1, coef_2)                       # [B, T] or [B, 1]
            aux_seq = (aux_tok * mask).sum(-1) / T                    # [B_local]

            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                world = dist.get_world_size()
                rank = dist.get_rank()
                Bl = aux_seq.size(0)
                aux_list = [torch.zeros_like(aux_seq) for _ in range(world)]
                dist.all_gather(aux_list, aux_seq.detach())
                aux_all_ng = torch.cat(aux_list, dim=0)
                base = aux_all_ng.detach()
                idx = torch.arange(rank * Bl, (rank + 1) * Bl, device=base.device)
                delta = torch.zeros_like(base).scatter(
                    0, idx, (aux_seq - base[idx]).to(base.dtype)
                )
                aux_all = base + delta
            else:
                aux_all = aux_seq

            aux_sums = torch.zeros_like(sums).scatter_add_(0, inv, aux_all)
            aux_group = aux_sums / cnts.to(aux_all.dtype)
            loss_reward = loss_reward - aux_group.mean()

        # ── scale to match GRPO's effective per-rank denominator ──────────────
        scale_to_grpo = cnts.sum().to(loss_reward.dtype) / float(B_loc)
        loss = (self.lamb if self.lamb > 1 else 1) * scale_to_grpo * loss_reward

        # ── KL stays as in GRPO (per-rank mean) ──────────────────────────────
        if self.beta != 0.0:
            kl_seq = (per_token_kl * mask).sum(-1) / T
            loss = loss + self.beta * kl_seq.mean()

        # ── metrics ───────────────────────────────────────────────────────────
        mode = "train" if self.model.training else "eval"
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.dim() > 1 and x.shape[1] == 1:
                return x.mean()
            return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )
        self._metrics[mode].setdefault("frpo/lambda", []).append(self.lamb)

        # Clip ratio metrics (recompute with original-scale coef for logging)
        # Note: advantages here are exp(-A/lambda), all positive; clipping
        # condition is still on the ratio coef_1 relative to 1 +/- epsilon.
        is_low_clipped = coef_1 < (1 - self.epsilon_low)
        is_high_clipped = coef_1 > (1 + self.epsilon_high)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(_nanmin(gathered_low).item())
        gathered_high = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(_nanmax(gathered_high).item())
        gathered_clip = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip.nanmean().item())

        return loss


# ── helpers ────────────────────────────────────────────────────────────────────

def load_and_merge(model_path: str, tokenizer, dtype):
    """Load model; if it's a PEFT adapter merge into base weights."""
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg) as f:
            base_path = json.load(f)["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(
            base_path, device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print(f"  ✓ Merged PEFT adapter from {base_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, torch_dtype=dtype,
        )
        print(f"  ✓ Loaded full model from {model_path}")
    return model


def run_eval(model, tokenizer, queries_by_ds, gen_kwargs, label):
    print(f"\n{'─'*60}")
    print(f"  EVAL: {label}")
    print(f"{'─'*60}")
    for ds, queries in queries_by_ds.items():
        scores = score_queries(model, tokenizer, queries, **gen_kwargs)
        s = summarize(scores)
        print(
            f"  [{ds}] mean={s['mean']:.4f}  std={s['std']:.4f}  "
            f"z={s['z']:.3f}  p={s['p']:.4e}"
        )


def build_frpo_trainer(
    model,
    training_args,
    train_dataset,
    tokenizer,
    reward_fn,
    lamb: float,
    do_offset: bool,
    jackknife: bool,
    frpo_delta: float | None = None,
):
    """Version-compatible FRPOTrainer constructor."""
    sig = inspect.signature(GRPOTrainer.__init__)
    accepted = {name for name in sig.parameters if name != "self"}

    kwargs: dict = {}

    if "model" in accepted:
        kwargs["model"] = model
    elif "policy" in accepted:
        kwargs["policy"] = model
    else:
        raise TypeError("Unsupported GRPOTrainer: no model/policy arg")

    if "args" in accepted:
        kwargs["args"] = training_args
    elif "config" in accepted:
        kwargs["config"] = training_args

    if "train_dataset" in accepted:
        kwargs["train_dataset"] = train_dataset
    elif "dataset" in accepted:
        kwargs["dataset"] = train_dataset

    if "tokenizer" in accepted:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in accepted:
        kwargs["processing_class"] = tokenizer
    elif "processor" in accepted:
        kwargs["processor"] = tokenizer

    reward_attempts = []
    if "reward_function" in accepted:
        reward_attempts.append(("reward_function", reward_fn))
    if "reward_funcs" in accepted:
        reward_attempts.append(("reward_funcs", reward_fn))
        reward_attempts.append(("reward_funcs", [reward_fn]))
    if "reward_fn" in accepted:
        reward_attempts.append(("reward_fn", reward_fn))

    if not reward_attempts:
        raise TypeError("Unsupported GRPOTrainer: no reward_function/reward_funcs/reward_fn arg")

    last_err = None
    for reward_key, reward_value in reward_attempts:
        try:
            trainer = FRPOTrainer(
                **kwargs,
                **{reward_key: reward_value},
                lamb=lamb,
                do_offset=do_offset,
                jackknife=jackknife,
                delta=frpo_delta,
            )
            return trainer
        except TypeError as exc:
            last_err = exc

    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to initialize FRPOTrainer")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FRPO-GRPO: fine-tuning-robust watermark training (correct implementation)"
    )

    # Model
    parser.add_argument("--warm-start-model", required=True,
                        help="Starting GRPO/SFT model checkpoint")
    parser.add_argument("--base-model", default=BASE_MODEL)

    # GRPO training
    parser.add_argument("--samples",          type=int,   default=1000)
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--batch-size",       type=int,   default=4)
    parser.add_argument("--learning-rate",    type=float, default=1e-5)
    parser.add_argument("--num-generations",  type=int,   default=4)
    parser.add_argument("--max-new-tokens",   type=int,   default=200)
    parser.add_argument("--temperature",      type=float, default=0.7)
    parser.add_argument("--top-p",            type=float, default=0.9)
    parser.add_argument("--beta",             type=float, default=0.04)
    parser.add_argument("--train-dataset",    default="mixed",
                        choices=["eli5", "alpaca", "mixed", "gsm8k"])
    parser.add_argument("--implicit-fraction", type=float, default=0.4)

    # LoRA
    parser.add_argument("--lora-rank",  type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # FRPO
    parser.add_argument("--lambda-frpo", type=float, default=0.5,
                        help=(
                            "FRPO lambda. Small = strong robustness (focus on worst "
                            "completions). Large (e.g. 1e6) = approximate GRPO. "
                            "Paper rule of thumb: std(reward)/sqrt(2*0.3). Default: 0.5"
                        ))
    parser.add_argument("--no-offset",   dest="do_offset",  action="store_false",
                        help="Disable grouped offset baseline")
    parser.add_argument("--no-jackknife", dest="jackknife", action="store_false",
                        help="Disable jackknife bias correction")
    parser.add_argument("--frpo-delta",  type=float, default=None,
                        help="Hard cap on importance weights (default: no cap)")
    parser.set_defaults(do_offset=True, jackknife=True)

    # Eval
    parser.add_argument("--eval-samples",    type=int, default=100)
    parser.add_argument("--gen-batch",       type=int, default=4)
    parser.add_argument("--eval-max-tokens", type=int, default=512)
    parser.add_argument("--eval-min-tokens", type=int, default=256)

    # Output
    parser.add_argument("--output-dir",
                        default=f"frpo_grpo_logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("\n" + "=" * 70)
    print("FRPO-GRPO: Fine-tuning-Robust Watermark Training (paper implementation)")
    print("=" * 70)
    print(f"Warm start:        {args.warm_start_model}")
    print(f"GRPO dataset:      {args.train_dataset} ({args.samples} samples x {args.epochs} epochs)")
    print(f"GRPO LoRA rank:    {args.lora_rank}")
    print(f"FRPO lambda:       {args.lambda_frpo}  (small = stronger robustness)")
    print(f"Jackknife:         {args.jackknife}")
    print(f"Offset baseline:   {args.do_offset}")
    print(f"Num generations:   {args.num_generations}")
    print(f"Output:            {args.output_dir}")
    print("=" * 70 + "\n")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.warm_start_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("✓ Tokenizer loaded\n")

    # ── model ─────────────────────────────────────────────────────────────────
    print("Loading model...")
    model = load_and_merge(args.warm_start_model, tokenizer, dtype)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print("✓ GRPO LoRA attached\n")

    # ── eval queries ──────────────────────────────────────────────────────────
    print(f"Loading eval queries ({args.eval_samples} per dataset)...")
    queries_by_ds = {
        ds: load_eval_queries(ds, args.eval_samples)
        for ds in ["eli5", "alpaca", "gsm8k"]
    }
    eval_gen_kwargs = dict(
        gen_batch=args.gen_batch,
        max_new_tokens=args.eval_max_tokens,
        min_new_tokens=args.eval_min_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    print()

    # ── baseline eval ─────────────────────────────────────────────────────────
    run_eval(model, tokenizer, queries_by_ds, eval_gen_kwargs, "Before FRPO-GRPO")

    # ── GRPO training dataset ─────────────────────────────────────────────────
    print(f"\nLoading GRPO training dataset ({args.train_dataset}, {args.samples} samples)...")
    train_dataset_raw = prepare_dataset(
        num_samples=args.samples,
        split="train",
        dataset_name=args.train_dataset,
        seed=args.seed,
    )

    prompt_fn = get_prompt_function("acrostics")
    _implicit_rng = np.random.default_rng(args.seed + 1000)

    def tokenize_function(examples):
        prompts = []
        for query in examples["query"]:
            if _implicit_rng.random() < args.implicit_fraction:
                messages = build_messages(query, include_instruction=False)
            else:
                messages = prompt_fn(query)
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt_text)
        return {"prompt": prompts}

    tokenized_dataset = train_dataset_raw.map(tokenize_function, batched=True)
    print(f"✓ {len(tokenized_dataset)} prompts tokenized\n")

    # ── baseline reward statistics ────────────────────────────────────────────
    print("Computing baseline reward statistics...")
    baseline_reward_fn = WatermarkRewardFunction("acrostics")
    baseline_mean, baseline_std = compute_baseline_statistics(
        model, tokenizer, train_dataset_raw, "acrostics",
        num_samples=min(50, args.samples),
        generation_batch_size=args.gen_batch,
        reward_override_fn=baseline_reward_fn._acrostics_training_score,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"  Baseline mean={baseline_mean:.3f}  std={baseline_std:.3f}")

    suggested_lambda = baseline_std / (2 * 0.3) ** 0.5
    print(f"  Suggested lambda (std/sqrt(2*0.3)): {suggested_lambda:.3f}")
    print(f"  Using lambda: {args.lambda_frpo}\n")

    # ── reward function (standard — FRPO logic lives in loss, not reward) ─────
    reward_fn = WatermarkRewardFunction(
        "acrostics", baseline_mean, baseline_std,
        reward_shaping=False,
    )

    # ── GRPO trainer config ───────────────────────────────────────────────────
    base_training_args = {
        "output_dir": os.path.join(args.output_dir, "grpo_checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "max_grad_norm": 1.0,
        "seed": args.seed,
        "beta": args.beta,
        "bf16": use_bf16,
        "fp16": bool(use_cuda and not use_bf16),
        # Use grpo loss_type so the base GRPOConfig is happy;
        # FRPOTrainer._compute_loss overrides the actual loss computation.
        "loss_type": "grpo",
    }
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "remove_invalid_values": True,
    }
    training_args = build_grpo_config(
        base_training_args, generation_args, args.num_generations
    )

    print("Initializing FRPOTrainer (loss-override implementation)...")
    trainer = build_frpo_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        lamb=args.lambda_frpo,
        do_offset=args.do_offset,
        jackknife=args.jackknife,
        frpo_delta=args.frpo_delta,
    )
    print(f"  lamb={trainer.lamb}  jackknife={trainer.jackknife}  "
          f"do_offset={trainer.do_offset}  delta={trainer.frpo_delta}")
    print("✓ FRPOTrainer initialized\n")

    # ── train ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Starting FRPO-GRPO training...")
    print("=" * 70 + "\n")
    trainer.train()
    print("\n✓ Training complete\n")

    # ── save (merge LoRA into weights) ────────────────────────────────────────
    final_model_path = os.path.join(args.output_dir, "final_model")
    print(f"Saving final model (merged) -> {final_model_path}")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    patch_saved_model_config(final_model_path, args.base_model)
    print("✓ Saved\n")

    # ── post-training eval ────────────────────────────────────────────────────
    run_eval(merged, tokenizer, queries_by_ds, eval_gen_kwargs, "After FRPO-GRPO")

    # ── save run config ───────────────────────────────────────────────────────
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run config -> {config_path}")

    print("\n" + "=" * 70)
    print("Next step - test fine-tuning robustness:")
    print(f"  python finetune_robustness.py \\")
    print(f"      --grpo-model {final_model_path} \\")
    print(f"      --base-model {args.base_model} \\")
    print(f"      --finetune-dataset alpaca \\")
    print(f"      --finetune-samples 200 \\")
    print(f"      --max-steps 200 \\")
    print(f"      --lora-rank 4 \\")
    print(f"      --skip-base \\")
    print(f"      --output-dir {args.output_dir}/robustness_test")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
