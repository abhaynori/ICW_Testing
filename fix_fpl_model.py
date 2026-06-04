#!/usr/bin/env python3
"""
Rescue script: the FPL model was saved as a PEFT adapter pointing to
Qwen/Qwen2.5-7B-Instruct, but the true effective base was the merged
warm-start (multidomain GRPO) model.  This script reconstructs the correct
merged weights and saves them as a plain full model.

Usage:
  python fix_fpl_model.py \
      --warm-start rerun_acrostics_multidomain_42_20260525_112640/grpo_models/acrostics_full_20260525_230104/final_model \
      --fpl-model  fpl_grpo_logs/run_no_recovery_20260602_235150/final_model \
      --out        fpl_grpo_logs/run_no_recovery_20260602_235150/final_model_merged
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_and_merge(base_path: str, adapter_path: str, dtype) -> AutoModelForCausalLM:
    adapter_cfg = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg) as f:
            recorded_base = json.load(f)["base_model_name_or_path"]
        print(f"  Loading base: {base_path}  (adapter_config says: {recorded_base})")
        base = AutoModelForCausalLM.from_pretrained(
            base_path, device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        return model.merge_and_unload()
    else:
        print(f"  Loading full model: {adapter_path}")
        return AutoModelForCausalLM.from_pretrained(
            adapter_path, device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, torch_dtype=dtype,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-start", required=True,
                        help="Path to warm-start PEFT adapter (the multidomain GRPO model)")
    parser.add_argument("--fpl-model", required=True,
                        help="Path to FPL final_model (saved with wrong base)")
    parser.add_argument("--out", required=True,
                        help="Output path for the corrected merged model")
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    dtype = torch.bfloat16
    os.makedirs(args.out, exist_ok=True)

    print("Step 1: Load warm-start and merge into base weights...")
    warm_merged = load_and_merge(args.base_model, args.warm_start, dtype)
    print("  ✓ Warm-start merged")

    print("Step 2: Load FPL LoRA delta on top and merge...")
    fpl_adapter_cfg = os.path.join(args.fpl_model, "adapter_config.json")
    if os.path.exists(fpl_adapter_cfg):
        model = PeftModel.from_pretrained(warm_merged, args.fpl_model)
        model = model.merge_and_unload()
        print("  ✓ FPL LoRA merged")
    else:
        print("  FPL model has no adapter_config — treating as full model, skipping merge")
        model = warm_merged

    print(f"Step 3: Saving merged model to {args.out} ...")
    model.save_pretrained(args.out)

    print("Step 4: Saving tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.save_pretrained(args.out)

    print(f"\n✓ Done.  Corrected model saved to: {args.out}")
    print("\nRun the robustness test with:")
    print(f"  python finetune_robustness.py \\")
    print(f"      --grpo-model {args.out} \\")
    print(f"      --base-model {args.base_model} \\")
    print(f"      --finetune-dataset alpaca \\")
    print(f"      --finetune-samples 200 \\")
    print(f"      --max-steps 200 \\")
    print(f"      --lora-rank 4 \\")
    print(f"      --skip-base \\")
    print(f"      --output-dir fpl_grpo_logs/robustness_test_fixed")


if __name__ == "__main__":
    main()
