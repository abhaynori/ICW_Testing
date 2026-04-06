#!/usr/bin/env python3
"""
Canonical experiment runner for NeurIPS-oriented ICW studies.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from memory_config import get_model_config
from research_utils import build_lm_eval_model_args


SUPPORTED_ALGORITHMS = {
    "base",
    "regular_sft",
    "sft",
    "dpo",
    "grpo",
    "sft_grpo",
}

MODEL_PATTERNS = {
    "regular_sft": "sft_{method}_{model}_*/final_model",
    "sft": "sft_{method}_{model}_*/final_model",
    "dpo": "{method}_{model}_*/final_model",
    "grpo": "{method}_{model}_*/final_model",
    "sft_grpo": "{method}_{model}_*/final_model",
}


@dataclass
class ExperimentSpec:
    run_id: str
    algorithm: str
    method: str
    seed: int


def parse_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str | None) -> list[int]:
    return [int(item) for item in parse_list(raw)]


def command_to_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        process = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process.returncode


def resolve_model_path(parent_dir: Path, algorithm: str, method: str, model: str) -> str | None:
    pattern = MODEL_PATTERNS.get(algorithm)
    if pattern is None:
        return None
    matches = sorted(glob.glob(str(parent_dir / pattern.format(method=method, model=model))))
    if not matches:
        return None
    return matches[-1]


def resolve_lm_eval_launcher() -> list[str] | None:
    binary = shutil.which("lm_eval")
    if binary:
        return [binary]
    if importlib.util.find_spec("lm_eval") is not None:
        return [sys.executable, "-m", "lm_eval"]
    return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def write_manifest_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as handle:
            handle.write("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    algorithms = parse_list(args.algorithms)
    methods = parse_list(args.methods)
    seeds = parse_int_list(args.seeds)

    invalid_algorithms = [algorithm for algorithm in algorithms if algorithm not in SUPPORTED_ALGORITHMS]
    if invalid_algorithms:
        raise ValueError(f"Unsupported algorithms: {', '.join(invalid_algorithms)}")

    specs = []
    for algorithm in algorithms:
        for method in methods:
            for seed in seeds:
                run_id = f"{method}_{algorithm}_seed{seed}"
                specs.append(
                    ExperimentSpec(
                        run_id=run_id,
                        algorithm=algorithm,
                        method=method,
                        seed=seed,
                    )
                )
    if args.max_runs and len(specs) > args.max_runs:
        specs = specs[: args.max_runs]
    return specs


def maybe_extend_prompt_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(["--prompt-variant", args.prompt_variant])
    cmd.extend(["--rules-variant", args.rules_variant])
    if args.base_system_prompt:
        cmd.extend(["--base-system-prompt", args.base_system_prompt])
    if args.system_prompt_prefix:
        cmd.extend(["--system-prompt-prefix", args.system_prompt_prefix])


def maybe_extend_lora_args(cmd: list[str], args: argparse.Namespace) -> None:
    if args.use_lora:
        cmd.extend(
            [
                "--use-lora",
                "--lora-rank",
                str(args.lora_rank),
                "--lora-alpha",
                str(args.lora_alpha),
                "--lora-dropout",
                str(args.lora_dropout),
                "--lora-target-modules",
                args.lora_target_modules,
            ]
        )
    else:
        cmd.append("--no-lora")


def build_sft_command(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    output_parent: Path,
    include_instruction: bool,
) -> list[str]:
    cmd = [
        "python",
        "sft_train.py",
        "--model",
        args.model,
        "--method",
        spec.method,
        "--samples",
        str(args.sft_samples),
        "--epochs",
        str(args.sft_epochs),
        "--batch-size",
        str(args.sft_batch_size),
        "--learning-rate",
        str(args.sft_learning_rate),
        "--train-dataset",
        args.train_dataset,
        "--output-dir",
        str(output_parent),
        "--seed",
        str(spec.seed),
    ]
    cmd.append("--wm-instruction" if include_instruction else "--no-wm-instruction")
    maybe_extend_prompt_args(cmd, args)
    maybe_extend_lora_args(cmd, args)
    return cmd


def build_dpo_command(spec: ExperimentSpec, args: argparse.Namespace, output_parent: Path) -> list[str]:
    cmd = [
        "python",
        "dpo_train.py",
        "--model",
        args.model,
        "--method",
        spec.method,
        "--samples",
        str(args.dpo_samples),
        "--epochs",
        str(args.dpo_epochs),
        "--batch-size",
        str(args.dpo_batch_size),
        "--learning-rate",
        str(args.dpo_learning_rate),
        "--train-dataset",
        args.train_dataset,
        "--n-candidates",
        str(args.dpo_n_candidates),
        "--implicit-fraction",
        str(args.dpo_implicit_fraction),
        "--beta",
        str(args.dpo_beta),
        "--output-dir",
        str(output_parent),
        "--eval-splits",
        "",
        "--eval-datasets",
        args.eval_datasets,
        "--eval-samples",
        str(args.eval_samples),
        "--gen-batch-size",
        str(args.gen_batch_size),
        "--seed",
        str(spec.seed),
    ]
    maybe_extend_prompt_args(cmd, args)
    maybe_extend_lora_args(cmd, args)
    return cmd


def build_grpo_train_command(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    output_parent: Path,
    warm_start_model: str | None = None,
) -> list[str]:
    cmd = [
        "python",
        "grpo_train.py",
        "--model",
        args.model,
        "--method",
        spec.method,
        "--samples",
        str(args.grpo_samples),
        "--epochs",
        str(args.grpo_epochs),
        "--batch-size",
        str(args.grpo_batch_size),
        "--learning-rate",
        str(args.grpo_learning_rate),
        "--train-dataset",
        args.train_dataset,
        "--output-dir",
        str(output_parent),
        "--eval-splits",
        "",
        "--eval-datasets",
        args.eval_datasets,
        "--eval-samples",
        str(args.eval_samples),
        "--gen-batch-size",
        str(args.gen_batch_size),
        "--seed",
        str(spec.seed),
        "--implicit-fraction",
        str(args.grpo_implicit_fraction),
        "--num-generations",
        str(args.grpo_num_generations),
        "--beta",
        str(args.grpo_beta),
        "--max-new-tokens",
        str(args.eval_max_new_tokens),
        "--eval-profiles",
        args.eval_profiles,
        "--eval-modes",
        args.eval_modes,
    ]
    if args.eval_controlled_min_new_tokens is not None:
        cmd.extend(
            ["--controlled-min-new-tokens", str(args.eval_controlled_min_new_tokens)]
        )
    if args.eval_natural_max_new_tokens is not None:
        cmd.extend(["--natural-max-new-tokens", str(args.eval_natural_max_new_tokens)])
    if args.eval_controlled_max_new_tokens is not None:
        cmd.extend(["--controlled-max-new-tokens", str(args.eval_controlled_max_new_tokens)])
    if warm_start_model:
        cmd.extend(["--warm-start-model", warm_start_model])
    maybe_extend_prompt_args(cmd, args)
    maybe_extend_lora_args(cmd, args)
    return cmd


def build_eval_command(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    model_source: str,
    run_dir: Path,
) -> list[str]:
    cmd = [
        "python",
        "grpo_train.py",
        "--model",
        args.model,
        "--method",
        spec.method,
        "--eval-only",
        model_source,
        "--eval-output-dir",
        str(run_dir / "eval"),
        "--train-dataset",
        args.train_dataset,
        "--eval-datasets",
        args.eval_datasets,
        "--eval-splits",
        args.eval_splits,
        "--eval-samples",
        str(args.eval_samples),
        "--eval-profiles",
        args.eval_profiles,
        "--eval-modes",
        args.eval_modes,
        "--samples",
        str(args.eval_baseline_samples),
        "--gen-batch-size",
        str(args.gen_batch_size),
        "--seed",
        str(spec.seed),
        "--max-new-tokens",
        str(args.eval_max_new_tokens),
    ]
    if args.eval_controlled_min_new_tokens is not None:
        cmd.extend(
            ["--controlled-min-new-tokens", str(args.eval_controlled_min_new_tokens)]
        )
    if args.eval_natural_max_new_tokens is not None:
        cmd.extend(["--natural-max-new-tokens", str(args.eval_natural_max_new_tokens)])
    if args.eval_controlled_max_new_tokens is not None:
        cmd.extend(["--controlled-max-new-tokens", str(args.eval_controlled_max_new_tokens)])
    return cmd


def build_robustness_commands(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    run_dir: Path,
) -> list[tuple[str, list[str]]]:
    commands = []
    datasets = parse_list(args.eval_datasets)
    splits = parse_list(args.robustness_splits)
    modes = parse_list(args.robustness_modes)
    profiles = parse_list(args.robustness_profiles)
    for dataset in datasets:
        for split in splits:
            for mode in modes:
                for profile in profiles:
                    eval_path = run_dir / "eval" / f"eval_{dataset}_{split}_{mode}_{profile}.json"
                    output_path = run_dir / "robustness" / f"eval_{dataset}_{split}_{mode}_{profile}_robustness.json"
                    label = f"{dataset}_{split}_{mode}_{profile}"
                    cmd = [
                        "python",
                        "robustness_eval.py",
                        "--eval-json",
                        str(eval_path),
                        "--method",
                        spec.method,
                        "--attacks",
                        args.robustness_attacks,
                        "--min-similarity-proxy",
                        str(args.robustness_min_similarity),
                        "--output",
                        str(output_path),
                    ]
                    commands.append((label, cmd))
    return commands


def build_utility_commands(
    args: argparse.Namespace,
    run_dir: Path,
    model_source: str,
    lm_eval_launcher: list[str],
) -> dict[str, list[str]]:
    commands = {}
    model_args = build_lm_eval_model_args(model_source, trust_remote_code=True)
    for task in parse_list(args.utility_tasks):
        commands[task] = [
            *lm_eval_launcher,
            "--model",
            "hf",
            "--apply_chat_template",
            "--model_args",
            model_args,
            "--tasks",
            task,
            "--batch_size",
            "auto",
            "--output_path",
            str(run_dir / "utility" / f"{task}.json"),
        ]
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run canonical NeurIPS-oriented ICW experiments."
    )
    parser.add_argument("--execute", action="store_true", help="Execute commands")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--max-runs", type=int, default=0, help="Cap total runs")
    parser.add_argument("--output-root", default="neurips_runs")

    parser.add_argument("--model", default="8bit")
    parser.add_argument("--methods", default="acrostics,lexical")
    parser.add_argument(
        "--algorithms",
        default="base,regular_sft,sft,dpo,grpo,sft_grpo",
        help="Comma-separated algorithms",
    )
    parser.add_argument("--seeds", default="41,42,43", help="Comma-separated seeds")

    parser.add_argument("--train-dataset", choices=["eli5", "alpaca"], default="eli5")
    parser.add_argument("--prompt-variant", choices=["paper", "concise", "strict"], default="paper")
    parser.add_argument("--rules-variant", choices=["paper", "minimal", "none"], default="paper")
    parser.add_argument("--base-system-prompt", default=None)
    parser.add_argument("--system-prompt-prefix", default=None)

    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    )

    parser.add_argument("--sft-samples", type=int, default=500)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--sft-batch-size", type=int, default=2)
    parser.add_argument("--sft-learning-rate", type=float, default=2e-5)

    parser.add_argument("--dpo-samples", type=int, default=200)
    parser.add_argument("--dpo-epochs", type=int, default=3)
    parser.add_argument("--dpo-batch-size", type=int, default=2)
    parser.add_argument("--dpo-learning-rate", type=float, default=5e-6)
    parser.add_argument("--dpo-n-candidates", type=int, default=8)
    parser.add_argument("--dpo-implicit-fraction", type=float, default=0.4)
    parser.add_argument("--dpo-beta", type=float, default=0.1)

    parser.add_argument("--grpo-samples", type=int, default=200)
    parser.add_argument("--grpo-epochs", type=int, default=3)
    parser.add_argument("--grpo-batch-size", type=int, default=4)
    parser.add_argument("--grpo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--grpo-num-generations", type=int, default=4)
    parser.add_argument("--grpo-implicit-fraction", type=float, default=0.4)
    parser.add_argument("--grpo-beta", type=float, default=0.04)

    parser.add_argument("--eval-datasets", default="eli5,alpaca")
    parser.add_argument("--eval-splits", default="validation,test")
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--eval-baseline-samples", type=int, default=50)
    parser.add_argument("--eval-profiles", default="natural,controlled")
    parser.add_argument("--eval-modes", default="implicit,explicit")
    parser.add_argument("--eval-max-new-tokens", type=int, default=200)
    parser.add_argument("--eval-natural-max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-controlled-max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-controlled-min-new-tokens", type=int, default=128)
    parser.add_argument("--gen-batch-size", type=int, default=4)

    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--robustness-splits", default="test")
    parser.add_argument("--robustness-modes", default="implicit")
    parser.add_argument("--robustness-profiles", default="natural")
    parser.add_argument(
        "--robustness-attacks",
        default="format_cleanup,truncate_sentence_50,truncate_word_50,sentence_merge,sentence_split,word_dropout,compression",
    )
    parser.add_argument("--robustness-min-similarity", type=float, default=0.0)

    parser.add_argument("--skip-utility", action="store_true")
    parser.add_argument("--allow-missing-utility", action="store_true")
    parser.add_argument("--utility-tasks", default="ifeval,gsm8k,mmlu,simpleqa")

    args = parser.parse_args()

    if args.model in {"small", "cpu"}:
        raise SystemExit(
            "neurips_experiment_runner.py is intended for 7B experiments only. "
            "Use --model 4bit, --model 8bit, or --model full."
        )

    if args.dry_run:
        args.execute = False
    elif not args.execute:
        args.dry_run = True

    specs = build_specs(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / f"neurips_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    lm_eval_launcher = resolve_lm_eval_launcher()
    utility_available = lm_eval_launcher is not None
    if (
        args.execute
        and not args.skip_utility
        and not utility_available
        and not args.allow_missing_utility
    ):
        raise SystemExit(
            "Utility evaluation requested but lm_eval is not available. "
            "Install with `pip install lm-eval`, or rerun with --skip-utility "
            "(or --allow-missing-utility to keep planning without utility)."
        )

    print(f"Planned runs: {len(specs)}")
    print(f"Run root: {run_root}")

    summary_rows = []
    top_level_manifest = {
        "created_at": datetime.now().isoformat(),
        "args": vars(args),
        "planned_runs": len(specs),
        "runs": [],
    }

    for index, spec in enumerate(specs, start=1):
        run_dir = run_root / f"{index:03d}_{spec.run_id}"
        train_artifacts_dir = run_dir / "train_artifacts"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{index}/{len(specs)}] {spec.run_id}")
        run_manifest = {
            "created_at": datetime.now().isoformat(),
            "run_id": spec.run_id,
            "algorithm": spec.algorithm,
            "method": spec.method,
            "seed": spec.seed,
            "run_dir": str(run_dir),
            "status": "planned",
            "train_steps": [],
            "model_source": None,
            "eval": None,
            "robustness": [],
            "utility": {},
        }

        model_source = None

        if spec.algorithm == "base":
            model_source = get_model_config(args.model)["model_name"]
            print(f"  base model: {model_source}")
        elif spec.algorithm in {"regular_sft", "sft"}:
            sft_cmd = build_sft_command(
                spec,
                args,
                train_artifacts_dir,
                include_instruction=(spec.algorithm == "sft"),
            )
            print(f"  train: {command_to_str(sft_cmd)}")
            run_manifest["train_steps"].append(
                {"name": "sft", "command": sft_cmd, "status": "planned"}
            )
            if args.execute:
                rc = run_command(sft_cmd, Path.cwd(), run_dir / "logs" / "train_sft.log")
                run_manifest["train_steps"][-1]["status"] = "ok" if rc == 0 else "failed"
                if rc == 0:
                    model_source = resolve_model_path(
                        train_artifacts_dir, spec.algorithm, spec.method, args.model
                    )
                if rc != 0 or not model_source:
                    run_manifest["status"] = "train_failed"
                    write_json(run_dir / "run_manifest.json", run_manifest)
                    top_level_manifest["runs"].append(run_manifest)
                    summary_rows.append(
                        {
                            "run_id": spec.run_id,
                            "algorithm": spec.algorithm,
                            "method": spec.method,
                            "seed": spec.seed,
                            "status": run_manifest["status"],
                            "run_dir": str(run_dir),
                            "model_source": "",
                        }
                    )
                    print("  status: train failed")
                    continue
        elif spec.algorithm == "dpo":
            dpo_cmd = build_dpo_command(spec, args, train_artifacts_dir)
            print(f"  train: {command_to_str(dpo_cmd)}")
            run_manifest["train_steps"].append(
                {"name": "dpo", "command": dpo_cmd, "status": "planned"}
            )
            if args.execute:
                rc = run_command(dpo_cmd, Path.cwd(), run_dir / "logs" / "train_dpo.log")
                run_manifest["train_steps"][-1]["status"] = "ok" if rc == 0 else "failed"
                if rc == 0:
                    model_source = resolve_model_path(train_artifacts_dir, "dpo", spec.method, args.model)
                if rc != 0 or not model_source:
                    run_manifest["status"] = "train_failed"
                    write_json(run_dir / "run_manifest.json", run_manifest)
                    top_level_manifest["runs"].append(run_manifest)
                    summary_rows.append(
                        {
                            "run_id": spec.run_id,
                            "algorithm": spec.algorithm,
                            "method": spec.method,
                            "seed": spec.seed,
                            "status": run_manifest["status"],
                            "run_dir": str(run_dir),
                            "model_source": "",
                        }
                    )
                    print("  status: train failed")
                    continue
        elif spec.algorithm in {"grpo", "sft_grpo"}:
            warm_start_model = None
            if spec.algorithm == "sft_grpo":
                sft_cmd = build_sft_command(spec, args, train_artifacts_dir, include_instruction=True)
                print(f"  sft: {command_to_str(sft_cmd)}")
                run_manifest["train_steps"].append(
                    {"name": "sft", "command": sft_cmd, "status": "planned"}
                )
                if args.execute:
                    rc = run_command(sft_cmd, Path.cwd(), run_dir / "logs" / "train_sft.log")
                    run_manifest["train_steps"][-1]["status"] = "ok" if rc == 0 else "failed"
                    if rc == 0:
                        warm_start_model = resolve_model_path(train_artifacts_dir, "sft", spec.method, args.model)
                    if rc != 0 or not warm_start_model:
                        run_manifest["status"] = "train_failed"
                        write_json(run_dir / "run_manifest.json", run_manifest)
                        top_level_manifest["runs"].append(run_manifest)
                        summary_rows.append(
                            {
                                "run_id": spec.run_id,
                                "algorithm": spec.algorithm,
                                "method": spec.method,
                                "seed": spec.seed,
                                "status": run_manifest["status"],
                                "run_dir": str(run_dir),
                                "model_source": "",
                            }
                        )
                        print("  status: warm-start SFT failed")
                        continue
            grpo_cmd = build_grpo_train_command(spec, args, train_artifacts_dir, warm_start_model)
            print(f"  train: {command_to_str(grpo_cmd)}")
            run_manifest["train_steps"].append(
                {"name": "grpo", "command": grpo_cmd, "status": "planned"}
            )
            if args.execute:
                rc = run_command(grpo_cmd, Path.cwd(), run_dir / "logs" / "train_grpo.log")
                run_manifest["train_steps"][-1]["status"] = "ok" if rc == 0 else "failed"
                if rc == 0:
                    model_source = resolve_model_path(
                        train_artifacts_dir, spec.algorithm, spec.method, args.model
                    )
                if rc != 0 or not model_source:
                    run_manifest["status"] = "train_failed"
                    write_json(run_dir / "run_manifest.json", run_manifest)
                    top_level_manifest["runs"].append(run_manifest)
                    summary_rows.append(
                        {
                            "run_id": spec.run_id,
                            "algorithm": spec.algorithm,
                            "method": spec.method,
                            "seed": spec.seed,
                            "status": run_manifest["status"],
                            "run_dir": str(run_dir),
                            "model_source": "",
                        }
                    )
                    print("  status: train failed")
                    continue

        if not args.execute and model_source is None:
            if spec.algorithm == "base":
                model_source = get_model_config(args.model)["model_name"]
            else:
                model_source = "<TRAINED_MODEL_PATH>"

        run_manifest["model_source"] = model_source

        eval_cmd = build_eval_command(spec, args, model_source, run_dir)
        print(f"  eval: {command_to_str(eval_cmd)}")
        run_manifest["eval"] = {"command": eval_cmd, "status": "planned"}
        if args.execute:
            rc = run_command(eval_cmd, Path.cwd(), run_dir / "logs" / "eval.log")
            run_manifest["eval"]["status"] = "ok" if rc == 0 else "failed"
            if rc != 0:
                run_manifest["status"] = "eval_failed"
                write_json(run_dir / "run_manifest.json", run_manifest)
                top_level_manifest["runs"].append(run_manifest)
                summary_rows.append(
                    {
                        "run_id": spec.run_id,
                        "algorithm": spec.algorithm,
                        "method": spec.method,
                        "seed": spec.seed,
                        "status": run_manifest["status"],
                        "run_dir": str(run_dir),
                        "model_source": model_source or "",
                    }
                )
                print("  status: eval failed")
                continue

        if args.skip_robustness:
            run_manifest["robustness"].append({"status": "skipped"})
        else:
            for label, robustness_cmd in build_robustness_commands(spec, args, run_dir):
                print(f"  robustness[{label}]: {command_to_str(robustness_cmd)}")
                robustness_record = {"label": label, "command": robustness_cmd, "status": "planned"}
                run_manifest["robustness"].append(robustness_record)
                if args.execute:
                    rc = run_command(
                        robustness_cmd,
                        Path.cwd(),
                        run_dir / "logs" / f"robustness_{label}.log",
                    )
                    robustness_record["status"] = "ok" if rc == 0 else "failed"

        if args.skip_utility:
            run_manifest["utility"]["status"] = "skipped"
        elif not utility_available:
            run_manifest["utility"]["status"] = "lm_eval_not_found"
            print("  utility: lm_eval not found, skipping utility evaluation")
        else:
            utility_cmds = build_utility_commands(args, run_dir, model_source, lm_eval_launcher)
            for task_name, utility_cmd in utility_cmds.items():
                print(f"  utility[{task_name}]: {command_to_str(utility_cmd)}")
                run_manifest["utility"][task_name] = {
                    "command": utility_cmd,
                    "status": "planned",
                }
                if args.execute:
                    rc = run_command(
                        utility_cmd,
                        Path.cwd(),
                        run_dir / "logs" / f"utility_{task_name}.log",
                    )
                    run_manifest["utility"][task_name]["status"] = "ok" if rc == 0 else "failed"

        if args.execute:
            robustness_failed = any(
                record.get("status") == "failed" for record in run_manifest["robustness"]
            )
            utility_failed = any(
                isinstance(record, dict) and record.get("status") == "failed"
                for record in run_manifest["utility"].values()
            )
            run_manifest["status"] = (
                "completed_with_failures"
                if robustness_failed or utility_failed
                else "completed"
            )

        write_json(run_dir / "run_manifest.json", run_manifest)
        top_level_manifest["runs"].append(run_manifest)
        summary_rows.append(
            {
                "run_id": spec.run_id,
                "algorithm": spec.algorithm,
                "method": spec.method,
                "seed": spec.seed,
                "status": run_manifest["status"],
                "run_dir": str(run_dir),
                "model_source": model_source or "",
            }
        )

    write_json(run_root / "manifest.json", top_level_manifest)
    write_manifest_csv(run_root / "manifest.csv", summary_rows)

    print("\nArtifacts:")
    print(f"  - {run_root / 'manifest.json'}")
    print(f"  - {run_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
