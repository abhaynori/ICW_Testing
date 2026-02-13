#!/usr/bin/env python3
"""
Mentor-feedback experiment runner for ICW/GRPO.

Covers:
1) Training-factor sweeps: learning rate, LoRA on/off, sample count, rules, system prompt styles
2) Validation generalization:
   - ELI5 with same prompts
   - ELI5 with alternate prompts
   - Alpaca with same prompts
3) Utility evaluation hooks:
   - IFEval, GSM8K, MMLU, SimpleQA (via lm-eval-harness if available)

Usage examples:
  python mentor_experiment_runner.py --phase screen --dry-run
  python mentor_experiment_runner.py --phase screen --execute --max-runs 8
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


SYSTEM_PROMPT_PROFILES = {
    "default": {
        "base_system_prompt": None,
        "system_prompt_prefix": None,
    },
    "tutor": {
        "base_system_prompt": (
            "You are a patient tutor. Give clear, accurate, step-by-step explanations."
        ),
        "system_prompt_prefix": (
            "Prioritize pedagogical clarity and concrete examples while following all constraints."
        ),
    },
    "direct": {
        "base_system_prompt": (
            "You are a direct expert assistant. Respond precisely with minimal fluff."
        ),
        "system_prompt_prefix": (
            "Favor concise, factual phrasing while preserving instruction fidelity."
        ),
    },
}


PHASE_CONFIG = {
    "screen": {
        "methods": ["acrostics", "lexical"],
        "learning_rates": [1e-5, 2e-5],
        "samples": [100, 200],
        "use_lora": [False, True],
        "rules_variants": ["paper", "minimal"],
        "prompt_variants": ["paper", "strict"],
        "system_profiles": ["default", "tutor"],
    },
    "full": {
        "methods": ["unicode", "initials", "lexical", "acrostics"],
        "learning_rates": [5e-6, 1e-5, 2e-5],
        "samples": [100, 200, 400],
        "use_lora": [False, True],
        "rules_variants": ["paper", "minimal", "none"],
        "prompt_variants": ["paper", "concise", "strict"],
        "system_profiles": ["default", "tutor", "direct"],
    },
}


@dataclass
class ExperimentSpec:
    run_id: str
    method: str
    learning_rate: float
    samples: int
    use_lora: bool
    rules_variant: str
    prompt_variant: str
    system_profile: str


def _slug_float(value: float) -> str:
    text = f"{value:.0e}" if value < 0.001 else f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def _parse_float_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_str_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def build_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    phase = PHASE_CONFIG[args.phase]
    methods = _parse_str_list(args.methods) or phase["methods"]
    learning_rates = _parse_float_list(args.learning_rates) or phase["learning_rates"]
    samples = _parse_int_list(args.samples) or phase["samples"]
    use_lora_values = [False, True] if args.use_lora_both else [args.use_lora]
    if args.use_lora_both:
        use_lora_values = phase["use_lora"]
    rules_variants = _parse_str_list(args.rules_variants) or phase["rules_variants"]
    prompt_variants = _parse_str_list(args.prompt_variants) or phase["prompt_variants"]
    system_profiles = _parse_str_list(args.system_profiles) or phase["system_profiles"]

    for profile in system_profiles:
        if profile not in SYSTEM_PROMPT_PROFILES:
            raise ValueError(f"Unknown system profile: {profile}")

    specs: list[ExperimentSpec] = []
    for values in itertools.product(
        methods,
        learning_rates,
        samples,
        use_lora_values,
        rules_variants,
        prompt_variants,
        system_profiles,
    ):
        method, lr, n_samples, use_lora, rules_v, prompt_v, sys_p = values
        run_id = (
            f"{method}_lr{_slug_float(lr)}_n{n_samples}"
            f"_lora{int(use_lora)}_rules-{rules_v}_prompt-{prompt_v}_sys-{sys_p}"
        )
        specs.append(
            ExperimentSpec(
                run_id=run_id,
                method=method,
                learning_rate=lr,
                samples=n_samples,
                use_lora=use_lora,
                rules_variant=rules_v,
                prompt_variant=prompt_v,
                system_profile=sys_p,
            )
        )

    if args.max_runs and len(specs) > args.max_runs:
        specs = specs[: args.max_runs]

    return specs


def _command_to_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_command(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process.returncode


def _resolve_trained_model(parent_dir: Path, method: str, model: str) -> str | None:
    pattern = str(parent_dir / f"{method}_{model}_*" / "final_model")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def build_train_command(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    run_dir: Path,
) -> list[str]:
    profile = SYSTEM_PROMPT_PROFILES[spec.system_profile]
    model_output_parent = run_dir / "train_artifacts"

    cmd = [
        "python",
        "grpo_train.py",
        "--model",
        args.model,
        "--method",
        spec.method,
        "--samples",
        str(spec.samples),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(spec.learning_rate),
        "--train-dataset",
        args.train_dataset,
        "--eval-datasets",
        args.train_dataset,
        "--eval-splits",
        "validation",
        "--eval-with-instruction",
        "--eval-samples",
        str(args.eval_samples),
        "--prompt-variant",
        spec.prompt_variant,
        "--rules-variant",
        spec.rules_variant,
        "--output-dir",
        str(model_output_parent),
        "--gen-batch-size",
        str(args.gen_batch_size),
    ]

    if profile["base_system_prompt"]:
        cmd.extend(["--base-system-prompt", profile["base_system_prompt"]])
    if profile["system_prompt_prefix"]:
        cmd.extend(["--system-prompt-prefix", profile["system_prompt_prefix"]])
    if spec.use_lora:
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

    return cmd


def build_eval_commands(
    trained_model_path: str,
    spec: ExperimentSpec,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, list[str]]:
    profile = SYSTEM_PROMPT_PROFILES[spec.system_profile]
    commands: dict[str, list[str]] = {}

    common = [
        "python",
        "cli.py",
        "--model-path",
        trained_model_path,
        "--samples",
        str(args.eval_samples),
    ]

    # 1) ELI5 same prompts
    same_prompt_cmd = common + [
        "--dataset",
        "eli5",
        "--split",
        "validation",
        "--prompt-variant",
        spec.prompt_variant,
        "--rules-variant",
        spec.rules_variant,
        "--output",
        str(run_dir / "eval_eli5_same"),
    ]
    if profile["base_system_prompt"]:
        same_prompt_cmd.extend(["--base-system-prompt", profile["base_system_prompt"]])
    if profile["system_prompt_prefix"]:
        same_prompt_cmd.extend(["--system-prompt-prefix", profile["system_prompt_prefix"]])
    commands["eli5_same_prompts"] = same_prompt_cmd

    # 2) ELI5 alternate prompts
    other_profile = SYSTEM_PROMPT_PROFILES[args.other_system_profile]
    other_prompt_cmd = common + [
        "--dataset",
        "eli5",
        "--split",
        "validation",
        "--prompt-variant",
        args.other_prompt_variant,
        "--rules-variant",
        args.other_rules_variant,
        "--output",
        str(run_dir / "eval_eli5_other_prompts"),
    ]
    if other_profile["base_system_prompt"]:
        other_prompt_cmd.extend(
            ["--base-system-prompt", other_profile["base_system_prompt"]]
        )
    if other_profile["system_prompt_prefix"]:
        other_prompt_cmd.extend(
            ["--system-prompt-prefix", other_profile["system_prompt_prefix"]]
        )
    commands["eli5_other_prompts"] = other_prompt_cmd

    # 3) Alpaca generalization (dataset shift)
    alpaca_cmd = common + [
        "--dataset",
        "alpaca",
        "--split",
        "validation",
        "--prompt-variant",
        spec.prompt_variant,
        "--rules-variant",
        spec.rules_variant,
        "--output",
        str(run_dir / "eval_alpaca"),
    ]
    if profile["base_system_prompt"]:
        alpaca_cmd.extend(["--base-system-prompt", profile["base_system_prompt"]])
    if profile["system_prompt_prefix"]:
        alpaca_cmd.extend(["--system-prompt-prefix", profile["system_prompt_prefix"]])
    commands["alpaca_generalization"] = alpaca_cmd

    return commands


def build_utility_commands(
    trained_model_path: str,
    tasks: list[str],
    run_dir: Path,
) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for task in tasks:
        task_clean = task.strip()
        if not task_clean:
            continue
        commands[f"utility_{task_clean}"] = [
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={trained_model_path},trust_remote_code=True",
            "--tasks",
            task_clean,
            "--batch_size",
            "auto",
            "--output_path",
            str(run_dir / "utility" / f"{task_clean}.json"),
        ]
    return commands


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mentor-feedback ICW experiments."
    )
    parser.add_argument("--phase", choices=["screen", "full"], default="screen")
    parser.add_argument("--execute", action="store_true", help="Execute commands")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--max-runs", type=int, default=0, help="Cap total runs")

    parser.add_argument("--model", default="small")
    parser.add_argument("--train-dataset", choices=["eli5", "alpaca"], default="eli5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--gen-batch-size", type=int, default=4)

    parser.add_argument("--methods", default=None, help="Comma-separated methods")
    parser.add_argument(
        "--learning-rates", default=None, help="Comma-separated learning rates"
    )
    parser.add_argument("--samples", default=None, help="Comma-separated sample counts")
    parser.add_argument("--rules-variants", default=None, help="Comma-separated variants")
    parser.add_argument(
        "--prompt-variants", default=None, help="Comma-separated prompt variants"
    )
    parser.add_argument(
        "--system-profiles",
        default=None,
        help=f"Comma-separated profiles: {','.join(SYSTEM_PROMPT_PROFILES.keys())}",
    )

    parser.add_argument("--use-lora", action="store_true", help="Use LoRA only")
    parser.add_argument(
        "--use-lora-both",
        action="store_true",
        help="Use both LoRA=False and LoRA=True settings",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    )

    parser.add_argument(
        "--other-prompt-variant",
        choices=["paper", "concise", "strict"],
        default="concise",
    )
    parser.add_argument(
        "--other-rules-variant",
        choices=["paper", "minimal", "none"],
        default="minimal",
    )
    parser.add_argument(
        "--other-system-profile",
        choices=list(SYSTEM_PROMPT_PROFILES.keys()),
        default="direct",
    )

    parser.add_argument(
        "--utility-tasks",
        default="ifeval,gsm8k,mmlu,simpleqa",
        help="Comma-separated utility tasks for lm_eval",
    )
    parser.add_argument("--skip-utility", action="store_true")
    parser.add_argument("--output-root", default="mentor_runs")
    args = parser.parse_args()

    if args.dry_run:
        args.execute = False
    elif not args.execute:
        # Default behavior: command preview only.
        args.dry_run = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / f"{args.phase}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args)
    print(f"Planned runs: {len(specs)}")
    print(f"Run root: {run_root}")

    manifest_rows: list[dict] = []
    summary = {
        "created_at": datetime.now().isoformat(),
        "phase": args.phase,
        "execute": args.execute,
        "args": vars(args),
        "planned_runs": len(specs),
        "runs": [],
    }

    utility_available = shutil.which("lm_eval") is not None
    utility_tasks = [task.strip() for task in args.utility_tasks.split(",") if task.strip()]

    for index, spec in enumerate(specs, start=1):
        run_dir = run_root / f"{index:03d}_{spec.run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{index}/{len(specs)}] {spec.run_id}")

        train_cmd = build_train_command(spec, args, run_dir)
        print(f"  train: {_command_to_str(train_cmd)}")

        run_record = asdict(spec)
        run_record["run_dir"] = str(run_dir)
        run_record["train_command"] = train_cmd
        run_record["status"] = "planned"
        run_record["trained_model_path"] = None
        run_record["evaluations"] = {}
        run_record["utility"] = {}

        if not args.execute:
            placeholder_model = "<TRAINED_MODEL_PATH>"
            eval_commands = build_eval_commands(placeholder_model, spec, args, run_dir)
            for eval_name, eval_cmd in eval_commands.items():
                print(f"  {eval_name}: {_command_to_str(eval_cmd)}")
                run_record["evaluations"][eval_name] = {
                    "command": eval_cmd,
                    "status": "planned",
                }

            if args.skip_utility:
                run_record["utility"]["status"] = "skipped"
            elif not utility_available:
                run_record["utility"]["status"] = "lm_eval_not_found"
                print("  utility: lm_eval not found, skipping utility evaluation")
            else:
                utility_cmds = build_utility_commands(placeholder_model, utility_tasks, run_dir)
                for util_name, util_cmd in utility_cmds.items():
                    print(f"  {util_name}: {_command_to_str(util_cmd)}")
                    run_record["utility"][util_name] = {
                        "command": util_cmd,
                        "status": "planned",
                    }

            summary["runs"].append(run_record)
            manifest_rows.append(
                {
                    "run_id": spec.run_id,
                    "status": run_record["status"],
                    "trained_model_path": "",
                    "run_dir": str(run_dir),
                }
            )
            continue

        if args.execute:
            train_rc = _run_command(train_cmd, Path.cwd(), run_dir / "logs" / "train.log")
            if train_rc != 0:
                run_record["status"] = "train_failed"
                summary["runs"].append(run_record)
                manifest_rows.append(
                    {
                        "run_id": spec.run_id,
                        "status": run_record["status"],
                        "trained_model_path": "",
                        "run_dir": str(run_dir),
                    }
                )
                print("  status: train failed")
                continue

        trained_model_path = _resolve_trained_model(
            run_dir / "train_artifacts",
            spec.method,
            args.model,
        )
        if not trained_model_path:
            run_record["status"] = "model_not_found"
            summary["runs"].append(run_record)
            manifest_rows.append(
                {
                    "run_id": spec.run_id,
                    "status": run_record["status"],
                    "trained_model_path": "",
                    "run_dir": str(run_dir),
                }
            )
            print("  status: trained model path not found")
            continue

        run_record["trained_model_path"] = trained_model_path
        eval_commands = build_eval_commands(trained_model_path, spec, args, run_dir)
        for eval_name, eval_cmd in eval_commands.items():
            print(f"  {eval_name}: {_command_to_str(eval_cmd)}")
            run_record["evaluations"][eval_name] = {
                "command": eval_cmd,
                "status": "planned",
            }
            if args.execute:
                rc = _run_command(
                    eval_cmd,
                    Path.cwd(),
                    run_dir / "logs" / f"{eval_name}.log",
                )
                run_record["evaluations"][eval_name]["status"] = (
                    "ok" if rc == 0 else "failed"
                )

        if args.skip_utility:
            run_record["utility"]["status"] = "skipped"
        elif not utility_available:
            run_record["utility"]["status"] = "lm_eval_not_found"
            print("  utility: lm_eval not found, skipping utility evaluation")
        else:
            utility_cmds = build_utility_commands(trained_model_path, utility_tasks, run_dir)
            for util_name, util_cmd in utility_cmds.items():
                print(f"  {util_name}: {_command_to_str(util_cmd)}")
                run_record["utility"][util_name] = {"command": util_cmd, "status": "planned"}
                if args.execute:
                    rc = _run_command(
                        util_cmd,
                        Path.cwd(),
                        run_dir / "logs" / f"{util_name}.log",
                    )
                    run_record["utility"][util_name]["status"] = (
                        "ok" if rc == 0 else "failed"
                    )

        if args.execute:
            run_record["status"] = "completed"
        summary["runs"].append(run_record)
        manifest_rows.append(
            {
                "run_id": spec.run_id,
                "status": run_record["status"],
                "trained_model_path": trained_model_path,
                "run_dir": str(run_dir),
            }
        )

    write_manifest(run_root / "manifest.csv", manifest_rows)
    with (run_root / "manifest.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nArtifacts:")
    print(f"  - {run_root / 'manifest.csv'}")
    print(f"  - {run_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
