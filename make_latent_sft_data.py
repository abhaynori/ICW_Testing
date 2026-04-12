#!/usr/bin/env python3
"""
Build a mixed explicit/implicit SFT dataset from generated watermark records.

Usage:
  python make_latent_sft_data.py \
      --input sft_data_probe_sec_14b/acrostics_8bit_20260410_094506.json \
      --output sft_data_probe_sec_14b/acrostics_8bit_20260410_094506_latent.json \
      --implicit-copies 1
"""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Duplicate generated SFT records with include_instruction=false "
        "to create a latent-distillation dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to generated SFT JSON from generate_sft_data.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the mixed explicit/implicit SFT JSON",
    )
    parser.add_argument(
        "--implicit-copies",
        type=int,
        default=1,
        help="How many implicit copies to create per source record (default: 1)",
    )
    parser.add_argument(
        "--explicit-copies",
        type=int,
        default=1,
        help="How many explicit copies to create per source record (default: 1)",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=41,
        help="Seed for output record shuffling (default: 41)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.implicit_copies < 0:
        raise ValueError("--implicit-copies must be >= 0")
    if args.explicit_copies < 0:
        raise ValueError("--explicit-copies must be >= 0")
    if args.implicit_copies == 0 and args.explicit_copies == 0:
        raise ValueError("At least one of --implicit-copies or --explicit-copies must be > 0")

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r") as f:
        payload = json.load(f)

    records = list(payload.get("records", []))
    if not records:
        raise ValueError("Input JSON contains no records.")

    explicit_records = []
    implicit_records = []

    for record in records:
        for _ in range(args.explicit_copies):
            explicit_record = deepcopy(record)
            explicit_record["include_instruction"] = True
            explicit_records.append(explicit_record)

        for _ in range(args.implicit_copies):
            implicit_record = deepcopy(record)
            implicit_record["include_instruction"] = False
            implicit_records.append(implicit_record)

    mixed_records = explicit_records + implicit_records
    random.Random(args.shuffle_seed).shuffle(mixed_records)

    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "source_path": str(input_path),
            "source_num_records": len(records),
            "explicit_records": len(explicit_records),
            "implicit_records": len(implicit_records),
            "explicit_copies_per_record": args.explicit_copies,
            "implicit_copies_per_record": args.implicit_copies,
            "num_records": len(mixed_records),
            "latent_sft_mixture": True,
        }
    )

    output = {
        "metadata": metadata,
        "records": mixed_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print("Built latent SFT dataset")
    print(f"  Source records: {len(records)}")
    print(f"  Explicit copies: {len(explicit_records)}")
    print(f"  Implicit copies: {len(implicit_records)}")
    print(f"  Total records: {len(mixed_records)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
