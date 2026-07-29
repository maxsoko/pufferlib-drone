#!/usr/bin/env python3
"""Train the compact actor on the SF030 causal-prefix DAgger distribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_vq2_recurrent_dagger_aggregate as aggregate
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_sf031_recurrent_dagger_causal_prefix_next_001"
SCHEMA = "vq2_recurrent_dagger_causal_prefix_next_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT
    / "docs/vq2_sf031_recurrent_dagger_causal_prefix_next_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf028_recurrent_dagger_causal_prefix_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "b74e2f1c792fbd0e250a06a9c65b4fdaa3cc4f132e5e6bbab0a4c07a240b0403"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "c1ebd267ca0d35ed873287da1378720c316eb8957115465c5813bf284a4c7ccd"
)
NEXT_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf030_recurrent_dagger_causal_prefix_next_512"
)
NEXT_REPORT_SHA256 = (
    "7c4d0d632d03e8713c0c4bfbfc083e66b680aa98e6553ba17876c44cfe6f2393"
)
NEXT_METADATA_SHA256 = (
    "99ebb650731ab5dd725fed704c1703a8ded308464fe4c893713117038938e3d2"
)
CONFIG = aggregate.AggregateTrainConfig(
    seed=42031,
    epochs=6,
    learning_rate=5e-5,
)


def configure_aggregate_module() -> None:
    aggregate.TAG = TAG
    aggregate.SCHEMA = SCHEMA
    aggregate.PREREGISTRATION = PREREGISTRATION
    aggregate.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    aggregate.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    aggregate.PARENT_REPORT = PARENT_REPORT
    aggregate.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    aggregate.NEXT_DATASET = NEXT_DATASET
    aggregate.NEXT_REPORT_SHA256 = NEXT_REPORT_SHA256
    aggregate.NEXT_METADATA_SHA256 = NEXT_METADATA_SHA256
    aggregate.DAGGER_SOURCE_PATTERN = ("broad", "next", "next", "next")


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda"
) -> dict:
    configure_aggregate_module()
    report = aggregate.train(output=output, device_name=device_name, config=CONFIG)
    wrapper_key = str(Path(__file__).resolve().relative_to(ROOT))
    wrapper_sha256 = sha256_path(Path(__file__).resolve())
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["source_sha256"][wrapper_key] = wrapper_sha256
    checkpoint["causal_prefix_dataset_report_sha256"] = NEXT_REPORT_SHA256
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"][wrapper_key] = wrapper_sha256
    report["causal_prefix_dataset_report_sha256"] = NEXT_REPORT_SHA256
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(json.dumps(train(output=args.output.resolve(), device_name=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
