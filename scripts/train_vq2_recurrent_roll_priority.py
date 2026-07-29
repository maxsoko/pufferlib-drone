#!/usr/bin/env python3
"""Continue SF031 with an SF032-justified roll-priority training loss."""

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


TAG = "vq2_sf033_recurrent_roll_priority_001"
SCHEMA = "vq2_recurrent_roll_priority_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf033_recurrent_roll_priority_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf031_recurrent_dagger_causal_prefix_next_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "b7c0e1b0516a2ae410b77a4558c0298fd2fe8c11793e7e814d2009cc88b336fc"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "8c22f07a40ce12737a88db930d92c9e1a7f0298d2392884b9cb0dfbbf2380649"
)
DIAGNOSIS_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf032_recurrent_roll_localization/report.json"
)
DIAGNOSIS_REPORT_SHA256 = (
    "3c4a7f287278d89d54ab4eeeef59953c375018a7fd7770c709768c6b99c0b940"
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
    seed=42033,
    epochs=6,
    learning_rate=2e-5,
    action_weights=(1.0, 1.0, 4.0, 1.0),
    loss_action_weights=(1.0, 4.0, 4.0, 1.0),
    prefer_admitted=True,
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


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict:
    if sha256_path(DIAGNOSIS_REPORT) != DIAGNOSIS_REPORT_SHA256:
        raise RuntimeError("SF032 diagnosis report hash mismatch")
    configure_aggregate_module()
    report = aggregate.train(output=output, device_name=device_name, config=CONFIG)
    wrapper_key = str(Path(__file__).resolve().relative_to(ROOT))
    diagnosis_key = str(DIAGNOSIS_REPORT.relative_to(ROOT))
    wrapper_sha256 = sha256_path(Path(__file__).resolve())
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["source_sha256"][wrapper_key] = wrapper_sha256
    checkpoint["source_sha256"][diagnosis_key] = DIAGNOSIS_REPORT_SHA256
    checkpoint["diagnosis_report_sha256"] = DIAGNOSIS_REPORT_SHA256
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"][wrapper_key] = wrapper_sha256
    report["source_sha256"][diagnosis_key] = DIAGNOSIS_REPORT_SHA256
    report["diagnosis_report_sha256"] = DIAGNOSIS_REPORT_SHA256
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(
        json.dumps(
            train(output=args.output.resolve(), device_name=args.device),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

