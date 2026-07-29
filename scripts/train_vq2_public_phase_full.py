#!/usr/bin/env python3
"""Train the full single phase-aware actor on fresh SF044 prefixes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_vq2_public_phase_recurrent as trainer
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_public_phase_adapter import PublicPhaseDataset


TAG = "vq2_sf045_public_phase_full_fit_001"
SCHEMA = "vq2_public_phase_recurrent_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf045_public_phase_full_fit_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf042_public_phase_recurrent_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "fe5cd944a588b574e3106c72c7859d0e044efd827871f47358b21473bcddb03c"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "08f76283b79f1be0370ff6403a96e85a33d042e2bbe3d95b9d6a461cb067e800"
)
DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf044_public_phase_prefix_dagger_512"
)
DATASET_REPORT_SHA256 = (
    "0bac52c2a019ce9ea7dfcd33e79aab7bbd031622a648c26d21e1dc98aac55782"
)
DATASET_METADATA_SHA256 = (
    "9a0ee45d4be7250532870197903192e7bc3d8fc3950c4f1c52f08d7d8e466b48"
)
CONFIG = replace(
    trainer.RecurrentPhaseConfig(),
    seed=42045,
    learning_rate=2e-5,
    train_encoder=True,
)
_ORIGINAL_DATASET = PublicPhaseDataset


def dataset_factory():
    return _ORIGINAL_DATASET(
        DATASET,
        verify_hashes=True,
        expected_report_sha256=DATASET_REPORT_SHA256,
        expected_metadata_sha256=DATASET_METADATA_SHA256,
        horizon=768,
    )


def configure_trainer() -> None:
    trainer.TAG = TAG
    trainer.SCHEMA = SCHEMA
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.PREREGISTRATION = PREREGISTRATION
    trainer.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    trainer.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    trainer.PARENT_REPORT = PARENT_REPORT
    trainer.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    trainer.DATASET = DATASET
    trainer.DATASET_REPORT_SHA256 = DATASET_REPORT_SHA256
    trainer.DATASET_METADATA_SHA256 = DATASET_METADATA_SHA256
    trainer.PublicPhaseDataset = dataset_factory


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict:
    configure_trainer()
    report = trainer.train(
        output_path=output, device_name=device_name, config=CONFIG
    )
    wrapper = Path(__file__).resolve()
    wrapper_key = str(wrapper.relative_to(ROOT))
    wrapper_sha = sha256_path(wrapper)
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["source_sha256"][wrapper_key] = wrapper_sha
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"][wrapper_key] = wrapper_sha
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

