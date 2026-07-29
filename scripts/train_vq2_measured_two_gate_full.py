#!/usr/bin/env python3
"""Fit the one phase-aware actor on exact measured two-gate oracle prefixes."""

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


TAG = "vq2_sf050_measured_two_gate_full_fit_001"
SCHEMA = "vq2_public_phase_recurrent_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf050_measured_two_gate_full_fit_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf045_public_phase_full_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "760101030db10e1dfbf2c31fcadf8fbbf01555105748ecbf4e20abcc638c819b"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "0bcf6acb4c6d0334de95e10759335071d73b58143272fcabadeb52063d0aaf96"
)
DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf049_measured_two_gate_oracle_prefix_64"
)
DATASET_REPORT_SHA256 = (
    "0231eb0f93aa589a8b96bf14c9764dea42ce626ec688293a37b9559a7f63452a"
)
DATASET_METADATA_SHA256 = (
    "e9f94910a415083e41e467090aaa9b5662d6df1dd6cfc5aea8244595c60bf339"
)
CAUSAL_HORIZON = 1472
CONFIG = replace(
    trainer.RecurrentPhaseConfig(),
    seed=42050,
    epochs=16,
    validation_agents=8,
    learning_rate=3e-5,
    train_encoder=True,
)
_ORIGINAL_DATASET = PublicPhaseDataset


def dataset_factory():
    return _ORIGINAL_DATASET(
        DATASET,
        verify_hashes=True,
        expected_report_sha256=DATASET_REPORT_SHA256,
        expected_metadata_sha256=DATASET_METADATA_SHA256,
        horizon=CAUSAL_HORIZON,
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
    key = str(wrapper.relative_to(ROOT))
    digest = sha256_path(wrapper)
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["source_sha256"][key] = digest
    checkpoint["causal_horizon_steps"] = CAUSAL_HORIZON
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"][key] = digest
    report["causal_horizon_steps"] = CAUSAL_HORIZON
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

