#!/usr/bin/env python3
"""Source-balance the short on-policy Gate-2 DAgger distribution."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_vq2_public_phase_recurrent as trainer
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_measured_two_gate_dagger import TwoSourcePublicPhaseDataset
from scripts.train_vq2_measured_two_gate_dagger_phase1 import (
    DAGGER_ONE,
    DAGGER_ONE_METADATA_SHA256,
    DAGGER_ONE_REPORT_SHA256,
    DAGGER_THREE,
    DAGGER_THREE_METADATA_SHA256,
    DAGGER_THREE_REPORT_SHA256,
    DAGGER_TWO,
    DAGGER_TWO_METADATA_SHA256,
    DAGGER_TWO_REPORT_SHA256,
    ORACLE_DATASET,
    ORACLE_HORIZON,
    ORACLE_METADATA_SHA256,
    ORACLE_REPORT_SHA256,
)
from scripts.train_vq2_public_phase_adapter import PublicPhaseDataset


TAG = "vq2_sf061_source_balanced_dagger_fit_001"
SCHEMA = "vq2_public_phase_recurrent_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf061_source_balanced_dagger_fit_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf059_measured_two_gate_dagger_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "00886d94a825798d9884be11f07819852fc278b4bf275c99ed262537ab891354"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "8281e2030197395ba91fcd8b383172dd85d7ecc4b248cfbfee0e4fd12d15e4b8"
)
CONFIG = replace(
    trainer.RecurrentPhaseConfig(),
    seed=42061,
    epochs=12,
    validation_agents=64,
    learning_rate=1e-5,
    phase_zero_loss_weight=1.0,
    gate2_loss_weight=2.0,
    train_encoder=True,
)


def _load_sources() -> tuple[PublicPhaseDataset, ...]:
    oracle = PublicPhaseDataset(
        ORACLE_DATASET,
        verify_hashes=True,
        expected_report_sha256=ORACLE_REPORT_SHA256,
        expected_metadata_sha256=ORACLE_METADATA_SHA256,
        horizon=ORACLE_HORIZON,
    )
    dagger_one = PublicPhaseDataset(
        DAGGER_ONE,
        verify_hashes=True,
        expected_report_sha256=DAGGER_ONE_REPORT_SHA256,
        expected_metadata_sha256=DAGGER_ONE_METADATA_SHA256,
        horizon=494,
    )
    dagger_two = PublicPhaseDataset(
        DAGGER_TWO,
        verify_hashes=True,
        expected_report_sha256=DAGGER_TWO_REPORT_SHA256,
        expected_metadata_sha256=DAGGER_TWO_METADATA_SHA256,
        horizon=462,
    )
    dagger_three = PublicPhaseDataset(
        DAGGER_THREE,
        verify_hashes=True,
        expected_report_sha256=DAGGER_THREE_REPORT_SHA256,
        expected_metadata_sha256=DAGGER_THREE_METADATA_SHA256,
        horizon=983,
    )
    return (oracle, dagger_one, dagger_two, *(dagger_three for _ in range(5)))


class BalancedPublicPhaseDataset:
    """Eight interleaved groups with fivefold SF058 sampling weight."""

    chunk = TwoSourcePublicPhaseDataset.chunk

    def __init__(self) -> None:
        self.sources = _load_sources()
        if len(self.sources) != 8 or len({source.agents for source in self.sources}) != 1:
            raise RuntimeError("source-balanced DAgger group contract changed")
        source_agents = self.sources[0].agents
        count = len(self.sources)
        self.agents = source_agents * count
        self.time_steps = max(source.time_steps for source in self.sources)
        self.source_for_agent = np.tile(np.arange(count, dtype=np.int64), source_agents)
        self.local_agent = np.repeat(np.arange(source_agents, dtype=np.int64), count)
        self.lengths = np.asarray(
            [
                self.sources[source].lengths[local]
                for source, local in zip(
                    self.source_for_agent, self.local_agent, strict=True
                )
            ],
            dtype=np.int64,
        )


def dataset_factory() -> BalancedPublicPhaseDataset:
    return BalancedPublicPhaseDataset()


def configure_trainer() -> None:
    trainer.TAG = TAG
    trainer.SCHEMA = SCHEMA
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.PREREGISTRATION = PREREGISTRATION
    trainer.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    trainer.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    trainer.PARENT_REPORT = PARENT_REPORT
    trainer.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    trainer.DATASET = ORACLE_DATASET
    trainer.DATASET_REPORT_SHA256 = ORACLE_REPORT_SHA256
    trainer.DATASET_METADATA_SHA256 = ORACLE_METADATA_SHA256
    trainer.PublicPhaseDataset = dataset_factory


def sf058_audit(
    checkpoint_path: Path, *, device: torch.device
) -> dict[str, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VQ2PhaseResidualActor(hidden_size=256, initial_std=0.15).to(device)
    model.load_state_dict(checkpoint["model_state"])
    source = _load_sources()[-1]
    return trainer.evaluate_partitions(
        model, source, np.arange(56, 64), CONFIG, device
    )


def sf058_audit_passes(audit: dict[str, dict]) -> bool:
    values = []
    for partition in ("phase_zero", "gate2"):
        values.append(float(audit[partition]["weighted_mse"]))
        values.extend(map(float, audit[partition]["mse"]))
    return bool(
        np.isfinite(values).all()
        and float(audit["phase_zero"]["weighted_mse"]) <= 0.01
        and float(audit["gate2"]["weighted_mse"]) <= 0.01
        and all(value <= 0.05 for value in values[1:5] + values[6:])
    )


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict:
    extra_frozen = {
        DAGGER_ONE / "report.json": DAGGER_ONE_REPORT_SHA256,
        DAGGER_ONE / "metadata.json": DAGGER_ONE_METADATA_SHA256,
        DAGGER_TWO / "report.json": DAGGER_TWO_REPORT_SHA256,
        DAGGER_TWO / "metadata.json": DAGGER_TWO_METADATA_SHA256,
        DAGGER_THREE / "report.json": DAGGER_THREE_REPORT_SHA256,
        DAGGER_THREE / "metadata.json": DAGGER_THREE_METADATA_SHA256,
    }
    for path, expected in extra_frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    configure_trainer()
    report = trainer.train(output_path=output, device_name=device_name, config=CONFIG)
    device = torch.device(device_name)
    checkpoint_path = output / "policy_best.pt"
    audit = sf058_audit(checkpoint_path, device=device)
    audit_passed = sf058_audit_passes(audit)
    wrapper = Path(__file__).resolve()
    extra_sources = {
        str(wrapper.relative_to(ROOT)): sha256_path(wrapper),
        **{
            str(path.relative_to(ROOT)): expected
            for path, expected in extra_frozen.items()
        },
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["source_sha256"].update(extra_sources)
    checkpoint["source_balancing"] = {
        "groups": ["sf049", "sf052", "sf055", *(["sf058"] * 5)],
        "validation_agents_per_group": 8,
        "sf058_only_audit": audit,
        "sf058_only_audit_passed": audit_passed,
    }
    checkpoint["combined_numerical_admission"] = bool(
        report["numerically_admitted"] and audit_passed
    )
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"].update(extra_sources)
    report["source_balancing"] = checkpoint["source_balancing"]
    report["combined_numerical_admission"] = checkpoint[
        "combined_numerical_admission"
    ]
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
    return 0 if report["combined_numerical_admission"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
