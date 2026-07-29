#!/usr/bin/env python3
"""Fit the phase actor on four exact-course oracle/DAgger distributions."""

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
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_measured_two_gate_dagger import (
    DAGGER_DATASET as DAGGER_ONE,
    DAGGER_METADATA_SHA256 as DAGGER_ONE_METADATA_SHA256,
    DAGGER_REPORT_SHA256 as DAGGER_ONE_REPORT_SHA256,
    ORACLE_DATASET,
    ORACLE_HORIZON,
    ORACLE_METADATA_SHA256,
    ORACLE_REPORT_SHA256,
    TwoSourcePublicPhaseDataset,
)
from scripts.train_vq2_measured_two_gate_dagger_second import (
    DAGGER_TWO,
    DAGGER_TWO_METADATA_SHA256,
    DAGGER_TWO_REPORT_SHA256,
)
from scripts.train_vq2_public_phase_adapter import PublicPhaseDataset


TAG = "vq2_sf059_measured_two_gate_dagger_fit_001"
SCHEMA = "vq2_public_phase_recurrent_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf059_measured_two_gate_dagger_fit_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf056_measured_two_gate_dagger_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "3b2f736fb8fae233d64415e07c2dcc3cb7e2613749ae1742dfdda14707989409"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "e21a0fb8a01a96f781379bcf6a8db93877dbdf6d5c9b80c88b3f9ed185ab35f5"
)
DAGGER_THREE = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf058_measured_two_gate_dagger_64"
)
DAGGER_THREE_REPORT_SHA256 = (
    "3b213089a8aa93d33a693fcf3db5e8076a370de60dc77989e1938897894684d6"
)
DAGGER_THREE_METADATA_SHA256 = (
    "c42131ed9d61329670c220101ccb51fa8f8343306723511c94becca327e4ee04"
)
CONFIG = replace(
    trainer.RecurrentPhaseConfig(),
    seed=42059,
    epochs=12,
    validation_agents=32,
    learning_rate=3e-5,
    phase_zero_loss_weight=1.0,
    gate2_loss_weight=2.0,
    train_encoder=True,
)


class FourSourcePublicPhaseDataset:
    """Four source-balanced agent groups with genuine recurrent boundaries."""

    chunk = TwoSourcePublicPhaseDataset.chunk

    def __init__(self) -> None:
        specifications = (
            (ORACLE_DATASET, ORACLE_REPORT_SHA256, ORACLE_METADATA_SHA256, ORACLE_HORIZON),
            (DAGGER_ONE, DAGGER_ONE_REPORT_SHA256, DAGGER_ONE_METADATA_SHA256, 494),
            (DAGGER_TWO, DAGGER_TWO_REPORT_SHA256, DAGGER_TWO_METADATA_SHA256, 462),
            (
                DAGGER_THREE,
                DAGGER_THREE_REPORT_SHA256,
                DAGGER_THREE_METADATA_SHA256,
                983,
            ),
        )
        self.sources = tuple(
            PublicPhaseDataset(
                root,
                verify_hashes=True,
                expected_report_sha256=report_sha,
                expected_metadata_sha256=metadata_sha,
                horizon=horizon,
            )
            for root, report_sha, metadata_sha, horizon in specifications
        )
        if len({source.agents for source in self.sources}) != 1:
            raise RuntimeError("four-source DAgger agent counts differ")
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


def dataset_factory() -> FourSourcePublicPhaseDataset:
    return FourSourcePublicPhaseDataset()


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
    wrapper = Path(__file__).resolve()
    extra_sources = {
        str(wrapper.relative_to(ROOT)): sha256_path(wrapper),
        **{
            str(path.relative_to(ROOT)): expected
            for path, expected in extra_frozen.items()
        },
    }
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["source_sha256"].update(extra_sources)
    checkpoint["aggregate_dataset_sources"] = {
        "oracle_report_sha256": ORACLE_REPORT_SHA256,
        "dagger_one_report_sha256": DAGGER_ONE_REPORT_SHA256,
        "dagger_two_report_sha256": DAGGER_TWO_REPORT_SHA256,
        "dagger_three_report_sha256": DAGGER_THREE_REPORT_SHA256,
        "logical_agents": 256,
        "validation_agents_per_source": 8,
    }
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"].update(extra_sources)
    report["aggregate_dataset_sources"] = checkpoint["aggregate_dataset_sources"]
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
