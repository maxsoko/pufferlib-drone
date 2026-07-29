#!/usr/bin/env python3
"""Fit clean, prior-crash, and safe-under-turn exact DAgger histories."""

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
from scripts.train_vq2_measured_two_gate_dagger import (
    ORACLE_DATASET,
    ORACLE_HORIZON,
    ORACLE_METADATA_SHA256,
    ORACLE_REPORT_SHA256,
    TwoSourcePublicPhaseDataset,
)
from scripts.train_vq2_public_phase_adapter import PublicPhaseDataset


TAG = "vq2_sf066_aggregate_dagger_fit_001"
SCHEMA = "vq2_public_phase_recurrent_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = ROOT / "docs/vq2_sf066_aggregate_dagger_fit_preregistration_2026-07-28.md"
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf063_current_dagger_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "c27c28bd26e937fc8023ceca6f8880fa6f3d4320bf6db55ce9837518b1868ebb"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "fa738aee5d5c79080d0e7ada1c69d8cce074ef5ec9ed03d8a5684a0c0357db2e"
)
PRIOR_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf062_measured_two_gate_dagger_phase1_second_64"
)
PRIOR_REPORT_SHA256 = (
    "4029758898e8ce0a6f8afbeeadfea6d8ef6088e9d2428ee12f3a309e2312a21f"
)
PRIOR_METADATA_SHA256 = (
    "767cd19eca3c0c2f6722b5c4c9bc3c980e826af7b89d1cd78780a5f819b7f16e"
)
PRIOR_HORIZON = 1493
UNDERTURN_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf065_underturn_dagger_64"
)
UNDERTURN_REPORT_SHA256 = (
    "30ca311f43000a757bb2fd3656f00ac9d956e55a194ba0ac256a778bfb346c7d"
)
UNDERTURN_METADATA_SHA256 = (
    "62edffae8ed13858ed88fef891f2865bf31d775f0824fe9b397c4301ea670605"
)
UNDERTURN_HORIZON = 461
REPETITIONS = 3
CONFIG = replace(
    trainer.RecurrentPhaseConfig(),
    seed=42066,
    epochs=16,
    validation_agents=56,
    learning_rate=2e-5,
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
    prior = PublicPhaseDataset(
        PRIOR_DATASET,
        verify_hashes=True,
        expected_report_sha256=PRIOR_REPORT_SHA256,
        expected_metadata_sha256=PRIOR_METADATA_SHA256,
        horizon=PRIOR_HORIZON,
    )
    underturn = PublicPhaseDataset(
        UNDERTURN_DATASET,
        verify_hashes=True,
        expected_report_sha256=UNDERTURN_REPORT_SHA256,
        expected_metadata_sha256=UNDERTURN_METADATA_SHA256,
        horizon=UNDERTURN_HORIZON,
    )
    return (
        oracle,
        *(prior for _ in range(REPETITIONS)),
        *(underturn for _ in range(REPETITIONS)),
    )


class AggregateDaggerDataset:
    """Seven balanced logical groups with intact source histories."""

    chunk = TwoSourcePublicPhaseDataset.chunk

    def __init__(self) -> None:
        self.sources = _load_sources()
        if len(self.sources) != 7 or len({source.agents for source in self.sources}) != 1:
            raise RuntimeError("aggregate DAgger source-group contract changed")
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


def dataset_factory() -> AggregateDaggerDataset:
    return AggregateDaggerDataset()


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


def source_audits(
    checkpoint_path: Path, *, device: torch.device
) -> dict[str, dict[str, dict]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VQ2PhaseResidualActor(hidden_size=256, initial_std=0.15).to(device)
    model.load_state_dict(checkpoint["model_state"])
    sources = _load_sources()
    held_out = np.arange(56, 64)
    return {
        "sf049_clean": trainer.evaluate_partitions(model, sources[0], held_out, CONFIG, device),
        "sf062_prior_crash": trainer.evaluate_partitions(model, sources[1], held_out, CONFIG, device),
        "sf065_underturn": trainer.evaluate_partitions(model, sources[-1], held_out, CONFIG, device),
    }


def source_audits_pass(audits: dict[str, dict[str, dict]]) -> bool:
    values: list[float] = []
    for source in ("sf049_clean", "sf062_prior_crash", "sf065_underturn"):
        for partition in ("phase_zero", "gate2"):
            result = audits[source][partition]
            weighted = float(result["weighted_mse"])
            mse = list(map(float, result["mse"]))
            values.extend((weighted, *mse))
            if weighted > 0.01 or any(value > 0.05 for value in mse):
                return False
    return bool(np.isfinite(values).all())


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict:
    extra_frozen = {
        PRIOR_DATASET / "report.json": PRIOR_REPORT_SHA256,
        PRIOR_DATASET / "metadata.json": PRIOR_METADATA_SHA256,
        UNDERTURN_DATASET / "report.json": UNDERTURN_REPORT_SHA256,
        UNDERTURN_DATASET / "metadata.json": UNDERTURN_METADATA_SHA256,
    }
    for path, expected in extra_frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    configure_trainer()
    report = trainer.train(output_path=output, device_name=device_name, config=CONFIG)
    device = torch.device(device_name)
    checkpoint_path = output / "policy_best.pt"
    audits = source_audits(checkpoint_path, device=device)
    audits_passed = source_audits_pass(audits)
    wrapper = Path(__file__).resolve()
    extra_sources = {
        str(wrapper.relative_to(ROOT)): sha256_path(wrapper),
        **{str(path.relative_to(ROOT)): expected for path, expected in extra_frozen.items()},
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["source_sha256"].update(extra_sources)
    checkpoint["source_balancing"] = {
        "groups": ["sf049", *("sf062" for _ in range(REPETITIONS)), *("sf065" for _ in range(REPETITIONS))],
        "validation_agents_per_group": 8,
        "source_specific_audits": audits,
        "source_specific_audits_passed": audits_passed,
    }
    checkpoint["combined_numerical_admission"] = bool(
        report["numerically_admitted"] and audits_passed
    )
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"].update(extra_sources)
    report["source_balancing"] = checkpoint["source_balancing"]
    report["combined_numerical_admission"] = checkpoint["combined_numerical_admission"]
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
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
