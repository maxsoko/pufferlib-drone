#!/usr/bin/env python3
"""Fit one phase actor on clean and policy-state measured-course episodes."""

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
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_public_phase_adapter import PublicPhaseDataset


TAG = "vq2_sf053_measured_two_gate_dagger_fit_001"
SCHEMA = "vq2_public_phase_recurrent_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf053_measured_two_gate_dagger_fit_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf050_measured_two_gate_full_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "61631d5bbcfce1eb75522d49804214c91b33930545db6815c2d8c09ea5adaa57"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "d78e6bcbb00e6ca087dee4ed48f5d9de466b08735db63cf073826283a7cfda0c"
)
ORACLE_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf049_measured_two_gate_oracle_prefix_64"
)
ORACLE_REPORT_SHA256 = (
    "0231eb0f93aa589a8b96bf14c9764dea42ce626ec688293a37b9559a7f63452a"
)
ORACLE_METADATA_SHA256 = (
    "e9f94910a415083e41e467090aaa9b5662d6df1dd6cfc5aea8244595c60bf339"
)
DAGGER_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf052_measured_two_gate_dagger_64"
)
DAGGER_REPORT_SHA256 = (
    "188b8afd8d4e9d42db8fdee032cd910d0362d612248d1aa108ea293accd9e81c"
)
DAGGER_METADATA_SHA256 = (
    "ba48d09f4b89995bb335eed27f703157423331a3f9d917081b365f9aaee9ac06"
)
ORACLE_HORIZON = 1472
CONFIG = replace(
    trainer.RecurrentPhaseConfig(),
    seed=42053,
    epochs=16,
    validation_agents=16,
    learning_rate=3e-5,
    train_encoder=True,
)


class TwoSourcePublicPhaseDataset:
    """Virtual agent-axis union that preserves each source's episode history."""

    def __init__(self) -> None:
        self.sources = (
            PublicPhaseDataset(
                ORACLE_DATASET,
                verify_hashes=True,
                expected_report_sha256=ORACLE_REPORT_SHA256,
                expected_metadata_sha256=ORACLE_METADATA_SHA256,
                horizon=ORACLE_HORIZON,
            ),
            PublicPhaseDataset(
                DAGGER_DATASET,
                verify_hashes=True,
                expected_report_sha256=DAGGER_REPORT_SHA256,
                expected_metadata_sha256=DAGGER_METADATA_SHA256,
                horizon=494,
            ),
        )
        if self.sources[0].agents != self.sources[1].agents:
            raise RuntimeError("two-source DAgger agent counts differ")
        source_agents = self.sources[0].agents
        self.agents = source_agents * 2
        self.time_steps = max(source.time_steps for source in self.sources)
        self.source_for_agent = np.tile(np.arange(2, dtype=np.int64), source_agents)
        self.local_agent = np.repeat(np.arange(source_agents, dtype=np.int64), 2)
        self.lengths = np.asarray(
            [
                self.sources[source].lengths[local]
                for source, local in zip(
                    self.source_for_agent, self.local_agent, strict=True
                )
            ],
            dtype=np.int64,
        )

    def chunk(
        self,
        agent_indices: np.ndarray,
        start: int,
        end: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.asarray(agent_indices, dtype=np.int64)
        if (
            indices.ndim != 1
            or start < 0
            or end <= start
            or end > self.time_steps
            or np.any(indices < 0)
            or np.any(indices >= self.agents)
        ):
            raise ValueError("invalid two-source public-phase chunk")
        batch = len(indices)
        steps = end - start
        observation = torch.zeros(
            (batch, steps, PHASE_LEGAL_OBS_SIZE), device=device
        )
        action = torch.zeros((batch, steps, ACTION_SIZE), device=device)
        valid = torch.zeros((batch, steps), dtype=torch.bool, device=device)
        gate2 = torch.zeros_like(valid)
        selected_sources = self.source_for_agent[indices]
        selected_locals = self.local_agent[indices]
        for source_index, source in enumerate(self.sources):
            positions = np.flatnonzero(selected_sources == source_index)
            source_end = min(end, source.time_steps)
            if not len(positions) or start >= source_end:
                continue
            source_observation, source_action, source_valid, source_gate2 = (
                source.chunk(
                    selected_locals[positions], start, source_end, device=device
                )
            )
            width = source_end - start
            position_tensor = torch.from_numpy(positions).to(device)
            observation[position_tensor, :width] = source_observation
            action[position_tensor, :width] = source_action
            valid[position_tensor, :width] = source_valid
            gate2[position_tensor, :width] = source_gate2
        return observation, action, valid, gate2


def dataset_factory() -> TwoSourcePublicPhaseDataset:
    return TwoSourcePublicPhaseDataset()


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
    for path, expected in {
        DAGGER_DATASET / "report.json": DAGGER_REPORT_SHA256,
        DAGGER_DATASET / "metadata.json": DAGGER_METADATA_SHA256,
    }.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    configure_trainer()
    report = trainer.train(
        output_path=output, device_name=device_name, config=CONFIG
    )
    wrapper = Path(__file__).resolve()
    wrapper_key = str(wrapper.relative_to(ROOT))
    wrapper_sha = sha256_path(wrapper)
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extra_sources = {
        wrapper_key: wrapper_sha,
        str((DAGGER_DATASET / "report.json").relative_to(ROOT)): (
            DAGGER_REPORT_SHA256
        ),
        str((DAGGER_DATASET / "metadata.json").relative_to(ROOT)): (
            DAGGER_METADATA_SHA256
        ),
    }
    checkpoint["source_sha256"].update(extra_sources)
    checkpoint["aggregate_dataset_sources"] = {
        "oracle_report_sha256": ORACLE_REPORT_SHA256,
        "oracle_metadata_sha256": ORACLE_METADATA_SHA256,
        "dagger_report_sha256": DAGGER_REPORT_SHA256,
        "dagger_metadata_sha256": DAGGER_METADATA_SHA256,
        "logical_agents": 128,
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
