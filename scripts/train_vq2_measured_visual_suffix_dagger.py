#!/usr/bin/env python3
"""Fit one SF066 child on C006 with three retained recurrent anchors."""

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

import scripts.train_vq2_public_phase_adapter as phase_adapter
import scripts.train_vq2_public_phase_recurrent as trainer
from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_measured_two_gate_aggregate_dagger import (
    PRIOR_DATASET,
    PRIOR_HORIZON,
    PRIOR_METADATA_SHA256,
    PRIOR_REPORT_SHA256,
    UNDERTURN_DATASET,
    UNDERTURN_HORIZON,
    UNDERTURN_METADATA_SHA256,
    UNDERTURN_REPORT_SHA256,
)
from scripts.train_vq2_measured_two_gate_dagger import (
    ORACLE_DATASET,
    ORACLE_HORIZON,
    ORACLE_METADATA_SHA256,
    ORACLE_REPORT_SHA256,
)
from scripts.train_vq2_public_phase_adapter import (
    PublicPhaseDataset,
    reconstruct_phase_batch,
)


TAG = "vq2_c007_measured_visual_suffix_dagger_fit_001"
SCHEMA = "vq2_public_phase_recurrent_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_c007_measured_visual_suffix_dagger_fit_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf066_aggregate_dagger_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = "f4ee6782de66736110c79a892efaa70635a2ad6c14f0bfaeeaba9812da0ae7a8"
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = "2ad76bc51417dd2160ef44ec4d4029870d3d9b189efb45d795aacd133e9e226e"
C006_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c006_measured_visual_suffix_dagger_128"
)
C006_REPORT_SHA256 = "e3c26dc8d8b6812a56994212ee1b392260cbc24bcb112a11140547bdd7ab4f79"
C006_METADATA_SHA256 = "77c0fe74a93ebc1afd501109e5cb3080a254a5758db7db7b5b4361fc37d8768f"
C006_HORIZON = 687
GROUP_NAMES = ("sf049_clean", "sf062_prior_crash", "sf065_underturn", "c006_true", "c006_alias")
GROUP_AGENTS = 64
CONFIG = replace(
    trainer.RecurrentPhaseConfig(),
    seed=43007,
    epochs=16,
    validation_agents=40,
    learning_rate=2e-5,
    phase_zero_loss_weight=1.0,
    gate2_loss_weight=2.0,
    train_encoder=True,
)


class C006PublicPhaseDataset:
    """Load the legal C006 schema without weakening the legacy loader."""

    def __init__(self, *, verify_hashes: bool = True) -> None:
        self.root = C006_DATASET
        if sha256_path(self.root / "report.json") != C006_REPORT_SHA256:
            raise RuntimeError("C006 report hash mismatch")
        if sha256_path(self.root / "metadata.json") != C006_METADATA_SHA256:
            raise RuntimeError("C006 metadata hash mismatch")
        self.report = json.loads((self.root / "report.json").read_text())
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        observation = self.metadata.get("observation", {})
        if not self.report.get("admitted") or (
            observation.get("mask_width") != MASK_SIZE
            or observation.get("tail_width") != PHASE_LEGAL_OBS_SIZE - MASK_SIZE
            or observation.get("stored_training_only_privileged_values_per_record") != 0
        ):
            raise RuntimeError("C006 legal storage boundary changed")
        if verify_hashes:
            for name, contract in self.metadata["files"].items():
                if sha256_path(self.root / name) != contract["sha256"]:
                    raise RuntimeError(f"C006 data hash mismatch for {name}")
        self.mask = np.load(self.root / "mask.npy", mmap_mode="r")
        self.tail = np.load(self.root / "tail.npy", mmap_mode="r")
        self.action = np.load(self.root / "action.npy", mmap_mode="r")
        self.valid = np.load(self.root / "valid.npy", mmap_mode="r")
        full_lengths = np.asarray(self.metadata["episode_lengths"], dtype=np.int64)
        full_steps, self.agents = self.valid.shape
        self.time_steps = C006_HORIZON
        self.lengths = np.minimum(full_lengths, self.time_steps)
        expected = {
            "mask": (full_steps, self.agents, MASK_SIZE),
            "tail": (full_steps, self.agents, PHASE_LEGAL_OBS_SIZE - MASK_SIZE),
            "action": (full_steps, self.agents, ACTION_SIZE),
        }
        if self.time_steps != full_steps or full_lengths.shape != (self.agents,):
            raise RuntimeError("C006 episode shape changed")
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise RuntimeError(f"C006 {name} shape changed")

    def chunk(
        self,
        agent_indices: np.ndarray,
        start: int,
        end: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.asarray(agent_indices, dtype=np.int64)
        if start < 0 or end <= start or end > self.time_steps:
            raise ValueError("invalid C006 sequence chunk")
        mask = np.take(self.mask[start:end], indices, axis=1)
        tail = np.take(self.tail[start:end], indices, axis=1)
        observation = reconstruct_phase_batch(mask, tail, device=device)
        action_np = np.take(self.action[start:end], indices, axis=1)
        valid_np = np.take(self.valid[start:end], indices, axis=1)
        action = torch.from_numpy(np.asarray(action_np, dtype=np.float32).copy())
        action = action.transpose(0, 1).contiguous().to(device)
        valid = torch.from_numpy(np.asarray(valid_np, dtype=np.uint8).copy())
        valid = valid.transpose(0, 1).contiguous().bool().to(device)
        gate2 = valid & torch.isclose(
            observation[..., -1],
            torch.tensor(float(phase_adapter.GATE2_PHASE), device=device),
            atol=1e-7,
            rtol=0.0,
        )
        return observation, action, valid, gate2


def _load_sources(*, verify_c006_hashes: bool = True) -> tuple[Any, ...]:
    return (
        PublicPhaseDataset(
            ORACLE_DATASET,
            verify_hashes=True,
            expected_report_sha256=ORACLE_REPORT_SHA256,
            expected_metadata_sha256=ORACLE_METADATA_SHA256,
            horizon=ORACLE_HORIZON,
        ),
        PublicPhaseDataset(
            PRIOR_DATASET,
            verify_hashes=True,
            expected_report_sha256=PRIOR_REPORT_SHA256,
            expected_metadata_sha256=PRIOR_METADATA_SHA256,
            horizon=PRIOR_HORIZON,
        ),
        PublicPhaseDataset(
            UNDERTURN_DATASET,
            verify_hashes=True,
            expected_report_sha256=UNDERTURN_REPORT_SHA256,
            expected_metadata_sha256=UNDERTURN_METADATA_SHA256,
            horizon=UNDERTURN_HORIZON,
        ),
        C006PublicPhaseDataset(verify_hashes=verify_c006_hashes),
        C006PublicPhaseDataset(verify_hashes=False),
    )


class BalancedSuffixDataset:
    """Five intact 64-agent recurrent groups with a split C006 source."""

    def __init__(self, *, verify_c006_hashes: bool = True) -> None:
        self.sources = _load_sources(verify_c006_hashes=verify_c006_hashes)
        if tuple(source.agents for source in self.sources[:3]) != (64, 64, 64):
            raise RuntimeError("anchor source agent counts changed")
        if self.sources[3].agents != 128 or self.sources[4].agents != 128:
            raise RuntimeError("C006 source agent count changed")
        count = len(self.sources)
        self.agents = GROUP_AGENTS * count
        self.time_steps = max(source.time_steps for source in self.sources)
        self.source_for_agent = np.tile(np.arange(count, dtype=np.int64), GROUP_AGENTS)
        local = np.repeat(np.arange(GROUP_AGENTS, dtype=np.int64), count)
        self.local_agent = local.copy()
        self.local_agent[self.source_for_agent == 4] += GROUP_AGENTS
        self.lengths = np.asarray(
            [
                self.sources[source].lengths[agent]
                for source, agent in zip(
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
            raise ValueError("invalid balanced suffix chunk")
        batch = len(indices)
        steps = end - start
        observation = torch.zeros((batch, steps, PHASE_LEGAL_OBS_SIZE), device=device)
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
            source_values = source.chunk(
                selected_locals[positions], start, source_end, device=device
            )
            width = source_end - start
            position_tensor = torch.from_numpy(positions).to(device)
            for destination, supplied in zip(
                (observation, action, valid, gate2), source_values, strict=True
            ):
                destination[position_tensor, :width] = supplied
        return observation, action, valid, gate2


def dataset_factory() -> BalancedSuffixDataset:
    return BalancedSuffixDataset()


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
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VQ2PhaseResidualActor(hidden_size=256, initial_std=0.15).to(device)
    model.load_state_dict(checkpoint["model_state"])
    sources = _load_sources(verify_c006_hashes=False)
    held_out = np.arange(56, 64)
    audits: dict[str, Any] = {
        GROUP_NAMES[index]: trainer.evaluate_partitions(
            model, sources[index], held_out, CONFIG, device
        )
        for index in range(3)
    }
    audits["c006_true"] = phase_adapter.evaluate_gate2(
        model, sources[3], held_out, CONFIG, device
    )
    audits["c006_alias"] = phase_adapter.evaluate_gate2(
        model, sources[4], held_out + GROUP_AGENTS, CONFIG, device
    )
    return audits


def source_audits_pass(audits: dict[str, Any]) -> bool:
    values: list[float] = []
    for source in GROUP_NAMES[:3]:
        for partition in ("phase_zero", "gate2"):
            result = audits[source][partition]
            weighted = float(result["weighted_mse"])
            mse = list(map(float, result["mse"]))
            values.extend((weighted, *mse))
            if weighted > 0.02 or any(value > 0.05 for value in mse):
                return False
    for source in GROUP_NAMES[3:]:
        result = audits[source]
        weighted = float(result["weighted_mse"])
        mse = list(map(float, result["mse"]))
        values.extend((weighted, *mse))
        if weighted > 0.02 or any(value > 0.05 for value in mse):
            return False
    return bool(np.isfinite(values).all())


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict[str, Any]:
    extra_frozen = {
        PRIOR_DATASET / "report.json": PRIOR_REPORT_SHA256,
        PRIOR_DATASET / "metadata.json": PRIOR_METADATA_SHA256,
        UNDERTURN_DATASET / "report.json": UNDERTURN_REPORT_SHA256,
        UNDERTURN_DATASET / "metadata.json": UNDERTURN_METADATA_SHA256,
        C006_DATASET / "report.json": C006_REPORT_SHA256,
        C006_DATASET / "metadata.json": C006_METADATA_SHA256,
    }
    for path, expected in extra_frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    configure_trainer()
    report = trainer.train(output_path=output, device_name=device_name, config=CONFIG)
    checkpoint_path = output / "policy_best.pt"
    audits = source_audits(checkpoint_path, device=torch.device(device_name))
    audits_passed = source_audits_pass(audits)
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
        "groups": list(GROUP_NAMES),
        "logical_agents": 320,
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
