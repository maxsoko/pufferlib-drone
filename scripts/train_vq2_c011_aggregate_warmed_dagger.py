#!/usr/bin/env python3
"""Fit one C009 child on the aggregate warmed C006 and C011 DAgger data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_vq2_measured_visual_suffix_dagger as c007
import scripts.train_vq2_public_phase_adapter as phase_adapter
import scripts.train_vq2_public_phase_recurrent as trainer
from pufferlib.vq2_informed import MASK_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.eval_vq2_measured_visual_suffix_handoff import load_warm_prefix


TAG = "vq2_c012_c011_aggregate_warmed_dagger_fit_001"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_c012_c011_aggregate_warmed_dagger_fit_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c009_measured_visual_suffix_warmed_dagger_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "f037e56ec07ae134a87b3cf86bc9e1f3312d2310918cdf4f6fb84a9e96b259f2"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "296adbfdb8c6e2c8360e872a47f604b54ac7265f7628944c4d8c3b7b426baf34"
)
C011_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c011_c009_measured_visual_suffix_dagger_128"
)
C011_REPORT_SHA256 = (
    "4264a4eef0644149d54a30e4e541478968ff84a6581fe55ea53a489a9b63d0e3"
)
C011_METADATA_SHA256 = (
    "e5910896a5cfaf50d50601b5f4a4a05020dbd4ab7eecb6bf9706a180987f3fbc"
)
C011_HORIZON = 283
GROUP_NAMES = (
    "sf049_clean",
    "sf062_prior_crash",
    "sf065_underturn",
    "c006_true",
    "c006_alias",
    "c011_true",
    "c011_alias",
)
GROUP_AGENTS = 64
CONFIG = replace(c007.CONFIG, seed=43012, validation_agents=56)


class C011PublicPhaseDataset:
    """Load the legal C011 time-major dataset."""

    def __init__(self, *, verify_hashes: bool = True) -> None:
        self.root = C011_DATASET
        if sha256_path(self.root / "report.json") != C011_REPORT_SHA256:
            raise RuntimeError("C011 report hash mismatch")
        if sha256_path(self.root / "metadata.json") != C011_METADATA_SHA256:
            raise RuntimeError("C011 metadata hash mismatch")
        self.report = json.loads((self.root / "report.json").read_text())
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        observation = self.metadata.get("observation", {})
        action = self.metadata.get("action", {})
        if (
            not self.report.get("admitted")
            or observation.get("mask_width") != MASK_SIZE
            or observation.get("tail_width") != PHASE_LEGAL_OBS_SIZE - MASK_SIZE
            or observation.get("stored_training_only_privileged_values_per_record")
            != 0
            or action.get("plant_source") != "c009_deterministic_mean"
            or action.get("teacher_blend") != 0.0
        ):
            raise RuntimeError("C011 legal storage boundary changed")
        if verify_hashes:
            for name, contract in self.metadata["files"].items():
                if sha256_path(self.root / name) != contract["sha256"]:
                    raise RuntimeError(f"C011 data hash mismatch for {name}")
        self.mask = np.load(self.root / "mask.npy", mmap_mode="r")
        self.tail = np.load(self.root / "tail.npy", mmap_mode="r")
        self.action = np.load(self.root / "action.npy", mmap_mode="r")
        self.valid = np.load(self.root / "valid.npy", mmap_mode="r")
        self.lengths = np.asarray(
            self.metadata["episode_lengths"], dtype=np.int64
        )
        self.time_steps, self.agents = self.valid.shape
        if self.time_steps != C011_HORIZON or self.lengths.shape != (self.agents,):
            raise RuntimeError("C011 episode shape changed")
        expected = {
            "mask": (self.time_steps, self.agents, MASK_SIZE),
            "tail": (
                self.time_steps,
                self.agents,
                PHASE_LEGAL_OBS_SIZE - MASK_SIZE,
            ),
            "action": (self.time_steps, self.agents, ACTION_SIZE),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise RuntimeError(f"C011 {name} shape changed")

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
            raise ValueError("invalid C011 sequence chunk")
        mask = np.take(self.mask[start:end], indices, axis=1)
        tail = np.take(self.tail[start:end], indices, axis=1)
        observation = phase_adapter.reconstruct_phase_batch(
            mask, tail, device=device
        )
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


class WarmedDataset:
    """Prepend the exact legal deployment warm sequence to a suffix dataset."""

    def __init__(self, suffix: Any) -> None:
        self.suffix = suffix
        prefix = load_warm_prefix()
        self.prefix_observation = np.asarray(
            prefix["visual_observation"], dtype=np.float32
        )
        self.prefix_action = np.asarray(prefix["suffix_action"], dtype=np.float32)
        self.prefix_steps = len(self.prefix_observation)
        self.agents = suffix.agents
        self.time_steps = self.prefix_steps + suffix.time_steps
        self.lengths = suffix.lengths + self.prefix_steps

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
        ):
            raise ValueError("invalid warmed sequence chunk")
        batch = len(indices)
        steps = end - start
        observation = torch.zeros(
            (batch, steps, PHASE_LEGAL_OBS_SIZE), device=device
        )
        action = torch.zeros((batch, steps, ACTION_SIZE), device=device)
        valid = torch.zeros((batch, steps), dtype=torch.bool, device=device)
        gate2 = torch.zeros_like(valid)
        prefix_end = min(end, self.prefix_steps)
        if start < prefix_end:
            source = slice(start, prefix_end)
            destination = slice(0, prefix_end - start)
            prefix_observation = torch.from_numpy(
                self.prefix_observation[source].copy()
            ).to(device)
            prefix_action = torch.from_numpy(self.prefix_action[source].copy()).to(
                device
            )
            observation[:, destination] = prefix_observation.unsqueeze(0)
            action[:, destination] = prefix_action.unsqueeze(0)
            valid[:, destination] = True
        suffix_start = max(start, self.prefix_steps)
        if suffix_start < end:
            supplied = self.suffix.chunk(
                indices,
                suffix_start - self.prefix_steps,
                end - self.prefix_steps,
                device=device,
            )
            destination = slice(suffix_start - start, end - start)
            for target, values in zip(
                (observation, action, valid, gate2), supplied, strict=True
            ):
                target[:, destination] = values
        return observation, action, valid, gate2


def load_sources(*, verify_hashes: bool = True) -> tuple[Any, ...]:
    anchors = c007._load_sources(verify_c006_hashes=verify_hashes)[:3]
    c006_true = WarmedDataset(
        c007.C006PublicPhaseDataset(verify_hashes=verify_hashes)
    )
    c006_alias = WarmedDataset(
        c007.C006PublicPhaseDataset(verify_hashes=False)
    )
    c011_true = WarmedDataset(C011PublicPhaseDataset(verify_hashes=verify_hashes))
    c011_alias = WarmedDataset(C011PublicPhaseDataset(verify_hashes=False))
    return (*anchors, c006_true, c006_alias, c011_true, c011_alias)


class AggregateWarmedDataset:
    """Seven equally weighted, intact 64-agent recurrent source groups."""

    def __init__(self, *, verify_hashes: bool = True) -> None:
        self.sources = load_sources(verify_hashes=verify_hashes)
        if len(self.sources) != len(GROUP_NAMES):
            raise RuntimeError("aggregate source count changed")
        self.agents = GROUP_AGENTS * len(self.sources)
        self.time_steps = max(source.time_steps for source in self.sources)
        self.source_for_agent = np.tile(
            np.arange(len(self.sources), dtype=np.int64), GROUP_AGENTS
        )
        self.local_agent = np.repeat(
            np.arange(GROUP_AGENTS, dtype=np.int64), len(self.sources)
        )
        self.local_agent[self.source_for_agent == 4] += GROUP_AGENTS
        self.local_agent[self.source_for_agent == 6] += GROUP_AGENTS
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
        if start < 0 or end <= start or end > self.time_steps:
            raise ValueError("invalid aggregate sequence chunk")
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
            supplied = source.chunk(
                selected_locals[positions], start, source_end, device=device
            )
            width = source_end - start
            position_tensor = torch.from_numpy(positions).to(device)
            for destination, values in zip(
                (observation, action, valid, gate2), supplied, strict=True
            ):
                destination[position_tensor, :width] = values
        return observation, action, valid, gate2


def dataset_factory() -> AggregateWarmedDataset:
    return AggregateWarmedDataset()


def configure_trainer() -> None:
    trainer.TAG = TAG
    trainer.SCHEMA = c007.SCHEMA
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.PREREGISTRATION = PREREGISTRATION
    trainer.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    trainer.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    trainer.PARENT_REPORT = PARENT_REPORT
    trainer.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    trainer.DATASET = c007.ORACLE_DATASET
    trainer.DATASET_REPORT_SHA256 = c007.ORACLE_REPORT_SHA256
    trainer.DATASET_METADATA_SHA256 = c007.ORACLE_METADATA_SHA256
    trainer.PublicPhaseDataset = dataset_factory


def source_audits(
    checkpoint_path: Path, *, device: torch.device
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VQ2PhaseResidualActor(hidden_size=256, initial_std=0.15).to(device)
    model.load_state_dict(checkpoint["model_state"])
    sources = load_sources(verify_hashes=False)
    held_out = np.arange(56, 64)
    audits = {
        GROUP_NAMES[index]: trainer.evaluate_partitions(
            model, sources[index], held_out, CONFIG, device
        )
        for index in range(3)
    }
    for index in range(3, len(sources)):
        local = held_out + (GROUP_AGENTS if index in (4, 6) else 0)
        audits[GROUP_NAMES[index]] = phase_adapter.evaluate_gate2(
            model, sources[index], local, CONFIG, device
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
    frozen = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        C011_DATASET / "report.json": C011_REPORT_SHA256,
        C011_DATASET / "metadata.json": C011_METADATA_SHA256,
        c007.C006_DATASET / "report.json": c007.C006_REPORT_SHA256,
        c007.C006_DATASET / "metadata.json": c007.C006_METADATA_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    configure_trainer()
    report = trainer.train(output_path=output, device_name=device_name, config=CONFIG)
    checkpoint_path = output / "policy_best.pt"
    audits = source_audits(checkpoint_path, device=torch.device(device_name))
    audits_passed = source_audits_pass(audits)
    wrapper = Path(__file__).resolve()
    added_sources = {
        str(wrapper.relative_to(ROOT)): sha256_path(wrapper),
        **{
            str(path.relative_to(ROOT)): expected for path, expected in frozen.items()
        },
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["source_sha256"].update(added_sources)
    checkpoint["source_balancing"] = {
        "groups": list(GROUP_NAMES),
        "logical_agents": GROUP_AGENTS * len(GROUP_NAMES),
        "validation_agents_per_group": 8,
        "source_specific_audits": audits,
        "source_specific_audits_passed": audits_passed,
    }
    checkpoint["combined_numerical_admission"] = bool(
        report["numerically_admitted"] and audits_passed
    )
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"].update(added_sources)
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
