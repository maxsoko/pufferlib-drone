#!/usr/bin/env python3
"""Refit C007 with the exact legal Gate-1 recurrent warm context prepended."""

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

import scripts.train_vq2_measured_visual_suffix_dagger as base
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.eval_vq2_measured_visual_suffix_handoff import (
    PREFIX_REPORT,
    PREFIX_REPORT_SHA256,
    PREFIX_TRACE,
    PREFIX_TRACE_SHA256,
    load_warm_prefix,
)


TAG = "vq2_c009_measured_visual_suffix_warmed_dagger_fit_001"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT
    / "docs/vq2_c009_measured_visual_suffix_warmed_dagger_fit_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c007_measured_visual_suffix_dagger_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "393f5de6f9dd97ae81d855b23d77c96e97926507e7c2b89611714df3d60a24c5"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "bfc99b3d5177c5e83d9d46653480c26c66e05eebbc8d1d1bbc960bacd77c6e49"
)
CONFIG = replace(base.CONFIG, seed=43009)
UNWARMED_C006_DATASET = base.C006PublicPhaseDataset


class WarmedC006PublicPhaseDataset:
    """C006 with the exact 208-row legal deployment warm prefix."""

    def __init__(self, *, verify_hashes: bool = True) -> None:
        self.suffix = UNWARMED_C006_DATASET(verify_hashes=verify_hashes)
        prefix = load_warm_prefix()
        self.prefix_observation = np.asarray(
            prefix["visual_observation"], dtype=np.float32
        )
        self.prefix_action = np.asarray(prefix["suffix_action"], dtype=np.float32)
        if (
            self.prefix_observation.ndim != 2
            or self.prefix_observation.shape[1] != PHASE_LEGAL_OBS_SIZE
            or self.prefix_action.shape != (len(self.prefix_observation), 4)
            or not np.all(self.prefix_observation[:, -1] == 0.0)
        ):
            raise RuntimeError("measured warm prefix ABI changed")
        self.prefix_steps = len(self.prefix_observation)
        self.agents = self.suffix.agents
        self.time_steps = self.prefix_steps + self.suffix.time_steps
        self.lengths = self.suffix.lengths + self.prefix_steps
        self.root = self.suffix.root
        self.report = self.suffix.report
        self.metadata = self.suffix.metadata

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
            raise ValueError("invalid warmed C006 sequence chunk")
        batch = len(indices)
        steps = end - start
        observation = torch.zeros(
            (batch, steps, PHASE_LEGAL_OBS_SIZE), device=device
        )
        action = torch.zeros((batch, steps, 4), device=device)
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


def configure_base() -> None:
    base.TAG = TAG
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.PREREGISTRATION = PREREGISTRATION
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT = PARENT_REPORT
    base.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.CONFIG = CONFIG
    base.C006PublicPhaseDataset = WarmedC006PublicPhaseDataset


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict[str, Any]:
    frozen = {
        PREFIX_REPORT: PREFIX_REPORT_SHA256,
        PREFIX_TRACE: PREFIX_TRACE_SHA256,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    configure_base()
    report = base.train(output=output, device_name=device_name)
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    wrapper = Path(__file__).resolve()
    added_sources = {
        str(wrapper.relative_to(ROOT)): sha256_path(wrapper),
        **{
            str(path.relative_to(ROOT)): expected for path, expected in frozen.items()
        },
    }
    checkpoint["source_sha256"].update(added_sources)
    checkpoint["measured_warm_prefix"] = {
        "records": 208,
        "observation_source": str(PREFIX_TRACE.relative_to(ROOT)),
        "label_source": "frozen_sf066_unselected_puffer_output",
        "stored_training_only_privileged_values_per_record": 0,
    }
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"].update(added_sources)
    report["measured_warm_prefix"] = checkpoint["measured_warm_prefix"]
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
