#!/usr/bin/env python3
"""Train the aggregate actor with a causal cap on only SF024 failure tails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_vq2_recurrent_dagger_aggregate as aggregate
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_sf028_recurrent_dagger_causal_prefix_001"
SCHEMA = "vq2_recurrent_dagger_causal_prefix_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf028_recurrent_dagger_causal_prefix_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf026_recurrent_dagger_aggregate_continuation_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "166c9b28bc1d4848e0e5e6d7028f9637835540cbe2f8e0ae883a8962d8e0e2b7"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "a1df63ceb0bf7fe85cc50a0010ed32d08395d76bc4f162b38aa900a67c643c03"
)
HORIZON_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf027_dagger_causal_horizon/report.json"
)
HORIZON_REPORT_SHA256 = (
    "527ab1be544634fa67bb9b411cd450f73552d3aee935e4b766cae3fb91c38809"
)
CAUSAL_HORIZON_STEPS = 384
DAGGER_SOURCE_PATTERN = ("broad", "next", "next")
CONFIG = aggregate.AggregateTrainConfig(
    seed=42028,
    epochs=4,
    learning_rate=2e-5,
)
_ORIGINAL_DATASET = aggregate.OracleBCDataset


class CausalHorizonDataset:
    """Read-only recurrent prefix view over an admitted legal dataset."""

    def __init__(self, dataset, *, horizon: int = CAUSAL_HORIZON_STEPS) -> None:
        if horizon <= 0 or horizon > dataset.time_steps:
            raise ValueError("causal horizon is outside the dataset")
        self.dataset = dataset
        self.root = dataset.root
        self.report = dataset.report
        self.metadata = dataset.metadata
        self.agents = dataset.agents
        self.time_steps = horizon
        self.lengths = np.minimum(dataset.lengths, horizon)

    def chunk(self, agent_indices, start, end, *, device):
        if end > self.time_steps:
            raise ValueError("chunk exceeds the causal horizon")
        return self.dataset.chunk(agent_indices, start, end, device=device)


def dataset_factory(root: Path, **kwargs):
    dataset = _ORIGINAL_DATASET(root, **kwargs)
    if Path(root).resolve() == aggregate.NEXT_DATASET.resolve():
        return CausalHorizonDataset(dataset)
    return dataset


def configure_aggregate_module() -> None:
    aggregate.TAG = TAG
    aggregate.SCHEMA = SCHEMA
    aggregate.PREREGISTRATION = PREREGISTRATION
    aggregate.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    aggregate.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    aggregate.PARENT_REPORT = PARENT_REPORT
    aggregate.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    aggregate.DAGGER_SOURCE_PATTERN = DAGGER_SOURCE_PATTERN
    aggregate.OracleBCDataset = dataset_factory


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda"
) -> dict:
    if sha256_path(HORIZON_REPORT) != HORIZON_REPORT_SHA256:
        raise RuntimeError("SF027 causal-horizon report hash mismatch")
    horizon_report = json.loads(HORIZON_REPORT.read_text())
    if horizon_report["horizons"][str(CAUSAL_HORIZON_STEPS)][
        "weighted_mse"
    ] > 0.02:
        raise RuntimeError("SF027 did not admit the causal prefix hypothesis")
    configure_aggregate_module()
    report = aggregate.train(
        output=output,
        device_name=device_name,
        config=CONFIG,
    )

    provenance_paths = [Path(__file__).resolve(), HORIZON_REPORT]
    provenance = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in provenance_paths
    }
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint["source_sha256"].update(provenance)
    checkpoint["causal_horizon_steps"] = CAUSAL_HORIZON_STEPS
    checkpoint["dagger_source_pattern"] = list(DAGGER_SOURCE_PATTERN)
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"].update(provenance)
    report["causal_horizon_steps"] = CAUSAL_HORIZON_STEPS
    report["dagger_source_pattern"] = list(DAGGER_SOURCE_PATTERN)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(json.dumps(train(output=args.output.resolve(), device_name=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
