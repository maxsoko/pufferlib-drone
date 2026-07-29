#!/usr/bin/env python3
"""Fit SF033 on the causal SF035 Gate-2 prefix with Gate-1 anchors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_vq2_recurrent_dagger_aggregate as aggregate
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_recurrent_dagger_causal_prefix import CausalHorizonDataset


TAG = "vq2_sf037_recurrent_gate2_causal_fit_001"
SCHEMA = "vq2_recurrent_gate2_causal_fit_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf037_recurrent_gate2_causal_fit_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf033_recurrent_roll_priority_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "91088ca96f432f57f1b85bbc521ddab29ba422fadf2b79b8d642ee426f09bea7"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "f54bf8512490a4629c08bf35ceae7c7fe07514e3424f89433f680536d96a0279"
)
HORIZON_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf036_gate2_causal_horizon/report.json"
)
HORIZON_REPORT_SHA256 = (
    "e81557cb912d8a01a9f4914be01b629828a9f04119c487cd23919f5962e39eb8"
)
BROAD_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf030_recurrent_dagger_causal_prefix_next_512"
)
BROAD_REPORT_SHA256 = (
    "7c4d0d632d03e8713c0c4bfbfc083e66b680aa98e6553ba17876c44cfe6f2393"
)
BROAD_METADATA_SHA256 = (
    "99ebb650731ab5dd725fed704c1703a8ded308464fe4c893713117038938e3d2"
)
NEXT_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf035_recurrent_dagger_gate2_512"
)
NEXT_REPORT_SHA256 = (
    "d07db36e11d321962defa217c80e06ff7678c8682f19ea6212d5fc8e3bb8313f"
)
NEXT_METADATA_SHA256 = (
    "8622f6c9be5cc15fbd46d9e5fd65d86b223406e820d658013f3478ebb8ec9851"
)
CAUSAL_HORIZON_STEPS = 768
DAGGER_SOURCE_PATTERN = ("broad", "next", "next")
CONFIG = aggregate.AggregateTrainConfig(
    seed=42037,
    epochs=6,
    learning_rate=5e-5,
    action_weights=(1.0, 1.0, 4.0, 1.0),
    loss_action_weights=(1.0, 4.0, 4.0, 1.0),
    prefer_admitted=True,
)
_ORIGINAL_DATASET = aggregate.OracleBCDataset


def dataset_factory(root: Path, **kwargs):
    dataset = _ORIGINAL_DATASET(root, **kwargs)
    if Path(root).resolve() == aggregate.NEXT_DATASET.resolve():
        return CausalHorizonDataset(dataset, horizon=CAUSAL_HORIZON_STEPS)
    return dataset


def configure_aggregate_module() -> None:
    aggregate.TAG = TAG
    aggregate.SCHEMA = SCHEMA
    aggregate.PREREGISTRATION = PREREGISTRATION
    aggregate.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    aggregate.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    aggregate.PARENT_REPORT = PARENT_REPORT
    aggregate.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    aggregate.BROAD_DATASET = BROAD_DATASET
    aggregate.BROAD_REPORT_SHA256 = BROAD_REPORT_SHA256
    aggregate.BROAD_METADATA_SHA256 = BROAD_METADATA_SHA256
    aggregate.NEXT_DATASET = NEXT_DATASET
    aggregate.NEXT_REPORT_SHA256 = NEXT_REPORT_SHA256
    aggregate.NEXT_METADATA_SHA256 = NEXT_METADATA_SHA256
    aggregate.DAGGER_SOURCE_PATTERN = DAGGER_SOURCE_PATTERN
    aggregate.OracleBCDataset = dataset_factory


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict:
    if sha256_path(HORIZON_REPORT) != HORIZON_REPORT_SHA256:
        raise RuntimeError("SF036 causal-horizon report hash mismatch")
    configure_aggregate_module()
    report = aggregate.train(output=output, device_name=device_name, config=CONFIG)
    provenance = {
        str(Path(__file__).resolve().relative_to(ROOT)): sha256_path(
            Path(__file__).resolve()
        ),
        str(HORIZON_REPORT.relative_to(ROOT)): HORIZON_REPORT_SHA256,
    }
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["source_sha256"].update(provenance)
    checkpoint["causal_horizon_steps"] = CAUSAL_HORIZON_STEPS
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"].update(provenance)
    report["causal_horizon_steps"] = CAUSAL_HORIZON_STEPS
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(
        json.dumps(
            train(output=args.output.resolve(), device_name=args.device),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

