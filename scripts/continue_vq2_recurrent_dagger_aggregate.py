#!/usr/bin/env python3
"""Continue the frozen SF025 aggregate fit from its selected checkpoint."""

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


TAG = "vq2_sf026_recurrent_dagger_aggregate_continuation_001"
SCHEMA = "vq2_recurrent_dagger_aggregate_continuation_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT
    / "docs/vq2_sf026_recurrent_dagger_aggregate_continuation_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf025_recurrent_dagger_aggregate_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "d0a4058fba5428b35514b3a7c74ab73f7f235e0fc4ac45f062d87591cda5523c"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "3b4d0bf47334f1e3576aced14c68178cd6ac9dab19a2ee6de3d27422cb345118"
)
CONFIG = aggregate.AggregateTrainConfig(
    seed=42026,
    epochs=6,
    learning_rate=5e-5,
)


def configure_aggregate_module() -> None:
    """Bind the frozen generic SF025 loop to the SF026 continuation lineage."""

    aggregate.TAG = TAG
    aggregate.SCHEMA = SCHEMA
    aggregate.PREREGISTRATION = PREREGISTRATION
    aggregate.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    aggregate.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    aggregate.PARENT_REPORT = PARENT_REPORT
    aggregate.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda"
) -> dict:
    configure_aggregate_module()
    report = aggregate.train(
        output=output,
        device_name=device_name,
        config=CONFIG,
    )

    wrapper_key = str(Path(__file__).resolve().relative_to(ROOT))
    wrapper_sha256 = sha256_path(Path(__file__).resolve())
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint["source_sha256"][wrapper_key] = wrapper_sha256
    checkpoint["continuation_wrapper_sha256"] = wrapper_sha256
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"][wrapper_key] = wrapper_sha256
    report["continuation_wrapper_sha256"] = wrapper_sha256
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
