#!/usr/bin/env python3
"""Continue the SF037 Gate-2 causal fit at a lower learning rate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_vq2_recurrent_gate2_causal_fit as fit
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_sf038_recurrent_gate2_causal_continuation_001"
SCHEMA = "vq2_recurrent_gate2_causal_continuation_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT
    / "docs/vq2_sf038_recurrent_gate2_causal_continuation_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf037_recurrent_gate2_causal_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "0526cb27c8667784bc377cde88dfc7af015f9af8d15ae6dd6de0547517d76eaf"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "7be2c4f5c5f0e575001d189f0e11ffd7c6dd8dd987b9d85ea160bdf241fde811"
)
CONFIG = fit.aggregate.AggregateTrainConfig(
    seed=42038,
    epochs=8,
    learning_rate=2e-5,
    action_weights=(1.0, 1.0, 4.0, 1.0),
    loss_action_weights=(1.0, 4.0, 4.0, 1.0),
    prefer_admitted=True,
)


def configure_fit_module() -> None:
    fit.TAG = TAG
    fit.SCHEMA = SCHEMA
    fit.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    fit.PREREGISTRATION = PREREGISTRATION
    fit.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    fit.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    fit.PARENT_REPORT = PARENT_REPORT
    fit.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    fit.CONFIG = CONFIG


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict:
    configure_fit_module()
    report = fit.train(output=output, device_name=device_name)
    wrapper_path = Path(__file__).resolve()
    wrapper_key = str(wrapper_path.relative_to(ROOT))
    wrapper_sha256 = sha256_path(wrapper_path)
    checkpoint_path = output / "policy_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["source_sha256"][wrapper_key] = wrapper_sha256
    torch.save(checkpoint, checkpoint_path)
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report["source_sha256"][wrapper_key] = wrapper_sha256
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

