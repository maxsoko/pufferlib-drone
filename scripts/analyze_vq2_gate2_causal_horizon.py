#!/usr/bin/env python3
"""Localize SF033 imitation error over the SF035 Gate-2 failure horizon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.analyze_vq2_dagger_causal_horizon as analysis
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_sf036_gate2_causal_horizon"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf036_gate2_causal_horizon_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf033_recurrent_roll_priority_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "91088ca96f432f57f1b85bbc521ddab29ba422fadf2b79b8d642ee426f09bea7"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "f54bf8512490a4629c08bf35ceae7c7fe07514e3424f89433f680536d96a0279"
)
DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf035_recurrent_dagger_gate2_512"
)
DATASET_REPORT_SHA256 = (
    "d07db36e11d321962defa217c80e06ff7678c8682f19ea6212d5fc8e3bb8313f"
)
DATASET_METADATA_SHA256 = (
    "8622f6c9be5cc15fbd46d9e5fd65d86b223406e820d658013f3478ebb8ec9851"
)
BINS = tuple((start, start + 256) for start in range(0, 2048, 256))
CAPS = tuple(end for _, end in BINS)


def configure_analysis_module() -> None:
    analysis.TAG = TAG
    analysis.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    analysis.PREREGISTRATION = PREREGISTRATION
    analysis.CHECKPOINT = CHECKPOINT
    analysis.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    analysis.TRAIN_REPORT = TRAIN_REPORT
    analysis.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    analysis.DATASET = DATASET
    analysis.DATASET_REPORT_SHA256 = DATASET_REPORT_SHA256
    analysis.DATASET_METADATA_SHA256 = DATASET_METADATA_SHA256
    analysis.BINS = BINS
    analysis.CAPS = CAPS


def analyze(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict:
    configure_analysis_module()
    report = analysis.analyze(output=output, device_name=device_name)
    wrapper_path = Path(__file__).resolve()
    wrapper_key = str(wrapper_path.relative_to(ROOT))
    wrapper_sha256 = sha256_path(wrapper_path)
    report["source_sha256"][wrapper_key] = wrapper_sha256
    report["analysis_wrapper_sha256"] = wrapper_sha256
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
            analyze(output=args.output.resolve(), device_name=args.device),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

