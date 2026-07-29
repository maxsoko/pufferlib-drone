#!/usr/bin/env python3
"""Collect labels on SF053's exact measured-course near-miss states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_measured_two_gate_dagger as iteration
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_sf055_measured_two_gate_dagger_64"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf055_measured_two_gate_dagger_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf053_measured_two_gate_dagger_fit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "fba4069f590beebfaf74f42ff0fb5a4e5258026fe165baa7c8a0ad605a9a01f6"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "c6dfb13c00183e8cc12fcb4df913953807a42ae8ecb4de225bff63f3d63d6ab6"
)
SCREEN_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf054_measured_two_gate_teacher_free_512/report.json"
)
SCREEN_REPORT_SHA256 = (
    "c31233ad2eb72a84c3dc9e52034982086194d38f3b6b5d0e1bb66884626765b8"
)
SEED = 42055


def configure_iteration() -> None:
    iteration.TAG = TAG
    iteration.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    iteration.PREREGISTRATION = PREREGISTRATION
    iteration.CHECKPOINT = CHECKPOINT
    iteration.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    iteration.TRAIN_REPORT = TRAIN_REPORT
    iteration.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    iteration.SF051_REPORT = SCREEN_REPORT
    iteration.SF051_REPORT_SHA256 = SCREEN_REPORT_SHA256
    iteration.SEED = SEED


def collect(output: Path = DEFAULT_OUTPUT, *, device_name: str = "cuda") -> dict:
    configure_iteration()
    report = iteration.collect(output=output, device_name=device_name)
    wrapper = Path(__file__).resolve()
    key = str(wrapper.relative_to(ROOT))
    digest = sha256_path(wrapper)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_sha256"][key] = digest
    metadata["action"]["plant_action_source"] = "sf053_phase_actor_mean"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    report["source_sha256"][key] = digest
    report["metadata_sha256"] = sha256_path(metadata_path)
    report["parent_screen_report_sha256"] = SCREEN_REPORT_SHA256
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
            collect(args.output.resolve(), device_name=args.device),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
