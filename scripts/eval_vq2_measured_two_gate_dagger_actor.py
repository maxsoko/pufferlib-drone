#!/usr/bin/env python3
"""Run the fresh SF054 teacher-free screen for the SF053 DAgger actor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_vq2_measured_two_gate_phase_actor as screen
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_sf054_measured_two_gate_teacher_free_512"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT
    / "docs/vq2_sf054_measured_two_gate_teacher_free_preregistration_2026-07-28.md"
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
SEED = 42054


def configure_screen() -> None:
    screen.TAG = TAG
    screen.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    screen.PREREGISTRATION = PREREGISTRATION
    screen.CHECKPOINT = CHECKPOINT
    screen.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    screen.TRAIN_REPORT = TRAIN_REPORT
    screen.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    screen.SEED = SEED


def run_screen(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict:
    configure_screen()
    report = screen.run_screen(output=output, device_name=device_name)
    wrapper = Path(__file__).resolve()
    key = str(wrapper.relative_to(ROOT))
    report["source_sha256"][key] = sha256_path(wrapper)
    report["screen_parent_tag"] = "vq2_sf053_measured_two_gate_dagger_fit_001"
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    report = run_screen(output=args.output.resolve(), device_name=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["two_gate_milestone_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
