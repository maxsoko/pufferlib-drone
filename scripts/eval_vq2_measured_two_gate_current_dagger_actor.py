#!/usr/bin/env python3
"""Run SF064 for the current-distribution DAgger checkpoint."""

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


TAG = "vq2_sf064_measured_two_gate_teacher_free_512"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT
    / "docs/vq2_sf064_measured_two_gate_teacher_free_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf063_current_dagger_fit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "c27c28bd26e937fc8023ceca6f8880fa6f3d4320bf6db55ce9837518b1868ebb"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "fa738aee5d5c79080d0e7ada1c69d8cce074ef5ec9ed03d8a5684a0c0357db2e"
)
SEED = 42064


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
    report["source_sha256"][str(wrapper.relative_to(ROOT))] = sha256_path(wrapper)
    report["screen_parent_tag"] = "vq2_sf063_current_dagger_fit_001"
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
