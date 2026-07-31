#!/usr/bin/env python3
"""Larger independent-seed confirmation and refinement of the LC059 bias."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc058_phase2_bias_milestone as base


TAG = "vq2_lc060_phase2_bias_confirmation_001"
SCHEMA = "vq2_lc060_phase2_bias_confirmation_report_v1"
GROUP_SIZE = 64
BIAS_CANDIDATES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("baseline", (0.0, 0.0, 0.0, 0.0)),
    ("pitch_m0p00125", (-0.00125, 0.0, 0.0, 0.0)),
    ("pitch_m0p00250", (-0.00250, 0.0, 0.0, 0.0)),
    ("pitch_m0p00375", (-0.00375, 0.0, 0.0, 0.0)),
)
SEED = 431600
MINIMUM_PASS_GAIN = 2
LC059_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc059_phase2_bias_milestone_001/report.json"
)
LC059_REPORT_SHA256 = "0e2d070706cb8b6a88be11ebb4e020065afeca1161e8d5d908d32e83e906a926"
PREREGISTRATION = ROOT / "docs/vq2_lc060_phase2_bias_confirmation_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc060_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc058_phase2_bias_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_BASE_VERIFY_INPUTS = base.verify_inputs


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.GROUP_SIZE = GROUP_SIZE
    base.BIAS_CANDIDATES = BIAS_CANDIDATES
    base.GROUPS = len(BIAS_CANDIDATES)
    base.TOTAL_AGENTS = GROUP_SIZE * base.GROUPS
    base.EPISODES = base.TOTAL_AGENTS
    base.SEED = SEED
    base.MINIMUM_PASS_GAIN = MINIMUM_PASS_GAIN
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), LC059_REPORT)
    base.NEXT_AUTHORITY_SELECTED = (
        "Run one different-seed paired 24-gate full-course screen of LC048 and "
        "the selected whole-Puffer phase-2 bias; do not promote from the milestone alone."
    )
    base.NEXT_AUTHORITY_NONE = "Reject the LC059 bias family and retain LC048."


def verify_inputs() -> dict[str, Any]:
    payload = _BASE_VERIFY_INPUTS()
    if sha256_path(LC059_REPORT) != LC059_REPORT_SHA256:
        raise RuntimeError("LC060 bound LC059 report changed")
    report = json.loads(LC059_REPORT.read_text())
    selected = report.get("causal_screen_selected", {})
    if (
        report.get("schema") != "vq2_lc059_phase2_bias_milestone_report_v1"
        or not report.get("diagnostic_valid")
        or report.get("numerically_admitted")
        or report.get("initial_seed_groups_exact") is not True
        or report.get("wall_time_seconds", 999.0) > 25.0
        or selected.get("bias") != [-0.0025, 0.0, 0.0, 0.0]
        or selected.get("gate3_passes") != 8
        or report.get("items", [{}])[0].get("gate3_passes") != 5
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC059 does not authorize LC060 confirmation")
    return payload


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    configure()
    base.verify_inputs = verify_inputs
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        base.verify_inputs = _BASE_VERIFY_INPUTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
