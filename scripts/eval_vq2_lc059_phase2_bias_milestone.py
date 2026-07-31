#!/usr/bin/env python3
"""Corrected LC058 paired Gate-3 milestone bias screen."""

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


TAG = "vq2_lc059_phase2_bias_milestone_001"
SCHEMA = "vq2_lc059_phase2_bias_milestone_report_v1"
REJECTION = ROOT / "docs/vq2_lc058_phase2_bias_milestone_rejection_2026-07-31.json"
REJECTION_SHA256 = "a18f160053721d6bf0ce6eae16004fbe4459753e28a6b6859668ae4df3c4f72a"
PREREGISTRATION = ROOT / "docs/vq2_lc059_phase2_bias_milestone_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc059_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc058_phase2_bias_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_BASE_VERIFY_INPUTS = base.verify_inputs


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), REJECTION,
        ROOT / "docs/vq2_lc058_phase2_bias_milestone_preregistration_2026-07-31.md",
    )


def verify_inputs() -> dict[str, Any]:
    payload = _BASE_VERIFY_INPUTS()
    if sha256_path(REJECTION) != REJECTION_SHA256:
        raise RuntimeError("LC059 bound LC058 rejection changed")
    rejection = json.loads(REJECTION.read_text())
    if (
        rejection.get("schema") != "vq2_lc058_phase2_bias_milestone_rejection_v1"
        or not rejection.get("rejected")
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("terminal_report_emitted")
        or rejection.get("minimum_observed_wall_time_seconds", 0.0) < 23.0
        or rejection.get("flight_sim_packets_sent") != 0
        or rejection.get("submission_authorized")
    ):
        raise RuntimeError("LC058 rejection does not authorize LC059")
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
