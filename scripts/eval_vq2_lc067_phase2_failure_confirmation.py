#!/usr/bin/env python3
"""Larger independent confirmation of failure-conditioned phase-2 pitch."""

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
import scripts.eval_vq2_lc066_phase2_failure_conditioned as base


TAG = "vq2_lc067_phase2_failure_confirmation_001"
SCHEMA = "vq2_lc067_phase2_failure_confirmation_report_v1"
GROUP_SIZE = 64
COEFFICIENTS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("baseline_lc062", (0.0, 0.0, 0.0, 0.0)),
    ("failure_pitch_m0p0075", (-0.0075, 0.0, 0.0, 0.0)),
    ("failure_pitch_m0p0100", (-0.0100, 0.0, 0.0, 0.0)),
    ("failure_pitch_m0p0150", (-0.0150, 0.0, 0.0, 0.0)),
)
SEED = 431670
MINIMUM_PASS_GAIN = 2
LC066_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc066_phase2_failure_conditioned_001/report.json"
)
LC066_REPORT_SHA256 = "4d7cbfa88e007dbde99b32a94199df5374ec50bf1502e863c0f9629f5a039413"
PREREGISTRATION = ROOT / "docs/vq2_lc067_phase2_failure_confirmation_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc067_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc067_phase2_failure_confirmation.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_BASE_VERIFY_INPUTS = base.verify_inputs


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.GROUP_SIZE = GROUP_SIZE
    base.COEFFICIENTS = COEFFICIENTS
    base.SEED = SEED
    base.MINIMUM_PASS_GAIN = MINIMUM_PASS_GAIN
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), LC066_REPORT)
    base.NEXT_AUTHORITY_SELECTED = (
        "Run one fresh-seed paired 24-gate full-course screen of LC062 and the "
        "selected failure-conditioned whole-Puffer candidate."
    )
    base.NEXT_AUTHORITY_NONE = "Reject the failure-conditioned pitch family and retain LC062."


def verify_inputs() -> dict[str, Any]:
    payload = _BASE_VERIFY_INPUTS()
    if sha256_path(LC066_REPORT) != LC066_REPORT_SHA256:
        raise RuntimeError("LC067 bound LC066 report changed")
    report = json.loads(LC066_REPORT.read_text())
    selected = report.get("causal_screen_selected", {})
    if (
        report.get("schema") != "vq2_lc066_phase2_failure_conditioned_report_v1"
        or not report.get("diagnostic_valid")
        or selected.get("failure_action_coefficient") != [-0.01, 0.0, 0.0, 0.0]
        or selected.get("gate3_passes") != 9
        or selected.get("paired_gate3_gains_vs_baseline") != 2
        or selected.get("paired_gate3_losses_vs_baseline") != 1
        or report.get("items", [{}])[0].get("gate3_passes") != 8
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC066 does not authorize LC067")
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
