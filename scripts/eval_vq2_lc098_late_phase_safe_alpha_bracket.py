#!/usr/bin/env python3
"""Refine the crash-free LC097 late-phase interpolation boundary."""

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
import scripts.eval_vq2_lc097_late_phase_endpoint_local_bracket as base


TAG = "vq2_lc098_late_phase_safe_alpha_bracket_001"
SCHEMA = "vq2_lc098_late_phase_safe_alpha_bracket_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc098_late_phase_safe_alpha_bracket_checkpoint_v1"
ALPHAS = (0.0, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
GROUP_SIZE = 96
SEED = 431_980
MINIMUM_MEAN_ADVANCE_GAIN = 0.03
MINIMUM_ONE_GATE_PASS_GAIN = 2
LC097_REPORT = base.DEFAULT_OUTPUT / "report.json"
LC097_REPORT_SHA256 = "35a1eb3132848b867fd17d37a429a9cc550d1d91f6d04beb0c822dac11dba3b4"
PREREGISTRATION = ROOT / "docs/vq2_lc098_late_phase_safe_alpha_bracket_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc098_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc098_late_phase_safe_alpha_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_BASE_VERIFY_INPUTS = base.verify_inputs
_BASE_SOURCE_IDENTITY = base.source_identity


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = _BASE_VERIFY_INPUTS()
    if sha256_path(LC097_REPORT) != LC097_REPORT_SHA256:
        raise RuntimeError("LC098 bound LC097 report changed")
    report = json.loads(LC097_REPORT.read_text())
    by_alpha = {item["alpha"]: item for item in report.get("items", [])}
    safe = by_alpha.get(0.1, {})
    unsafe = by_alpha.get(0.3, {})
    if (
        report.get("schema") != "vq2_lc097_late_phase_endpoint_local_bracket_report_v1"
        or not report.get("diagnostic_valid")
        or report.get("numerically_admitted")
        or report.get("selected_candidate") is not None
        or safe.get("mean_gate_advance") != 0.375
        or safe.get("one_gate_passes") != 19
        or safe.get("crash_rate") != 0.0
        or unsafe.get("mean_gate_advance") != 0.59375
        or unsafe.get("crash_rate") != 0.015625
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC097 does not authorize the safe-boundary bracket")
    return inputs


def source_identity() -> dict[str, Any]:
    identity = _BASE_SOURCE_IDENTITY()
    identity["source_sha256"].update({
        str(Path(__file__).relative_to(ROOT)): sha256_path(Path(__file__)),
        str(LC097_REPORT.relative_to(ROOT)): sha256_path(LC097_REPORT),
    })
    return identity


def configure() -> None:
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.ALPHAS = ALPHAS
    base.GROUP_SIZE = GROUP_SIZE
    base.GROUPS = len(ALPHAS)
    base.TOTAL_AGENTS = GROUP_SIZE * base.GROUPS
    base.EPISODES = base.TOTAL_AGENTS
    base.SEED = SEED
    base.MINIMUM_MEAN_ADVANCE_GAIN = MINIMUM_MEAN_ADVANCE_GAIN
    base.MINIMUM_ONE_GATE_PASS_GAIN = MINIMUM_ONE_GATE_PASS_GAIN
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = base.verify_inputs, base.source_identity
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        base.verify_inputs, base.source_identity = originals


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
