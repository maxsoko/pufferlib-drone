#!/usr/bin/env python3
"""Fast multiaxis phase-2 bias screen from the promoted LC062 Puffer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc058_phase2_bias_milestone as base


TAG = "vq2_lc063_phase2_bias_multiaxis_001"
SCHEMA = "vq2_lc063_phase2_bias_multiaxis_report_v1"
GROUP_SIZE = 32
# Deltas below are additional to LC062's retained [-0.0025, 0, 0, 0].
BIAS_CANDIDATES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("baseline_lc062", (0.0, 0.0, 0.0, 0.0)),
    ("pitch_m0p00125", (-0.00125, 0.0, 0.0, 0.0)),
    ("pitch_p0p00125", (0.00125, 0.0, 0.0, 0.0)),
    ("roll_m0p00250", (0.0, -0.00250, 0.0, 0.0)),
    ("thrust_p0p00250", (0.0, 0.0, 0.00250, 0.0)),
    ("thrust_m0p00250", (0.0, 0.0, -0.00250, 0.0)),
    ("yaw_p0p00250", (0.0, 0.0, 0.0, 0.00250)),
    ("pitch_m_roll_m_thrust_p", (-0.00125, -0.00250, 0.00250, 0.0)),
)
SEED = 431630
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc062_phase2_bias_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "06381161445f5f207a3b12f7c97b82c884f91a71fdb3e34e05189255cc58d04a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "5bf0fc1f95fec795dcf0c8212f7838a0f18a683baaece679228e87e667085af5"
PREREGISTRATION = ROOT / "docs/vq2_lc063_phase2_bias_multiaxis_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc063_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc063_phase2_bias_multiaxis.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.GROUP_SIZE = GROUP_SIZE
    base.BIAS_CANDIDATES = BIAS_CANDIDATES
    base.GROUPS = len(BIAS_CANDIDATES)
    base.TOTAL_AGENTS = GROUP_SIZE * base.GROUPS
    base.EPISODES = base.TOTAL_AGENTS
    base.SEED = SEED
    base.MINIMUM_PASS_GAIN = MINIMUM_PASS_GAIN
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT = PARENT_REPORT
    base.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.PARENT_CHILD_REPORT = PARENT_REPORT
    base.PARENT_CHILD_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)
    base.NEXT_AUTHORITY_SELECTED = (
        "Run one larger different-seed Gate-3 milestone confirmation of the selected "
        "LC062-plus-bias whole-Puffer candidate."
    )
    base.NEXT_AUTHORITY_NONE = "Reject the multiaxis family and retain LC062."


def verify_inputs() -> dict[str, Any]:
    if sha256_path(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("LC063 parent checkpoint changed")
    if sha256_path(PARENT_REPORT) != PARENT_REPORT_SHA256:
        raise RuntimeError("LC063 parent report changed")
    report = json.loads(PARENT_REPORT.read_text())
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        report.get("schema") != "vq2_lc062_phase2_bias_full_course_report_v1"
        or not report.get("numerically_admitted")
        or report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or report.get("mean_gate_gain") != 0.0625
        or report.get("gate3_pass_gain") != 1
        or report.get("selected_candidate", {}).get("bias") != [-0.0025, 0.0, 0.0, 0.0]
        or payload.get("schema") != "vq2_lc062_phase2_bias_full_course_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or payload.get("phase2_bias_surgery", {}).get("pre_tanh_output_bias_delta")
        != [-0.0025, 0.0, 0.0, 0.0]
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC062 does not authorize LC063")
    return payload


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    configure()
    original_verify = base.verify_inputs
    base.verify_inputs = verify_inputs
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        base.verify_inputs = original_verify


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
