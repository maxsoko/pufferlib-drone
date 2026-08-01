#!/usr/bin/env python3
"""Teacher-free repeated-source raw-15 screen of LC162 against LC160."""

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
import scripts.eval_vq2_lc154_phase11_failure_state_dagger_milestone as base


BASE_CONFIGURE = base.configure
TAG = "vq2_lc163_phase14_cem_milestone_001"
SCHEMA = "vq2_lc163_phase14_cem_milestone_report_v1"
TARGET_PHASE = 14
TARGET_RAW_INDEX = 15
MAX_STEPS = 24_000
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc160_phase13_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "8ba6516139e6eb2c50082728003066308f85bfe81d76446fb883241b3107ac57"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c11e7068f7fdf49a34b31ec10f9597c08ff77fcfd7b069cdff2cfac018c2325d"
CANDIDATE_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc162_phase14_split_batch_cem_001"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "4a45d1814e2f76b776ed2a278bd9a026dab5c1caf346792bc26032eea874770f"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "a85ae9c0da3c52110891d03cf67eafeebf76a3123b62a210d3740e541d32fabe"
SELECTED_DELTA = (-0.09961516410112381, 0.027437731623649597, -0.04881178215146065, -0.011450924910604954)
SELECTED_PARAMETER_DELTA_L2 = sum(value * value for value in SELECTED_DELTA) ** 0.5
BIASES = (
    ("parent_lc160", (0.0, 0.0, 0.0, 0.0)),
    ("lc162_phase14_cem", SELECTED_DELTA),
)
PREREGISTRATION = ROOT / "docs/vq2_lc163_phase14_cem_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc163_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc163_phase14_cem_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC163 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    candidate = torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    training = json.loads(CANDIDATE_REPORT.read_text())
    generation = training.get("generations", [{}])[0]
    if (
        parent.get("schema") != "vq2_lc160_phase13_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("candidate_selected_for_screen", {}).get("sha256")
        != PARENT_CHECKPOINT_SHA256
        or candidate.get("schema") != "vq2_lc162_phase14_split_batch_cem_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or not candidate.get("frozen_non_phase14_state_exact")
        or tuple(candidate.get("phase14_pre_tanh_output_bias_delta", ())) != SELECTED_DELTA
        or training.get("schema") != "vq2_lc162_phase14_split_batch_cem_report_v1"
        or not training.get("training_admitted")
        or training.get("candidate_selected_for_screen", {}).get("sha256")
        != CANDIDATE_CHECKPOINT_SHA256
        or generation.get("query_agents") != 512
        or generation.get("target_passes") != 2
        or not generation.get("transport_pass")
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC160/LC162 do not authorize LC163")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.MAX_STEPS = (
        TARGET_PHASE, TARGET_RAW_INDEX, MAX_STEPS,
    )
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.CANDIDATE_CHECKPOINT, base.CANDIDATE_CHECKPOINT_SHA256 = (
        CANDIDATE_CHECKPOINT, CANDIDATE_CHECKPOINT_SHA256,
    )
    base.CANDIDATE_REPORT, base.CANDIDATE_REPORT_SHA256 = (
        CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256,
    )
    base.SELECTED_PARAMETER_DELTA_L2, base.BIASES = SELECTED_PARAMETER_DELTA_L2, BIASES
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    BASE_CONFIGURE()
    base.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        ROOT / "scripts/train_vq2_lc162_phase14_split_batch_cem.py",
    )
    base.milestone.NEXT_AUTHORITY_SELECTED = (
        "Use LC162 as the phase-15 source-trajectory parent; broad admission remains mandatory."
    )
    base.milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC162 and collect a phase-14 state-dependent rescue; do not run FlightSim."
    )
    base.milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form Puffer actors over 256 identical seed-15 rows; LC162 changes only phase-14 output bias"
    )


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    originals = (base.verify_inputs, base.configure)
    base.verify_inputs = verify_inputs
    try:
        configure()
        base.configure = lambda: None
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        base.verify_inputs, base.configure = originals


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
