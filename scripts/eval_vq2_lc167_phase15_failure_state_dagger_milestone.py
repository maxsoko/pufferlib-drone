#!/usr/bin/env python3
"""Teacher-free repeated-source raw-16 screen of LC166 against LC162."""

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
from scripts.train_vq2_lc111_phase8_9_full_residual_endpoint import PARAMETER_NAMES
import scripts.eval_vq2_lc157_phase11_failure_state_dagger2_milestone as previous


BASE_PREVIOUS_CONFIGURE = previous.configure
TAG = "vq2_lc167_phase15_failure_state_dagger_milestone_001"
SCHEMA = "vq2_lc167_phase15_failure_state_dagger_milestone_report_v1"
TARGET_PHASE = 15
TARGET_RAW_INDEX = 16
MAX_STEPS = 27_000
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc162_phase14_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "4a45d1814e2f76b776ed2a278bd9a026dab5c1caf346792bc26032eea874770f"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "a85ae9c0da3c52110891d03cf67eafeebf76a3123b62a210d3740e541d32fabe"
CANDIDATE_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc166_phase15_failure_state_dagger_fit_001"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "b9a6fc73bd7eaa81987f436d33c44325b6d0f5aef9f3394da9e99a0d6f29e518"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "825cae461cf4b095e45b441e072df13adaf5435c424f78304d909cc06c698048"
SELECTED_PARAMETER_DELTA_L2 = 9.545120120398119
BIASES = (
    ("parent_lc162", (0.0, 0.0, 0.0, 0.0)),
    ("lc166_failure_state_dagger", (0.0, 0.0, 0.0, 0.0)),
)
PREREGISTRATION = ROOT / "docs/vq2_lc167_phase15_failure_state_dagger_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc167_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc167_phase15_failure_state_dagger_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC167 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    candidate = torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    training = json.loads(CANDIDATE_REPORT.read_text())
    selected = training.get("selected", {})
    if (
        parent.get("schema") != "vq2_lc162_phase14_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc162_phase14_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or candidate.get("schema") != "vq2_lc166_phase15_failure_state_dagger_fit_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or training.get("schema") != "vq2_lc166_phase15_failure_state_dagger_fit_report_v1"
        or not training.get("numerically_admitted")
        or not training.get("frozen_non_phase15_state_exact")
        or training.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or selected.get("step") != 512
        or selected.get("scale") != 1.0
        or selected.get("parameter_delta_l2") != SELECTED_PARAMETER_DELTA_L2
        or selected.get("success_improvement_factor", 0.0) < 189.5
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC162/LC166 do not authorize LC167")
    for name in parent["model_state"]:
        if name in PARAMETER_NAMES:
            keep = torch.arange(parent["model_state"][name].shape[0]) != TARGET_PHASE
            if (
                not torch.equal(parent["model_state"][name][keep], candidate["model_state"][name][keep])
                or torch.equal(parent["model_state"][name][TARGET_PHASE], candidate["model_state"][name][TARGET_PHASE])
            ):
                raise RuntimeError(f"LC166 phase-15 isolation changed: {name}")
        elif not torch.equal(parent["model_state"][name], candidate["model_state"][name]):
            raise RuntimeError(f"LC166 changed frozen state: {name}")
    return parent


def configure() -> None:
    previous.TAG, previous.SCHEMA = TAG, SCHEMA
    previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    previous.CANDIDATE_CHECKPOINT, previous.CANDIDATE_CHECKPOINT_SHA256 = CANDIDATE_CHECKPOINT, CANDIDATE_CHECKPOINT_SHA256
    previous.CANDIDATE_REPORT, previous.CANDIDATE_REPORT_SHA256 = CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256
    previous.SELECTED_PARAMETER_DELTA_L2 = SELECTED_PARAMETER_DELTA_L2
    previous.BIASES = BIASES
    previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    previous.prior.TARGET_PHASE = TARGET_PHASE
    previous.prior.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    previous.prior.MAX_STEPS = MAX_STEPS
    BASE_PREVIOUS_CONFIGURE()
    previous.prior.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        ROOT / "scripts/train_vq2_lc166_phase15_failure_state_dagger_fit.py",
        ROOT / "scripts/collect_vq2_lc165_phase15_failure_state_dagger.py",
    )
    previous.prior.milestone.NEXT_AUTHORITY_SELECTED = (
        "Use LC166 only as the phase-16 source-trajectory parent; broad admission remains mandatory."
    )
    previous.prior.milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC166 and run a second phase-15 on-policy DAgger iteration; do not run FlightSim."
    )
    previous.prior.milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form Puffer actors over 256 identical seed-15 rows; LC166 changes only phase-15 residual parameter rows"
    )


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    originals = (previous.verify_inputs, previous.configure)
    previous.verify_inputs = verify_inputs
    try:
        configure()
        previous.configure = lambda: None
        return previous.run(output=output, device_name=device_name, resume=resume)
    finally:
        previous.verify_inputs, previous.configure = originals


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
