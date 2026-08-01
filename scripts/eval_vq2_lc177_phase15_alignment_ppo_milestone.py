#!/usr/bin/env python3
"""Teacher-free deterministic raw-16 screen of LC175 against LC173."""

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
import scripts.eval_vq2_lc167_phase15_failure_state_dagger_milestone as previous


BASE_PREVIOUS_CONFIGURE = previous.configure
TAG = "vq2_lc177_phase15_alignment_ppo_milestone_001"
SCHEMA = "vq2_lc177_phase15_alignment_ppo_milestone_report_v1"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc173_phase15_failure_state_dagger3_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "e3c65e89ce928bff59ffe05a7b01f2f9abde7467d728c791944b33d7f3924958"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "a48a4514e2c0240cb0f375357d6299a6cdeb4bb9162e8223862017d2f3a1ec7d"
CANDIDATE_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc175_phase15_alignment_return_ppo_001"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_rollout_03.pt"
CANDIDATE_CHECKPOINT_SHA256 = "d4f59e06dc98a80c01b28097be38f9e80af61d725f31f887a911058acb09d950"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "87bba2917e8a9e63d6370dbaa5cbc3be3a5aaa046faa8b732c70bb8e19f20e2f"
SELECTED_PARAMETER_DELTA_L2 = 0.03156687903502732
BIASES = (
    ("parent_lc173", (0.0, 0.0, 0.0, 0.0)),
    ("lc175_alignment_ppo_mean", (0.0, 0.0, 0.0, 0.0)),
)
PREREGISTRATION = ROOT / "docs/vq2_lc177_phase15_alignment_ppo_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc177_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc177_phase15_alignment_ppo_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC177 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    candidate = torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    training = json.loads(CANDIDATE_REPORT.read_text())
    selected = training.get("candidate_selected_for_screen", {})
    selected_metrics = training.get("selected_rollout_metrics", {})
    if (
        parent.get("schema") != "vq2_lc173_phase15_failure_state_dagger3_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc173_phase15_failure_state_dagger3_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or candidate.get("schema") != "vq2_lc175_phase15_alignment_return_ppo_checkpoint_v1"
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or candidate.get("onpolicy_rollout_iteration") != 3
        or training.get("schema") != "vq2_lc175_phase15_alignment_return_ppo_report_v1"
        or not training.get("training_admitted")
        or not training.get("frozen_non_target_phase_state_exact")
        or selected.get("sha256") != CANDIDATE_CHECKPOINT_SHA256
        or selected.get("iteration") != 3
        or selected_metrics.get("target_passes") != 6
        or len(training.get("updates", [])) != 2
        or any(not item.get("update_admitted") for item in training.get("updates", []))
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC173/LC175 do not authorize LC177")
    for name in parent["model_state"]:
        if name in PARAMETER_NAMES:
            keep = torch.arange(parent["model_state"][name].shape[0]) != previous.TARGET_PHASE
            if (
                not torch.equal(parent["model_state"][name][keep], candidate["model_state"][name][keep])
                or torch.equal(parent["model_state"][name][previous.TARGET_PHASE], candidate["model_state"][name][previous.TARGET_PHASE])
            ):
                raise RuntimeError(f"LC175 phase-15 isolation changed: {name}")
        elif not torch.equal(parent["model_state"][name], candidate["model_state"][name]):
            raise RuntimeError(f"LC175 changed frozen state: {name}")
    return parent


def configure() -> None:
    previous.TAG, previous.SCHEMA = TAG, SCHEMA
    previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    previous.CANDIDATE_CHECKPOINT, previous.CANDIDATE_CHECKPOINT_SHA256 = CANDIDATE_CHECKPOINT, CANDIDATE_CHECKPOINT_SHA256
    previous.CANDIDATE_REPORT, previous.CANDIDATE_REPORT_SHA256 = CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256
    previous.SELECTED_PARAMETER_DELTA_L2 = SELECTED_PARAMETER_DELTA_L2
    previous.BIASES = BIASES
    previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    BASE_PREVIOUS_CONFIGURE()
    previous.previous.prior.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        ROOT / "scripts/train_vq2_lc175_phase15_alignment_return_ppo.py",
    )
    previous.previous.prior.milestone.NEXT_AUTHORITY_SELECTED = (
        "Use LC175 only as the phase-16 source-trajectory parent; broad admission remains mandatory."
    )
    previous.previous.prior.milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC175 and fit the preregistered phase-local recurrent adapter; do not run FlightSim."
    )
    previous.previous.prior.milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form Puffer actors over 256 identical seed-15 rows; LC175 changes only phase-15 residual parameter rows"
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
