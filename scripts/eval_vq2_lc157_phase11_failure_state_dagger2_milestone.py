#!/usr/bin/env python3
"""Teacher-free repeated-source raw-12 screen of LC156 against LC153."""

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
import scripts.eval_vq2_lc154_phase11_failure_state_dagger_milestone as prior


BASE_CONFIGURE = prior.configure

TAG = "vq2_lc157_phase11_failure_state_dagger2_milestone_001"
SCHEMA = "vq2_lc157_phase11_failure_state_dagger2_milestone_report_v1"
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc153_phase11_failure_state_dagger_fit_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "d7d379e5bdc7fe2489cfa6ca58aec443a27dcfe47b3cf6b422fb165e6dbb1ff9"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "7dc5cde27a9a854577e58dc4eee55766c8c7f42d6cd16cefe5f3d852299ae485"
CANDIDATE_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc156_phase11_failure_state_dagger2_fit_001"
)
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "525747adc6700ca8a4fb22eeaa81c6533b55ca196b0164b795a778f2ff1c083b"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "3b795d9c4fec70856fb10166b55b26ba5b6222fd8da57f9c88d344aa2f6f9d65"
SELECTED_PARAMETER_DELTA_L2 = 3.4041691809558157
BIASES = (
    ("parent_lc153", (0.0, 0.0, 0.0, 0.0)),
    ("lc156_failure_state_dagger2", (0.0, 0.0, 0.0, 0.0)),
)
PREREGISTRATION = ROOT / "docs/vq2_lc157_phase11_failure_state_dagger2_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc157_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc157_phase11_failure_state_dagger2_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC157 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    candidate = torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    training = json.loads(CANDIDATE_REPORT.read_text())
    selected = training.get("selected", {})
    if (
        parent.get("schema") != "vq2_lc153_phase11_failure_state_dagger_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc153_phase11_failure_state_dagger_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or candidate.get("schema") != "vq2_lc156_phase11_failure_state_dagger2_fit_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or training.get("schema") != "vq2_lc156_phase11_failure_state_dagger2_fit_report_v1"
        or not training.get("numerically_admitted")
        or not training.get("frozen_non_phase11_state_exact")
        or training.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or selected.get("step") != 512
        or selected.get("scale") != 1.0
        or selected.get("parameter_delta_l2") != SELECTED_PARAMETER_DELTA_L2
        or selected.get("success_improvement_factor", 0.0) < 187.8
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC153/LC156 do not authorize LC157")
    for name in parent["model_state"]:
        if name in PARAMETER_NAMES:
            keep = torch.arange(parent["model_state"][name].shape[0]) != prior.TARGET_PHASE
            if (
                not torch.equal(
                    parent["model_state"][name][keep], candidate["model_state"][name][keep]
                )
                or torch.equal(
                    parent["model_state"][name][prior.TARGET_PHASE],
                    candidate["model_state"][name][prior.TARGET_PHASE],
                )
            ):
                raise RuntimeError(f"LC156 phase-11 isolation changed: {name}")
        elif not torch.equal(parent["model_state"][name], candidate["model_state"][name]):
            raise RuntimeError(f"LC156 changed frozen state: {name}")
    return parent


def configure() -> None:
    prior.TAG, prior.SCHEMA = TAG, SCHEMA
    prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    prior.CANDIDATE_CHECKPOINT, prior.CANDIDATE_CHECKPOINT_SHA256 = (
        CANDIDATE_CHECKPOINT, CANDIDATE_CHECKPOINT_SHA256,
    )
    prior.CANDIDATE_REPORT, prior.CANDIDATE_REPORT_SHA256 = (
        CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256,
    )
    prior.SELECTED_PARAMETER_DELTA_L2 = SELECTED_PARAMETER_DELTA_L2
    prior.BIASES = BIASES
    prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    BASE_CONFIGURE()
    prior.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        ROOT / "scripts/train_vq2_lc156_phase11_failure_state_dagger2_fit.py",
        ROOT / "scripts/collect_vq2_lc155_phase11_failure_state_dagger2.py",
    )
    prior.milestone.NEXT_AUTHORITY_SELECTED = (
        "Use LC156 only as the source-trajectory parent for phase-12 offline search; broad admission remains mandatory."
    )
    prior.milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC156 and continue phase-11 on-policy DAgger; do not run FlightSim."
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    originals = (prior.verify_inputs, prior.configure)
    prior.verify_inputs = verify_inputs
    try:
        configure()
        prior.configure = lambda: None
        return prior.run(output=output, device_name=device_name, resume=resume)
    finally:
        prior.verify_inputs, prior.configure = originals


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
