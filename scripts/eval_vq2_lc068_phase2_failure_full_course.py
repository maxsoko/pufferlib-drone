#!/usr/bin/env python3
"""Paired full-course screen of the confirmed failure-conditioned Puffer."""

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
import scripts.eval_vq2_lc061_phase2_bias_full_course as base
import scripts.eval_vq2_lc064_phase2_residual_direction as residual
import scripts.eval_vq2_lc066_phase2_failure_conditioned as failure


TAG = "vq2_lc068_phase2_failure_full_course_001"
SCHEMA = "vq2_lc068_phase2_failure_full_course_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc068_phase2_failure_full_course_checkpoint_v1"
SEED = 431680
COEFFICIENT = (-0.01, 0.0, 0.0, 0.0)
PARENT_CHECKPOINT = failure.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = failure.PARENT_CHECKPOINT_SHA256
PARENT_REPORT = failure.PARENT_REPORT
PARENT_REPORT_SHA256 = failure.PARENT_REPORT_SHA256
LC067_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc067_phase2_failure_confirmation_001/report.json"
)
LC067_REPORT_SHA256 = "7a32aa61239cb540832322665c775da59ef81257c063c4c50598024c0ba3a914"
PREREGISTRATION = ROOT / "docs/vq2_lc068_phase2_failure_full_course_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc068_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc068_phase2_failure_full_course.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_BASE_VERIFY_INPUTS = base.verify_inputs


def build_selected_candidate_state(
    parent_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    coefficient = torch.tensor(COEFFICIENT, dtype=torch.float32)
    weight, bias = failure.failure_direction()
    state["indexed_phase_residual_output"][base.TARGET_PHASE].add_(
        coefficient[:, None] * weight[None, :]
    )
    state["indexed_phase_residual_output_bias"][base.TARGET_PHASE].add_(
        coefficient * bias
    )
    return state


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, torch.Tensor]:
    states = [
        {name: value.detach().cpu().clone() for name, value in payload["model_state"].items()},
        build_selected_candidate_state(payload["model_state"]),
    ]
    context: dict[str, torch.Tensor] = {}
    for name in residual.RESIDUAL_NAMES:
        rows = torch.stack([state[name][base.TARGET_PHASE] for state in states])
        context[name] = rows.repeat_interleave(base.GROUP_SIZE, dim=0).to(device)
        context[f"parent_{name}"] = payload["model_state"][name][base.TARGET_PHASE].to(device)
    return context


def selected_candidate_metadata() -> dict[str, Any]:
    return {
        "failure_action_coefficient": list(COEFFICIENT),
        "failure_direction_checkpoint_sha256": failure.FIT_CHECKPOINT_SHA256,
    }


def checkpoint_surgery_metadata(candidate_state_sha256: str) -> dict[str, Any]:
    return {
        "operation": "phase2_failure_score_outer_product",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "target_phase": base.TARGET_PHASE,
        "failure_action_coefficient": list(COEFFICIENT),
        "failure_direction_checkpoint_sha256": failure.FIT_CHECKPOINT_SHA256,
        "candidate_state_sha256": candidate_state_sha256,
    }


def configure() -> None:
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.SEED = SEED
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.LC060_REPORT = LC067_REPORT
    base.LC060_REPORT_SHA256 = LC067_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.CANDIDATE_NAME = "failure_pitch_m0p0100"
    base.NEXT_AUTHORITY_SELECTED = (
        "Retain LC068 as the offline whole-Puffer frontier and rediagnose the "
        "earliest remaining high-mass phase on the 24-gate proxy."
    )
    base.NEXT_AUTHORITY_NONE = "Reject the failure-conditioned candidate and retain LC062."
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), PARENT_REPORT, LC067_REPORT,
        failure.FIT_CHECKPOINT, failure.FIT_REPORT,
        ROOT / "scripts/eval_vq2_lc066_phase2_failure_conditioned.py",
        ROOT / "scripts/eval_vq2_lc067_phase2_failure_confirmation.py",
        ROOT / "scripts/train_vq2_lc065_phase2_failure_direction.py",
    )


def verify_inputs() -> dict[str, Any]:
    parent = failure.verify_inputs()
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC067_REPORT: LC067_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC068 bound input changed: {path}")
    report = json.loads(LC067_REPORT.read_text())
    selected = report.get("causal_screen_selected", {})
    if (
        report.get("schema") != "vq2_lc067_phase2_failure_confirmation_report_v1"
        or not report.get("diagnostic_valid")
        or selected.get("failure_action_coefficient") != list(COEFFICIENT)
        or selected.get("gate3_passes") != 19
        or selected.get("paired_gate3_gains_vs_baseline") != 8
        or selected.get("paired_gate3_losses_vs_baseline") != 1
        or report.get("items", [{}])[0].get("gate3_passes") != 12
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC067 does not authorize LC068")
    return parent


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    configure()
    originals = (
        base.verify_inputs, base.build_candidate_context,
        base.apply_candidate_actions, base.build_selected_candidate_state,
        base.selected_candidate_metadata, base.checkpoint_surgery_metadata,
    )
    (
        base.verify_inputs, base.build_candidate_context,
        base.apply_candidate_actions, base.build_selected_candidate_state,
        base.selected_candidate_metadata, base.checkpoint_surgery_metadata,
    ) = (
        verify_inputs, build_candidate_context, residual.apply_candidate_actions,
        build_selected_candidate_state, selected_candidate_metadata,
        checkpoint_surgery_metadata,
    )
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            base.verify_inputs, base.build_candidate_context,
            base.apply_candidate_actions, base.build_selected_candidate_state,
            base.selected_candidate_metadata, base.checkpoint_surgery_metadata,
        ) = originals


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
