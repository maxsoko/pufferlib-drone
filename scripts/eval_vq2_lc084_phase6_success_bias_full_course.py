#!/usr/bin/env python3
"""Paired 24-gate screen of the confirmed phase-6 success-action bias."""

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

from pufferlib.vq2_public_phase import OFFICIAL_PROGRESS_SCALE
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc061_phase2_bias_full_course as base


TAG = "vq2_lc084_phase6_success_bias_full_course_001"
SCHEMA = "vq2_lc084_phase6_success_bias_full_course_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc084_phase6_success_bias_full_course_checkpoint_v1"
SEED = 431840
TARGET_PHASE = 6
TARGET_RAW_INDEX = 7
BIAS = (0.010, -0.050, 0.025, 0.0)
MINIMUM_MEAN_GATE_GAIN = 1.0 / 64.0
MINIMUM_TARGET_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc073_phase2_constrained_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "5614a95cd2f9b428a99ddd43ec5e324c095344cae51b8d771aea3d5f05d35be8"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "ae68af50524b02425f9a970c7f87dccbcbb6b60a0debb9899fcf1baa167b5d75"
LC083_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc083_phase6_success_bias_confirmation_001/report.json"
)
LC083_REPORT_SHA256 = "1592b0041e46d8bda8317d14a5bfbcc97054c5d2070bf27101cd408a99080877"
PREREGISTRATION = ROOT / "docs/vq2_lc084_phase6_success_bias_full_course_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc084_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc084_phase6_success_bias_full_course.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> torch.Tensor:
    del payload
    deltas = torch.zeros(
        (base.TOTAL_AGENTS, 4), dtype=torch.float32, device=device
    )
    deltas[base.group_slice(1)] = torch.tensor(BIAS, dtype=torch.float32, device=device)
    return deltas


def apply_candidate_actions(
    actor_output: Any,
    next_recurrent: torch.Tensor,
    held_progress: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    del next_recurrent
    phase = torch.round(held_progress[:, 0] * OFFICIAL_PROGRESS_SCALE).to(torch.long)
    return torch.tanh(
        actor_output.pre_tanh_mean
        + (phase == TARGET_PHASE)[:, None] * context
    )


def build_selected_candidate_state(
    parent_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    state["indexed_phase_residual_output_bias"][TARGET_PHASE].add_(
        torch.tensor(BIAS, dtype=torch.float32)
    )
    return state


def selected_candidate_metadata() -> dict[str, Any]:
    return {"phase6_output_bias": list(BIAS)}


def checkpoint_surgery_metadata(candidate_state_sha256: str) -> dict[str, Any]:
    return {
        "operation": "phase6_success_action_pre_tanh_output_bias",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "target_phase": TARGET_PHASE,
        "pre_tanh_output_bias_delta": list(BIAS),
        "candidate_state_sha256": candidate_state_sha256,
    }


def choose_candidate(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return bool(
        parent["transport_pass"] and candidate["transport_pass"]
        and candidate["mean_gates_passed"]
        >= parent["mean_gates_passed"] + MINIMUM_MEAN_GATE_GAIN
        and candidate["promotion_target_passes"]
        >= parent["promotion_target_passes"] + MINIMUM_TARGET_PASS_GAIN
        and candidate["crash_rate"] <= parent["crash_rate"]
        and candidate["maximum_raw_index"] >= parent["maximum_raw_index"]
    )


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC083_REPORT: LC083_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC084 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    confirmation = json.loads(LC083_REPORT.read_text())
    selected = confirmation.get("causal_screen_selected", {})
    if (
        parent.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or confirmation.get("schema") != "vq2_lc083_phase6_success_bias_confirmation_report_v1"
        or not confirmation.get("diagnostic_valid")
        or tuple(selected.get("phase6_output_bias", ())) != BIAS
        or selected.get("paired_target_gains_vs_baseline") != 3
        or selected.get("paired_target_losses_vs_baseline") != 0
        or selected.get("target_passes") != 4
        or confirmation.get("items", [{}])[0].get("target_passes") != 1
        or confirmation.get("safety", {}).get("flight_sim_packets_sent") != 0
        or confirmation.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC073/LC083 do not authorize LC084")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.SEED = SEED
    base.TARGET_PHASE = TARGET_PHASE
    base.BIAS = BIAS
    base.MINIMUM_MEAN_GATE_GAIN = MINIMUM_MEAN_GATE_GAIN
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.LC060_REPORT = LC083_REPORT
    base.LC060_REPORT_SHA256 = LC083_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.CANDIDATE_NAME = "phase6_success_action_bias"
    base.PROMOTION_TARGET_RAW_INDEX = TARGET_RAW_INDEX
    base.SURGERY_PAYLOAD_KEY = "phase6_surgery"
    base.NEXT_AUTHORITY_SELECTED = (
        "Retain LC084 as the offline whole-Puffer frontier and diagnose the next 24-gate bottleneck."
    )
    base.NEXT_AUTHORITY_NONE = "Reject the phase-6 bias and retain LC073."
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), PARENT_REPORT, LC083_REPORT,
        ROOT / "scripts/eval_vq2_lc083_phase6_success_bias_confirmation.py",
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        base.verify_inputs, base.build_candidate_context,
        base.apply_candidate_actions, base.build_selected_candidate_state,
        base.selected_candidate_metadata, base.checkpoint_surgery_metadata,
        base.choose_candidate,
    )
    (
        base.verify_inputs, base.build_candidate_context,
        base.apply_candidate_actions, base.build_selected_candidate_state,
        base.selected_candidate_metadata, base.checkpoint_surgery_metadata,
        base.choose_candidate,
    ) = (
        verify_inputs, build_candidate_context, apply_candidate_actions,
        build_selected_candidate_state, selected_candidate_metadata,
        checkpoint_surgery_metadata, choose_candidate,
    )
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            base.verify_inputs, base.build_candidate_context,
            base.apply_candidate_actions, base.build_selected_candidate_state,
            base.selected_candidate_metadata, base.checkpoint_surgery_metadata,
            base.choose_candidate,
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
