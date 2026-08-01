#!/usr/bin/env python3
"""Higher-power 24-gate screen of the confirmed phase-6 half-bias."""

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
import scripts.eval_vq2_lc061_phase2_bias_full_course as full
import scripts.eval_vq2_lc084_phase6_success_bias_full_course as phase6


TAG = "vq2_lc087_phase6_half_bias_full_course_001"
SCHEMA = "vq2_lc087_phase6_half_bias_full_course_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc087_phase6_half_bias_full_course_checkpoint_v1"
GROUP_SIZE = 128
SEED = 431870
TARGET_PHASE = 6
TARGET_RAW_INDEX = 7
BIAS = (0.005, -0.025, 0.0125, 0.0)
MINIMUM_MEAN_GATE_GAIN = 1.0 / GROUP_SIZE
MINIMUM_TARGET_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc073_phase2_constrained_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "5614a95cd2f9b428a99ddd43ec5e324c095344cae51b8d771aea3d5f05d35be8"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "ae68af50524b02425f9a970c7f87dccbcbb6b60a0debb9899fcf1baa167b5d75"
LC086_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc086_phase6_half_bias_confirmation_001/report.json"
)
LC086_REPORT_SHA256 = "12d2db6be8a4b312dfb3b274c61c29bd775c45600cd4f2fc4a65b73610a15927"
PREREGISTRATION = ROOT / "docs/vq2_lc087_phase6_half_bias_full_course_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc087_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc087_phase6_half_bias_full_course.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC086_REPORT: LC086_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC087 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    confirmation = json.loads(LC086_REPORT.read_text())
    selected = confirmation.get("causal_screen_selected", {})
    if (
        parent.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or confirmation.get("schema") != "vq2_lc086_phase6_half_bias_confirmation_report_v1"
        or not confirmation.get("diagnostic_valid")
        or tuple(selected.get("phase6_output_bias", ())) != BIAS
        or selected.get("direction_scale") != 0.5
        or selected.get("paired_target_gains_vs_baseline") != 2
        or selected.get("paired_target_losses_vs_baseline") != 0
        or selected.get("target_passes") != 3
        or confirmation.get("items", [{}])[0].get("target_passes") != 1
        or confirmation.get("safety", {}).get("flight_sim_packets_sent") != 0
        or confirmation.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC073/LC086 do not authorize LC087")
    return parent


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


def selected_candidate_metadata() -> dict[str, Any]:
    return {"direction_scale": 0.5, "phase6_output_bias": list(BIAS)}


def build_selected_candidate_state(
    parent_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    state["indexed_phase_residual_output_bias"][TARGET_PHASE].add_(
        torch.tensor(BIAS, dtype=torch.float32)
    )
    return state


def checkpoint_surgery_metadata(candidate_state_sha256: str) -> dict[str, Any]:
    return {
        "operation": "phase6_half_success_action_pre_tanh_output_bias",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "target_phase": TARGET_PHASE,
        "pre_tanh_output_bias_delta": list(BIAS),
        "candidate_state_sha256": candidate_state_sha256,
    }


def configure() -> None:
    phase6.BIAS = BIAS
    phase6.TARGET_PHASE = TARGET_PHASE
    phase6.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    phase6.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    phase6.MINIMUM_MEAN_GATE_GAIN = MINIMUM_MEAN_GATE_GAIN
    phase6.MINIMUM_TARGET_PASS_GAIN = MINIMUM_TARGET_PASS_GAIN
    full.TAG, full.SCHEMA, full.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    full.GROUP_SIZE = GROUP_SIZE
    full.GROUPS = 2
    full.TOTAL_AGENTS = GROUP_SIZE * full.GROUPS
    full.EPISODES = full.TOTAL_AGENTS
    full.SEED = SEED
    full.TARGET_PHASE = TARGET_PHASE
    full.BIAS = BIAS
    full.MINIMUM_MEAN_GATE_GAIN = MINIMUM_MEAN_GATE_GAIN
    full.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    full.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    full.LC060_REPORT = LC086_REPORT
    full.LC060_REPORT_SHA256 = LC086_REPORT_SHA256
    full.PREREGISTRATION, full.RUNNER, full.TEST = PREREGISTRATION, RUNNER, TEST
    full.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    full.CANDIDATE_NAME = "phase6_half_success_action_bias"
    full.PROMOTION_TARGET_RAW_INDEX = TARGET_RAW_INDEX
    full.SURGERY_PAYLOAD_KEY = "phase6_surgery"
    full.NEXT_AUTHORITY_SELECTED = (
        "Retain LC087 as the offline whole-Puffer frontier and diagnose the next 24-gate bottleneck."
    )
    full.NEXT_AUTHORITY_NONE = "Reject the half-bias and retain LC073."
    full.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), PARENT_REPORT, LC086_REPORT,
        ROOT / "scripts/eval_vq2_lc084_phase6_success_bias_full_course.py",
        ROOT / "scripts/eval_vq2_lc086_phase6_half_bias_confirmation.py",
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        full.verify_inputs, full.build_candidate_context,
        full.apply_candidate_actions, full.build_selected_candidate_state,
        full.selected_candidate_metadata, full.checkpoint_surgery_metadata,
        full.choose_candidate,
    )
    (
        full.verify_inputs, full.build_candidate_context,
        full.apply_candidate_actions, full.build_selected_candidate_state,
        full.selected_candidate_metadata, full.checkpoint_surgery_metadata,
        full.choose_candidate,
    ) = (
        verify_inputs, phase6.build_candidate_context,
        phase6.apply_candidate_actions, build_selected_candidate_state,
        selected_candidate_metadata, checkpoint_surgery_metadata,
        choose_candidate,
    )
    try:
        return full.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            full.verify_inputs, full.build_candidate_context,
            full.apply_candidate_actions, full.build_selected_candidate_state,
            full.selected_candidate_metadata, full.checkpoint_surgery_metadata,
            full.choose_candidate,
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
