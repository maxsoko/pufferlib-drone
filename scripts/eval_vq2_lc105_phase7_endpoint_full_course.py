#!/usr/bin/env python3
"""Serialization-exact 24-gate comparison of LC094 and confirmed LC101."""

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

from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc061_phase2_bias_full_course as full
import scripts.eval_vq2_lc102_phase7_endpoint_scale_screen as endpoint


TAG = "vq2_lc105_phase7_endpoint_full_course_001"
SCHEMA = "vq2_lc105_phase7_endpoint_full_course_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
GROUP_SIZE = 128
TOTAL_AGENTS = 256
SEED = 432_050
TARGET_RAW_INDEX = 8
MINIMUM_MEAN_GATE_GAIN = 1.0 / GROUP_SIZE
MINIMUM_TARGET_PASS_GAIN = 1
PARENT_CHECKPOINT = endpoint.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = endpoint.PARENT_CHECKPOINT_SHA256
PARENT_REPORT = endpoint.PARENT_REPORT
PARENT_REPORT_SHA256 = endpoint.PARENT_REPORT_SHA256
FIT_CHECKPOINT = endpoint.FIT_CHECKPOINT
FIT_CHECKPOINT_SHA256 = endpoint.FIT_CHECKPOINT_SHA256
FIT_REPORT = endpoint.FIT_REPORT
FIT_REPORT_SHA256 = endpoint.FIT_REPORT_SHA256
LC104_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc104_phase7_pairwise_confirmation_001/report.json"
)
LC104_REPORT_SHA256 = "3f53e5421462b5a6f2fd345753d97ce89ca2405cd7b37420546637ed44713848"
PREREGISTRATION = ROOT / "docs/vq2_lc105_phase7_endpoint_full_course_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc105_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc105_phase7_endpoint_full_course.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def group_slice(group: int) -> slice:
    if not 0 <= group < 2:
        raise ValueError("LC105 group index is outside the exact pair")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
        LC104_REPORT: LC104_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC105 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    fitted = endpoint.endpoint_decoder()
    fit_report = json.loads(FIT_REPORT.read_text())
    confirmation = json.loads(LC104_REPORT.read_text())
    selected = confirmation.get("causal_screen_selected", {})
    if (
        parent.get("schema") != "vq2_lc094_full_batch_serialized_checkpoint_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or fitted.get("schema") != "vq2_lc101_phase7_success_anchored_endpoint_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("target_phase") != 7
        or fitted.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or not fit_report.get("numerically_admitted")
        or fit_report.get("checkpoint_sha256") != FIT_CHECKPOINT_SHA256
        or confirmation.get("schema") != "vq2_lc104_phase7_pairwise_confirmation_report_v1"
        or not confirmation.get("diagnostic_valid")
        or selected.get("endpoint_scale") != 1.0
        or selected.get("target_passes") != 2
        or selected.get("paired_target_gains_vs_baseline") != 1
        or selected.get("paired_target_losses_vs_baseline") != 0
        or confirmation.get("items", [{}])[0].get("target_passes") != 1
        or confirmation.get("safety", {}).get("flight_sim_packets_sent") != 0
        or confirmation.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC094/LC101/LC104 do not authorize LC105")
    return parent


def candidate_state(
    parent_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    fitted = endpoint.endpoint_decoder()
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    state["indexed_phase_residual_output"][7].copy_(fitted["output_weight"])
    state["indexed_phase_residual_output_bias"][7].copy_(fitted["output_bias"])
    return state


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> None:
    del payload, device
    return None


def initialize_actor_execution(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    candidate_payload = {
        **payload, "model_state": candidate_state(payload["model_state"]),
    }
    actors = (
        milestone.load_actor(payload, device),
        milestone.load_actor(candidate_payload, device),
    )
    return {
        "actors": actors,
        "recurrent": [
            actor.initial_state(TOTAL_AGENTS, device=device) for actor in actors
        ],
    }


def execute_actor_actions(
    execution: dict[str, Any],
    actor_input: torch.Tensor,
    active: torch.Tensor,
    held_progress: torch.Tensor,
    candidate_context: Any,
) -> torch.Tensor:
    del held_progress, candidate_context
    outputs = []
    for group, actor in enumerate(execution["actors"]):
        output, next_recurrent = actor.forward_step(
            actor_input, execution["recurrent"][group]
        )
        execution["recurrent"][group] = preserve_frozen_state(
            execution["recurrent"][group], next_recurrent, active
        )
        outputs.append(output.mean)
    actions = torch.empty_like(outputs[0])
    actions[group_slice(0)] = outputs[0][group_slice(0)]
    actions[group_slice(1)] = outputs[1][group_slice(1)]
    return actions


def build_selected_candidate_state(
    parent_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return candidate_state(parent_state)


def selected_candidate_metadata() -> dict[str, Any]:
    return {
        "execution": "complete_saved_form_checkpoint_exact",
        "phase7_endpoint_checkpoint_sha256": FIT_CHECKPOINT_SHA256,
        "endpoint_scale": 1.0,
    }


def checkpoint_surgery_metadata(candidate_state_sha256: str) -> dict[str, Any]:
    return {
        "operation": "exact_phase7_endpoint_row_replacement",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "endpoint_checkpoint_sha256": FIT_CHECKPOINT_SHA256,
        "candidate_state_sha256": candidate_state_sha256,
        "target_phase": 7,
        "endpoint_scale": 1.0,
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


def configure() -> None:
    full.TAG, full.SCHEMA, full.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    full.GROUP_SIZE, full.GROUPS = GROUP_SIZE, 2
    full.TOTAL_AGENTS, full.EPISODES = TOTAL_AGENTS, TOTAL_AGENTS
    full.SEED = SEED
    full.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    full.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    full.LC060_REPORT, full.LC060_REPORT_SHA256 = LC104_REPORT, LC104_REPORT_SHA256
    full.PREREGISTRATION, full.RUNNER, full.TEST = PREREGISTRATION, RUNNER, TEST
    full.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    full.CANDIDATE_NAME = "lc101_exact_phase7_endpoint"
    full.PROMOTION_TARGET_RAW_INDEX = TARGET_RAW_INDEX
    full.SURGERY_PAYLOAD_KEY = "phase7_endpoint_replacement"
    full.ACTOR_EXECUTION_CONTRACT = (
        "two complete saved-form Puffer checkpoints, each executed over the full 256-agent batch"
    )
    full.NEXT_AUTHORITY_SELECTED = (
        "Retain LC105 as the offline full-start frontier and diagnose raw-index-8 continuation."
    )
    full.NEXT_AUTHORITY_NONE = "Reject LC101 and retain LC094."
    full.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), PARENT_REPORT, FIT_CHECKPOINT, FIT_REPORT,
        LC104_REPORT,
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        full.verify_inputs, full.build_candidate_context,
        full.initialize_actor_execution, full.execute_actor_actions,
        full.build_selected_candidate_state, full.selected_candidate_metadata,
        full.checkpoint_surgery_metadata, full.choose_candidate,
    )
    (
        full.verify_inputs, full.build_candidate_context,
        full.initialize_actor_execution, full.execute_actor_actions,
        full.build_selected_candidate_state, full.selected_candidate_metadata,
        full.checkpoint_surgery_metadata, full.choose_candidate,
    ) = (
        verify_inputs, build_candidate_context,
        initialize_actor_execution, execute_actor_actions,
        build_selected_candidate_state, selected_candidate_metadata,
        checkpoint_surgery_metadata, choose_candidate,
    )
    try:
        return full.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            full.verify_inputs, full.build_candidate_context,
            full.initialize_actor_execution, full.execute_actor_actions,
            full.build_selected_candidate_state, full.selected_candidate_metadata,
            full.checkpoint_surgery_metadata, full.choose_candidate,
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
