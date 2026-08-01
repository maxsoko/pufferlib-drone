#!/usr/bin/env python3
"""Exact saved-checkpoint 24-gate replay of LC073 versus LC087."""

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


TAG = "vq2_lc093_serialized_checkpoint_full_course_001"
SCHEMA = "vq2_lc093_serialized_checkpoint_full_course_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc093_serialized_checkpoint_full_course_checkpoint_v1"
GROUP_SIZE = 128
SEED = 431_930
TARGET_RAW_INDEX = 7
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
CANDIDATE_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc087_phase6_half_bias_full_course_001"
)
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "128bc970f169da81b4bd6b44dfbf73a9bf5ba0c66085d3e0cd13ca48f7f26b86"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "5023b42ef9b9bb2fbc4d016e4b5f5681e659da8c97217816769d3f2c9d21f6d5"
LC092_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc092_phase7_paired_seed_recapture_001/report.json"
)
LC092_REPORT_SHA256 = "3e39304784119dac92d793aeb96ebbf5f1c61fbba20fbb754fa589f8897bf5b7"
PREREGISTRATION = ROOT / "docs/vq2_lc093_serialized_checkpoint_full_course_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc093_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc093_serialized_checkpoint_full_course.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
        LC092_REPORT: LC092_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC093 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    candidate = torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    candidate_report = json.loads(CANDIDATE_REPORT.read_text())
    parity = json.loads(LC092_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or candidate.get("schema") != "vq2_lc087_phase6_half_bias_full_course_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or candidate_report.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or not candidate_report.get("numerically_admitted")
        or parity.get("schema") != "vq2_lc092_phase7_paired_seed_recapture_report_v1"
        or parity.get("training_dataset_admitted")
        or parity.get("query_agents") != 8
        or parity.get("query_outcome_success_agents") != 2
        or parity.get("query_outcome_failure_agents") != 6
        or parity.get("metrics", {}).get("env/gates_passed") != 3.421875
        or parity.get("safety", {}).get("flight_sim_packets_sent") != 0
        or parity.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC073/LC087/LC092 do not authorize LC093")
    return parent


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> None:
    del payload, device
    return None


def initialize_actor_execution(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    candidate = torch.load(
        CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    actors = (
        milestone.load_actor(payload, device),
        milestone.load_actor(candidate, device),
    )
    return {
        "actors": actors,
        "recurrent": [
            actor.initial_state(GROUP_SIZE, device=device) for actor in actors
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
    actions = torch.empty(
        (full.TOTAL_AGENTS, 4), dtype=actor_input.dtype, device=actor_input.device
    )
    for group, actor in enumerate(execution["actors"]):
        selected = full.group_slice(group)
        output, candidate_recurrent = actor.forward_step(
            actor_input[selected], execution["recurrent"][group]
        )
        execution["recurrent"][group] = preserve_frozen_state(
            execution["recurrent"][group],
            candidate_recurrent,
            active[selected],
        )
        actions[selected] = output.mean
    return actions


def build_selected_candidate_state(
    parent_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    del parent_state
    payload = torch.load(
        CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    return {
        name: value.detach().cpu().clone()
        for name, value in payload["model_state"].items()
    }


def selected_candidate_metadata() -> dict[str, Any]:
    return {
        "execution": "saved_checkpoint_exact",
        "source_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
    }


def checkpoint_surgery_metadata(candidate_state_sha256: str) -> dict[str, Any]:
    return {
        "operation": "exact_saved_checkpoint_state_repromotion",
        "source_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
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


def configure() -> None:
    full.TAG, full.SCHEMA, full.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    full.GROUP_SIZE = GROUP_SIZE
    full.GROUPS = 2
    full.TOTAL_AGENTS = GROUP_SIZE * 2
    full.EPISODES = full.TOTAL_AGENTS
    full.SEED = SEED
    full.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    full.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    full.LC060_REPORT = LC092_REPORT
    full.LC060_REPORT_SHA256 = LC092_REPORT_SHA256
    full.PREREGISTRATION, full.RUNNER, full.TEST = PREREGISTRATION, RUNNER, TEST
    full.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    full.CANDIDATE_NAME = "lc087_saved_checkpoint"
    full.PROMOTION_TARGET_RAW_INDEX = TARGET_RAW_INDEX
    full.SURGERY_PAYLOAD_KEY = "serialized_checkpoint_repromotion"
    full.ACTOR_EXECUTION_CONTRACT = (
        "two independently loaded saved Puffer checkpoints; no post-forward surgery"
    )
    full.NEXT_AUTHORITY_SELECTED = (
        "Retain LC093 as the serialization-exact offline frontier and diagnose its phase-7 outcomes."
    )
    full.NEXT_AUTHORITY_NONE = "Reject LC087 serialization and retain LC073."
    full.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), PARENT_REPORT, CANDIDATE_CHECKPOINT,
        CANDIDATE_REPORT, LC092_REPORT,
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
