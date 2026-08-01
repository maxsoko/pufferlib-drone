#!/usr/bin/env python3
"""Full-start 24-gate comparison of saved LC094 and LC098 Puffer policies."""

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
import scripts.eval_vq2_lc097_late_phase_endpoint_local_bracket as late


TAG = "vq2_lc099_late_phase_safe_alpha_full_course_001"
SCHEMA = "vq2_lc099_late_phase_safe_alpha_full_course_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc099_late_phase_safe_alpha_full_course_checkpoint_v1"
GROUP_SIZE = 128
GROUPS = 2
TOTAL_AGENTS = GROUP_SIZE * GROUPS
SEED = 431_990
TARGET_RAW_INDEX = 7
MINIMUM_MEAN_GATE_GAIN = 1.0 / GROUP_SIZE
MINIMUM_TARGET_PASS_GAIN = 1
SELECTED_ALPHA = 0.15
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc094_full_batch_serialized_checkpoint_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "14a5f90e8a7a80d3535ef0d86d3f9b5d75d42317c387a0554cf59ecf283e340a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "639edd98ec0d99c981d16837c802d758adc6622736314b3e6afe09a58af21139"
CANDIDATE_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc098_late_phase_safe_alpha_bracket_001"
)
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "fe0b684b6ae160e595902b031d9ddde154aaf96398e3c2f00c0a8293558fd9c6"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "4de2ae18d1cdc336fa9e1ec3601102c5e3d2465044c7dc2f1ba3a0bfdd9b6c1b"
PREREGISTRATION = ROOT / "docs/vq2_lc099_late_phase_safe_alpha_full_course_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc099_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc099_late_phase_safe_alpha_full_course.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def group_slice(group: int) -> slice:
    if not 0 <= group < GROUPS:
        raise ValueError("LC099 group index is outside the exact pair")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC099 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    candidate = torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    candidate_report = json.loads(CANDIDATE_REPORT.read_text())
    selected = candidate_report.get("selected_candidate", {})
    interpolation = candidate.get("late_phase_interpolation", {})
    if (
        parent.get("schema") != "vq2_lc094_full_batch_serialized_checkpoint_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("diagnostic_valid")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or parent_report.get("items", [{}])[0].get("mean_gates_passed") != 3.3828125
        or parent_report.get("selected_candidate", {}).get("mean_gates_passed") != 3.421875
        or candidate.get("schema") != "vq2_lc098_late_phase_safe_alpha_bracket_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or interpolation.get("alpha") != SELECTED_ALPHA
        or tuple(interpolation.get("phases", ())) != late.ADMISSIBLE_PHASES
        or not candidate_report.get("diagnostic_valid")
        or not candidate_report.get("numerically_admitted")
        or candidate_report.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or selected.get("alpha") != SELECTED_ALPHA
        or selected.get("mean_gate_advance") != 0.4270833333333333
        or selected.get("one_gate_passes") != 32
        or selected.get("crash_rate") != 0.0
        or candidate_report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or candidate_report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC094/LC098 do not authorize LC099")
    return parent


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> None:
    del payload, device
    return None


def initialize_actor_execution(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    candidate = torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    actors = (
        milestone.load_actor(payload, device),
        milestone.load_actor(candidate, device),
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
    del parent_state
    payload = torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    return {
        name: value.detach().cpu().clone()
        for name, value in payload["model_state"].items()
    }


def selected_candidate_metadata() -> dict[str, Any]:
    return {
        "execution": "saved_checkpoint_exact",
        "source_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
        "late_phase_alpha": SELECTED_ALPHA,
        "late_phase_rows": list(late.ADMISSIBLE_PHASES),
    }


def checkpoint_surgery_metadata(candidate_state_sha256: str) -> dict[str, Any]:
    return {
        "operation": "exact_saved_checkpoint_state_repromotion",
        "source_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
        "candidate_state_sha256": candidate_state_sha256,
        "late_phase_alpha": SELECTED_ALPHA,
        "late_phase_rows": list(late.ADMISSIBLE_PHASES),
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
    full.GROUP_SIZE, full.GROUPS = GROUP_SIZE, GROUPS
    full.TOTAL_AGENTS, full.EPISODES = TOTAL_AGENTS, TOTAL_AGENTS
    full.SEED = SEED
    full.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    full.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    full.LC060_REPORT = CANDIDATE_REPORT
    full.LC060_REPORT_SHA256 = CANDIDATE_REPORT_SHA256
    full.PREREGISTRATION, full.RUNNER, full.TEST = PREREGISTRATION, RUNNER, TEST
    full.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    full.CANDIDATE_NAME = "lc098_safe_alpha_0p15"
    full.PROMOTION_TARGET_RAW_INDEX = TARGET_RAW_INDEX
    full.SURGERY_PAYLOAD_KEY = "late_phase_safe_checkpoint"
    full.ACTOR_EXECUTION_CONTRACT = (
        "two independently loaded complete saved Puffer checkpoints, each executed over the full 256-agent batch"
    )
    full.NEXT_AUTHORITY_SELECTED = (
        "Retain LC099 as the offline full-start frontier; diagnose its next bottleneck and require broader robustness before any live authority."
    )
    full.NEXT_AUTHORITY_NONE = (
        "Reject LC098 on the full-start proxy and retain the LC094 frontier."
    )
    full.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), PARENT_REPORT, CANDIDATE_CHECKPOINT,
        CANDIDATE_REPORT,
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
