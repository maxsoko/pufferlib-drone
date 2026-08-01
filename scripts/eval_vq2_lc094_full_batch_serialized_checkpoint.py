#!/usr/bin/env python3
"""Full-batch exact saved-checkpoint replay of LC073 versus LC087."""

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
import scripts.eval_vq2_lc093_serialized_checkpoint_full_course as partitioned


TAG = "vq2_lc094_full_batch_serialized_checkpoint_001"
SCHEMA = "vq2_lc094_full_batch_serialized_checkpoint_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc094_full_batch_serialized_checkpoint_checkpoint_v1"
SEED = 431_940
LC093_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc093_serialized_checkpoint_full_course_001/report.json"
)
LC093_REPORT_SHA256 = "457d5f920e6aacccb8cf12754b56b504c082bf535894968e2aabead008877c88"
PREREGISTRATION = ROOT / "docs/vq2_lc094_full_batch_serialized_checkpoint_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc094_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc094_full_batch_serialized_checkpoint.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
GROUP_SIZE = partitioned.GROUP_SIZE


def group_slice(group: int) -> slice:
    if not 0 <= group < 2:
        raise ValueError("LC094 group index is outside the exact pair")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def verify_inputs() -> dict[str, Any]:
    parent = partitioned.verify_inputs()
    if sha256_path(LC093_REPORT) != LC093_REPORT_SHA256:
        raise RuntimeError("LC094 bound LC093 report changed")
    rejected = json.loads(LC093_REPORT.read_text())
    if (
        rejected.get("schema") != "vq2_lc093_serialized_checkpoint_full_course_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("numerically_admitted")
        or rejected.get("actor_execution_contract")
        != "two independently loaded saved Puffer checkpoints; no post-forward surgery"
        or rejected.get("group_size") != partitioned.GROUP_SIZE
        or rejected.get("mean_gate_gain") != 0.0
        or rejected.get("promotion_target_pass_gain") != 0
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC093 does not authorize the full-batch parity audit")
    return parent


def initialize_actor_execution(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    candidate = torch.load(
        partitioned.CANDIDATE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    actors = (
        milestone.load_actor(payload, device),
        milestone.load_actor(candidate, device),
    )
    return {
        "actors": actors,
        "recurrent": [
            actor.initial_state(full.TOTAL_AGENTS, device=device)
            for actor in actors
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
        output, candidate_recurrent = actor.forward_step(
            actor_input, execution["recurrent"][group]
        )
        execution["recurrent"][group] = preserve_frozen_state(
            execution["recurrent"][group], candidate_recurrent, active
        )
        outputs.append(output.mean)
    actions = torch.empty_like(outputs[0])
    actions[group_slice(0)] = outputs[0][group_slice(0)]
    actions[group_slice(1)] = outputs[1][group_slice(1)]
    return actions


def configure() -> None:
    partitioned.configure()
    full.TAG, full.SCHEMA, full.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    full.SEED = SEED
    full.PREREGISTRATION, full.RUNNER, full.TEST = PREREGISTRATION, RUNNER, TEST
    full.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    full.ACTOR_EXECUTION_CONTRACT = (
        "two independently loaded saved Puffer checkpoints, each executed over the full 256-agent batch"
    )
    full.NEXT_AUTHORITY_SELECTED = (
        "Retain LC094 as the full-batch serialization-exact offline frontier; require batch-size robustness before deployment."
    )
    full.NEXT_AUTHORITY_NONE = "Reject LC087 serialization and retain LC073."
    full.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), partitioned.PARENT_REPORT,
        partitioned.CANDIDATE_CHECKPOINT, partitioned.CANDIDATE_REPORT,
        partitioned.LC092_REPORT, LC093_REPORT,
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
        verify_inputs, partitioned.build_candidate_context,
        initialize_actor_execution, execute_actor_actions,
        partitioned.build_selected_candidate_state,
        partitioned.selected_candidate_metadata,
        partitioned.checkpoint_surgery_metadata,
        partitioned.choose_candidate,
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
