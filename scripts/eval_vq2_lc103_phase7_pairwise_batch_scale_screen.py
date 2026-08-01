#!/usr/bin/env python3
"""Phase-7 endpoint scales in the established pairwise 256-row actor context."""

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
import scripts.eval_vq2_lc102_phase7_endpoint_scale_screen as prior


TAG = "vq2_lc103_phase7_pairwise_batch_scale_screen_001"
SCHEMA = "vq2_lc103_phase7_pairwise_batch_scale_screen_report_v1"
GROUP_SIZE = 128
PAIR_SIZE = GROUP_SIZE * 2
ALPHAS = (0.0, 0.50, 1.0, 2.0)
CANDIDATES = tuple(
    ("baseline_lc094" if alpha == 0.0 else f"lc101_scale_{alpha:g}", (0.0,) * 4)
    for alpha in ALPHAS
)
SEED = 432_030
TARGET_PHASE = 7
TARGET_RAW_INDEX = 8
MAX_STEPS = 12_000
MINIMUM_PASS_GAIN = 1
LC102_REPORT = prior.DEFAULT_OUTPUT / "report.json"
LC102_REPORT_SHA256 = "48dc42a8ba5aead9e3ebfd2d14f1153d882d67c633832b08d516d4e6ab2cf67b"
PREREGISTRATION = ROOT / "docs/vq2_lc103_phase7_pairwise_batch_scale_screen_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc103_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc103_phase7_pairwise_batch_scale_screen.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def group_slice(group: int) -> slice:
    if not 0 <= group < len(ALPHAS):
        raise ValueError("LC103 group index is outside the scale bracket")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def pair_indices(group: int, *, device: torch.device) -> torch.Tensor:
    baseline = torch.arange(0, GROUP_SIZE, device=device)
    candidate = torch.arange(
        group * GROUP_SIZE, (group + 1) * GROUP_SIZE, device=device
    )
    return torch.cat((baseline, candidate))


def verify_inputs() -> dict[str, Any]:
    parent = prior.verify_inputs()
    if sha256_path(LC102_REPORT) != LC102_REPORT_SHA256:
        raise RuntimeError("LC103 bound LC102 report changed")
    rejected = json.loads(LC102_REPORT.read_text())
    if (
        rejected.get("schema") != "vq2_lc102_phase7_endpoint_scale_screen_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or rejected.get("group_size") != 64
        or rejected.get("total_agents") != 448
        or any(item.get("target_passes") != 0 for item in rejected.get("items", []))
        or any(item.get("pre_target_terminals") != 64 for item in rejected.get("items", []))
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC102 does not authorize pairwise batch correction")
    return parent


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> None:
    del payload, device
    return None


def initialize_actor_execution(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    states = [
        prior.candidate_state_for_index(payload["model_state"], index)
        for index in range(len(ALPHAS))
    ]
    actors = [
        milestone.load_actor({**payload, "model_state": state}, device)
        for state in states
    ]
    return {
        "actors": actors,
        "recurrent": [
            actor.initial_state(PAIR_SIZE, device=device) for actor in actors
        ],
        "indices": [pair_indices(group, device=device) for group in range(len(ALPHAS))],
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
        (milestone.TOTAL_AGENTS, 4), dtype=actor_input.dtype,
        device=actor_input.device,
    )
    for group, actor in enumerate(execution["actors"]):
        indices = execution["indices"][group]
        pair_active = active.index_select(0, indices)
        output, next_recurrent = actor.forward_step(
            actor_input.index_select(0, indices), execution["recurrent"][group]
        )
        execution["recurrent"][group] = preserve_frozen_state(
            execution["recurrent"][group], next_recurrent, pair_active
        )
        source = slice(0, GROUP_SIZE) if group == 0 else slice(GROUP_SIZE, PAIR_SIZE)
        actions[group_slice(group)] = output.mean[source]
    return actions


def configure() -> None:
    prior.ALPHAS, prior.CANDIDATES = ALPHAS, CANDIDATES
    prior.GROUP_SIZE = GROUP_SIZE
    prior.SEED = SEED
    prior.TARGET_PHASE, prior.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    prior.MAX_STEPS, prior.MINIMUM_PASS_GAIN = MAX_STEPS, MINIMUM_PASS_GAIN
    prior.TAG, prior.SCHEMA = TAG, SCHEMA
    prior.PREREGISTRATION, prior.RUNNER, prior.TEST = PREREGISTRATION, RUNNER, TEST
    prior.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    prior.configure()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), LC102_REPORT,
        prior.FIT_CHECKPOINT, prior.FIT_REPORT,
        ROOT / "scripts/eval_vq2_lc102_phase7_endpoint_scale_screen.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one independent 128-pair confirmation in the same 256-row actor context."
    )
    milestone.NEXT_AUTHORITY_NONE = "Reject LC101 and retain LC094."
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "four complete saved-form Puffer actors; each executes a baseline-plus-candidate 256-row pair"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
    )
    (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
    ) = (
        verify_inputs, build_candidate_context,
        initialize_actor_execution, execute_actor_actions,
        prior.candidate_state_for_index, prior.candidate_metadata_for_index,
    )
    try:
        return milestone.run(
            output=output, device_name=device_name, resume=resume
        )
    finally:
        (
            milestone.verify_inputs, milestone.build_candidate_context,
            milestone.initialize_actor_execution, milestone.execute_actor_actions,
            milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
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
