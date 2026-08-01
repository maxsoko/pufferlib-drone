#!/usr/bin/env python3
"""Pairwise screen of phase-8 biases measured from successful LC105 actions."""

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


TAG = "vq2_lc110_phase8_success_action_bias_001"
SCHEMA = "vq2_lc110_phase8_success_action_bias_report_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
TARGET_PHASE = 8
TARGET_RAW_INDEX = 9
MAX_STEPS = 12_000
BIASES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("baseline_lc105", (0.0, 0.0, 0.0, 0.0)),
    ("roll_p0p050", (0.0, 0.050, 0.0, 0.0)),
    ("success_combo_weak", (-0.005, 0.025, -0.010, 0.0)),
    ("success_combo_strong", (-0.010, 0.050, -0.025, 0.0)),
)
SEED = 432_100
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc105_phase7_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "005e5e7929258fd282ab390fd230fda817700afaee790e78144200e67b3e10a4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c614e6929282399cac2f18465084f8337ed8770389a74e48b86ff1fcf4315e7d"
FEATURES = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc106_phase8_full_batch_recapture_001/features.bin"
)
FEATURES_SHA256 = "e13fc384a159e92d0ef59733360d11c01d570fdd1044504c46793bde40e97a33"
LC106_REPORT = FEATURES.parent / "report.json"
LC106_REPORT_SHA256 = "2bfa88c9caa53c4cde6a0ae51a79413f1c1bfe5e2f70f5e960b27b607f2b8999"
LC109_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc109_phase8_large_scale_screen_001/report.json"
)
LC109_REPORT_SHA256 = "53ca3940eb652e3d6eb35ef76df7720e8a14bcfe7700e5fffff9846170124879"
PREREGISTRATION = ROOT / "docs/vq2_lc110_phase8_success_action_bias_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc110_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc110_phase8_success_action_bias.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def group_slice(group: int) -> slice:
    if not 0 <= group < len(BIASES):
        raise ValueError("LC110 group index is outside the measured bracket")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def pair_indices(group: int, *, device: torch.device) -> torch.Tensor:
    return torch.cat((
        torch.arange(GROUP_SIZE, device=device),
        torch.arange(group * GROUP_SIZE, (group + 1) * GROUP_SIZE, device=device),
    ))


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    state["indexed_phase_residual_output_bias"][TARGET_PHASE].add_(
        torch.tensor(BIASES[candidate_index][1], dtype=torch.float32)
    )
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    bias = BIASES[candidate_index][1]
    return {
        "phase8_output_bias": list(bias),
        "bias_l2": sum(value * value for value in bias) ** 0.5,
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        LC106_REPORT: LC106_REPORT_SHA256,
        LC109_REPORT: LC109_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC110 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(LC106_REPORT.read_text())
    rejected = json.loads(LC109_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema") != "vq2_lc106_phase8_full_batch_recapture_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 4_736
        or dataset.get("query_outcome_success_agents") != 2
        or dataset.get("query_outcome_failure_agents") != 2
        or rejected.get("schema") != "vq2_lc109_phase8_large_scale_screen_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or rejected.get("items", [{}, {}])[2].get("paired_target_losses_vs_baseline") != 1
        or rejected.get("items", [{}, {}, {}, {}])[3].get("paired_target_losses_vs_baseline") != 1
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC106/LC109 do not authorize LC110")
    return parent


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> None:
    del payload, device
    return None


def initialize_actor_execution(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    actors = [
        milestone.load_actor(
            {
                **payload,
                "model_state": candidate_state_for_index(
                    payload["model_state"], group
                ),
            },
            device,
        )
        for group in range(len(BIASES))
    ]
    return {
        "actors": actors,
        "recurrent": [
            actor.initial_state(PAIR_SIZE, device=device) for actor in actors
        ],
        "indices": [
            pair_indices(group, device=device) for group in range(len(BIASES))
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
    milestone.TAG, milestone.SCHEMA = TAG, SCHEMA
    milestone.GROUP_SIZE = GROUP_SIZE
    milestone.BIAS_CANDIDATES = BIASES
    milestone.GROUPS = len(BIASES)
    milestone.TOTAL_AGENTS = GROUP_SIZE * len(BIASES)
    milestone.EPISODES = milestone.TOTAL_AGENTS
    milestone.SEED = SEED
    milestone.TARGET_PHASE, milestone.TARGET_RAW_INDEX = (
        TARGET_PHASE, TARGET_RAW_INDEX,
    )
    milestone.MAX_STEPS, milestone.MINIMUM_PASS_GAIN = MAX_STEPS, MINIMUM_PASS_GAIN
    milestone.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    milestone.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    milestone.PARENT_REPORT, milestone.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    milestone.PARENT_CHILD_REPORT, milestone.PARENT_CHILD_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    milestone.PREREGISTRATION, milestone.RUNNER, milestone.TEST = (
        PREREGISTRATION, RUNNER, TEST,
    )
    milestone.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FEATURES, LC106_REPORT, LC109_REPORT,
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one independent pairwise-256 confirmation of the selected success-action bias."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject the phase-8 measured-bias family and retain LC105."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "four complete saved-form Puffer actors with source-measured phase-8 output biases, each executing a 256-row pair"
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
        candidate_state_for_index, candidate_metadata_for_index,
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
