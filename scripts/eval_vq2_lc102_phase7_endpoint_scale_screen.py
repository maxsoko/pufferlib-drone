#!/usr/bin/env python3
"""Teacher-free full-start scale screen of the LC101 phase-7 endpoint."""

from __future__ import annotations

import argparse
import functools
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
import scripts.eval_vq2_lc058_phase2_bias_milestone as base


TAG = "vq2_lc102_phase7_endpoint_scale_screen_001"
SCHEMA = "vq2_lc102_phase7_endpoint_scale_screen_report_v1"
GROUP_SIZE = 64
TARGET_PHASE = 7
TARGET_RAW_INDEX = 8
MAX_STEPS = 12_000
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0, 1.50, 2.0)
CANDIDATES = tuple(
    ("baseline_lc094" if alpha == 0.0 else f"lc101_scale_{alpha:g}", (0.0,) * 4)
    for alpha in ALPHAS
)
SEED = 432_020
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc094_full_batch_serialized_checkpoint_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "14a5f90e8a7a80d3535ef0d86d3f9b5d75d42317c387a0554cf59ecf283e340a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "639edd98ec0d99c981d16837c802d758adc6622736314b3e6afe09a58af21139"
FIT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc101_phase7_success_anchored_endpoint_001"
)
FIT_CHECKPOINT = FIT_DIR / "decoder_endpoint.pt"
FIT_CHECKPOINT_SHA256 = "fd9feb93c2ce04092f3b800a94c1bac1891403d73a7a5cf01bff38c83b670812"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "0bd35fcae29a5341f57f52a12410f8be698106bed60fde8eeeaf1c3426763ea8"
PREREGISTRATION = ROOT / "docs/vq2_lc102_phase7_endpoint_scale_screen_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc102_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc102_phase7_endpoint_scale_screen.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def group_slice(group: int) -> slice:
    if not 0 <= group < len(ALPHAS):
        raise ValueError("LC102 group index is outside the scale bracket")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


@functools.lru_cache(maxsize=1)
def endpoint_decoder() -> dict[str, Any]:
    return torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    endpoint = endpoint_decoder()
    alpha = ALPHAS[candidate_index]
    state["indexed_phase_residual_output"][TARGET_PHASE].lerp_(
        endpoint["output_weight"], alpha
    )
    state["indexed_phase_residual_output_bias"][TARGET_PHASE].lerp_(
        endpoint["output_bias"], alpha
    )
    return state


def parameter_delta_l2(parent_state: dict[str, torch.Tensor], alpha: float) -> float:
    endpoint = endpoint_decoder()
    delta = torch.cat((
        (endpoint["output_weight"]
         - parent_state["indexed_phase_residual_output"][TARGET_PHASE]).flatten(),
        (endpoint["output_bias"]
         - parent_state["indexed_phase_residual_output_bias"][TARGET_PHASE]).flatten(),
    )).double()
    return float(alpha * delta.norm())


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    alpha = ALPHAS[candidate_index]
    delta_l2 = parameter_delta_l2(parent["model_state"], alpha)
    return {
        "endpoint_scale": alpha,
        "parameter_delta_l2": delta_l2,
        "bias_l2": delta_l2,
    }


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> None:
    del payload, device
    return None


def initialize_actor_execution(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    states = [
        candidate_state_for_index(payload["model_state"], index)
        for index in range(len(ALPHAS))
    ]
    actors = [
        base.load_actor({**payload, "model_state": state}, device) for state in states
    ]
    return {
        "actors": actors,
        "recurrent": [
            actor.initial_state(base.TOTAL_AGENTS, device=device) for actor in actors
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
    for group, output in enumerate(outputs):
        selected = group_slice(group)
        actions[selected] = output[selected]
    return actions


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC102 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    fit = endpoint_decoder()
    fit_report = json.loads(FIT_REPORT.read_text())
    selected = fit_report.get("selected", {})
    if (
        parent.get("schema") != "vq2_lc094_full_batch_serialized_checkpoint_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or fit.get("schema") != "vq2_lc101_phase7_success_anchored_endpoint_checkpoint_v1"
        or not fit.get("numerically_admitted")
        or fit.get("target_phase") != TARGET_PHASE
        or fit.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or fit_report.get("schema") != "vq2_lc101_phase7_success_anchored_endpoint_report_v1"
        or not fit_report.get("numerically_admitted")
        or fit_report.get("checkpoint_sha256") != FIT_CHECKPOINT_SHA256
        or selected.get("ridge") != 1e-6
        or selected.get("alpha") != 0.025
        or selected.get("failure_improvement_factor", 0.0) < 1.10
        or selected.get("success_parent_action_drift_mse", 1.0) > 0.00025
        or fit_report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit_report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC094/LC101 do not authorize LC102")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.GROUP_SIZE = GROUP_SIZE
    base.BIAS_CANDIDATES = CANDIDATES
    base.GROUPS = len(CANDIDATES)
    base.TOTAL_AGENTS = GROUP_SIZE * base.GROUPS
    base.EPISODES = base.TOTAL_AGENTS
    base.SEED = SEED
    base.TARGET_PHASE, base.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    base.MAX_STEPS, base.MINIMUM_PASS_GAIN = MAX_STEPS, MINIMUM_PASS_GAIN
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    base.PARENT_CHILD_REPORT, base.PARENT_CHILD_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT,
        ROOT / "scripts/fit_vq2_lc101_phase7_success_anchored_endpoint.py",
    )
    base.NEXT_AUTHORITY_SELECTED = (
        "Run one larger full-batch confirmation of the selected phase-7 scale."
    )
    base.NEXT_AUTHORITY_NONE = "Reject LC101 and retain LC094."
    base.VECTORIZED_CHECKPOINT_SURGERY = (
        "seven independently loaded complete Puffer checkpoints, each executed over the full 448-agent batch"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        base.verify_inputs, base.build_candidate_context,
        base.initialize_actor_execution, base.execute_actor_actions,
        base.candidate_state_for_index, base.candidate_metadata_for_index,
    )
    (
        base.verify_inputs, base.build_candidate_context,
        base.initialize_actor_execution, base.execute_actor_actions,
        base.candidate_state_for_index, base.candidate_metadata_for_index,
    ) = (
        verify_inputs, build_candidate_context,
        initialize_actor_execution, execute_actor_actions,
        candidate_state_for_index, candidate_metadata_for_index,
    )
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            base.verify_inputs, base.build_candidate_context,
            base.initialize_actor_execution, base.execute_actor_actions,
            base.candidate_state_for_index, base.candidate_metadata_for_index,
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
