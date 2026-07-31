#!/usr/bin/env python3
"""Fast closed-loop screen of the constrained failure-teacher endpoint."""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc058_phase2_bias_milestone as base
import scripts.eval_vq2_lc064_phase2_residual_direction as residual


TAG = "vq2_lc071_phase2_constrained_endpoint_screen_001"
SCHEMA = "vq2_lc071_phase2_constrained_endpoint_screen_report_v1"
GROUP_SIZE = 32
TARGET_PHASE = 2
ALPHAS = (0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 1.0)
CANDIDATES = tuple(
    ("baseline_lc062" if alpha == 0.0 else f"endpoint_a{str(alpha).replace('.', 'p')}",
     (0.0, 0.0, 0.0, 0.0))
    for alpha in ALPHAS
)
SEED = 431710
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc062_phase2_bias_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "06381161445f5f207a3b12f7c97b82c884f91a71fdb3e34e05189255cc58d04a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "5bf0fc1f95fec795dcf0c8212f7838a0f18a683baaece679228e87e667085af5"
FIT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc070_phase2_constrained_endpoint_001"
)
FIT_CHECKPOINT = FIT_DIR / "decoder_endpoint.pt"
FIT_CHECKPOINT_SHA256 = "6ba7c1a0e77d6d3bd2518d623e4a95be124c1ffe5385592ed52da0601a1155e0"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "fdf93225efe090fa9cf170910846f0b99c578f4f7a0b1c202e2ed3ce966a2986"
PREREGISTRATION = ROOT / "docs/vq2_lc071_phase2_constrained_endpoint_screen_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc071_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc071_phase2_constrained_endpoint_screen.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


@functools.lru_cache(maxsize=1)
def endpoint_decoder() -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    return payload["output_weight"].detach().cpu(), payload["output_bias"].detach().cpu()


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    alpha = ALPHAS[candidate_index]
    endpoint_weight, endpoint_bias = endpoint_decoder()
    output_name = "indexed_phase_residual_output"
    bias_name = "indexed_phase_residual_output_bias"
    if alpha == 1.0:
        state[output_name][TARGET_PHASE].copy_(endpoint_weight)
        state[bias_name][TARGET_PHASE].copy_(endpoint_bias)
    elif alpha != 0.0:
        state[output_name][TARGET_PHASE].lerp_(endpoint_weight, alpha)
        state[bias_name][TARGET_PHASE].lerp_(endpoint_bias, alpha)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    alpha = ALPHAS[candidate_index]
    endpoint_weight, endpoint_bias = endpoint_decoder()
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)["model_state"]
    delta = torch.cat((
        (endpoint_weight - parent["indexed_phase_residual_output"][TARGET_PHASE]).flatten(),
        (endpoint_bias - parent["indexed_phase_residual_output_bias"][TARGET_PHASE]).flatten(),
    )).double()
    return {
        "endpoint_interpolation_alpha": alpha,
        "parameter_delta_l2": abs(alpha) * float(delta.norm()),
        "bias_l2": abs(alpha) * float(delta.norm()),
    }


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, torch.Tensor]:
    states = [
        candidate_state_for_index(payload["model_state"], index)
        for index in range(len(ALPHAS))
    ]
    context: dict[str, torch.Tensor] = {}
    for name in residual.RESIDUAL_NAMES:
        rows = torch.stack([state[name][TARGET_PHASE] for state in states])
        context[name] = rows.repeat_interleave(GROUP_SIZE, dim=0).to(device)
        context[f"parent_{name}"] = payload["model_state"][name][TARGET_PHASE].to(device)
    return context


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.GROUP_SIZE = GROUP_SIZE
    base.BIAS_CANDIDATES = CANDIDATES
    base.GROUPS = len(CANDIDATES)
    base.TOTAL_AGENTS = GROUP_SIZE * base.GROUPS
    base.EPISODES = base.TOTAL_AGENTS
    base.SEED = SEED
    base.MINIMUM_PASS_GAIN = MINIMUM_PASS_GAIN
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT = PARENT_REPORT
    base.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.PARENT_CHILD_REPORT = PARENT_REPORT
    base.PARENT_CHILD_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT,
        ROOT / "scripts/fit_vq2_lc070_phase2_constrained_endpoint.py",
        ROOT / "scripts/eval_vq2_lc064_phase2_residual_direction.py",
    )
    base.NEXT_AUTHORITY_SELECTED = (
        "Run one larger different-seed Gate-3 confirmation of the selected constrained endpoint interpolation."
    )
    base.NEXT_AUTHORITY_NONE = "Reject the constrained endpoint family and retain LC062."
    base.VECTORIZED_CHECKPOINT_SURGERY = (
        "exact phase-2 Puffer decoder interpolation toward the held-success constrained failure-teacher endpoint"
    )


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC071 bound input changed: {path}")
    parent_report = json.loads(PARENT_REPORT.read_text())
    fit_report = json.loads(FIT_REPORT.read_text())
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    fitted = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        parent_report.get("schema") != "vq2_lc062_phase2_bias_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent.get("schema") != "vq2_lc062_phase2_bias_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or fit_report.get("schema") != "vq2_lc070_phase2_constrained_endpoint_report_v1"
        or not fit_report.get("numerically_admitted")
        or fit_report.get("selected", {}).get("alpha") != 0.075
        or fit_report.get("selected", {}).get("success_parent_action_drift_mse", 1.0) > 0.0025
        or fitted.get("schema") != "vq2_lc070_phase2_constrained_endpoint_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or fit_report.get("safety", {}).get("runtime_privileged_values") != 0
        or fit_report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit_report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC062/LC070 do not authorize LC071")
    return parent


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        base.verify_inputs, base.build_candidate_context,
        base.apply_candidate_actions, base.candidate_state_for_index,
        base.candidate_metadata_for_index,
    )
    (
        base.verify_inputs, base.build_candidate_context,
        base.apply_candidate_actions, base.candidate_state_for_index,
        base.candidate_metadata_for_index,
    ) = (
        verify_inputs, build_candidate_context, residual.apply_candidate_actions,
        candidate_state_for_index, candidate_metadata_for_index,
    )
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            base.verify_inputs, base.build_candidate_context,
            base.apply_candidate_actions, base.candidate_state_for_index,
            base.candidate_metadata_for_index,
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
