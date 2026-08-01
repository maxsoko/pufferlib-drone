#!/usr/bin/env python3
"""Paired raw-index-7 screen of the success-anchored phase-6 endpoint."""

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

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc058_phase2_bias_milestone as base
import scripts.eval_vq2_lc064_phase2_residual_direction as residual


TAG = "vq2_lc081_phase6_success_endpoint_milestone_001"
SCHEMA = "vq2_lc081_phase6_success_endpoint_milestone_report_v1"
GROUP_SIZE = 128
TARGET_PHASE = 6
TARGET_RAW_INDEX = 7
MAX_STEPS = 12_000
ALPHAS = (0.0, 1.0)
CANDIDATES = (
    ("baseline_lc073", (0.0, 0.0, 0.0, 0.0)),
    ("lc080_exact", (0.0, 0.0, 0.0, 0.0)),
)
SEED = 431810
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc073_phase2_constrained_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "5614a95cd2f9b428a99ddd43ec5e324c095344cae51b8d771aea3d5f05d35be8"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "ae68af50524b02425f9a970c7f87dccbcbb6b60a0debb9899fcf1baa167b5d75"
FIT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc080_phase6_success_anchored_endpoint_001"
)
FIT_CHECKPOINT = FIT_DIR / "decoder_endpoint.pt"
FIT_CHECKPOINT_SHA256 = "53281b52b311f5833ef3abab00995ff3a5a55899a12e0214f6edfaf44acd0553"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "5c48293c915e2d29957cf87e0c80104d6b05bbadfe62af72e691d20946af0df0"
PREREGISTRATION = ROOT / "docs/vq2_lc081_phase6_success_endpoint_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc081_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc081_phase6_success_endpoint_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


@functools.lru_cache(maxsize=1)
def endpoint_decoder() -> dict[str, Any]:
    return torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    if candidate_index == 1:
        endpoint = endpoint_decoder()
        state["indexed_phase_residual_output"][TARGET_PHASE].copy_(endpoint["output_weight"])
        state["indexed_phase_residual_output_bias"][TARGET_PHASE].copy_(endpoint["output_bias"])
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    endpoint = endpoint_decoder()
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)["model_state"]
    delta = torch.cat((
        (endpoint["output_weight"] - parent["indexed_phase_residual_output"][TARGET_PHASE]).flatten(),
        (endpoint["output_bias"] - parent["indexed_phase_residual_output_bias"][TARGET_PHASE]).flatten(),
    )).double()
    alpha = ALPHAS[candidate_index]
    return {
        "endpoint_interpolation_alpha": alpha,
        "parameter_delta_l2": alpha * float(delta.norm()),
        "bias_l2": alpha * float(delta.norm()),
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


def apply_candidate_actions(
    actor_output: Any,
    next_recurrent: torch.Tensor,
    held_progress: torch.Tensor,
    context: dict[str, torch.Tensor],
) -> torch.Tensor:
    hidden = next_recurrent[0]
    feature = torch.tanh(
        torch.einsum("bh,brh->br", hidden, context[residual.RESIDUAL_NAMES[0]])
        + context[residual.RESIDUAL_NAMES[1]]
    )
    candidate_residual = (
        torch.einsum("br,bor->bo", feature, context[residual.RESIDUAL_NAMES[2]])
        + context[residual.RESIDUAL_NAMES[3]]
    )
    parent_feature = torch.tanh(
        hidden @ context[f"parent_{residual.RESIDUAL_NAMES[0]}"].T
        + context[f"parent_{residual.RESIDUAL_NAMES[1]}"]
    )
    parent_residual = (
        parent_feature @ context[f"parent_{residual.RESIDUAL_NAMES[2]}"].T
        + context[f"parent_{residual.RESIDUAL_NAMES[3]}"]
    )
    phase = torch.round(
        held_progress[:, 0] * base.OFFICIAL_PROGRESS_SCALE
    ).to(torch.long)
    delta = candidate_residual - parent_residual
    return torch.tanh(
        actor_output.pre_tanh_mean + (phase == TARGET_PHASE)[:, None] * delta
    )


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.GROUP_SIZE = GROUP_SIZE
    base.BIAS_CANDIDATES = CANDIDATES
    base.GROUPS = len(CANDIDATES)
    base.TOTAL_AGENTS = GROUP_SIZE * base.GROUPS
    base.EPISODES = base.TOTAL_AGENTS
    base.SEED = SEED
    base.TARGET_PHASE = TARGET_PHASE
    base.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    base.MAX_STEPS = MAX_STEPS
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
        ROOT / "scripts/fit_vq2_lc080_phase6_success_anchored_endpoint.py",
        ROOT / "scripts/eval_vq2_lc064_phase2_residual_direction.py",
    )
    base.NEXT_AUTHORITY_SELECTED = (
        "Run one larger different-seed raw-index-7 confirmation of LC080."
    )
    base.NEXT_AUTHORITY_NONE = "Reject LC080 and retain LC073."
    base.VECTORIZED_CHECKPOINT_SURGERY = (
        "exact phase-6 Puffer decoder replacement by the constrained LC080 endpoint"
    )


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC081 bound input changed: {path}")
    parent_report = json.loads(PARENT_REPORT.read_text())
    fit_report = json.loads(FIT_REPORT.read_text())
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    fitted = endpoint_decoder()
    if (
        parent_report.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or fit_report.get("schema") != "vq2_lc080_phase6_success_anchored_endpoint_report_v1"
        or not fit_report.get("numerically_admitted")
        or fit_report.get("selected", {}).get("ridge") != 0.01
        or fit_report.get("selected", {}).get("alpha") != 0.025
        or fit_report.get("selected", {}).get("failure_improvement_factor", 0.0) < 1.037
        or fit_report.get("selected", {}).get("success_parent_action_drift_mse", 1.0) > 0.00025
        or fitted.get("schema") != "vq2_lc080_phase6_success_anchored_endpoint_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("target_phase") != TARGET_PHASE
        or fitted.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or fit_report.get("safety", {}).get("runtime_privileged_values") != 0
        or fit_report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit_report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC073/LC080 do not authorize LC081")
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
        verify_inputs, build_candidate_context, apply_candidate_actions,
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
