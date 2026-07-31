#!/usr/bin/env python3
"""Fast closed-loop screen of failure-conditioned phase-2 Puffer actions."""

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


TAG = "vq2_lc066_phase2_failure_conditioned_001"
SCHEMA = "vq2_lc066_phase2_failure_conditioned_report_v1"
GROUP_SIZE = 32
TARGET_PHASE = 2
COEFFICIENTS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("baseline_lc062", (0.0, 0.0, 0.0, 0.0)),
    ("failure_pitch_m0p0025", (-0.0025, 0.0, 0.0, 0.0)),
    ("failure_pitch_m0p0050", (-0.0050, 0.0, 0.0, 0.0)),
    ("failure_pitch_m0p0100", (-0.0100, 0.0, 0.0, 0.0)),
    ("failure_roll_p0p0025", (0.0, 0.0025, 0.0, 0.0)),
    ("failure_pitch_m_roll_p", (-0.0025, 0.0025, 0.0, 0.0)),
    ("failure_thrust_p0p0025", (0.0, 0.0, 0.0025, 0.0)),
    ("failure_pitch_m_thrust_p", (-0.0025, 0.0, 0.0025, 0.0)),
)
SEED = 431660
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
    / "vq2_lc065_phase2_failure_direction_001"
)
FIT_CHECKPOINT = FIT_DIR / "failure_direction.pt"
FIT_CHECKPOINT_SHA256 = "59bffd577c9ef9cc30e943fb1dfff83fe7de71e191222045c0f3593e538e75fa"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "3a3eb85a84bbd340e8b192a1e2cf7159f17c5525bf49ee7b907c60ef75844460"
PREREGISTRATION = ROOT / "docs/vq2_lc066_phase2_failure_conditioned_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc066_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc066_phase2_failure_conditioned.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()
NEXT_AUTHORITY_SELECTED = (
    "Run one larger different-seed Gate-3 confirmation of the selected "
    "failure-conditioned whole-Puffer decoder surgery."
)
NEXT_AUTHORITY_NONE = "Reject the failure-conditioned actions and retain LC062."


@functools.lru_cache(maxsize=1)
def failure_direction() -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    return payload["feature_weight"].detach().cpu(), payload["feature_bias"].detach().cpu()


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    coefficient = torch.tensor(COEFFICIENTS[candidate_index][1], dtype=torch.float32)
    weight, bias = failure_direction()
    output_name = "indexed_phase_residual_output"
    bias_name = "indexed_phase_residual_output_bias"
    state[output_name][TARGET_PHASE].add_(coefficient[:, None] * weight[None, :])
    state[bias_name][TARGET_PHASE].add_(coefficient * bias)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    coefficient = torch.tensor(COEFFICIENTS[candidate_index][1], dtype=torch.float64)
    weight, bias = failure_direction()
    squared = float(weight.double().square().sum() + bias.double().square())
    delta_l2 = float(coefficient.norm()) * math.sqrt(squared)
    return {
        "failure_action_coefficient": coefficient.tolist(),
        "parameter_delta_l2": delta_l2,
        "bias_l2": delta_l2,
    }


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, torch.Tensor]:
    states = [
        candidate_state_for_index(payload["model_state"], index)
        for index in range(len(COEFFICIENTS))
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
    base.BIAS_CANDIDATES = COEFFICIENTS
    base.GROUPS = len(COEFFICIENTS)
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
        ROOT / "scripts/train_vq2_lc065_phase2_failure_direction.py",
        ROOT / "scripts/eval_vq2_lc064_phase2_residual_direction.py",
        *EXTRA_SOURCE_PATHS,
    )
    base.NEXT_AUTHORITY_SELECTED = NEXT_AUTHORITY_SELECTED
    base.NEXT_AUTHORITY_NONE = NEXT_AUTHORITY_NONE
    base.VECTORIZED_CHECKPOINT_SURGERY = (
        "failure-score outer product merged exactly into phase-2 Puffer output weight and bias"
    )


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC066 bound input changed: {path}")
    parent_report = json.loads(PARENT_REPORT.read_text())
    fit_report = json.loads(FIT_REPORT.read_text())
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    fitted = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        parent_report.get("schema") != "vq2_lc062_phase2_bias_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent.get("schema") != "vq2_lc062_phase2_bias_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or fit_report.get("schema") != "vq2_lc065_phase2_failure_direction_report_v1"
        or not fit_report.get("numerically_admitted")
        or fit_report.get("validation_agent_auc", 0.0) < 0.86
        or fitted.get("schema") != "vq2_lc065_phase2_failure_direction_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("feature_size") != 64
        or fit_report.get("safety", {}).get("runtime_privileged_values") != 0
        or fit_report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit_report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC062/LC065 do not authorize LC066")
    return parent


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
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
