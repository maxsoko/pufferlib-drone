#!/usr/bin/env python3
"""Pairwise-256 full-start scale screen of the LC111 phase-8/9 endpoint."""

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
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc110_phase8_success_action_bias as pairwise
import scripts.train_vq2_lc111_phase8_9_full_residual_endpoint as training


TAG = "vq2_lc112_phase8_9_full_residual_scale_screen_001"
SCHEMA = "vq2_lc112_phase8_9_full_residual_scale_screen_report_v1"
ALPHAS = (0.0, 0.01, 0.03, 0.10)
CANDIDATES = tuple(
    ("baseline_lc105" if alpha == 0.0 else f"lc111_scale_{alpha:g}", (0.0,) * 4)
    for alpha in ALPHAS
)
SEED = 432_120
TARGET_PHASE = 9
TARGET_RAW_INDEX = 10
MINIMUM_PASS_GAIN = 1
ENDPOINT = training.DEFAULT_OUTPUT / "policy_endpoint.pt"
ENDPOINT_SHA256 = "64fa2e09b7f91bea2f0fe733125916d2069cbefeff8479b9248dc2302c7932ab"
ENDPOINT_REPORT = training.DEFAULT_OUTPUT / "report.json"
ENDPOINT_REPORT_SHA256 = "ebaddc1c4b6be2fbd6d6010b07fe58d32fe12d30e1d86e75080cd30e82e8b2af"
LC110_REPORT = pairwise.DEFAULT_OUTPUT / "report.json"
LC110_REPORT_SHA256 = "0310afe43e4cd80d9b880fc57d0d31df9b222a708449636827b37440bf5c4d39"
PREREGISTRATION = ROOT / "docs/vq2_lc112_phase8_9_full_residual_scale_screen_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc112_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc112_phase8_9_full_residual_scale_screen.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


@functools.lru_cache(maxsize=1)
def endpoint_payload() -> dict[str, Any]:
    return torch.load(ENDPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    endpoint_state = endpoint_payload()["model_state"]
    alpha = ALPHAS[candidate_index]
    for name in training.PARAMETER_NAMES:
        for phase in training.PHASES:
            state[name][phase].lerp_(endpoint_state[name][phase], alpha)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    parent = torch.load(
        pairwise.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )["model_state"]
    endpoint_state = endpoint_payload()["model_state"]
    full_delta = torch.sqrt(sum(
        (endpoint_state[name][phase] - parent[name][phase]).double().square().sum()
        for name in training.PARAMETER_NAMES
        for phase in training.PHASES
    ))
    alpha = ALPHAS[candidate_index]
    return {
        "endpoint_scale": alpha,
        "parameter_delta_l2": float(alpha * full_delta),
        "bias_l2": float(alpha * full_delta),
        "endpoint_phases": list(training.PHASES),
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        pairwise.PARENT_CHECKPOINT: pairwise.PARENT_CHECKPOINT_SHA256,
        pairwise.PARENT_REPORT: pairwise.PARENT_REPORT_SHA256,
        ENDPOINT: ENDPOINT_SHA256,
        ENDPOINT_REPORT: ENDPOINT_REPORT_SHA256,
        LC110_REPORT: LC110_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC112 bound input changed: {path}")
    parent = torch.load(
        pairwise.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    parent_report = json.loads(pairwise.PARENT_REPORT.read_text())
    endpoint = endpoint_payload()
    fit = json.loads(ENDPOINT_REPORT.read_text())
    rejected = json.loads(LC110_REPORT.read_text())
    phase_reports = fit.get("phase_reports", {})
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != pairwise.PARENT_CHECKPOINT_SHA256
        or endpoint.get("schema") != "vq2_lc111_phase8_9_full_residual_endpoint_checkpoint_v1"
        or not endpoint.get("numerically_admitted")
        or endpoint.get("parent_checkpoint_sha256") != pairwise.PARENT_CHECKPOINT_SHA256
        or fit.get("schema") != "vq2_lc111_phase8_9_full_residual_endpoint_report_v1"
        or not fit.get("numerically_admitted")
        or fit.get("checkpoint_sha256") != ENDPOINT_SHA256
        or phase_reports.get("8", {}).get("improvement_factor", 0.0) < 10.7
        or phase_reports.get("9", {}).get("improvement_factor", 0.0) < 9.0
        or rejected.get("schema") != "vq2_lc110_phase8_success_action_bias_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or fit.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC110/LC111 do not authorize LC112")
    return parent


def configure() -> None:
    pairwise.TAG, pairwise.SCHEMA = TAG, SCHEMA
    pairwise.BIASES = CANDIDATES
    pairwise.SEED = SEED
    pairwise.TARGET_PHASE, pairwise.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    pairwise.MINIMUM_PASS_GAIN = MINIMUM_PASS_GAIN
    pairwise.PREREGISTRATION, pairwise.RUNNER, pairwise.TEST = (
        PREREGISTRATION, RUNNER, TEST,
    )
    pairwise.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    pairwise.configure()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), ENDPOINT, ENDPOINT_REPORT, LC110_REPORT,
        ROOT / "scripts/train_vq2_lc111_phase8_9_full_residual_endpoint.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one independent pairwise-256 confirmation of the selected full-residual scale."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject the small LC111 scales and retain LC105."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "four complete saved-form Puffer actors with interpolated phase-8/9 residual MLP heads, each executing a 256-row pair"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    original_state = pairwise.candidate_state_for_index
    original_metadata = pairwise.candidate_metadata_for_index
    pairwise.candidate_state_for_index = candidate_state_for_index
    pairwise.candidate_metadata_for_index = candidate_metadata_for_index
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
        verify_inputs, pairwise.build_candidate_context,
        pairwise.initialize_actor_execution, pairwise.execute_actor_actions,
        candidate_state_for_index, candidate_metadata_for_index,
    )
    try:
        return milestone.run(
            output=output, device_name=device_name, resume=resume
        )
    finally:
        pairwise.candidate_state_for_index = original_state
        pairwise.candidate_metadata_for_index = original_metadata
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
