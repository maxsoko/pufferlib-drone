#!/usr/bin/env python3
"""Pairwise screen of LC111's phase-9 head with LC105 phase 8 preserved."""

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

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc112_phase8_9_full_residual_scale_screen as base


TAG = "vq2_lc113_phase9_only_full_residual_screen_001"
SCHEMA = "vq2_lc113_phase9_only_full_residual_screen_report_v1"
ALPHAS = (0.0, 0.01, 0.03, 0.10)
CANDIDATES = tuple(
    ("baseline_lc105" if alpha == 0.0 else f"lc111_phase9_scale_{alpha:g}", (0.0,) * 4)
    for alpha in ALPHAS
)
SEED = 432_130
TARGET_PHASE = 9
LC112_REPORT = base.DEFAULT_OUTPUT / "report.json"
LC112_REPORT_SHA256 = "e84a5604cf251201f69ca93b0ffb6e8917f004f3ef17f4f47a532b4a2261bcca"
PREREGISTRATION = ROOT / "docs/vq2_lc113_phase9_only_full_residual_screen_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc113_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc113_phase9_only_full_residual_screen.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    endpoint_state = base.endpoint_payload()["model_state"]
    alpha = ALPHAS[candidate_index]
    for name in base.training.PARAMETER_NAMES:
        state[name][TARGET_PHASE].lerp_(endpoint_state[name][TARGET_PHASE], alpha)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    parent = torch.load(
        base.pairwise.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )["model_state"]
    endpoint_state = base.endpoint_payload()["model_state"]
    full_delta = torch.sqrt(sum(
        (endpoint_state[name][TARGET_PHASE] - parent[name][TARGET_PHASE])
        .double().square().sum()
        for name in base.training.PARAMETER_NAMES
    ))
    alpha = ALPHAS[candidate_index]
    return {
        "endpoint_scale": alpha,
        "parameter_delta_l2": float(alpha * full_delta),
        "bias_l2": float(alpha * full_delta),
        "endpoint_phases": [TARGET_PHASE],
    }


def verify_inputs() -> dict[str, Any]:
    parent = base.verify_inputs()
    if sha256_path(LC112_REPORT) != LC112_REPORT_SHA256:
        raise RuntimeError("LC113 bound LC112 report changed")
    rejected = json.loads(LC112_REPORT.read_text())
    items = rejected.get("items", [])
    if (
        rejected.get("schema") != "vq2_lc112_phase8_9_full_residual_scale_screen_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or items[0].get("maximum_raw_index_distribution", {}).get("9") != 1
        or any(
            item.get("maximum_raw_index_distribution", {}).get("9") != 0
            for item in items[1:]
        )
        or any(item.get("target_passes") != 0 for item in items)
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC112 does not authorize phase-9 isolation")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.ALPHAS, base.CANDIDATES = ALPHAS, CANDIDATES
    base.SEED = SEED
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.configure()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), LC112_REPORT,
        base.ENDPOINT, base.ENDPOINT_REPORT,
        ROOT / "scripts/eval_vq2_lc112_phase8_9_full_residual_scale_screen.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one independent pairwise-256 confirmation of the selected phase-9-only scale."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject the LC111 phase-9 head and retain LC105."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "four complete saved-form Puffer actors with LC105 phase 8 exact and interpolated LC111 phase 9, each executing a 256-row pair"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    original_state = base.pairwise.candidate_state_for_index
    original_metadata = base.pairwise.candidate_metadata_for_index
    base.pairwise.candidate_state_for_index = candidate_state_for_index
    base.pairwise.candidate_metadata_for_index = candidate_metadata_for_index
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
        verify_inputs, base.pairwise.build_candidate_context,
        base.pairwise.initialize_actor_execution,
        base.pairwise.execute_actor_actions,
        candidate_state_for_index, candidate_metadata_for_index,
    )
    try:
        return milestone.run(
            output=output, device_name=device_name, resume=resume
        )
    finally:
        base.pairwise.candidate_state_for_index = original_state
        base.pairwise.candidate_metadata_for_index = original_metadata
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
