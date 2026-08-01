#!/usr/bin/env python3
"""Compose LC120's admitted phase-6 and phase-8 sub-heads teacher-free."""

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
from scripts.train_vq2_lc111_phase8_9_full_residual_endpoint import PARAMETER_NAMES
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc121_phase6_partial_endpoint_scale_screen as base


TAG = "vq2_lc122_phase6_8_composite_scale_screen_001"
SCHEMA = "vq2_lc122_phase6_8_composite_scale_screen_report_v1"
PHASE6_SCALE = 0.30
PHASE8_SCALES = (0.0, 0.10, 0.30, 1.0)
CANDIDATES = (
    ("baseline_lc105", (0.0,) * 4),
    *((f"lc120_phase6_0.3_phase8_{scale:g}", (0.0,) * 4)
      for scale in PHASE8_SCALES[1:]),
)
LC121_REPORT = base.DEFAULT_OUTPUT / "report.json"
LC121_REPORT_SHA256 = "a23ec72e636626c9b318cb79775e3d67585a2789d533225d512b9628a4d30841"
PREREGISTRATION = ROOT / "docs/vq2_lc122_phase6_8_composite_scale_screen_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc122_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc122_phase6_8_composite_scale_screen.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    if candidate_index == 0:
        return state
    endpoint = base.endpoint_payload()["model_state"]
    phase8_scale = PHASE8_SCALES[candidate_index]
    for name in PARAMETER_NAMES:
        state[name][6].lerp_(endpoint[name][6], PHASE6_SCALE)
        state[name][8].lerp_(endpoint[name][8], phase8_scale)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    parent = torch.load(
        base.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )["model_state"]
    endpoint = base.endpoint_payload()["model_state"]
    phase8_scale = PHASE8_SCALES[candidate_index]
    phase6_scale = 0.0 if candidate_index == 0 else PHASE6_SCALE
    delta = torch.sqrt(sum(
        (phase6_scale * (endpoint[name][6] - parent[name][6]))
        .double().square().sum()
        + (phase8_scale * (endpoint[name][8] - parent[name][8]))
        .double().square().sum()
        for name in PARAMETER_NAMES
    ))
    return {
        "phase6_endpoint_scale": phase6_scale,
        "phase8_endpoint_scale": phase8_scale,
        "parameter_delta_l2": float(delta), "bias_l2": float(delta),
        "endpoint_phases": [] if candidate_index == 0 else [6, 8],
    }


def verify_inputs() -> dict[str, Any]:
    parent = base.verify_inputs()
    if sha256_path(LC121_REPORT) != LC121_REPORT_SHA256:
        raise RuntimeError("LC122 bound LC121 report changed")
    rejected = json.loads(LC121_REPORT.read_text())
    items = rejected.get("items", [])
    phase8 = json.loads(base.FIT_REPORT.read_text()).get("phase_reports", {}).get("8", {})
    if (
        rejected.get("schema") != "vq2_lc121_phase6_partial_endpoint_scale_screen_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or len(items) != 4
        or any(item.get("target_passes") != 0 for item in items)
        or items[2].get("endpoint_scale") != PHASE6_SCALE
        or items[2].get("maximum_raw_index_distribution", {}).get("9") != 1
        or items[2].get("maximum_raw_index_distribution", {}).get("8") != 2
        or phase8.get("numerically_admitted") is not True
        or phase8.get("overall_improvement_factor", 0.0) < 1.40
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC120/LC121 do not authorize LC122")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.ALPHAS = PHASE8_SCALES
    base.CANDIDATES = CANDIDATES
    base.TARGET_PHASE = 8
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.configure()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), LC121_REPORT,
        base.FIT_CHECKPOINT, base.FIT_REPORT,
        ROOT / "scripts/eval_vq2_lc121_phase6_partial_endpoint_scale_screen.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one independent pairwise-256 confirmation of the selected phase-6/8 composite."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject the LC120 phase-6/8 composite and retain LC105."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "four complete saved-form Puffer actors; each executes a baseline-plus-candidate 256-row pair and only phase-6/8 residual tensors differ"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    originals = (
        base.verify_inputs, base.configure,
        base.candidate_state_for_index, base.candidate_metadata_for_index,
    )
    original_configure = base.configure
    base.verify_inputs = verify_inputs
    base.candidate_state_for_index = candidate_state_for_index
    base.candidate_metadata_for_index = candidate_metadata_for_index
    base.configure = configure
    try:
        # configure() calls the saved LC121 configuration through this temporary restore.
        base.configure = original_configure
        configure()
        base.configure = lambda: None
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            base.verify_inputs, base.configure,
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
