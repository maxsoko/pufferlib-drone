#!/usr/bin/env python3
"""Reduced parent-versus-LC123 teacher-free raw-index-10 milestone."""

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

from scripts.eval_vq2_lc015_restore_vg071_head1 import state_sha256
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc121_phase6_partial_endpoint_scale_screen as base


BASE_VERIFY_INPUTS = base.verify_inputs
BASE_CONFIGURE = base.configure

TAG = "vq2_lc124_phase6_success_rescue_milestone_001"
SCHEMA = "vq2_lc124_phase6_success_rescue_milestone_report_v1"
ALPHAS = (0.0, 1.0)
CANDIDATES = (
    ("baseline_lc105", (0.0,) * 4),
    ("lc123_success_rescue_anchor", (0.0,) * 4),
)
TARGET_RAW_INDEX = 10
FIT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc123_phase6_success_rescue_anchor_001"
)
FIT_CHECKPOINT = FIT_DIR / "policy_selected.pt"
FIT_CHECKPOINT_SHA256 = "0d0d9f6ba796dbf1803521214a0c030a79673f24cff538ae05338183ded4f558"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "92f29cdd0f3c76b2c031be3acdcf5bc23fecc9c7e499cafe3cf6edc5758c0bf9"
PREREGISTRATION = ROOT / "docs/vq2_lc124_phase6_success_rescue_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc124_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc124_phase6_success_rescue_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def candidate_payload() -> dict[str, Any]:
    return torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    if candidate_index not in (0, 1):
        raise ValueError("LC124 candidate index must be parent or LC123")
    if candidate_index == 0:
        return {
            name: value.detach().cpu().clone()
            for name, value in parent_state.items()
        }
    return {
        name: value.detach().cpu().clone()
        for name, value in candidate_payload()["model_state"].items()
    }


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    if candidate_index == 0:
        return {
            "fit_scale": 0.0, "parameter_delta_l2": 0.0,
            "bias_l2": 0.0, "endpoint_phases": [],
        }
    selected = json.loads(FIT_REPORT.read_text())["selected"]
    return {
        "fit_scale": selected["scale"],
        "fit_step": selected["step"],
        "parameter_delta_l2": selected["parameter_delta_l2"],
        "bias_l2": selected["parameter_delta_l2"],
        "endpoint_phases": [6],
    }


def verify_inputs() -> dict[str, Any]:
    parent = BASE_VERIFY_INPUTS()
    for path, digest in {
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC124 bound LC123 artifact changed: {path}")
    payload = candidate_payload()
    report = json.loads(FIT_REPORT.read_text())
    parent_hash = state_sha256(parent["model_state"])
    changed = [
        name for name in parent["model_state"]
        if not torch.equal(parent["model_state"][name], payload["model_state"][name])
    ]
    expected_changed = {
        "indexed_phase_residual_input",
        "indexed_phase_residual_input_bias",
        "indexed_phase_residual_output",
        "indexed_phase_residual_output_bias",
    }
    if (
        payload.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or report.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_report_v1"
        or not report.get("numerically_admitted")
        or report.get("checkpoint_sha256") != FIT_CHECKPOINT_SHA256
        or report.get("parent_state_sha256") != parent_hash
        or report.get("selected", {}).get("success_improvement_factor", 0.0) < 1.20
        or report.get("selected", {}).get(
            "validation_failure_parent_action_drift_mse", 1.0
        ) > 0.00025
        or set(changed) != expected_changed
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC123 does not authorize LC124")
    for name in expected_changed:
        parent_tensor = parent["model_state"][name]
        candidate_tensor = payload["model_state"][name]
        phase_equal = torch.tensor([
            torch.equal(parent_tensor[index], candidate_tensor[index])
            for index in range(parent_tensor.shape[0])
        ])
        if not bool(phase_equal[:6].all()) or bool(phase_equal[6]) or not bool(
            phase_equal[7:].all()
        ):
            raise RuntimeError(f"LC123 changed a non-phase-6 tensor row: {name}")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.ALPHAS, base.CANDIDATES = ALPHAS, CANDIDATES
    base.TARGET_PHASE, base.TARGET_RAW_INDEX = 6, TARGET_RAW_INDEX
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    BASE_CONFIGURE()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT,
        ROOT / "scripts/train_vq2_lc123_phase6_success_rescue_anchor.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one independent parent-versus-candidate confirmation of LC123."
    )
    milestone.NEXT_AUTHORITY_NONE = "Reject LC123 and retain LC105."
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form Puffer actors; each executes a baseline-plus-candidate 256-row pair and only LC123 phase-6 tensors differ"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    originals = (
        base.verify_inputs, base.configure,
        base.candidate_state_for_index, base.candidate_metadata_for_index,
    )
    base.verify_inputs = verify_inputs
    base.candidate_state_for_index = candidate_state_for_index
    base.candidate_metadata_for_index = candidate_metadata_for_index
    try:
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
