#!/usr/bin/env python3
"""Teacher-free LC123-versus-LC137 raw-index-9 milestone screen."""

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


BASE_CONFIGURE = base.configure

TAG = "vq2_lc138_phase8_success_fit_milestone_001"
SCHEMA = "vq2_lc138_phase8_success_fit_milestone_report_v1"
ALPHAS = (0.0, 1.0)
CANDIDATES = (
    ("baseline_lc123", (0.0,) * 4),
    ("lc137_phase8_success_fit", (0.0,) * 4),
)
TARGET_PHASE = 8
TARGET_RAW_INDEX = 9
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc123_phase6_success_rescue_anchor_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "0d0d9f6ba796dbf1803521214a0c030a79673f24cff538ae05338183ded4f558"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "92f29cdd0f3c76b2c031be3acdcf5bc23fecc9c7e499cafe3cf6edc5758c0bf9"
FIT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc137_phase8_success_fit_threshold_001"
)
FIT_CHECKPOINT = FIT_DIR / "policy_selected.pt"
FIT_CHECKPOINT_SHA256 = "2534d80f40e20298f96a50bac77813e148446cad827ca8852977b9e8831e6a5d"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "4119229336bf88d32070803919277ce2b94b1d05a4c0b1290caee1c62be984ec"
PREREGISTRATION = ROOT / "docs/vq2_lc138_phase8_success_fit_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc138_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc138_phase8_success_fit_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def parent_payload() -> dict[str, Any]:
    return torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_payload() -> dict[str, Any]:
    return torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    if candidate_index not in (0, 1):
        raise ValueError("LC138 candidate index must be parent or LC137")
    source = parent_state if candidate_index == 0 else candidate_payload()["model_state"]
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    if candidate_index == 0:
        return {
            "fit_scale": 0.0, "parameter_delta_l2": 0.0,
            "bias_l2": 0.0, "endpoint_phases": [],
        }
    selected = json.loads(FIT_REPORT.read_text())["selected"]
    return {
        "fit_scale": selected["scale"], "fit_step": selected["step"],
        "parameter_delta_l2": selected["parameter_delta_l2"],
        "bias_l2": selected["parameter_delta_l2"],
        "endpoint_phases": [TARGET_PHASE],
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC138 bound artifact changed: {path}")
    parent = parent_payload()
    parent_report = json.loads(PARENT_REPORT.read_text())
    candidate = candidate_payload()
    fit_report = json.loads(FIT_REPORT.read_text())
    changed = [
        name for name in parent["model_state"]
        if not torch.equal(parent["model_state"][name], candidate["model_state"][name])
    ]
    expected_changed = {
        "indexed_phase_residual_input",
        "indexed_phase_residual_input_bias",
        "indexed_phase_residual_output",
        "indexed_phase_residual_output_bias",
    }
    if (
        parent.get("schema") != "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or candidate.get("schema")
        != "vq2_lc137_phase8_success_fit_threshold_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or fit_report.get("schema")
        != "vq2_lc137_phase8_success_fit_threshold_report_v1"
        or not fit_report.get("numerically_admitted")
        or fit_report.get("checkpoint_sha256") != FIT_CHECKPOINT_SHA256
        or fit_report.get("parent_state_sha256") != state_sha256(parent["model_state"])
        or fit_report.get("selected", {}).get("success_improvement_factor", 0.0) < 1.4
        or fit_report.get("selected", {}).get(
            "paired_control_parent_action_drift_mse", 1.0
        ) > 0.02
        or set(changed) != expected_changed
        or fit_report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit_report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC123/LC137 do not authorize LC138")
    for name in expected_changed:
        parent_tensor = parent["model_state"][name]
        candidate_tensor = candidate["model_state"][name]
        equal = torch.tensor([
            torch.equal(parent_tensor[index], candidate_tensor[index])
            for index in range(parent_tensor.shape[0])
        ])
        if not bool(equal[:TARGET_PHASE].all()) or bool(equal[TARGET_PHASE]) or not bool(
            equal[TARGET_PHASE + 1 :].all()
        ):
            raise RuntimeError(f"LC137 changed a non-phase-8 tensor row: {name}")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.ALPHAS, base.CANDIDATES = ALPHAS, CANDIDATES
    base.TARGET_PHASE, base.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.FIT_CHECKPOINT, base.FIT_CHECKPOINT_SHA256 = FIT_CHECKPOINT, FIT_CHECKPOINT_SHA256
    base.FIT_REPORT, base.FIT_REPORT_SHA256 = FIT_REPORT, FIT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    BASE_CONFIGURE()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT,
        ROOT / "scripts/train_vq2_lc137_phase8_success_fit_threshold.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Promote LC137 only as the offline raw-9 phase milestone and collect a new phase-9 training bridge; no FlightSim authority."
    )
    milestone.NEXT_AUTHORITY_NONE = "Reject LC137 and retain LC105."
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form Puffer actors; each executes a baseline-plus-candidate 256-row pair and only LC137 phase-8 tensors differ"
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
