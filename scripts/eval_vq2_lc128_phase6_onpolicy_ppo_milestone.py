#!/usr/bin/env python3
"""Deterministic parent-versus-LC127 raw-index-10 milestone screen."""

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
from scripts.train_vq2_lc111_phase8_9_full_residual_endpoint import PARAMETER_NAMES


BASE_VERIFY_INPUTS = base.verify_inputs
BASE_CONFIGURE = base.configure

TAG = "vq2_lc128_phase6_onpolicy_ppo_milestone_001"
SCHEMA = "vq2_lc128_phase6_onpolicy_ppo_milestone_report_v1"
ALPHAS = (0.0, 1.0)
CANDIDATES = (
    ("baseline_lc105", (0.0,) * 4),
    ("lc127_rollout_04", (0.0,) * 4),
)
TARGET_RAW_INDEX = 10
TRAIN_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc127_phase6_onpolicy_ppo_001"
)
TRAIN_CHECKPOINT = TRAIN_DIR / "policy_rollout_04.pt"
TRAIN_CHECKPOINT_SHA256 = "31bc19b4ef57de61ca08f17600968cabd77b34051b01352a17e3a9ce57147dee"
TRAIN_REPORT = TRAIN_DIR / "report.json"
TRAIN_REPORT_SHA256 = "fa2458e2788939497c1890db21224384db5f5c49222c2ab8a851bbfa19d704b6"
PREREGISTRATION = ROOT / "docs/vq2_lc128_phase6_onpolicy_ppo_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc128_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc128_phase6_onpolicy_ppo_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def candidate_payload() -> dict[str, Any]:
    return torch.load(TRAIN_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    if candidate_index not in (0, 1):
        raise ValueError("LC128 candidate index must be parent or LC127")
    source = parent_state if candidate_index == 0 else candidate_payload()["model_state"]
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    if candidate_index == 0:
        return {
            "onpolicy_rollout_iteration": 0,
            "parameter_delta_l2": 0.0, "bias_l2": 0.0,
            "endpoint_phases": [],
        }
    parent = torch.load(
        base.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )["model_state"]
    candidate = candidate_payload()["model_state"]
    delta = torch.sqrt(sum(
        (candidate[name][6] - parent[name][6]).double().square().sum()
        for name in PARAMETER_NAMES
    ))
    return {
        "onpolicy_rollout_iteration": 4,
        "parameter_delta_l2": float(delta), "bias_l2": float(delta),
        "endpoint_phases": [6],
    }


def verify_inputs() -> dict[str, Any]:
    parent = BASE_VERIFY_INPUTS()
    for path, digest in {
        TRAIN_CHECKPOINT: TRAIN_CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC128 bound LC127 artifact changed: {path}")
    payload = candidate_payload()
    report = json.loads(TRAIN_REPORT.read_text())
    selected = report.get("candidate_selected_for_screen", {})
    parent_hash = state_sha256(parent["model_state"])
    changed = [
        name for name in parent["model_state"]
        if not torch.equal(parent["model_state"][name], payload["model_state"][name])
    ]
    if (
        payload.get("schema") != "vq2_lc127_phase6_onpolicy_ppo_checkpoint_v1"
        or payload.get("onpolicy_rollout_iteration") != 4
        or payload.get("numerically_admitted")
        or payload.get("deployment_candidate")
        or report.get("schema") != "vq2_lc127_phase6_onpolicy_ppo_report_v1"
        or not report.get("training_admitted")
        or selected.get("iteration") != 4
        or selected.get("sha256") != TRAIN_CHECKPOINT_SHA256
        or report.get("parent_state_sha256") != parent_hash
        or report.get("selected_rollout_metrics", {}).get(
            "mean_maximum_raw_index", 0.0
        ) != 3.375
        or set(changed) != set(PARAMETER_NAMES)
        or not report.get("frozen_non_phase6_state_exact")
        or report.get("safety", {}).get("offline_training_teacher_actions") != 0
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC127 does not authorize LC128")
    for name in PARAMETER_NAMES:
        parent_tensor = parent["model_state"][name]
        candidate_tensor = payload["model_state"][name]
        equal = torch.tensor([
            torch.equal(parent_tensor[index], candidate_tensor[index])
            for index in range(parent_tensor.shape[0])
        ])
        if not bool(equal[:6].all()) or bool(equal[6]) or not bool(equal[7:].all()):
            raise RuntimeError(f"LC127 changed a non-phase-6 row: {name}")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.ALPHAS, base.CANDIDATES = ALPHAS, CANDIDATES
    base.TARGET_PHASE, base.TARGET_RAW_INDEX = 6, TARGET_RAW_INDEX
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    BASE_CONFIGURE()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), TRAIN_CHECKPOINT, TRAIN_REPORT,
        ROOT / "scripts/train_vq2_lc127_phase6_onpolicy_ppo.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one independent parent-versus-candidate confirmation of LC127."
    )
    milestone.NEXT_AUTHORITY_NONE = "Reject LC127 and retain LC105."
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form Puffer actors; each executes a baseline-plus-candidate 256-row pair and only LC127 phase-6 tensors differ"
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
