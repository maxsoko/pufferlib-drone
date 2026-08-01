#!/usr/bin/env python3
"""Teacher-free pairwise scale screen of LC120's admitted phase-6 sub-head."""

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
import scripts.eval_vq2_lc103_phase7_pairwise_batch_scale_screen as pairwise


TAG = "vq2_lc121_phase6_partial_endpoint_scale_screen_001"
SCHEMA = "vq2_lc121_phase6_partial_endpoint_scale_screen_report_v1"
GROUP_SIZE = 128
ALPHAS = (0.0, 0.10, 0.30, 1.0)
CANDIDATES = tuple(
    ("baseline_lc105" if alpha == 0.0 else f"lc120_phase6_scale_{alpha:g}", (0.0,) * 4)
    for alpha in ALPHAS
)
SEED = 432_050
TARGET_PHASE = 6
TARGET_RAW_INDEX = 10
MAX_STEPS = 12_000
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc105_phase7_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "005e5e7929258fd282ab390fd230fda817700afaee790e78144200e67b3e10a4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c614e6929282399cac2f18465084f8337ed8770389a74e48b86ff1fcf4315e7d"
FIT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc120_phase6_9_outcome_balanced_distillation_001"
)
FIT_CHECKPOINT = FIT_DIR / "policy_endpoint.pt"
FIT_CHECKPOINT_SHA256 = "d30928abe640de7edfcf79900c034ac0be6c24410b196e61c0b5a6d6d6b7eaf1"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "5fd1ed3ebb6540a02e3826676021ce67e7662f60ff87e369154e23718fbfc48a"
PREREGISTRATION = ROOT / "docs/vq2_lc121_phase6_partial_endpoint_scale_screen_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc121_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc121_phase6_partial_endpoint_scale_screen.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def endpoint_payload() -> dict[str, Any]:
    return torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    endpoint = endpoint_payload()["model_state"]
    alpha = ALPHAS[candidate_index]
    for name in PARAMETER_NAMES:
        state[name][TARGET_PHASE].lerp_(endpoint[name][TARGET_PHASE], alpha)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)[
        "model_state"
    ]
    endpoint = endpoint_payload()["model_state"]
    full_delta = torch.sqrt(sum(
        (endpoint[name][TARGET_PHASE] - parent[name][TARGET_PHASE])
        .double().square().sum()
        for name in PARAMETER_NAMES
    ))
    alpha = ALPHAS[candidate_index]
    return {
        "endpoint_scale": alpha,
        "parameter_delta_l2": float(alpha * full_delta),
        "bias_l2": float(alpha * full_delta),
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
            raise RuntimeError(f"LC121 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    fitted = endpoint_payload()
    fit_report = json.loads(FIT_REPORT.read_text())
    phases = fit_report.get("phase_reports", {})
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or fitted.get("schema")
        != "vq2_lc120_phase6_9_outcome_balanced_distillation_checkpoint_v1"
        or fitted.get("numerically_admitted")
        or fit_report.get("schema")
        != "vq2_lc120_phase6_9_outcome_balanced_distillation_report_v1"
        or fit_report.get("numerically_admitted")
        or phases.get("6", {}).get("numerically_admitted") is not True
        or phases.get("6", {}).get("overall_improvement_factor", 0.0) < 3.39
        or phases.get("6", {}).get("success_improvement_factor", 0.0) < 2.77
        or phases.get("6", {}).get("failure_improvement_factor", 0.0) < 3.55
        or phases.get("7", {}).get("best_step") != 0
        or phases.get("9", {}).get("best_step") != 0
        or fit_report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit_report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC120 do not authorize the phase-6 partial screen")
    return parent


def configure() -> None:
    pairwise.TAG, pairwise.SCHEMA = TAG, SCHEMA
    pairwise.GROUP_SIZE, pairwise.PAIR_SIZE = GROUP_SIZE, GROUP_SIZE * 2
    pairwise.ALPHAS, pairwise.CANDIDATES = ALPHAS, CANDIDATES
    pairwise.SEED = SEED
    pairwise.TARGET_PHASE, pairwise.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    pairwise.MAX_STEPS, pairwise.MINIMUM_PASS_GAIN = MAX_STEPS, MINIMUM_PASS_GAIN
    pairwise.PREREGISTRATION, pairwise.RUNNER, pairwise.TEST = (
        PREREGISTRATION, RUNNER, TEST,
    )
    pairwise.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    pairwise.prior.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    pairwise.prior.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    pairwise.prior.PARENT_REPORT = PARENT_REPORT
    pairwise.prior.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    pairwise.prior.FIT_CHECKPOINT = FIT_CHECKPOINT
    pairwise.prior.FIT_CHECKPOINT_SHA256 = FIT_CHECKPOINT_SHA256
    pairwise.prior.FIT_REPORT = FIT_REPORT
    pairwise.prior.FIT_REPORT_SHA256 = FIT_REPORT_SHA256
    pairwise.configure()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT,
        ROOT / "scripts/eval_vq2_lc103_phase7_pairwise_batch_scale_screen.py",
        ROOT / "scripts/train_vq2_lc120_phase6_9_outcome_balanced_distillation.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one independent pairwise-256 confirmation of the selected phase-6 scale."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject the LC120 phase-6 partial endpoint and retain LC105."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "four complete saved-form Puffer actors; each executes a baseline-plus-candidate 256-row pair and only phase-6 residual tensors differ"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    original_state = pairwise.prior.candidate_state_for_index
    original_metadata = pairwise.prior.candidate_metadata_for_index
    pairwise.prior.candidate_state_for_index = candidate_state_for_index
    pairwise.prior.candidate_metadata_for_index = candidate_metadata_for_index
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
        pairwise.prior.candidate_state_for_index = original_state
        pairwise.prior.candidate_metadata_for_index = original_metadata
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
