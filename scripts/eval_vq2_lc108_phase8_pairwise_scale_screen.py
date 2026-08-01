#!/usr/bin/env python3
"""Pairwise-256 teacher-free scale screen of the LC107 phase-8 endpoint."""

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
import scripts.eval_vq2_lc103_phase7_pairwise_batch_scale_screen as pairwise


TAG = "vq2_lc108_phase8_pairwise_scale_screen_001"
SCHEMA = "vq2_lc108_phase8_pairwise_scale_screen_report_v1"
GROUP_SIZE = 128
ALPHAS = (0.0, 0.25, 0.50, 1.0)
CANDIDATES = tuple(
    ("baseline_lc105" if alpha == 0.0 else f"lc107_scale_{alpha:g}", (0.0,) * 4)
    for alpha in ALPHAS
)
SEED = 432_080
TARGET_PHASE = 8
TARGET_RAW_INDEX = 9
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
    / "vq2_lc107_phase8_success_anchored_endpoint_001"
)
FIT_CHECKPOINT = FIT_DIR / "decoder_endpoint.pt"
FIT_CHECKPOINT_SHA256 = "18884e5245d595d949bfe71a463a58fb3bdbde09caa82ec7ec80366426ad0e24"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "fc77e78669412a05be9bab2b5ea4ca309a78c776646f2558590f266bea5b9bff"
PREREGISTRATION = ROOT / "docs/vq2_lc108_phase8_pairwise_scale_screen_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc108_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc108_phase8_pairwise_scale_screen.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC108 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    fitted = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    fit_report = json.loads(FIT_REPORT.read_text())
    selected = fit_report.get("selected", {})
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or fitted.get("schema") != "vq2_lc107_phase8_success_anchored_endpoint_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("target_phase") != TARGET_PHASE
        or fitted.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or fit_report.get("schema") != "vq2_lc107_phase8_success_anchored_endpoint_report_v1"
        or not fit_report.get("numerically_admitted")
        or fit_report.get("checkpoint_sha256") != FIT_CHECKPOINT_SHA256
        or selected.get("ridge") != 1e-6
        or selected.get("alpha") != 0.05
        or selected.get("failure_improvement_factor", 0.0) < 1.25
        or selected.get("success_parent_action_drift_mse", 1.0) > 0.00025
        or fit_report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit_report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC107 do not authorize LC108")
    return parent


def configure() -> None:
    pairwise.TAG, pairwise.SCHEMA = TAG, SCHEMA
    pairwise.GROUP_SIZE, pairwise.PAIR_SIZE = GROUP_SIZE, GROUP_SIZE * 2
    pairwise.ALPHAS, pairwise.CANDIDATES = ALPHAS, CANDIDATES
    pairwise.SEED = SEED
    pairwise.TARGET_PHASE, pairwise.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    pairwise.MINIMUM_PASS_GAIN = MINIMUM_PASS_GAIN
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
    pairwise.prior.endpoint_decoder.cache_clear()
    pairwise.configure()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT,
        ROOT / "scripts/eval_vq2_lc103_phase7_pairwise_batch_scale_screen.py",
        ROOT / "scripts/fit_vq2_lc107_phase8_success_anchored_endpoint.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one independent pairwise-256 confirmation of the selected phase-8 scale."
    )
    milestone.NEXT_AUTHORITY_NONE = "Reject LC107 and retain LC105."
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "four complete saved-form Puffer actors; each executes a baseline-plus-candidate 256-row pair"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
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
        pairwise.prior.candidate_state_for_index,
        pairwise.prior.candidate_metadata_for_index,
    )
    try:
        return milestone.run(
            output=output, device_name=device_name, resume=resume
        )
    finally:
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
