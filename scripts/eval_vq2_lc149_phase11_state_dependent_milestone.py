#!/usr/bin/env python3
"""Teacher-free paired raw-12 screen of LC148 against LC143."""

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
import scripts.eval_vq2_lc110_phase8_success_action_bias as pairwise


TAG = "vq2_lc149_phase11_state_dependent_milestone_001"
SCHEMA = "vq2_lc149_phase11_state_dependent_milestone_report_v1"
GROUP_SIZE = 128
TARGET_PHASE = 11
TARGET_RAW_INDEX = 12
MAX_STEPS = 17_000
SEED = 432_170
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc143_phase10_split_batch_cem_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "c4fe8fc5e069ea7696f52a81a9dd55a702957b92ad73285d55c06c6783e46ea6"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "8ad28f6c61589e9430e2813a6b7c7b37538cdd885c12c2d0778d0dbf09800bd9"
CANDIDATE_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc148_phase11_state_dependent_fit_001"
)
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "1a3fe9762f3cd2884aefc88c2e70eb69225d8d21bda9e9a0134e30d021a18b58"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "bb9733ac17a600cdfed4e35144e00c6233d3a7e4daeebd053e7bbd86c7b23e41"
SELECTED_PARAMETER_DELTA_L2 = 2.0097120667435378
SELECTED_SUCCESS_IMPROVEMENT = 2.4414455888558093
SELECTED_CONTROL_DRIFT = 0.015593620477063867
BIASES = (
    ("parent_lc143", (0.0, 0.0, 0.0, 0.0)),
    ("lc148_phase11_state_dependent", (0.0, 0.0, 0.0, 0.0)),
)
PREREGISTRATION = ROOT / "docs/vq2_lc149_phase11_state_dependent_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc149_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc149_phase11_state_dependent_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def candidate_payload() -> dict[str, Any]:
    return torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    if candidate_index == 0:
        source = parent_state
    elif candidate_index == 1:
        source = candidate_payload()["model_state"]
    else:
        raise ValueError("LC149 has only parent and candidate")
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    parameter_delta = 0.0 if candidate_index == 0 else SELECTED_PARAMETER_DELTA_L2
    return {
        "phase11_parameter_delta_l2": parameter_delta,
        # The shared milestone ranks this generic bounded-change field.
        "bias_l2": parameter_delta,
        "checkpoint_sha256": (
            PARENT_CHECKPOINT_SHA256 if candidate_index == 0
            else CANDIDATE_CHECKPOINT_SHA256
        ),
    }


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline, candidate = items
    if (
        candidate["transport_pass"]
        and candidate["target_passes"] >= baseline["target_passes"] + MINIMUM_PASS_GAIN
        and candidate["paired_target_losses_vs_baseline"] == 0
        and candidate["pre_target_terminals"] <= baseline["pre_target_terminals"]
    ):
        return candidate
    return None


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC149 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    candidate = candidate_payload()
    training = json.loads(CANDIDATE_REPORT.read_text())
    selected = training.get("selected", {})
    if (
        parent.get("schema") != "vq2_lc143_phase10_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc143_phase10_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or candidate.get("schema") != "vq2_lc148_phase11_state_dependent_fit_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or candidate.get("dataset_feature_sha256")
        != "dd080c1f32f8667b6426f2b243d47d1f3873b94609368907f29be8c593cc6969"
        or candidate.get("success_only_phase11_fit", {}).get("target_phase")
        != TARGET_PHASE
        or not training.get("frozen_non_phase11_state_exact")
        or training.get("schema") != "vq2_lc148_phase11_state_dependent_fit_report_v1"
        or not training.get("numerically_admitted")
        or training.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or selected.get("step") != 480
        or selected.get("scale") != 0.5
        or selected.get("parameter_delta_l2") != SELECTED_PARAMETER_DELTA_L2
        or selected.get("success_improvement_factor") != SELECTED_SUCCESS_IMPROVEMENT
        or selected.get("paired_control_parent_action_drift_mse") != SELECTED_CONTROL_DRIFT
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC143/LC148 do not authorize LC149")

    parent_state = parent["model_state"]
    candidate_state = candidate["model_state"]
    changed_rows = 0
    for name in parent_state:
        if name in PARAMETER_NAMES:
            keep = torch.arange(parent_state[name].shape[0]) != TARGET_PHASE
            if not torch.equal(parent_state[name][keep], candidate_state[name][keep]):
                raise RuntimeError(f"LC148 changed non-phase-11 state: {name}")
            changed_rows += int(
                not torch.equal(
                    parent_state[name][TARGET_PHASE], candidate_state[name][TARGET_PHASE]
                )
            )
        elif not torch.equal(parent_state[name], candidate_state[name]):
            raise RuntimeError(f"LC148 changed frozen state: {name}")
    if changed_rows != len(PARAMETER_NAMES):
        raise RuntimeError("LC148 must change all four phase-11 residual parameter rows")
    return parent


def configure() -> None:
    pairwise.TAG, pairwise.SCHEMA = TAG, SCHEMA
    pairwise.GROUP_SIZE, pairwise.PAIR_SIZE = GROUP_SIZE, 2 * GROUP_SIZE
    pairwise.BIASES = BIASES
    pairwise.SEED = SEED
    pairwise.TARGET_PHASE, pairwise.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    pairwise.MAX_STEPS, pairwise.MINIMUM_PASS_GAIN = MAX_STEPS, MINIMUM_PASS_GAIN
    pairwise.PARENT_CHECKPOINT, pairwise.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    pairwise.PARENT_REPORT, pairwise.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    pairwise.PREREGISTRATION, pairwise.RUNNER, pairwise.TEST = (
        PREREGISTRATION, RUNNER, TEST,
    )
    pairwise.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    pairwise.configure()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        ROOT / "scripts/train_vq2_lc148_phase11_state_dependent_fit.py",
        ROOT / "scripts/eval_vq2_lc110_phase8_success_action_bias.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Advance the source-locked trajectory with phase-12 Puffer search; no FlightSim authority."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC148 and retain LC105 as the safe frontier; do not run FlightSim."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form Puffer actors, each executing a 256-row paired batch; LC148 changes only the four phase-11 residual parameter rows"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
        milestone.choose_candidate,
        pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
    )
    (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
        milestone.choose_candidate,
        pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
    ) = (
        verify_inputs, pairwise.build_candidate_context,
        pairwise.initialize_actor_execution, pairwise.execute_actor_actions,
        candidate_state_for_index, candidate_metadata_for_index,
        choose_candidate,
        candidate_state_for_index, candidate_metadata_for_index,
    )
    try:
        return milestone.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            milestone.verify_inputs, milestone.build_candidate_context,
            milestone.initialize_actor_execution, milestone.execute_actor_actions,
            milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
            milestone.choose_candidate,
            pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
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
