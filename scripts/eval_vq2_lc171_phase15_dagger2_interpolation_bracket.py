#!/usr/bin/env python3
"""Teacher-free interpolation bracket of the LC166-to-LC169 phase-15 fit."""

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
from scripts.train_vq2_lc111_phase8_9_full_residual_endpoint import PARAMETER_NAMES
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc110_phase8_success_action_bias as pairwise


TAG = "vq2_lc171_phase15_dagger2_interpolation_bracket_001"
SCHEMA = "vq2_lc171_phase15_dagger2_interpolation_bracket_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc171_phase15_source_trajectory_checkpoint_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
TARGET_PHASE = 15
TARGET_RAW_INDEX = 16
MAX_STEPS = 27_000
SEED = 432_190
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
MINIMUM_PASS_GAIN = GROUP_SIZE
FIT_SCALES = (0.0, 0.25, 0.50, 0.75, 1.0)
CANDIDATES = tuple(
    ("parent_lc166" if scale == 0.0 else f"lc169_fit_scale_{scale:g}", (0.0,) * 4)
    for scale in FIT_SCALES
)
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc166_phase15_failure_state_dagger_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "b9a6fc73bd7eaa81987f436d33c44325b6d0f5aef9f3394da9e99a0d6f29e518"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "825cae461cf4b095e45b441e072df13adaf5435c424f78304d909cc06c698048"
FIT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc169_phase15_failure_state_dagger2_fit_001"
FIT_CHECKPOINT = FIT_DIR / "policy_selected.pt"
FIT_CHECKPOINT_SHA256 = "8ae1a6d9aebd01d584344f006724f5a81699a011a759356283ada1f271ece1b7"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "182e81da9cb41a7f8479f0a2cfa99baef9b408a63cdfd1024130e38349cccf9b"
LC170_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc170_phase15_failure_state_dagger2_milestone_001/report.json"
LC170_REPORT_SHA256 = "77d8ef5c4f1fcad2a9087bcb990d00329d6833c7216c194a2dfa2cd9b9751dab"
SELECTED_PARAMETER_DELTA_L2 = 2.7561824584048114
PREREGISTRATION = ROOT / "docs/vq2_lc171_phase15_dagger2_interpolation_bracket_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc171_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc171_phase15_dagger2_interpolation_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def parent_payload() -> dict[str, Any]:
    return torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)


def fit_payload() -> dict[str, Any]:
    return torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    fitted = fit_payload()["model_state"]
    scale = FIT_SCALES[candidate_index]
    for name in PARAMETER_NAMES:
        parent_row = parent_state[name][TARGET_PHASE]
        direction = fitted[name][TARGET_PHASE] - parent_row
        state[name][TARGET_PHASE].copy_(parent_row + scale * direction)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    scale = FIT_SCALES[candidate_index]
    delta_l2 = SELECTED_PARAMETER_DELTA_L2 * scale
    return {
        "fit_scale": scale,
        "phase15_parameter_delta_l2": delta_l2,
        "bias_l2": delta_l2,
        "endpoint_phases": [TARGET_PHASE] if scale else [],
    }


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline = items[0]
    eligible = [
        item for item in items[1:]
        if baseline["target_passes"] == 0
        and item["target_passes"] == GROUP_SIZE
        and item["paired_target_gains_vs_baseline"] == GROUP_SIZE
        and item["paired_target_losses_vs_baseline"] == 0
        and item["transport_pass"]
        and item["pre_target_terminals"] <= baseline["pre_target_terminals"]
    ]
    return min(eligible, key=lambda item: item["fit_scale"]) if eligible else None


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
        LC170_REPORT: LC170_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC171 bound input changed: {path}")
    parent = parent_payload()
    fitted = fit_payload()
    parent_report = json.loads(PARENT_REPORT.read_text())
    fit_report = json.loads(FIT_REPORT.read_text())
    rejected = json.loads(LC170_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc166_phase15_failure_state_dagger_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or fitted.get("schema") != "vq2_lc169_phase15_failure_state_dagger2_fit_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or not fit_report.get("numerically_admitted")
        or fit_report.get("selected", {}).get("scale") != 1.0
        or fit_report.get("selected", {}).get("parameter_delta_l2") != SELECTED_PARAMETER_DELTA_L2
        or rejected.get("schema") != "vq2_lc170_phase15_failure_state_dagger2_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or any(item.get("target_passes") != 0 for item in rejected.get("items", []))
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC166/LC169/LC170 do not authorize LC171")
    exact_fit = candidate_state_for_index(parent["model_state"], len(FIT_SCALES) - 1)
    if any(not torch.equal(exact_fit[name], fitted["model_state"][name]) for name in exact_fit):
        raise RuntimeError("LC171 scale 1 does not reconstruct LC169 exactly")
    return parent


def configure() -> None:
    pairwise.TAG, pairwise.SCHEMA = TAG, SCHEMA
    pairwise.GROUP_SIZE, pairwise.PAIR_SIZE = GROUP_SIZE, PAIR_SIZE
    pairwise.BIASES = CANDIDATES
    pairwise.SEED = SEED
    pairwise.TARGET_PHASE, pairwise.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    pairwise.MAX_STEPS, pairwise.MINIMUM_PASS_GAIN = MAX_STEPS, MINIMUM_PASS_GAIN
    pairwise.PARENT_CHECKPOINT, pairwise.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    pairwise.PARENT_REPORT, pairwise.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    pairwise.PREREGISTRATION, pairwise.RUNNER, pairwise.TEST = PREREGISTRATION, RUNNER, TEST
    pairwise.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    pairwise.configure()
    milestone.ENV_SEED_GROUP_SIZE = ENV_SEED_GROUP_SIZE
    milestone.ENV_SEED_INDEX_OFFSET = ENV_SEED_INDEX_OFFSET
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT, LC170_REPORT,
        ROOT / "scripts/train_vq2_lc169_phase15_failure_state_dagger2_fit.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Use the selected checkpoint only as the phase-16 source parent; broad admission remains mandatory."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject the LC169 fitted direction and continue phase-15 state-dependent optimization; do not run FlightSim."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "five complete saved-form Puffer actors over identical seed-15 rows; only phase-15 residual parameter rows vary"
    )


def save_selected_checkpoint(
    report: dict[str, Any], *, output: Path, parent: dict[str, Any]
) -> None:
    selected = report.get("causal_screen_selected")
    if selected is None:
        return
    state = candidate_state_for_index(parent["model_state"], selected["candidate_index"])
    if state_sha256(state) != selected["candidate_state_sha256"]:
        raise RuntimeError("LC171 selected state changed after screen")
    checkpoint = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": TAG,
        "model_state": state,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "fit_checkpoint_sha256": FIT_CHECKPOINT_SHA256,
        "fit_scale": selected["fit_scale"],
        "source_trajectory_admitted": True,
        "independent_screen_admitted": False,
        "deployment_candidate": False,
        "safety": report["safety"],
    }
    atomic_torch_save(output / "policy_source_selected.pt", checkpoint)


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    parent = verify_inputs()
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
        candidate_state_for_index, candidate_metadata_for_index, choose_candidate,
        candidate_state_for_index, candidate_metadata_for_index,
    )
    try:
        report = milestone.run(output=output, device_name=device_name, resume=resume)
        save_selected_checkpoint(report, output=output, parent=parent)
        return report
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
