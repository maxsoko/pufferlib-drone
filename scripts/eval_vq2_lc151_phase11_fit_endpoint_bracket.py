#!/usr/bin/env python3
"""Teacher-free repeated-source bracket of the LC148 phase-11 fit direction."""

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
import scripts.eval_vq2_lc149_phase11_state_dependent_milestone as lc149


TAG = "vq2_lc151_phase11_fit_endpoint_bracket_001"
SCHEMA = "vq2_lc151_phase11_fit_endpoint_bracket_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc151_phase11_source_trajectory_checkpoint_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
TARGET_PHASE = 11
TARGET_RAW_INDEX = 12
MAX_STEPS = 17_000
SEED = 432_190
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
MINIMUM_PASS_GAIN = GROUP_SIZE
FIT_SCALES = (0.0, 0.50, 0.75, 1.0, 1.25)
CANDIDATES = tuple(
    ("parent_lc143" if scale == 0.0 else f"lc148_fit_scale_{scale:g}", (0.0,) * 4)
    for scale in FIT_SCALES
)
PARENT_CHECKPOINT = lc149.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = lc149.PARENT_CHECKPOINT_SHA256
PARENT_REPORT = lc149.PARENT_REPORT
PARENT_REPORT_SHA256 = lc149.PARENT_REPORT_SHA256
FIT_CHECKPOINT = lc149.CANDIDATE_CHECKPOINT
FIT_CHECKPOINT_SHA256 = lc149.CANDIDATE_CHECKPOINT_SHA256
FIT_REPORT = lc149.CANDIDATE_REPORT
FIT_REPORT_SHA256 = lc149.CANDIDATE_REPORT_SHA256
LC150_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc150_phase11_repeated_source_milestone_001/report.json"
)
LC150_REPORT_SHA256 = "a91fc9048d0853efc85cf6068454de34dea1d90ef658aecb27633323a3e913af"
PREREGISTRATION = ROOT / "docs/vq2_lc151_phase11_fit_endpoint_bracket_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc151_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc151_phase11_fit_endpoint_bracket.py"
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
    ratio = scale / 0.50
    for name in PARAMETER_NAMES:
        parent_row = parent_state[name][TARGET_PHASE]
        direction = fitted[name][TARGET_PHASE] - parent_row
        state[name][TARGET_PHASE].copy_(parent_row + ratio * direction)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    scale = FIT_SCALES[candidate_index]
    delta_l2 = lc149.SELECTED_PARAMETER_DELTA_L2 * scale / 0.50
    return {
        "fit_scale": scale,
        "phase11_parameter_delta_l2": delta_l2,
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
    lc149.verify_inputs()
    if sha256_path(LC150_REPORT) != LC150_REPORT_SHA256:
        raise RuntimeError("LC151 bound LC150 report changed")
    repeated = json.loads(LC150_REPORT.read_text())
    baseline, fitted = repeated.get("items", [{}, {}])
    fit_report = json.loads(FIT_REPORT.read_text())
    selected = fit_report.get("selected", {})
    if (
        repeated.get("schema") != "vq2_lc150_phase11_repeated_source_milestone_report_v1"
        or not repeated.get("diagnostic_valid")
        or repeated.get("causal_screen_selected") is not None
        or repeated.get("environment_seed_group_size") != 1
        or repeated.get("environment_seed_index_offset") != 15
        or baseline.get("target_passes") != 0
        or fitted.get("target_passes") != 0
        or selected.get("step") != 480
        or selected.get("scale") != 0.5
        or selected.get("parameter_delta_l2") != lc149.SELECTED_PARAMETER_DELTA_L2
        or repeated.get("safety", {}).get("flight_sim_packets_sent") != 0
        or repeated.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC148/LC150 do not authorize LC151")
    parent = parent_payload()
    exact_fit = candidate_state_for_index(parent["model_state"], 1)
    if any(
        not torch.equal(exact_fit[name], fit_payload()["model_state"][name])
        for name in exact_fit
    ):
        raise RuntimeError("LC151 scale 0.5 does not reconstruct LC148 exactly")
    for index in range(len(FIT_SCALES)):
        state = candidate_state_for_index(parent["model_state"], index)
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise RuntimeError("LC151 generated a nonfinite candidate")
    return parent


def configure() -> None:
    pairwise.TAG, pairwise.SCHEMA = TAG, SCHEMA
    pairwise.GROUP_SIZE, pairwise.PAIR_SIZE = GROUP_SIZE, PAIR_SIZE
    pairwise.BIASES = CANDIDATES
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
    milestone.ENV_SEED_GROUP_SIZE = ENV_SEED_GROUP_SIZE
    milestone.ENV_SEED_INDEX_OFFSET = ENV_SEED_INDEX_OFFSET
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT, LC150_REPORT,
        ROOT / "scripts/train_vq2_lc148_phase11_state_dependent_fit.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Use the selected whole-Puffer checkpoint only for phase-12 source-trajectory offline search; independent admission remains failed."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject the LC148 fitted direction and return to phase-11 state/action representation diagnosis; do not run FlightSim."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "five complete saved-form Puffer actors over 256 identical seed-15 rows; only the four phase-11 residual parameter rows vary"
    )


def save_selected_checkpoint(
    report: dict[str, Any], *, output: Path, parent: dict[str, Any]
) -> None:
    selected = report.get("causal_screen_selected")
    if selected is None:
        return
    state = candidate_state_for_index(parent["model_state"], selected["candidate_index"])
    if state_sha256(state) != selected["candidate_state_sha256"]:
        raise RuntimeError("LC151 selected state changed after the screen")
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


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
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
        candidate_state_for_index, candidate_metadata_for_index,
        choose_candidate,
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
