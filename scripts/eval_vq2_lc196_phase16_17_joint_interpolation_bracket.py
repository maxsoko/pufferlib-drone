#!/usr/bin/env python3
"""Teacher-free joint interpolation bracket from LC189 toward LC194."""

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
from scripts.eval_vq2_lc178_phase15_recurrent_adapter_milestone import load_actor
from scripts.eval_vq2_variable_gate_oracle import sha256_path
from scripts.train_vq2_lc111_phase8_9_full_residual_endpoint import PARAMETER_NAMES
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc110_phase8_success_action_bias as pairwise


TAG = "vq2_lc196_phase16_17_joint_interpolation_bracket_001"
SCHEMA = "vq2_lc196_phase16_17_joint_interpolation_bracket_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc196_phase16_17_joint_interpolation_checkpoint_v1"
GROUP_SIZE = 32
PAIR_SIZE = 64
TARGET_PHASE = 17
TARGET_RAW_INDEX = 18
MAX_STEPS = 33_000
SEED = 432_195
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
MINIMUM_PASS_GAIN = 1
SCALE_PAIRS = (
    (0.0, 0.0),
    (0.0, 0.25), (0.0, 0.50), (0.0, 1.0),
    (0.0025, 0.25), (0.0025, 0.50), (0.0025, 1.0),
    (0.0050, 0.25), (0.0050, 0.50), (0.0050, 1.0),
    (0.0100, 0.25), (0.0100, 0.50), (0.0100, 1.0),
)
CANDIDATES = tuple(
    ("parent_lc189" if pair == (0.0, 0.0) else f"phase16_{pair[0]:g}_phase17_{pair[1]:g}", (0.0,) * 4)
    for pair in SCALE_PAIRS
)
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc189_phase16_adapter_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "944be397e95d2bcc427e99d09e0b7885f1b0f1e7f2c2bacaf51a6e10398fc0b2"
FIT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc194_phase16_17_full_residual_fit_001"
FIT_CHECKPOINT = FIT_DIR / "policy_endpoint.pt"
FIT_CHECKPOINT_SHA256 = "f175dfc7a568cd8dd2ed95e9a9baf558e5351dd6f3df1f065a1ce1ceece7158b"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "5fe3258953eaf75d15059800939771e88ccf97599d1229032b6346d2d75be796"
LC195_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc195_phase16_17_teacher_free_milestone_001/report.json"
LC195_REPORT_SHA256 = "b046188466d304ed900853499ae9ba2084370870c5bd4902313c1ee716b66a5b"
PREREGISTRATION = ROOT / "docs/vq2_lc196_phase16_17_joint_interpolation_bracket_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc196_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc196_phase16_17_joint_interpolation_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def parent_payload() -> dict[str, Any]:
    return torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)


def fit_payload() -> dict[str, Any]:
    return torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(parent_state: dict[str, torch.Tensor], candidate_index: int) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    fitted = fit_payload()["model_state"]
    scales = {16: SCALE_PAIRS[candidate_index][0], 17: SCALE_PAIRS[candidate_index][1]}
    for phase, scale in scales.items():
        for name in PARAMETER_NAMES:
            parent_row = parent_state[name][phase]
            if scale == 0.0:
                state[name][phase].copy_(parent_row)
            elif scale == 1.0:
                state[name][phase].copy_(fitted[name][phase])
            else:
                direction = fitted[name][phase] - parent_row
                state[name][phase].copy_(parent_row + scale * direction)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    scale16, scale17 = SCALE_PAIRS[candidate_index]
    return {
        "phase16_fit_scale": scale16,
        "phase17_fit_scale": scale17,
        "bias_l2": (scale16 * scale16 + scale17 * scale17) ** 0.5,
        "endpoint_phases": [phase for phase, scale in ((16, scale16), (17, scale17)) if scale],
    }


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline = items[0]
    eligible = [
        item for item in items[1:]
        if baseline["target_passes"] == 0
        and baseline.get("maximum_raw_index_distribution", {}).get("17") == GROUP_SIZE
        and item["target_passes"] == GROUP_SIZE
        and item["paired_target_gains_vs_baseline"] == GROUP_SIZE
        and item["paired_target_losses_vs_baseline"] == 0
        and item["transport_pass"]
        and item["pre_target_terminals"] <= baseline["pre_target_terminals"]
    ]
    return min(
        eligible,
        key=lambda item: (item["phase16_fit_scale"], item["phase17_fit_scale"]),
    ) if eligible else None


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
        LC195_REPORT: LC195_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC196 bound input changed: {path}")
    parent = parent_payload()
    fitted = fit_payload()
    parent_report = json.loads(PARENT_REPORT.read_text())
    fit_report = json.loads(FIT_REPORT.read_text())
    rejected = json.loads(LC195_REPORT.read_text())
    baseline, endpoint = rejected.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("training_admitted")
        or fitted.get("schema") != "vq2_lc194_phase16_17_full_residual_fit_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or not fit_report.get("numerically_admitted")
        or fit_report.get("phases") != [16, 17]
        or rejected.get("schema") != "vq2_lc195_phase16_17_teacher_free_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or baseline.get("maximum_raw_index_distribution", {}).get("17") != 128
        or endpoint.get("maximum_raw_index_distribution", {}).get("16") != 128
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC194/LC195 do not authorize LC196")
    exact_fit = candidate_state_for_index(parent["model_state"], 3)
    for name in exact_fit:
        if name in PARAMETER_NAMES:
            if not torch.equal(exact_fit[name][16], parent["model_state"][name][16]):
                raise RuntimeError("LC196 phase-16 zero scale changed parent")
            if not torch.equal(exact_fit[name][17], fitted["model_state"][name][17]):
                raise RuntimeError("LC196 phase-17 unit scale changed fit")
    return parent


def configure() -> None:
    pairwise.TAG, pairwise.SCHEMA = TAG, SCHEMA
    pairwise.GROUP_SIZE, pairwise.PAIR_SIZE = GROUP_SIZE, PAIR_SIZE
    pairwise.BIASES, pairwise.SEED = CANDIDATES, SEED
    pairwise.TARGET_PHASE, pairwise.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    pairwise.MAX_STEPS, pairwise.MINIMUM_PASS_GAIN = MAX_STEPS, MINIMUM_PASS_GAIN
    pairwise.PARENT_CHECKPOINT, pairwise.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    pairwise.PARENT_REPORT, pairwise.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    pairwise.PREREGISTRATION, pairwise.RUNNER, pairwise.TEST, pairwise.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    pairwise.configure()
    milestone.ENV_SEED_GROUP_SIZE = ENV_SEED_GROUP_SIZE
    milestone.ENV_SEED_INDEX_OFFSET = ENV_SEED_INDEX_OFFSET
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT, LC195_REPORT,
        ROOT / "scripts/train_vq2_lc194_phase16_17_full_residual_fit.py",
        ROOT / "scripts/eval_vq2_lc195_phase16_17_teacher_free_milestone.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = "Confirm the selected raw-18 source independently before extending to phase 18; no FlightSim authority."
    milestone.NEXT_AUTHORITY_NONE = "Reject this interpolation grid and retain LC189; do not run FlightSim."
    milestone.VECTORIZED_CHECKPOINT_SURGERY = "thirteen complete adapter Puffers over paired 32-row seed-15 groups; only phase-16/17 residual rows vary"


def save_selected_checkpoint(report: dict[str, Any], *, output: Path, parent: dict[str, Any]) -> None:
    selected = report.get("causal_screen_selected")
    if selected is None:
        return
    state = candidate_state_for_index(parent["model_state"], selected["candidate_index"])
    if state_sha256(state) != selected["candidate_state_sha256"]:
        raise RuntimeError("LC196 selected state changed after screen")
    checkpoint = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": TAG,
        "model_state": state,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "fit_checkpoint_sha256": FIT_CHECKPOINT_SHA256,
        "phase16_fit_scale": selected["phase16_fit_scale"],
        "phase17_fit_scale": selected["phase17_fit_scale"],
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
        milestone.choose_candidate, milestone.load_actor,
        pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
    )
    (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
        milestone.choose_candidate, milestone.load_actor,
        pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
    ) = (
        verify_inputs, pairwise.build_candidate_context,
        pairwise.initialize_actor_execution, pairwise.execute_actor_actions,
        candidate_state_for_index, candidate_metadata_for_index, choose_candidate,
        load_actor, candidate_state_for_index, candidate_metadata_for_index,
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
            milestone.choose_candidate, milestone.load_actor,
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
    return 0 if report.get("diagnostic_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
