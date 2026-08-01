#!/usr/bin/env python3
"""Teacher-free raw-16 screen of second DAgger LC187 against LC184."""

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

from scripts.eval_vq2_lc178_phase15_recurrent_adapter_milestone import load_actor
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc110_phase8_success_action_bias as pairwise


TAG = "vq2_lc188_phase15_adapter_onpolicy_dagger2_milestone_001"
SCHEMA = "vq2_lc188_phase15_adapter_onpolicy_dagger2_milestone_report_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
TARGET_PHASE = 15
TARGET_RAW_INDEX = 16
MAX_STEPS = 27_000
SEED = 432_190
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
MINIMUM_PASS_GAIN = GROUP_SIZE
BIASES = (
    ("parent_lc184", (0.0, 0.0, 0.0, 0.0)),
    ("lc187_adapter_onpolicy_dagger2", (0.0, 0.0, 0.0, 0.0)),
)
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc184_phase15_adapter_onpolicy_dagger_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "b63346551f3d43d39d6a7409fed7e28ce3e034e4313639489c24574c1a607629"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "818a7fc7c89832cd50d62141ffe2286bbfc19a339c042a5d046ce0d5e63422c3"
CANDIDATE_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_001"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "a51d2a065edcc1154d6262a87146183ab786a9111150274c476762724ac6e196"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "4269220c21f3fafc1b7ef72275e44a5f2e2757f901734bfd28b34d52ac80fa55"
ADAPTER_PARAMETER_DELTA_L2 = 4.293662065120229
PREREGISTRATION = ROOT / "docs/vq2_lc188_phase15_adapter_onpolicy_dagger2_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc188_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc188_phase15_adapter_onpolicy_dagger2_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def parent_payload() -> dict[str, Any]:
    return torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_payload() -> dict[str, Any]:
    return torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(parent_state: dict[str, torch.Tensor], candidate_index: int) -> dict[str, torch.Tensor]:
    if candidate_index not in (0, 1):
        raise IndexError("LC188 has only parent and second DAgger adapter")
    source = parent_state if candidate_index == 0 else candidate_payload()["model_state"]
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    return {
        "onpolicy_dagger2_candidate": candidate_index == 1,
        "adapter_size": 64,
        "adapter_parameter_delta_l2": ADAPTER_PARAMETER_DELTA_L2 if candidate_index == 1 else 0.0,
        "bias_l2": ADAPTER_PARAMETER_DELTA_L2 if candidate_index == 1 else 0.0,
        "endpoint_phases": [TARGET_PHASE] if candidate_index == 1 else [],
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC188 bound input changed: {path}")
    parent = parent_payload()
    candidate = candidate_payload()
    parent_report = json.loads(PARENT_REPORT.read_text())
    training = json.loads(CANDIDATE_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc184_phase15_adapter_onpolicy_dagger_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc184_phase15_adapter_onpolicy_dagger_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or candidate.get("schema") != "vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or training.get("schema") != "vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_report_v1"
        or not training.get("numerically_admitted")
        or not training.get("frozen_base_state_exact")
        or training.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or training.get("validation_improvement_factor", 0.0) < 243.9
        or training.get("best_validation_teacher_action_mse", 1.0) > 1.035e-4
        or training.get("adapter_parameter_delta_l2") != ADAPTER_PARAMETER_DELTA_L2
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC184/LC187 do not authorize LC188")
    prefixes = ("phase_adapter_cell.", "phase_adapter_output.")
    if set(parent["model_state"]) != set(candidate["model_state"]):
        raise RuntimeError("LC187 model-state key contract changed")
    for name, value in parent["model_state"].items():
        if not name.startswith(prefixes) and not torch.equal(value, candidate["model_state"][name]):
            raise RuntimeError(f"LC187 changed frozen base parameter {name}")
    return parent


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline, candidate = items
    if (
        baseline["target_passes"] == 0
        and candidate["target_passes"] == GROUP_SIZE
        and candidate["paired_target_gains_vs_baseline"] == GROUP_SIZE
        and candidate["paired_target_losses_vs_baseline"] == 0
        and candidate["transport_pass"]
        and candidate["pre_target_terminals"] <= baseline["pre_target_terminals"]
    ):
        return candidate
    return None


def configure() -> None:
    pairwise.TAG, pairwise.SCHEMA = TAG, SCHEMA
    pairwise.GROUP_SIZE, pairwise.PAIR_SIZE = GROUP_SIZE, PAIR_SIZE
    pairwise.BIASES, pairwise.SEED = BIASES, SEED
    pairwise.TARGET_PHASE, pairwise.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    pairwise.MAX_STEPS, pairwise.MINIMUM_PASS_GAIN = MAX_STEPS, MINIMUM_PASS_GAIN
    pairwise.PARENT_CHECKPOINT, pairwise.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    pairwise.PARENT_REPORT, pairwise.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    pairwise.PREREGISTRATION, pairwise.RUNNER, pairwise.TEST, pairwise.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    pairwise.configure()
    milestone.ENV_SEED_GROUP_SIZE = ENV_SEED_GROUP_SIZE
    milestone.ENV_SEED_INDEX_OFFSET = ENV_SEED_INDEX_OFFSET
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        ROOT / "scripts/train_vq2_lc187_phase15_adapter_onpolicy_dagger2.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = "Use LC187 only as the phase-16 source parent; broad admission remains mandatory."
    milestone.NEXT_AUTHORITY_NONE = "Reject LC187 and continue adapter-owned DAgger; do not run FlightSim."
    milestone.VECTORIZED_CHECKPOINT_SURGERY = "two complete 256-row phase-local-adapter Puffers with independent 320-state recurrent tensors"


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    configure()
    originals = (
        milestone.verify_inputs, milestone.initialize_actor_execution,
        milestone.execute_actor_actions, milestone.candidate_state_for_index,
        milestone.candidate_metadata_for_index, milestone.choose_candidate,
        milestone.load_actor, pairwise.candidate_state_for_index,
        pairwise.candidate_metadata_for_index,
    )
    (
        milestone.verify_inputs, milestone.initialize_actor_execution,
        milestone.execute_actor_actions, milestone.candidate_state_for_index,
        milestone.candidate_metadata_for_index, milestone.choose_candidate,
        milestone.load_actor, pairwise.candidate_state_for_index,
        pairwise.candidate_metadata_for_index,
    ) = (
        verify_inputs, pairwise.initialize_actor_execution,
        pairwise.execute_actor_actions, candidate_state_for_index,
        candidate_metadata_for_index, choose_candidate, load_actor,
        candidate_state_for_index, candidate_metadata_for_index,
    )
    try:
        return milestone.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            milestone.verify_inputs, milestone.initialize_actor_execution,
            milestone.execute_actor_actions, milestone.candidate_state_for_index,
            milestone.candidate_metadata_for_index, milestone.choose_candidate,
            milestone.load_actor, pairwise.candidate_state_for_index,
            pairwise.candidate_metadata_for_index,
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
