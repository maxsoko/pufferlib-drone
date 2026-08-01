#!/usr/bin/env python3
"""Teacher-free raw-16 screen of recurrent adapter LC176 against LC169."""

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

from pufferlib.vq2_recurrent_phase_residual import (
    VQ2PhaseLocalAdapterActor,
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc110_phase8_success_action_bias as pairwise


TAG = "vq2_lc178_phase15_recurrent_adapter_milestone_001"
SCHEMA = "vq2_lc178_phase15_recurrent_adapter_milestone_report_v1"
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
    ("parent_lc169", (0.0, 0.0, 0.0, 0.0)),
    ("lc176_phase15_recurrent_adapter", (0.0, 0.0, 0.0, 0.0)),
)
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc169_phase15_failure_state_dagger2_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "8ae1a6d9aebd01d584344f006724f5a81699a011a759356283ada1f271ece1b7"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "182e81da9cb41a7f8479f0a2cfa99baef9b408a63cdfd1024130e38349cccf9b"
CANDIDATE_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc176_phase15_recurrent_adapter_001"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "72f342fabdff61ab7a6baa77dda084d6e25badf3e49df13c2f889472be0436c9"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "c1ee64eafb41c81ac19f8283e00dd37476be801a3840e8cfa8faaac5064607a3"
ADAPTER_PARAMETER_DELTA_L2 = 6.262093971042356
PREREGISTRATION = ROOT / "docs/vq2_lc178_phase15_recurrent_adapter_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc178_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc178_phase15_recurrent_adapter_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def parent_payload() -> dict[str, Any]:
    return torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_payload() -> dict[str, Any]:
    return torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    if candidate_index == 0:
        return {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    if candidate_index == 1:
        return {
            name: value.detach().cpu().clone()
            for name, value in candidate_payload()["model_state"].items()
        }
    raise IndexError("LC178 has only parent and adapter candidates")


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    return {
        "adapter_enabled": candidate_index == 1,
        "adapter_size": 64 if candidate_index == 1 else 0,
        "adapter_parameter_delta_l2": ADAPTER_PARAMETER_DELTA_L2 if candidate_index == 1 else 0.0,
        "bias_l2": ADAPTER_PARAMETER_DELTA_L2 if candidate_index == 1 else 0.0,
        "endpoint_phases": [TARGET_PHASE] if candidate_index == 1 else [],
    }


def load_actor(payload: dict[str, Any], device: torch.device) -> Any:
    contract = payload["model"]
    state = payload["model_state"]
    if "phase_adapter_cell.weight_ih" in state:
        actor = VQ2PhaseLocalAdapterActor(
            target_phase=int(contract.get("adapter_target_phase", TARGET_PHASE)),
            hidden_size=int(contract["hidden_size"]),
            residual_size=int(contract["residual_size"]),
            adapter_size=int(
                contract.get("adapter_size", state["phase_adapter_cell.weight_hh"].shape[1])
            ),
            initial_std=float(contract["initial_std"]),
        ).to(device)
    else:
        actor = VQ2UnboundedProgressMLPResidualActor(
            hidden_size=int(contract["hidden_size"]),
            residual_size=int(contract["residual_size"]),
            initial_std=float(contract["initial_std"]),
        ).to(device)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC178 bound input changed: {path}")
    parent = parent_payload()
    candidate = candidate_payload()
    parent_report = json.loads(PARENT_REPORT.read_text())
    training = json.loads(CANDIDATE_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc169_phase15_failure_state_dagger2_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc169_phase15_failure_state_dagger2_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or candidate.get("schema") != "vq2_lc176_phase15_recurrent_adapter_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or candidate.get("model", {}).get("adapter_target_phase") != TARGET_PHASE
        or candidate.get("model", {}).get("adapter_size") != 64
        or training.get("schema") != "vq2_lc176_phase15_recurrent_adapter_report_v1"
        or not training.get("numerically_admitted")
        or not training.get("frozen_base_state_exact")
        or training.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or training.get("validation_improvement_factor", 0.0) < 96.4
        or training.get("adapter_parameter_delta_l2") != ADAPTER_PARAMETER_DELTA_L2
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC169/LC176 do not authorize LC178")
    for name, value in parent["model_state"].items():
        if not torch.equal(value, candidate["model_state"][name]):
            raise RuntimeError(f"LC176 changed frozen base parameter {name}")
    expected_extra = {
        "phase_adapter_cell.weight_ih", "phase_adapter_cell.weight_hh",
        "phase_adapter_cell.bias_ih", "phase_adapter_cell.bias_hh",
        "phase_adapter_output.weight", "phase_adapter_output.bias",
    }
    if set(candidate["model_state"]) - set(parent["model_state"]) != expected_extra:
        raise RuntimeError("LC176 adapter parameter contract changed")
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
        ROOT / "scripts/train_vq2_lc176_phase15_recurrent_adapter.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Use LC176 only as the phase-16 source parent; broad admission remains mandatory."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC176 and collect adapter-owned on-policy sequences; do not run FlightSim."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete 256-row saved-form Puffers with independent recurrent states; candidate adds one public-phase-15 recurrent adapter"
    )


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    configure()
    originals = (
        milestone.verify_inputs, milestone.initialize_actor_execution,
        milestone.execute_actor_actions, milestone.candidate_state_for_index,
        milestone.candidate_metadata_for_index, milestone.choose_candidate,
        milestone.load_actor,
        pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
    )
    (
        milestone.verify_inputs, milestone.initialize_actor_execution,
        milestone.execute_actor_actions, milestone.candidate_state_for_index,
        milestone.candidate_metadata_for_index, milestone.choose_candidate,
        milestone.load_actor,
        pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
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
            milestone.load_actor,
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
