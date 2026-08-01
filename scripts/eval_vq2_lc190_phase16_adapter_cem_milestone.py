#!/usr/bin/env python3
"""Teacher-free raw-17 screen of LC189 against source LC187."""

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


TAG = "vq2_lc190_phase16_adapter_cem_milestone_001"
SCHEMA = "vq2_lc190_phase16_adapter_cem_milestone_report_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
TARGET_PHASE = 16
TARGET_RAW_INDEX = 17
MAX_STEPS = 30_000
SEED = 432_190
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
MINIMUM_PASS_GAIN = GROUP_SIZE
BIASES = (("parent_lc187", (0.0, 0.0, 0.0, 0.0)), ("lc189_phase16_cem", (0.0, 0.0, 0.0, 0.0)))
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "a51d2a065edcc1154d6262a87146183ab786a9111150274c476762724ac6e196"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "4269220c21f3fafc1b7ef72275e44a5f2e2757f901734bfd28b34d52ac80fa55"
CANDIDATE_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc189_phase16_adapter_split_batch_cem_001"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "944be397e95d2bcc427e99d09e0b7885f1b0f1e7f2c2bacaf51a6e10398fc0b2"
BIAS_L2 = 0.17026823971043295
PREREGISTRATION = ROOT / "docs/vq2_lc190_phase16_adapter_cem_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc190_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc190_phase16_adapter_cem_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def parent_payload() -> dict[str, Any]:
    return torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_payload() -> dict[str, Any]:
    return torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(parent_state: dict[str, torch.Tensor], candidate_index: int) -> dict[str, torch.Tensor]:
    if candidate_index not in (0, 1):
        raise IndexError("LC190 has only parent and phase-16 CEM candidate")
    source = parent_state if candidate_index == 0 else candidate_payload()["model_state"]
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    return {
        "phase16_cem_candidate": candidate_index == 1,
        "checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256 if candidate_index == 1 else PARENT_CHECKPOINT_SHA256,
        "bias_l2": BIAS_L2 if candidate_index == 1 else 0.0,
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
            raise RuntimeError(f"LC190 bound input changed: {path}")
    parent = parent_payload()
    candidate = candidate_payload()
    parent_report = json.loads(PARENT_REPORT.read_text())
    training = json.loads(CANDIDATE_REPORT.read_text())
    selected = training.get("candidate_selected_for_screen", {})
    if (
        parent.get("schema") != "vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or candidate.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or not candidate.get("frozen_non_phase16_state_exact")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or training.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_report_v1"
        or not training.get("training_admitted")
        or selected.get("sha256") != CANDIDATE_CHECKPOINT_SHA256
        or training.get("selected_candidate", {}).get("generation") != 1
        or training.get("generations", [{}])[0].get("target_passes") != 1
        or training.get("configuration", {}).get("target_raw_index") != TARGET_RAW_INDEX
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC187/LC189 do not authorize LC190")
    changed = candidate["model_state"]["indexed_phase_residual_output_bias"] - parent["model_state"]["indexed_phase_residual_output_bias"]
    if torch.count_nonzero(changed[:TARGET_PHASE]) or torch.count_nonzero(changed[TARGET_PHASE + 1 :]):
        raise RuntimeError("LC189 changed a non-phase-16 residual row")
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
        ROOT / "scripts/train_vq2_lc189_phase16_adapter_split_batch_cem.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = "Use LC189 only as the phase-17 source parent; broad admission remains mandatory."
    milestone.NEXT_AUTHORITY_NONE = "Reject LC189 and start phase-16 adapter-owned DAgger; do not run FlightSim."
    milestone.VECTORIZED_CHECKPOINT_SURGERY = "two complete 256-row adapter Puffers with independent 320-state recurrent tensors"


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
