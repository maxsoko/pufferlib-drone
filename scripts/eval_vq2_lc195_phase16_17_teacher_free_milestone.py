#!/usr/bin/env python3
"""Teacher-free raw-18 screen of LC194 against its LC189 source parent."""

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


TAG = "vq2_lc195_phase16_17_teacher_free_milestone_001"
SCHEMA = "vq2_lc195_phase16_17_teacher_free_milestone_report_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
TARGET_PHASE = 17
TARGET_RAW_INDEX = 18
MAX_STEPS = 33_000
SEED = 432_195
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
MINIMUM_PASS_GAIN = GROUP_SIZE
BIASES = (("parent_lc189", (0.0, 0.0, 0.0, 0.0)), ("lc194_phase16_17_fit", (0.0, 0.0, 0.0, 0.0)))
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc189_phase16_adapter_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "944be397e95d2bcc427e99d09e0b7885f1b0f1e7f2c2bacaf51a6e10398fc0b2"
CANDIDATE_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc194_phase16_17_full_residual_fit_001"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_endpoint.pt"
CANDIDATE_CHECKPOINT_SHA256 = "f175dfc7a568cd8dd2ed95e9a9baf558e5351dd6f3df1f065a1ce1ceece7158b"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "5fe3258953eaf75d15059800939771e88ccf97599d1229032b6346d2d75be796"
PREREGISTRATION = ROOT / "docs/vq2_lc195_phase16_17_teacher_free_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc195_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc195_phase16_17_teacher_free_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def parent_payload() -> dict[str, Any]:
    return torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_payload() -> dict[str, Any]:
    return torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(parent_state: dict[str, torch.Tensor], candidate_index: int) -> dict[str, torch.Tensor]:
    if candidate_index not in (0, 1):
        raise IndexError("LC195 has only parent and phase-16/17 candidate")
    source = parent_state if candidate_index == 0 else candidate_payload()["model_state"]
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    return {
        "phase16_17_fit_candidate": candidate_index == 1,
        "checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256 if candidate_index == 1 else PARENT_CHECKPOINT_SHA256,
        "bias_l2": 0.0,
        "endpoint_phases": [16, 17] if candidate_index == 1 else [],
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC195 bound input changed: {path}")
    parent = parent_payload()
    candidate = candidate_payload()
    parent_report = json.loads(PARENT_REPORT.read_text())
    training = json.loads(CANDIDATE_REPORT.read_text())
    phase_reports = training.get("phase_reports", {})
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or candidate.get("schema") != "vq2_lc194_phase16_17_full_residual_fit_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or candidate.get("dataset_feature_sha256")
        != "699bd082aec4e0d6471377949639ed48c69329f39e47f28634d1c3ec5c739a81"
        or training.get("schema") != "vq2_lc194_phase16_17_full_residual_fit_report_v1"
        or not training.get("numerically_admitted")
        or training.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or training.get("phases") != [16, 17]
        or any(not phase_reports.get(str(phase), {}).get("numerically_admitted") for phase in (16, 17))
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC194 do not authorize LC195")
    changed_phases = {16, 17}
    if set(parent["model_state"]) != set(candidate["model_state"]):
        raise RuntimeError("LC194 model-state key contract changed")
    residual_names = set(prior_name for prior_name in (
        "indexed_phase_residual_input", "indexed_phase_residual_input_bias",
        "indexed_phase_residual_output", "indexed_phase_residual_output_bias",
    ))
    for name, value in parent["model_state"].items():
        other = candidate["model_state"][name]
        if name not in residual_names:
            if not torch.equal(value, other):
                raise RuntimeError(f"LC194 changed frozen parameter {name}")
            continue
        for phase in range(value.shape[0]):
            if phase not in changed_phases and not torch.equal(value[phase], other[phase]):
                raise RuntimeError(f"LC194 changed frozen residual phase {phase}")
    return parent


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline, candidate = items
    baseline_raw17 = baseline.get("maximum_raw_index_distribution", {}).get("17") == GROUP_SIZE
    if (
        baseline["target_passes"] == 0
        and baseline_raw17
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
        ROOT / "scripts/train_vq2_lc194_phase16_17_full_residual_fit.py",
        ROOT / "scripts/collect_vq2_lc193_phase16_17_rescue_dagger.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = "Use LC194 only as the phase-18 source parent and continue offline toward all 24 proxy gates; broad admission remains mandatory."
    milestone.NEXT_AUTHORITY_NONE = "Reject LC194 and continue source-owned DAgger; do not run FlightSim."
    milestone.VECTORIZED_CHECKPOINT_SURGERY = "two complete 128-row adapter Puffers with independent 320-state recurrent tensors"


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
