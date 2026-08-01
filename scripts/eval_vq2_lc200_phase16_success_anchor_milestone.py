#!/usr/bin/env python3
"""Exact-context teacher-free raw-18 screen of LC199 against LC189."""

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
import scripts.eval_vq2_lc195_phase16_17_teacher_free_milestone as prior


BASE_CONFIGURE = prior.configure
TAG = "vq2_lc200_phase16_success_anchor_milestone_001"
SCHEMA = "vq2_lc200_phase16_success_anchor_milestone_report_v1"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc189_phase16_adapter_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "944be397e95d2bcc427e99d09e0b7885f1b0f1e7f2c2bacaf51a6e10398fc0b2"
CANDIDATE_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc199_phase16_success_anchor_fit_001"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "ffceb4452072c87095728e3ffc4c6d68238383517a09b18dbbda9951cc25656c"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "f96e002f72e8a14fdea68c2becc55094ff3426bdd145571f5aca8b4ba54d5b0a"
BIASES = (("parent_lc189", (0.0,) * 4), ("lc199_phase16_success_anchor", (0.0,) * 4))
PREREGISTRATION = ROOT / "docs/vq2_lc200_phase16_success_anchor_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc200_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc200_phase16_success_anchor_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def parent_payload() -> dict[str, Any]:
    return torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_payload() -> dict[str, Any]:
    return torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(parent_state: dict[str, torch.Tensor], candidate_index: int) -> dict[str, torch.Tensor]:
    if candidate_index not in (0, 1):
        raise IndexError("LC200 has only parent and LC199")
    source = parent_state if candidate_index == 0 else candidate_payload()["model_state"]
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    return {
        "phase16_success_anchor_candidate": candidate_index == 1,
        "checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256 if candidate_index == 1 else PARENT_CHECKPOINT_SHA256,
        "bias_l2": 0.0,
        "endpoint_phases": [16] if candidate_index == 1 else [],
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC200 bound input changed: {path}")
    parent = parent_payload()
    candidate = candidate_payload()
    parent_report = json.loads(PARENT_REPORT.read_text())
    training = json.loads(CANDIDATE_REPORT.read_text())
    selected = training.get("selected", {})
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("training_admitted")
        or candidate.get("schema") != "vq2_lc199_phase16_success_anchor_fit_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or candidate.get("dataset_feature_sha256")
        != "699bd082aec4e0d6471377949639ed48c69329f39e47f28634d1c3ec5c739a81"
        or training.get("schema") != "vq2_lc199_phase16_success_anchor_fit_report_v1"
        or not training.get("numerically_admitted")
        or not training.get("frozen_non_phase16_state_exact")
        or training.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or selected.get("scale") != 0.3
        or selected.get("paired_control_parent_action_drift_mse", 1.0) > 2e-4
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC199 do not authorize LC200")
    if set(parent["model_state"]) != set(candidate["model_state"]):
        raise RuntimeError("LC199 model-state key contract changed")
    for name, value in parent["model_state"].items():
        other = candidate["model_state"][name]
        if name not in PARAMETER_NAMES:
            if not torch.equal(value, other):
                raise RuntimeError(f"LC199 changed frozen parameter {name}")
            continue
        for phase in range(value.shape[0]):
            if phase != 16 and not torch.equal(value[phase], other[phase]):
                raise RuntimeError(f"LC199 changed frozen residual phase {phase}")
    return parent


def configure() -> None:
    BASE_CONFIGURE()
    prior.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        ROOT / "scripts/train_vq2_lc199_phase16_success_anchor_fit.py",
        ROOT / "scripts/collect_vq2_lc193_phase16_17_rescue_dagger.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    prior.milestone.NEXT_AUTHORITY_SELECTED = "Use LC199 only as the raw-18 source parent and continue offline toward all 24 proxy gates; broad admission remains mandatory."
    prior.milestone.NEXT_AUTHORITY_NONE = "Reject LC199 and retain LC189; do not run FlightSim."
    prior.milestone.VECTORIZED_CHECKPOINT_SURGERY = "two complete adapter Puffers, each executing the established 256-row recurrent context over paired 128-row groups"


def configure_wrapper() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "BIASES", "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT", "PARENT_REPORT_SHA256",
        "CANDIDATE_CHECKPOINT", "CANDIDATE_CHECKPOINT_SHA256",
        "CANDIDATE_REPORT", "CANDIDATE_REPORT_SHA256",
        "PREREGISTRATION", "RUNNER", "TEST", "DEFAULT_OUTPUT",
        "parent_payload", "candidate_payload", "candidate_state_for_index",
        "candidate_metadata_for_index", "verify_inputs", "configure",
    )
    originals = tuple(getattr(prior, name) for name in names)
    values = (
        TAG, SCHEMA, BIASES, PARENT_CHECKPOINT,
        PARENT_CHECKPOINT_SHA256, PARENT_REPORT, PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT, CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256,
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
        parent_payload, candidate_payload, candidate_state_for_index,
        candidate_metadata_for_index, verify_inputs, configure,
    )
    for name, value in zip(names, values):
        setattr(prior, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(prior, name, value)


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_wrapper()
    try:
        return prior.run(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


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
