#!/usr/bin/env python3
"""Exact-context teacher-free raw-18 screen of LC206 against LC189."""

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

from scripts.eval_vq2_lc178_phase15_recurrent_adapter_milestone import (
    load_actor as shared_load_actor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc203_stacked_adapter_raw18_milestone as base


BASE_CONFIGURE = base.configure
TAG = "vq2_lc207_stacked_adapter_onpolicy_milestone_001"
SCHEMA = "vq2_lc207_stacked_adapter_onpolicy_milestone_report_v1"
PARENT_CHECKPOINT = base.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = base.PARENT_CHECKPOINT_SHA256
PARENT_REPORT = base.PARENT_REPORT
PARENT_REPORT_SHA256 = base.PARENT_REPORT_SHA256
CANDIDATE_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc206_stacked_adapter_onpolicy_dagger_fit_001"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "85bab7bf89ce1e140ebfc8cb714aedfb2427810f433ef7b1a524727ec47f51be"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "61032f4488c0e6b4f92858d2873a0810ea591006ac7025ac9db9baadfd50df0f"
BIASES = (
    ("parent_lc189", (0.0,) * 4),
    ("lc206_onpolicy_stacked_adapter", (0.0,) * 4),
)
PREREGISTRATION = ROOT / "docs/vq2_lc207_stacked_adapter_onpolicy_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc207_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc207_stacked_adapter_onpolicy_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def parent_payload() -> dict[str, Any]:
    return torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_payload() -> dict[str, Any]:
    return torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    if candidate_index not in (0, 1):
        raise IndexError("LC207 has only parent and LC206")
    source = parent_state if candidate_index == 0 else candidate_payload()["model_state"]
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    return {
        "onpolicy_stacked_adapter_candidate": candidate_index == 1,
        "checkpoint_sha256": (
            CANDIDATE_CHECKPOINT_SHA256 if candidate_index == 1
            else PARENT_CHECKPOINT_SHA256
        ),
        "bias_l2": 0.0,
        "endpoint_phases": [16, 17] if candidate_index == 1 else [],
    }


def load_actor(payload: dict[str, Any], device: torch.device) -> Any:
    if "continuation_adapter_cell.weight_ih" not in payload["model_state"]:
        return shared_load_actor(payload, device)
    contract = candidate_payload()["model"]
    corrected = {
        **payload,
        "model": {
            **payload["model"],
            "continuation_phase_min": contract["continuation_phase_min"],
            "continuation_phase_max_exclusive": contract[
                "continuation_phase_max_exclusive"
            ],
            "continuation_adapter_size": contract["continuation_adapter_size"],
        },
    }
    return shared_load_actor(corrected, device)


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC207 bound input changed: {path}")
    parent = parent_payload()
    candidate = candidate_payload()
    training = json.loads(CANDIDATE_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or candidate.get("schema")
        != "vq2_lc206_stacked_adapter_onpolicy_dagger_fit_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256")
        != "9574928925d778910729d5fa55a1bd86a05767e29ddff8564bb15ea375286a73"
        or candidate.get("model", {}).get("continuation_phase_min") != 16
        or candidate.get("model", {}).get("continuation_phase_max_exclusive") != 18
        or training.get("schema")
        != "vq2_lc206_stacked_adapter_onpolicy_dagger_fit_report_v1"
        or not training.get("numerically_admitted")
        or not training.get("frozen_lc189_state_exact")
        or training.get("validation_improvement_factor", 0.0) < 220.0
        or training.get("best_validation_teacher_action_mse", 1.0) > 0.0006
        or training.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC206 do not authorize LC207")
    for name, value in parent["model_state"].items():
        if not torch.equal(candidate["model_state"][name], value):
            raise RuntimeError(f"LC206 changed frozen LC189 tensor {name}")
    return parent


def configure() -> None:
    BASE_CONFIGURE()
    base.prior.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        ROOT / "scripts/train_vq2_lc206_stacked_adapter_onpolicy_dagger.py",
        ROOT / "scripts/collect_vq2_lc205_stacked_adapter_onpolicy_dagger.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    base.prior.milestone.NEXT_AUTHORITY_SELECTED = (
        "Confirm LC206 independently, then continue offline from raw 18; no FlightSim authority."
    )
    base.prior.milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC206 and retain LC189; continue adapter-owned DAgger offline."
    )
    base.prior.milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete recurrent Puffers, each executing the established "
        "256-row context; LC206 carries the on-policy-refit continuation adapter"
    )


def configure_outer() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "BIASES", "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT", "PARENT_REPORT_SHA256",
        "CANDIDATE_CHECKPOINT", "CANDIDATE_CHECKPOINT_SHA256",
        "CANDIDATE_REPORT", "CANDIDATE_REPORT_SHA256", "PREREGISTRATION",
        "RUNNER", "TEST", "DEFAULT_OUTPUT", "parent_payload",
        "candidate_payload", "candidate_state_for_index",
        "candidate_metadata_for_index", "verify_inputs", "configure", "load_actor",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG, SCHEMA, BIASES, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT, PARENT_REPORT_SHA256, CANDIDATE_CHECKPOINT,
        CANDIDATE_CHECKPOINT_SHA256, CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256,
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT, parent_payload,
        candidate_payload, candidate_state_for_index, candidate_metadata_for_index,
        verify_inputs, configure, load_actor,
    )
    for name, value in zip(names, values):
        setattr(base, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(base, name, value)


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_outer()
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
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
