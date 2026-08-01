#!/usr/bin/env python3
"""Fit a new phase-16/17 recurrent adapter while freezing LC189 exactly."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase_residual import (
    VQ2StackedPhaseRangeAdapterActor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_lc176_phase15_recurrent_adapter as prior


BASE_SEQUENCE_CONTRACT = prior.sequence_contract
TAG = "vq2_lc202_phase16_17_stacked_adapter_001"
SCHEMA = "vq2_lc202_phase16_17_stacked_adapter_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc202_phase16_17_stacked_adapter_checkpoint_v1"
PHASE_MIN = 16
PHASE_MAX_EXCLUSIVE = 18
ADAPTER_SIZE = 64
EPOCHS = 160
LEARNING_RATE = 5e-4
MINIMUM_VALIDATION_IMPROVEMENT = 2.0
MAXIMUM_VALIDATION_MSE = 2e-4
TRAINABLE_PREFIXES = (
    "continuation_adapter_cell.",
    "continuation_adapter_output.",
)
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc189_phase16_adapter_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "944be397e95d2bcc427e99d09e0b7885f1b0f1e7f2c2bacaf51a6e10398fc0b2"
DATASET_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc193_phase16_17_rescue_dagger_001"
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "699bd082aec4e0d6471377949639ed48c69329f39e47f28634d1c3ec5c739a81"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "50b1a311a153dc0277fa2ba85d97e043cd31b022aa01f5b519acd0df83c10b64"
LC201_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc201_phase16_success_anchor_scale_bracket_001/report.json"
LC201_REPORT_SHA256 = "ad6fc236afd51ebb9139dbccb5a773a427a5ad3a02688d1a657db3a8d2343a5c"
PREREGISTRATION = ROOT / "docs/vq2_lc202_phase16_17_stacked_adapter_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc202_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc202_phase16_17_stacked_adapter.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
NEXT_AUTHORITY_ADMITTED = (
    "Run one exact-256-context teacher-free LC189-versus-LC202 raw-18 screen; "
    "no FlightSim authority."
)
NEXT_AUTHORITY_REJECTED = "Reject LC202 and retain LC189; do not run FlightSim."
EXPECTED_SEQUENCE_CONTRACT = {
    "agents": 512,
    "control_length_min": 466,
    "control_length_max": 466,
    "rescue_length_min": 2329,
    "rescue_length_max": 2329,
    "start_step_min": 20_928,
    "start_step_max": 20_928,
    "end_step_control": 21_393,
    "end_step_rescue": 23_256,
    "contiguous": True,
}


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        LC201_REPORT: LC201_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC202 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC201_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or dataset.get("schema") != "vq2_lc193_phase16_17_rescue_dagger_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 715_520
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_phase_records", [])[16:18] != [454_656, 260_864]
        or dataset.get("teacher_phase_min") != PHASE_MIN
        or dataset.get("teacher_phase_max_exclusive") != PHASE_MAX_EXCLUSIVE
        or dataset.get("feature_hidden_contract")
        != "frozen 256-state base recurrent output; phase-15 adapter remains in trajectory only"
        or rejected.get("schema")
        != "vq2_lc201_phase16_success_anchor_scale_bracket_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC193/LC201 do not authorize LC202")
    return parent


def sequence_contract(records: np.ndarray) -> dict[str, Any]:
    contract = BASE_SEQUENCE_CONTRACT(records)
    if contract != EXPECTED_SEQUENCE_CONTRACT:
        raise RuntimeError(f"LC202 sequence contract changed: {contract}")
    return contract


def materialize_sequences(records: np.ndarray) -> tuple[np.ndarray, ...]:
    sequence_contract(records)
    maximum = EXPECTED_SEQUENCE_CONTRACT["rescue_length_max"]
    hidden = np.zeros((512, maximum, 256), dtype=np.float16)
    base_action = np.zeros((512, maximum, 4), dtype=np.float32)
    teacher = np.zeros((512, maximum, 4), dtype=np.float32)
    mask = np.zeros((512, maximum), dtype=bool)
    lengths = np.zeros(512, dtype=np.int64)
    agent_field = np.asarray(records["agent_index"])
    for agent in range(512):
        sequence = records[np.flatnonzero(agent_field == agent)]
        sequence = sequence[np.argsort(sequence["step"], kind="stable")]
        length = len(sequence)
        hidden[agent, :length] = sequence["hidden"]
        base_action[agent, :length] = sequence["base_pre_tanh"]
        teacher[agent, :length] = sequence["teacher_action"]
        mask[agent, :length] = True
        lengths[agent] = length
    return hidden, base_action, teacher, mask, lengths


def build_actor(parent: dict[str, Any]) -> VQ2StackedPhaseRangeAdapterActor:
    contract = parent["model"]
    actor = VQ2StackedPhaseRangeAdapterActor(
        target_phase=int(contract["adapter_target_phase"]),
        continuation_phase_min=PHASE_MIN,
        continuation_phase_max_exclusive=PHASE_MAX_EXCLUSIVE,
        hidden_size=int(contract["hidden_size"]),
        residual_size=int(contract["residual_size"]),
        adapter_size=int(contract["adapter_size"]),
        continuation_adapter_size=ADAPTER_SIZE,
        initial_std=float(contract["initial_std"]),
    )
    actor.load_phase_local_state(parent["model_state"])
    return actor


def adapter_modules(
    actor: VQ2StackedPhaseRangeAdapterActor,
) -> tuple[nn.GRUCell, nn.Linear]:
    return actor.continuation_adapter_cell, actor.continuation_adapter_output


def fitted_model_contract(
    actor: VQ2StackedPhaseRangeAdapterActor, contract: dict[str, Any]
) -> dict[str, Any]:
    return {
        **contract,
        "continuation_phase_min": PHASE_MIN,
        "continuation_phase_max_exclusive": PHASE_MAX_EXCLUSIVE,
        "continuation_adapter_size": ADAPTER_SIZE,
    }


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT, LC201_REPORT,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_lc193_phase16_17_rescue_dagger.py",
        ROOT / "scripts/train_vq2_lc176_phase15_recurrent_adapter.py",
        ROOT / "scripts/eval_vq2_lc178_phase15_recurrent_adapter_milestone.py",
    )
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in paths
        },
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def configure() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "CHECKPOINT_SCHEMA", "TARGET_PHASE", "ADAPTER_SIZE",
        "EPOCHS", "LEARNING_RATE", "MINIMUM_VALIDATION_IMPROVEMENT",
        "MAXIMUM_VALIDATION_MSE", "TRAINABLE_ADAPTER_PREFIXES",
        "FIT_METADATA_FIELD", "FROZEN_STATE_FIELD", "TEMPORAL_WEIGHT_SEGMENTS",
        "PARENT_CHECKPOINT", "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT",
        "PARENT_REPORT_SHA256", "FEATURES", "FEATURES_SHA256", "DATASET_REPORT",
        "DATASET_REPORT_SHA256", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "NEXT_AUTHORITY_ADMITTED", "NEXT_AUTHORITY_REJECTED",
        "verify_inputs", "source_identity", "sequence_contract",
        "materialize_sequences", "build_actor", "adapter_modules",
        "fitted_model_contract",
    )
    originals = tuple(getattr(prior, name) for name in names)
    values = (
        TAG, SCHEMA, CHECKPOINT_SCHEMA, PHASE_MIN, ADAPTER_SIZE,
        EPOCHS, LEARNING_RATE, MINIMUM_VALIDATION_IMPROVEMENT,
        MAXIMUM_VALIDATION_MSE, TRAINABLE_PREFIXES,
        "phase16_17_stacked_adapter", "frozen_lc189_state_exact", (),
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256, PARENT_REPORT,
        PARENT_REPORT_SHA256, FEATURES, FEATURES_SHA256, DATASET_REPORT,
        DATASET_REPORT_SHA256, PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
        NEXT_AUTHORITY_ADMITTED, NEXT_AUTHORITY_REJECTED, verify_inputs,
        source_identity, sequence_contract, materialize_sequences, build_actor,
        adapter_modules, fitted_model_contract,
    )
    for name, value in zip(names, values):
        setattr(prior, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(prior, name, value)


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure()
    try:
        return prior.fit(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = fit(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("numerically_admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
