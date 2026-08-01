#!/usr/bin/env python3
"""Continue LC184's adapter on the second adapter-owned DAgger corpus."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import FEATURE_DTYPE
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_lc184_phase15_adapter_onpolicy_dagger as prior


TAG = "vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_001"
SCHEMA = "vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_checkpoint_v1"
EPOCHS = 160
LEARNING_RATE = 5e-4
MINIMUM_VALIDATION_IMPROVEMENT = 2.0
MAXIMUM_VALIDATION_MSE = 2e-4
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc184_phase15_adapter_onpolicy_dagger_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "b63346551f3d43d39d6a7409fed7e28ce3e034e4313639489c24574c1a607629"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "818a7fc7c89832cd50d62141ffe2286bbfc19a339c042a5d046ce0d5e63422c3"
DATASET_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc186_phase15_adapter_onpolicy_dagger2_001"
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "2b1881ca669164a1780733ac1427f39b1b5d5a445dcffebf8aa9c26728eb7242"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "f912d9b8b95248d747143e02404f01f73770f7ab2a5c924280c49400a7cf522c"
PREREGISTRATION = ROOT / "docs/vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc187_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc187_phase15_adapter_onpolicy_dagger2.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
NEXT_AUTHORITY_ADMITTED = "Run one teacher-free LC184-versus-LC187 raw-16 screen; no FlightSim authority."
NEXT_AUTHORITY_REJECTED = "Reject LC187; do not screen or run FlightSim."
EXPECTED_SEQUENCE_CONTRACT = {
    "agents": 512,
    "control_length_min": 803,
    "control_length_max": 803,
    "rescue_length_min": 986,
    "rescue_length_max": 986,
    "start_step_min": 19_920,
    "start_step_max": 19_920,
    "end_step_control": 20_722,
    "end_step_rescue": 20_905,
    "contiguous": True,
}


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC187 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc184_phase15_adapter_onpolicy_dagger_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc184_phase15_adapter_onpolicy_dagger_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema") != "vq2_lc186_phase15_adapter_onpolicy_dagger2_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 457_984
        or dataset.get("query_outcome_success_agents") != list(range(256, 512))
        or dataset.get("query_outcome_failure_agents") != list(range(256))
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC184/LC186 do not authorize LC187")
    return parent


def sequence_contract(records: np.ndarray) -> dict[str, Any]:
    contract = prior.BASE_SEQUENCE_CONTRACT(records)
    if contract != EXPECTED_SEQUENCE_CONTRACT:
        raise RuntimeError(f"LC187 sequence contract changed: {contract}")
    return contract


def materialize_sequences(records: np.ndarray) -> tuple[np.ndarray, ...]:
    sequence_contract(records)
    maximum = 986
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


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_lc186_phase15_adapter_onpolicy_dagger2.py",
        ROOT / "scripts/train_vq2_lc184_phase15_adapter_onpolicy_dagger.py",
        ROOT / "scripts/train_vq2_lc176_phase15_recurrent_adapter.py",
    )
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_path(path) for path in paths},
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def configure() -> tuple[Any, ...]:
    originals = (
        prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA,
        prior.EPOCHS, prior.LEARNING_RATE,
        prior.MINIMUM_VALIDATION_IMPROVEMENT, prior.MAXIMUM_VALIDATION_MSE,
        prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.FEATURES, prior.FEATURES_SHA256,
        prior.DATASET_REPORT, prior.DATASET_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.NEXT_AUTHORITY_ADMITTED, prior.NEXT_AUTHORITY_REJECTED,
        prior.verify_inputs, prior.source_identity, prior.sequence_contract,
        prior.materialize_sequences,
    )
    prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    prior.EPOCHS, prior.LEARNING_RATE = EPOCHS, LEARNING_RATE
    prior.MINIMUM_VALIDATION_IMPROVEMENT = MINIMUM_VALIDATION_IMPROVEMENT
    prior.MAXIMUM_VALIDATION_MSE = MAXIMUM_VALIDATION_MSE
    prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    prior.FEATURES, prior.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    prior.DATASET_REPORT, prior.DATASET_REPORT_SHA256 = DATASET_REPORT, DATASET_REPORT_SHA256
    prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    prior.NEXT_AUTHORITY_ADMITTED, prior.NEXT_AUTHORITY_REJECTED = NEXT_AUTHORITY_ADMITTED, NEXT_AUTHORITY_REJECTED
    prior.verify_inputs, prior.source_identity = verify_inputs, source_identity
    prior.sequence_contract, prior.materialize_sequences = sequence_contract, materialize_sequences
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA,
        prior.EPOCHS, prior.LEARNING_RATE,
        prior.MINIMUM_VALIDATION_IMPROVEMENT, prior.MAXIMUM_VALIDATION_MSE,
        prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.FEATURES, prior.FEATURES_SHA256,
        prior.DATASET_REPORT, prior.DATASET_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.NEXT_AUTHORITY_ADMITTED, prior.NEXT_AUTHORITY_REJECTED,
        prior.verify_inputs, prior.source_identity, prior.sequence_contract,
        prior.materialize_sequences,
    ) = originals


def fit(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        return prior.fit(output=output, device_name=device_name, resume=resume)
    finally:
        restore(originals)


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
