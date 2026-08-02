#!/usr/bin/env python3
"""Continue LC206's adapter on the second on-policy DAgger corpus."""

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

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_lc206_stacked_adapter_onpolicy_dagger as base


TAG = "vq2_lc209_stacked_adapter_onpolicy_dagger2_fit_001"
SCHEMA = "vq2_lc209_stacked_adapter_onpolicy_dagger2_fit_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc209_stacked_adapter_onpolicy_dagger2_fit_checkpoint_v1"
EPOCHS = 160
LEARNING_RATE = 5e-4
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc206_stacked_adapter_onpolicy_dagger_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "85bab7bf89ce1e140ebfc8cb714aedfb2427810f433ef7b1a524727ec47f51be"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "61032f4488c0e6b4f92858d2873a0810ea591006ac7025ac9db9baadfd50df0f"
DATASET_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc208_stacked_adapter_onpolicy_dagger2_001"
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "b6eb8afe3432b610bf0879b90465358d2c22c17f9aa1a96935e71fabe2849332"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "fc041ae7471e1d51bf088475fc800f3dd41f13b8ab1be81e1684912cd10b4186"
PREREGISTRATION = ROOT / "docs/vq2_lc209_stacked_adapter_onpolicy_dagger2_fit_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc209_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc209_stacked_adapter_onpolicy_dagger2.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
NEXT_AUTHORITY_ADMITTED = (
    "Run one exact-256-context teacher-free LC189-versus-LC209 raw-18 screen; "
    "no FlightSim authority."
)
NEXT_AUTHORITY_REJECTED = "Reject LC209 and retain LC189; do not run FlightSim."
EXPECTED_SEQUENCE_CONTRACT = {
    "agents": 512,
    "control_length_min": 223,
    "control_length_max": 223,
    "rescue_length_min": 2329,
    "rescue_length_max": 2329,
    "start_step_min": 20_928,
    "start_step_max": 20_928,
    "end_step_control": 21_150,
    "end_step_rescue": 23_256,
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
            raise RuntimeError(f"LC209 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    if (
        parent.get("schema")
        != "vq2_lc206_stacked_adapter_onpolicy_dagger_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema")
        != "vq2_lc206_stacked_adapter_onpolicy_dagger_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or not parent_report.get("frozen_lc189_state_exact")
        or dataset.get("schema")
        != "vq2_lc208_stacked_adapter_onpolicy_dagger2_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 653_312
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_phase_records", [])[16:18] != [413_440, 239_872]
        or dataset.get("feature_hidden_contract")
        != "frozen LC189 base recurrent state only, 256 values"
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC206/LC208 do not authorize LC209")
    return parent


def sequence_contract(records: np.ndarray) -> dict[str, Any]:
    contract = base.BASE_SEQUENCE_CONTRACT(records)
    if contract != EXPECTED_SEQUENCE_CONTRACT:
        raise RuntimeError(f"LC209 sequence contract changed: {contract}")
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


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_lc208_stacked_adapter_onpolicy_dagger2.py",
        ROOT / "scripts/train_vq2_lc206_stacked_adapter_onpolicy_dagger.py",
        ROOT / "scripts/train_vq2_lc176_phase15_recurrent_adapter.py",
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


def configure_outer() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "CHECKPOINT_SCHEMA", "EPOCHS", "LEARNING_RATE",
        "PARENT_CHECKPOINT", "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT",
        "PARENT_REPORT_SHA256", "FEATURES", "FEATURES_SHA256", "DATASET_REPORT",
        "DATASET_REPORT_SHA256", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "NEXT_AUTHORITY_ADMITTED", "NEXT_AUTHORITY_REJECTED",
        "verify_inputs", "source_identity", "sequence_contract",
        "materialize_sequences",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG, SCHEMA, CHECKPOINT_SCHEMA, EPOCHS, LEARNING_RATE,
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256, PARENT_REPORT,
        PARENT_REPORT_SHA256, FEATURES, FEATURES_SHA256, DATASET_REPORT,
        DATASET_REPORT_SHA256, PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
        NEXT_AUTHORITY_ADMITTED, NEXT_AUTHORITY_REJECTED, verify_inputs,
        source_identity, sequence_contract, materialize_sequences,
    )
    for name, value in zip(names, values):
        setattr(base, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(base, name, value)


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_outer()
    try:
        return base.fit(output=output, device_name=device_name, resume=resume)
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
