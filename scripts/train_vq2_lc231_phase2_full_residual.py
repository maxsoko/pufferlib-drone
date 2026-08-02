#!/usr/bin/env python3
"""Fit LC216 phase 2 on the deployment-context LC230 paired corpus."""

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

from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save
import scripts.train_vq2_lc111_phase8_9_full_residual_endpoint as prior


TAG = "vq2_lc231_phase2_full_residual_001"
SCHEMA = "vq2_lc231_phase2_full_residual_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc231_phase2_full_residual_checkpoint_v1"
NUMPY_SCHEMA = "vq2_lc231_phase2_full_residual_numpy_checkpoint_v1"
EXPORT_SCHEMA = "vq2_lc231_phase2_full_residual_numpy_export_v1"
SEED = 432_231
PHASES = (2,)
OPTIMIZER_STEPS = 512
BATCH_SIZE = 4_096
EVALUATION_INTERVAL = 16
LEARNING_RATE = 3e-3
ANCHOR_COEFFICIENT = 1e-3
GRADIENT_CLIP = 1.0
TARGET_ACTION_CLIP = 0.999
MINIMUM_PHASE_IMPROVEMENT = 1.50
MAXIMUM_PHASE_DELTA_L2 = 64.0
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc216_all24_action_sequence_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "672af0c4b014bb7d7399e2d709f268a487a83294b7b2a72ebfc79a48896a95b9"
)
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = (
    "b112413c778d984f369717643cdd8d69e0ec984e28c47e1b36b20aed18503617"
)
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc230_numpy_batch1_phase2_rescue_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = (
    "761bfea1700a86c8a9923036c440d1fbb7e5c535fe36c8f5d7d23d1b35e1273a"
)
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = (
    "514325b30725ff39cfe3e141560f8e117fbbd7156d69b482ff9baac044b9264f"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc231_phase2_full_residual_preregistration_2026-08-02.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc231_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc231_phase2_full_residual.py"
CALLABLE = ROOT / "scripts/policy_callable_vq2_lc231_phase2.py"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC231 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    control, intervention = dataset.get("items", [{}, {}])
    phase_records = dataset.get("feature_phase_records", [])
    if (
        parent.get("schema") != "vq2_lc216_all24_action_sequence_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema")
        != "vq2_lc216_all24_action_sequence_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema")
        != "vq2_lc230_numpy_batch1_phase2_rescue_report_v1"
        or not dataset.get("diagnostic_valid")
        or dataset.get("training_dataset_admitted")
        or dataset.get("failed_admission_predicates")
        != ["exact_control_failure", "paired_oracle_rescue"]
        or not dataset.get("training_oracle_rescued_phase8_9")
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_records") != 46_804
        or len(phase_records) != 33 or phase_records[2] != 46_804
        or control.get("target_passes") != 4
        or control.get("pre_target_terminals") != 4
        or intervention.get("target_passes") != 8
        or intervention.get("paired_target_gains_vs_control") != 4
        or intervention.get("paired_target_losses_vs_control") != 0
        or intervention.get("pre_target_terminals") != 0
        or dataset.get("teacher_action_envelope_violations") != 0
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC216/LC230 do not authorize the LC231 offline fit")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST, CALLABLE,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc230_numpy_batch1_phase2_rescue.py",
        ROOT / "scripts/train_vq2_lc111_phase8_9_full_residual_endpoint.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
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
        "TAG", "SCHEMA", "CHECKPOINT_SCHEMA", "SEED", "PHASES",
        "OPTIMIZER_STEPS", "BATCH_SIZE", "EVALUATION_INTERVAL",
        "LEARNING_RATE", "ANCHOR_COEFFICIENT", "GRADIENT_CLIP",
        "TARGET_ACTION_CLIP", "MINIMUM_PHASE_IMPROVEMENT",
        "MAXIMUM_PHASE_DELTA_L2", "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT", "PARENT_REPORT_SHA256",
        "FEATURES", "FEATURES_SHA256", "DATASET_REPORT",
        "DATASET_REPORT_SHA256", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "verify_inputs", "source_identity",
    )
    originals = tuple(getattr(prior, name) for name in names)
    values = (
        TAG, SCHEMA, CHECKPOINT_SCHEMA, SEED, PHASES, OPTIMIZER_STEPS,
        BATCH_SIZE, EVALUATION_INTERVAL, LEARNING_RATE, ANCHOR_COEFFICIENT,
        GRADIENT_CLIP, TARGET_ACTION_CLIP, MINIMUM_PHASE_IMPROVEMENT,
        MAXIMUM_PHASE_DELTA_L2, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT, PARENT_REPORT_SHA256, FEATURES, FEATURES_SHA256,
        DATASET_REPORT, DATASET_REPORT_SHA256, PREREGISTRATION, RUNNER, TEST,
        DEFAULT_OUTPUT, verify_inputs, source_identity,
    )
    for name, value in zip(names, values):
        setattr(prior, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(prior, name, value)


def export_numpy(output: Path, report: dict[str, Any]) -> dict[str, Any]:
    checkpoint = output / str(report["checkpoint"])
    checkpoint_sha256 = sha256_path(checkpoint)
    if checkpoint_sha256 != report["checkpoint_sha256"]:
        raise RuntimeError("LC231 fitted checkpoint changed before export")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or not payload.get("numerically_admitted")
        or contract.get("class") != "VQ2PhaseActionSequenceActor"
        or int(contract.get("sequence_phase_min", -1)) != 16
        or int(contract.get("sequence_phase_max_exclusive", -1)) != 24
        or int(contract.get("sequence_length", 0)) != 9_592
    ):
        raise RuntimeError("LC231 fitted checkpoint is not exportable")
    arrays = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, value in payload["model_state"].items()
    }
    arrays["__metadata_json__"] = np.asarray(json.dumps({
        "schema": NUMPY_SCHEMA,
        "source_checkpoint_sha256": checkpoint_sha256,
        "model": contract,
    }, sort_keys=True), dtype=np.str_)
    archive = output / "policy_endpoint_numpy.npz"
    if not archive.exists():
        np.savez_compressed(archive, **arrays)
    export_report = {
        "schema": EXPORT_SCHEMA, "completed": True,
        "source_checkpoint": checkpoint.name,
        "source_checkpoint_sha256": checkpoint_sha256,
        "numpy_checkpoint": archive.name,
        "numpy_checkpoint_sha256": sha256_path(archive),
        "tensor_count": len(arrays) - 1,
        "callable_sha256": sha256_path(CALLABLE),
        "safety": {"flight_sim_packets_sent": 0, "submission_authorized": False},
    }
    write_json_once(output / "numpy_export.json", export_report)
    return export_report


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure()
    try:
        report = prior.fit(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)
    if report.get("numerically_admitted"):
        report = {**report, "numpy_export": export_numpy(output, report)}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = fit(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("numerically_admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
