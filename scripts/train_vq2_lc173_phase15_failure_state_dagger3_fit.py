#!/usr/bin/env python3
"""Fit the third phase-15 DAgger iteration from LC169 and LC172."""

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
import scripts.train_vq2_lc166_phase15_failure_state_dagger_fit as previous


TAG = "vq2_lc173_phase15_failure_state_dagger3_fit_001"
SCHEMA = "vq2_lc173_phase15_failure_state_dagger3_fit_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc173_phase15_failure_state_dagger3_fit_checkpoint_v1"
FIT_METADATA_FIELD = "phase15_failure_state_dagger3_fit"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc169_phase15_failure_state_dagger2_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "8ae1a6d9aebd01d584344f006724f5a81699a011a759356283ada1f271ece1b7"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "182e81da9cb41a7f8479f0a2cfa99baef9b408a63cdfd1024130e38349cccf9b"
DATASET_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc172_phase15_failure_state_dagger3_001"
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "ddb15553a498e325edf14078dab6a1ca596750604aca9b84e82d497295f13d32"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "6e0f176d7e0ff328e1cfe0f898525c5d93d6da1fa219d4905065bc90b1c3985c"
FEATURE_RECORDS = 375_552
PREREGISTRATION = ROOT / "docs/vq2_lc173_phase15_failure_state_dagger3_fit_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc173_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc173_phase15_failure_state_dagger3_fit.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC173 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    control, intervention = dataset.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc169_phase15_failure_state_dagger2_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc169_phase15_failure_state_dagger2_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or dataset.get("schema") != "vq2_lc172_phase15_failure_state_dagger3_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_records") != FEATURE_RECORDS
        or dataset.get("query_agents") != 512
        or dataset.get("query_outcome_success_agents") != list(range(256, 512))
        or dataset.get("query_outcome_failure_agents") != list(range(256))
        or not dataset.get("control_features_use_teacher_targets")
        or control.get("target_passes") != 0
        or intervention.get("target_passes") != 256
        or intervention.get("paired_target_losses_vs_control") != 0
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC169/LC172 do not authorize LC173")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc172_phase15_failure_state_dagger3.py",
        ROOT / "scripts/train_vq2_lc166_phase15_failure_state_dagger_fit.py",
        ROOT / "scripts/train_vq2_lc169_phase15_failure_state_dagger2_fit.py",
        ROOT / "scripts/train_vq2_lc153_phase11_failure_state_dagger_fit.py",
        ROOT / "scripts/train_vq2_lc136_phase8_success_only_fit.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
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


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        corrected["dataset_schema"] = "vq2_lc172_phase15_failure_state_dagger3_report_v1"
        corrected["whole_puffer_phase"] = previous.TARGET_PHASE
        corrected["dagger_iteration"] = 3
        corrected["next_authority"] = (
            "Run one teacher-free repeated-source LC169-versus-LC173 raw-16 screen; no FlightSim authority."
            if corrected.get("numerically_admitted") else
            "Reject LC173; do not run FlightSim."
        )
    previous.prior.BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        previous.TAG, previous.SCHEMA, previous.CHECKPOINT_SCHEMA, previous.FIT_METADATA_FIELD,
        previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256,
        previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256,
        previous.FEATURES, previous.FEATURES_SHA256,
        previous.DATASET_REPORT, previous.DATASET_REPORT_SHA256,
        previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT,
        previous.verify_inputs, previous.source_identity, previous.corrected_writer,
    )
    previous.TAG, previous.SCHEMA, previous.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    previous.FIT_METADATA_FIELD = FIT_METADATA_FIELD
    previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    previous.FEATURES, previous.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    previous.DATASET_REPORT, previous.DATASET_REPORT_SHA256 = DATASET_REPORT, DATASET_REPORT_SHA256
    previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    previous.verify_inputs, previous.source_identity, previous.corrected_writer = verify_inputs, source_identity, corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        previous.TAG, previous.SCHEMA, previous.CHECKPOINT_SCHEMA, previous.FIT_METADATA_FIELD,
        previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256,
        previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256,
        previous.FEATURES, previous.FEATURES_SHA256,
        previous.DATASET_REPORT, previous.DATASET_REPORT_SHA256,
        previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT,
        previous.verify_inputs, previous.source_identity, previous.corrected_writer,
    ) = originals


def fit(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        return previous.fit(output=output, device_name=device_name, resume=resume)
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
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
