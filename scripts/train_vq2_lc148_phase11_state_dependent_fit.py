#!/usr/bin/env python3
"""Fit LC143 phase 11 on LC147's state-dependent rescue trajectory."""

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
import scripts.train_vq2_lc136_phase8_success_only_fit as base


BASE_WRITE_JSON_ONCE = base.write_json_once

TAG = "vq2_lc148_phase11_state_dependent_fit_001"
SCHEMA = "vq2_lc148_phase11_state_dependent_fit_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc148_phase11_state_dependent_fit_checkpoint_v1"
TARGET_PHASE = 11
SUCCESS_AGENTS = tuple(range(256, 512))
CONTROL_AGENTS = tuple(range(256))
TRAINING_SUCCESS_COUNT = 192
FROZEN_STATE_FIELD = "frozen_non_phase11_state_exact"
FIT_METADATA_FIELD = "success_only_phase11_fit"
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc143_phase10_split_batch_cem_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "c4fe8fc5e069ea7696f52a81a9dd55a702957b92ad73285d55c06c6783e46ea6"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "8ad28f6c61589e9430e2813a6b7c7b37538cdd885c12c2d0778d0dbf09800bd9"
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc147_phase11_state_dependent_rescue_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "dd080c1f32f8667b6426f2b243d47d1f3873b94609368907f29be8c593cc6969"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "191507966d9dcd17ded44054bdadd0270cabef37070f8591f565ab543f25b883"
PREREGISTRATION = ROOT / "docs/vq2_lc148_phase11_state_dependent_fit_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc148_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc148_phase11_state_dependent_fit.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC148 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    control, intervention = dataset.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc143_phase10_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc143_phase10_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or dataset.get("schema") != "vq2_lc147_phase11_state_dependent_rescue_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_records") != 122_112
        or dataset.get("query_agents") != 512
        or dataset.get("query_outcome_success_agents") != list(SUCCESS_AGENTS)
        or dataset.get("query_outcome_failure_agents") != list(CONTROL_AGENTS)
        or control.get("target_passes") != 0
        or intervention.get("target_passes") != 256
        or intervention.get("paired_target_losses_vs_control") != 0
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC143/LC147 do not authorize LC148")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc147_phase11_state_dependent_rescue.py",
        ROOT / "scripts/train_vq2_lc136_phase8_success_only_fit.py",
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


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        corrected["dataset_schema"] = "vq2_lc147_phase11_state_dependent_rescue_report_v1"
        corrected["whole_puffer_phase"] = TARGET_PHASE
        corrected["next_authority"] = (
            "Run one teacher-free LC143-versus-LC148 raw-12 screen; no FlightSim authority."
            if corrected.get("numerically_admitted") else
            "Reject LC148 and retain LC105 as the safe frontier; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA, base.TARGET_PHASE,
        base.SUCCESS_AGENTS, base.CONTROL_AGENTS, base.TRAINING_SUCCESS_COUNT,
        base.FROZEN_STATE_FIELD, base.FIT_METADATA_FIELD,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.FEATURES, base.FEATURES_SHA256,
        base.DATASET_REPORT, base.DATASET_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.TARGET_PHASE = TARGET_PHASE
    base.SUCCESS_AGENTS, base.CONTROL_AGENTS = SUCCESS_AGENTS, CONTROL_AGENTS
    base.TRAINING_SUCCESS_COUNT = TRAINING_SUCCESS_COUNT
    base.FROZEN_STATE_FIELD, base.FIT_METADATA_FIELD = (
        FROZEN_STATE_FIELD, FIT_METADATA_FIELD,
    )
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.DATASET_REPORT, base.DATASET_REPORT_SHA256 = DATASET_REPORT, DATASET_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.write_json_once = corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA, base.TARGET_PHASE,
        base.SUCCESS_AGENTS, base.CONTROL_AGENTS, base.TRAINING_SUCCESS_COUNT,
        base.FROZEN_STATE_FIELD, base.FIT_METADATA_FIELD,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.FEATURES, base.FEATURES_SHA256,
        base.DATASET_REPORT, base.DATASET_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
    ) = originals


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        base.fit(output=output, device_name=device_name, resume=resume)
        return json.loads((output / "report.json").read_text())
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
