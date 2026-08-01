#!/usr/bin/env python3
"""Fit LC123 phase 8 on paired Puffer anchors and oracle rescues."""

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
import scripts.train_vq2_lc123_phase6_success_rescue_anchor as base


BASE_WRITE_JSON_ONCE = base.write_json_once

TAG = "vq2_lc135_phase8_paired_rescue_fit_001"
SCHEMA = "vq2_lc135_phase8_paired_rescue_fit_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc135_phase8_paired_rescue_fit_checkpoint_v1"
TARGET_PHASE = 8
EXPECTED_QUERY_AGENTS = 6
EXPECTED_SUCCESS_AGENTS = 3
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc123_phase6_success_rescue_anchor_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "0d0d9f6ba796dbf1803521214a0c030a79673f24cff538ae05338183ded4f558"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "92f29cdd0f3c76b2c031be3acdcf5bc23fecc9c7e499cafe3cf6edc5758c0bf9"
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc134_phase8_paired_anchor_rescue_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "e9454912542d6f42626f4763ea500b55ca618d201bb6a8eff85280b13da83f95"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "570e11e8b16f0191e64a4dc462349e31f6707c71eb2a17c2139ebd2cfde975a2"
PREREGISTRATION = ROOT / "docs/vq2_lc135_phase8_paired_rescue_fit_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc135_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc135_phase8_paired_rescue_fit.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC135 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    if (
        parent.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema")
        != "vq2_lc134_phase8_paired_anchor_rescue_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 7500
        or dataset.get("query_agents") != EXPECTED_QUERY_AGENTS
        or len(dataset.get("query_outcome_success_agents", []))
        != EXPECTED_SUCCESS_AGENTS
        or len(dataset.get("query_outcome_failure_agents", [])) != 3
        or dataset.get("items", [{}, {}])[0].get("target_passes") != 0
        or dataset.get("items", [{}, {}])[1].get("target_passes") != 3
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC123/LC134 do not authorize LC135")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc134_phase8_paired_anchor_rescue.py",
        ROOT / "scripts/train_vq2_lc123_phase6_success_rescue_anchor.py",
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
        corrected["dataset_schema"] = "vq2_lc134_phase8_paired_anchor_rescue_report_v1"
        corrected["whole_puffer_phase"] = TARGET_PHASE
        corrected["next_authority"] = (
            "Run exactly one reduced LC123-versus-LC135 teacher-free raw-9 screen; no FlightSim authority."
            if corrected.get("numerically_admitted")
            else "Reject LC135 and retain LC105; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TARGET_PHASE, base.EXPECTED_QUERY_AGENTS, base.EXPECTED_SUCCESS_AGENTS,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.FEATURES, base.FEATURES_SHA256,
        base.DATASET_REPORT, base.DATASET_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.TARGET_PHASE = TARGET_PHASE
    base.EXPECTED_QUERY_AGENTS = EXPECTED_QUERY_AGENTS
    base.EXPECTED_SUCCESS_AGENTS = EXPECTED_SUCCESS_AGENTS
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.DATASET_REPORT, base.DATASET_REPORT_SHA256 = DATASET_REPORT, DATASET_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.write_json_once = corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TARGET_PHASE, base.EXPECTED_QUERY_AGENTS, base.EXPECTED_SUCCESS_AGENTS,
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
