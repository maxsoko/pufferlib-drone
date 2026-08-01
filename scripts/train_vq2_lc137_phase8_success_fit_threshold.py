#!/usr/bin/env python3
"""Regenerate LC136 with an evidence-backed numerical screen threshold."""

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
BASE_VERIFY_INPUTS = base.verify_inputs

TAG = "vq2_lc137_phase8_success_fit_threshold_001"
SCHEMA = "vq2_lc137_phase8_success_fit_threshold_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc137_phase8_success_fit_threshold_checkpoint_v1"
MINIMUM_SUCCESS_IMPROVEMENT = 1.4
LC136_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc136_phase8_success_only_fit_001/report.json"
)
LC136_REPORT_SHA256 = "bf10140a299d837650a5f0091696d20a762ee1b6403480b6793510d0946dfdc8"
PREREGISTRATION = ROOT / "docs/vq2_lc137_phase8_success_fit_threshold_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc137_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc137_phase8_success_fit_threshold.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    parent = BASE_VERIFY_INPUTS()
    if sha256_path(LC136_REPORT) != LC136_REPORT_SHA256:
        raise RuntimeError("LC137 bound LC136 report changed")
    rejected = json.loads(LC136_REPORT.read_text())
    selected = rejected.get("selected", {})
    if (
        rejected.get("schema") != "vq2_lc136_phase8_success_only_fit_report_v1"
        or rejected.get("numerically_admitted")
        or selected.get("step") != 176
        or selected.get("scale") != 1.0
        or selected.get("success_improvement_factor") != 1.473931528893509
        or selected.get("paired_control_parent_action_drift_mse")
        != 0.004604219119829456
        or selected.get("parameter_delta_l2") != 4.844313185799752
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC136 does not authorize LC137")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        base.PARENT_CHECKPOINT, base.PARENT_REPORT,
        base.FEATURES, base.DATASET_REPORT, LC136_REPORT,
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
        corrected["lc136_rejected_report_sha256"] = LC136_REPORT_SHA256
        corrected["only_threshold_changed_from_lc136"] = True
        corrected["next_authority"] = (
            "Run exactly one reduced LC123-versus-LC137 teacher-free raw-9 screen; no FlightSim authority."
            if corrected.get("numerically_admitted")
            else "Reject LC137 and retain LC105; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.MINIMUM_SUCCESS_IMPROVEMENT,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.MINIMUM_SUCCESS_IMPROVEMENT = MINIMUM_SUCCESS_IMPROVEMENT
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.write_json_once = corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.MINIMUM_SUCCESS_IMPROVEMENT,
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
