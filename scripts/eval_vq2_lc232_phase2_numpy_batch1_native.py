#!/usr/bin/env python3
"""Screen LC231 through exact NumPy batch-1 native Gate 3 and all 24 gates."""

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
from scripts.policy_callable_vq2_lc231_phase2 import NumpyLC231Policy
import scripts.eval_vq2_lc229_numpy_batch1_native as base


TAG_PREFIX = "vq2_lc232_phase2_numpy_batch1_native"
SCHEMA = "vq2_lc232_phase2_numpy_batch1_native_report_v1"
PARENT_CHECKPOINT_SHA256 = (
    "672af0c4b014bb7d7399e2d709f268a487a83294b7b2a72ebfc79a48896a95b9"
)
CANDIDATE_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc231_phase2_full_residual_001"
)
CHECKPOINT = CANDIDATE_DIR / "policy_endpoint_numpy.npz"
CHECKPOINT_SHA256 = (
    "a0156a423fb06996221226f446599be827b34f12739524fbde15eab29061aa6a"
)
SOURCE_CHECKPOINT = CANDIDATE_DIR / "policy_endpoint.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "ba16dd1cfd06fbce985b73ba04bc800882dd48a212523ef3ecbd33d389e04600"
)
SOURCE_REPORT = CANDIDATE_DIR / "report.json"
SOURCE_REPORT_SHA256 = (
    "e48093be4c42352f66670b2f0115bcfee9dca2676009c9d19232a5a9f47e5b7a"
)
EXPORT_REPORT = CANDIDATE_DIR / "numpy_export.json"
EXPORT_REPORT_SHA256 = (
    "929f1f448b3f25f82e202d2370c20549ae3a97ff23f4197f954edee74cb79b99"
)
LC229_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc229_numpy_batch1_native_raw3_001/report.json"
)
LC229_REPORT_SHA256 = (
    "5b04510999480f2bae03d332377aaf30fb76e2d24902165be3950ab08b50dddd"
)
CALLABLE = ROOT / "scripts/policy_callable_vq2_lc231_phase2.py"
CALLABLE_SHA256 = (
    "0a7425c7b75f094529287914d0d8091af35da7b08c3df6af2e9f546010aaa6b8"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc232_phase2_numpy_batch1_native_preregistration_2026-08-02.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc232_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc232_phase2_numpy_batch1_native.py"


def bound_policy(checkpoint: Path | str) -> NumpyLC231Policy:
    return NumpyLC231Policy(
        checkpoint, expected_source_sha256=SOURCE_CHECKPOINT_SHA256
    )


def verify_inputs() -> None:
    for path, digest in {
        CHECKPOINT: CHECKPOINT_SHA256,
        SOURCE_CHECKPOINT: SOURCE_CHECKPOINT_SHA256,
        SOURCE_REPORT: SOURCE_REPORT_SHA256,
        EXPORT_REPORT: EXPORT_REPORT_SHA256,
        LC229_REPORT: LC229_REPORT_SHA256,
        CALLABLE: CALLABLE_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC232 bound input changed: {path}")
    candidate = torch.load(
        SOURCE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    fit = json.loads(SOURCE_REPORT.read_text())
    export = json.loads(EXPORT_REPORT.read_text())
    failed = json.loads(LC229_REPORT.read_text())
    if (
        candidate.get("schema")
        != "vq2_lc231_phase2_full_residual_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256")
        != PARENT_CHECKPOINT_SHA256
        or fit.get("schema") != "vq2_lc231_phase2_full_residual_report_v1"
        or not fit.get("numerically_admitted")
        or fit.get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
        or fit.get("phases") != [2]
        or fit.get("phase_reports", {}).get("2", {}).get("improvement_factor", 0)
        < 1.5
        or export.get("schema")
        != "vq2_lc231_phase2_full_residual_numpy_export_v1"
        or export.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
        or export.get("numpy_checkpoint_sha256") != CHECKPOINT_SHA256
        or export.get("callable_sha256") != CALLABLE_SHA256
        or failed.get("schema") != "vq2_lc229_numpy_batch1_native_report_v1"
        or failed.get("outcome", {}).get("target_passes") != 0
        or failed.get("outcome", {}).get("pre_target_terminals") != 8
        or failed.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("LC229/LC231 do not authorize LC232")


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST, CALLABLE,
        CHECKPOINT, SOURCE_CHECKPOINT, SOURCE_REPORT, EXPORT_REPORT,
        LC229_REPORT, ROOT / "scripts/eval_vq2_lc229_numpy_batch1_native.py",
        ROOT / "scripts/train_vq2_lc231_phase2_full_residual.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "scripts/eval_vq2_lc001_long_course_oracle.py",
        ROOT / "scripts/eval_vq2_lc007_converted_vg071_long_course.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h", ROOT / "src/vecenv.h",
        ROOT / "src/bindings.cu", ROOT / "src/bindings_cpu.cpp",
    )
    extension = Path(_C.__file__).resolve()
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {
            **{str(path.relative_to(ROOT)): sha256_path(path) for path in paths},
            "compiled_extension": sha256_path(extension),
        },
        "compiled_extension_path": str(extension),
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }


def configure() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG_PREFIX", "SCHEMA", "CHECKPOINT", "CHECKPOINT_SHA256",
        "SOURCE_CHECKPOINT", "SOURCE_CHECKPOINT_SHA256", "SOURCE_REPORT",
        "SOURCE_REPORT_SHA256", "CALLABLE", "CALLABLE_SHA256",
        "PREREGISTRATION", "RUNNER", "TEST", "NumpyLC216Policy",
        "verify_inputs", "source_identity",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG_PREFIX, SCHEMA, CHECKPOINT, CHECKPOINT_SHA256, SOURCE_CHECKPOINT,
        SOURCE_CHECKPOINT_SHA256, SOURCE_REPORT, SOURCE_REPORT_SHA256,
        CALLABLE, CALLABLE_SHA256, PREREGISTRATION, RUNNER, TEST,
        bound_policy, verify_inputs, source_identity,
    )
    for name, value in zip(names, values):
        setattr(base, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(base, name, value)


def run(
    *, target_raw_index: int, output: Path, resume: bool = False,
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure()
    try:
        return base.run(
            target_raw_index=target_raw_index, output=output, resume=resume
        )
    finally:
        restore(snapshot)


def default_output(target_raw_index: int) -> Path:
    suffix = "raw3_001" if target_raw_index == 3 else "all24_001"
    return base.LOG_ROOT / f"{TAG_PREFIX}_{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, choices=(3, 24), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = default_output(args.target) if args.output is None else args.output.resolve()
    report = run(
        target_raw_index=args.target, output=output, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("diagnostic_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
