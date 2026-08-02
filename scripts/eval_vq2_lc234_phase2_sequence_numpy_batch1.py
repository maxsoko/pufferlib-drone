#!/usr/bin/env python3
"""Screen LC233 through exact NumPy batch-1 Gate 3 and all 24 proxy gates."""

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
from scripts.policy_callable_vq2_lc233_phase2_sequence import NumpyLC233Policy
import scripts.eval_vq2_lc232_phase2_numpy_batch1_native as base


TAG_PREFIX = "vq2_lc234_phase2_sequence_numpy_batch1"
SCHEMA = "vq2_lc234_phase2_sequence_numpy_batch1_report_v1"
PARENT_CHECKPOINT_SHA256 = (
    "672af0c4b014bb7d7399e2d709f268a487a83294b7b2a72ebfc79a48896a95b9"
)
CANDIDATE_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc233_phase2_action_sequence_001"
)
CHECKPOINT = CANDIDATE_DIR / "policy_selected_numpy.npz"
CHECKPOINT_SHA256 = (
    "d3fd25442d2e0ad7b1fff7dca1a05f024aef01f9ea7b01cb2363ad8042f2c158"
)
SOURCE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "ebab0d42db832a31d91851f7514132bef1478b31a2fb902f3bd0a3978e9241c5"
)
SOURCE_REPORT = CANDIDATE_DIR / "report.json"
SOURCE_REPORT_SHA256 = (
    "01214b49488968001b7068abcdd2afde3c5b0414a2b7443d0d5c3edd8b5030f2"
)
LC232_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc232_phase2_numpy_batch1_native_raw3_001/report.json"
)
LC232_REPORT_SHA256 = (
    "40d2aaa929be12e42b7f99a5e623a19fa391a3e7f5d3f7ba4e3710780d907c01"
)
CALLABLE = ROOT / "scripts/policy_callable_vq2_lc233_phase2_sequence.py"
CALLABLE_SHA256 = (
    "67f1849239817dfb7bb331a844cb58e370b06b079602ca2a87c587a89e61fc44"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc234_phase2_sequence_numpy_batch1_preregistration_2026-08-02.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc234_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc234_phase2_sequence_numpy_batch1.py"


def bound_policy(checkpoint: Path | str) -> NumpyLC233Policy:
    return NumpyLC233Policy(
        checkpoint, expected_source_sha256=SOURCE_CHECKPOINT_SHA256
    )


def verify_inputs() -> None:
    for path, digest in {
        CHECKPOINT: CHECKPOINT_SHA256,
        SOURCE_CHECKPOINT: SOURCE_CHECKPOINT_SHA256,
        SOURCE_REPORT: SOURCE_REPORT_SHA256,
        LC232_REPORT: LC232_REPORT_SHA256,
        CALLABLE: CALLABLE_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC234 bound input changed: {path}")
    candidate = torch.load(
        SOURCE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    construction = json.loads(SOURCE_REPORT.read_text())
    rejected = json.loads(LC232_REPORT.read_text())
    if (
        candidate.get("schema")
        != "vq2_lc233_phase2_action_sequence_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or candidate.get("model", {}).get("class")
        != "VQ2Phase2AndLateActionSequenceActor"
        or candidate.get("model", {}).get("phase2_sequence_length") != 1433
        or candidate.get("model_state", {}).get("phase2_action_sequence").shape
        != (1433, 4)
        or construction.get("schema")
        != "vq2_lc233_phase2_action_sequence_report_v1"
        or not construction.get("numerically_admitted")
        or not construction.get("frozen_parent_state_exact")
        or construction.get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
        or construction.get("numpy_checkpoint_sha256") != CHECKPOINT_SHA256
        or construction.get("callable_sha256") != CALLABLE_SHA256
        or rejected.get("schema")
        != "vq2_lc232_phase2_numpy_batch1_native_report_v1"
        or rejected.get("milestone_admitted")
        or rejected.get("outcome", {}).get("target_passes") != 0
        or rejected.get("outcome", {}).get("pre_target_terminals") != 8
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("LC232/LC233 do not authorize LC234")


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST, CALLABLE,
        CHECKPOINT, SOURCE_CHECKPOINT, SOURCE_REPORT, LC232_REPORT,
        ROOT / "scripts/eval_vq2_lc229_numpy_batch1_native.py",
        ROOT / "scripts/build_vq2_lc233_phase2_action_sequence.py",
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
        "TAG_PREFIX", "SCHEMA", "PARENT_CHECKPOINT_SHA256", "CHECKPOINT",
        "CHECKPOINT_SHA256", "SOURCE_CHECKPOINT", "SOURCE_CHECKPOINT_SHA256",
        "SOURCE_REPORT", "SOURCE_REPORT_SHA256", "CALLABLE", "CALLABLE_SHA256",
        "PREREGISTRATION", "RUNNER", "TEST", "bound_policy", "verify_inputs",
        "source_identity",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG_PREFIX, SCHEMA, PARENT_CHECKPOINT_SHA256, CHECKPOINT,
        CHECKPOINT_SHA256, SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256,
        SOURCE_REPORT, SOURCE_REPORT_SHA256, CALLABLE, CALLABLE_SHA256,
        PREREGISTRATION, RUNNER, TEST, bound_policy, verify_inputs,
        source_identity,
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
    return base.base.LOG_ROOT / f"{TAG_PREFIX}_{suffix}"


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
