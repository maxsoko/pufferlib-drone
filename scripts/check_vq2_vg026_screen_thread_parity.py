#!/usr/bin/env python3
"""Prove 4-vs-32-thread native screen equivalence before VG026."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
import scripts.eval_vq2_vg026_variable_gate_recurrent_policy as vg026
from scripts.eval_vq2_variable_gate_oracle import write_json_atomic


TAG = "vq2_vg026_screen_thread_parity_001"
SCHEMA = "vq2_vg026_screen_thread_parity_v1"
THREADS = (4, vg026.SCREEN_THREADS)
COUNT = 5
STEPS = 128
MAXIMUM_NUMERIC_ERROR = 1e-7
DEFAULT_OUTPUT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG
    / "report.json"
)


def _thread_config(threads: int):
    def configured(
        pufferl_module: Any,
        *,
        num_gates: int,
    ) -> tuple[dict[str, Any], list[str]]:
        config, overrides = vg026._GENERIC_TEACHER_FREE_CONFIG(
            pufferl_module, num_gates=num_gates
        )
        config["vec"]["num_threads"] = threads
        config["env"]["max_steps"] = STEPS
        config["env"]["time_limit_seconds"] = STEPS / 64.0
        return config, [
            *overrides,
            "--vec.num-threads",
            str(threads),
            "--env.max-steps",
            str(STEPS),
        ]

    return configured


def _numeric_max_error(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf")
        return max(
            (_numeric_max_error(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max(
            (_numeric_max_error(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not np.isfinite(float(left)) or not np.isfinite(float(right)):
            return 0.0 if left == right else float("inf")
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(),
        vg026.PREREGISTRATION,
        Path(vg026.__file__).resolve(),
        ROOT / "scripts/eval_vq2_variable_gate_recurrent_policy.py",
        vg026.CHECKPOINT,
        vg026.TRAIN_REPORT,
        vg026.VG025_ADMISSION,
    )
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): evaluator.sha256_path(path) for path in paths
        },
        "compiled_extension_sha256": evaluator.sha256_path(
            Path(_C.__file__).resolve()
        ),
        "runtime": evaluator.runtime_manifest(),
    }


def check(*, output: Path, device_name: str) -> dict[str, Any]:
    vg026.verify_candidate()
    identity = source_identity()
    if output.is_file():
        report = json.loads(output.read_text())
        if report.get("source_identity") != identity or not report.get("admitted"):
            raise RuntimeError("existing VG026 thread parity report changed")
        return report
    if output.parent.exists():
        raise RuntimeError("VG026 parity output exists without its report")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG026 thread parity requires CUDA inference")

    vg026.configure_evaluator()
    device = torch.device(device_name)
    reports: dict[str, dict[str, Any]] = {}
    for threads in THREADS:
        evaluator.teacher_free_config = _thread_config(threads)
        reports[str(threads)] = evaluator.run_count(
            num_gates=COUNT,
            device=device,
            source_identity={"thread_parity": identity},
        )

    left, right = (reports[str(threads)] for threads in THREADS)
    exact_fields = (
        "completed",
        "num_gates",
        "agents",
        "episodes",
        "seed",
        "checkpoint_sha256",
        "checkpoint_best_epoch",
        "crossing_margin_admission_predicate",
        "vector_steps",
        "teacher_action_blend",
        "actor_input_width",
        "actor_input_privileged_values",
        "status_hold_steps",
        "phase_samples",
        "phase_changes_off_tick",
        "phase_decreases",
        "phase_skips",
        "nonfinite_action",
        "action_samples",
        "action_envelope_violations",
        "maximum_raw_public_index_distribution",
        "maximum_held_public_index_distribution",
        "safety",
    )
    exact_equal = all(left[field] == right[field] for field in exact_fields)
    numeric_fields = (
        "raw_phase_encoding_max_error",
        "executed_action_max_error",
        "action_min",
        "action_max",
        "action_mean",
        "action_std",
        "metrics",
    )
    numeric_errors = {
        field: _numeric_max_error(left[field], right[field])
        for field in numeric_fields
    }
    admitted = bool(
        exact_equal
        and max(numeric_errors.values(), default=float("inf"))
        <= MAXIMUM_NUMERIC_ERROR
        and left["vector_steps"] == STEPS
        and left["action_samples"] == STEPS * evaluator.AGENTS
        and left["executed_action_max_error"] == 0.0
        and right["executed_action_max_error"] == 0.0
    )
    report = {
        "schema": SCHEMA,
        "tag": TAG,
        "admitted": admitted,
        "threads": list(THREADS),
        "count": COUNT,
        "steps": STEPS,
        "agents": evaluator.AGENTS,
        "seed": vg026.SEEDS[COUNT],
        "checkpoint_sha256": vg026.CHECKPOINT_SHA256,
        "exact_fields_equal": exact_equal,
        "numeric_max_abs_error": numeric_errors,
        "maximum_allowed_numeric_error": MAXIMUM_NUMERIC_ERROR,
        "reports": reports,
        "optimizer_steps": 0,
        "state_writes": 0,
        "source_identity": identity,
        "safety": {
            "teacher_action_blend": 0.0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    output.parent.mkdir(parents=True)
    write_json_atomic(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    report = check(output=args.output.resolve(), device_name=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
