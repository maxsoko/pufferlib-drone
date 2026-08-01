#!/usr/bin/env python3
"""Move the exact LC115 training-oracle rescue horizon back to phases 8--9."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc115_phase9_teacher_rescue as base


TAG = "vq2_lc116_phase8_9_teacher_rescue_001"
SCHEMA = "vq2_lc116_phase8_9_teacher_rescue_report_v1"
TEACHER_PHASE_MIN = 8
TEACHER_PHASE_MAX_EXCLUSIVE = 10
LC115_REPORT = base.DEFAULT_OUTPUT / "report.json"
LC115_REPORT_SHA256 = "cd456b536da8cbc481995274c89f5642b53791da6fe453a33b943d13fa58fbbe"
PREREGISTRATION = ROOT / "docs/vq2_lc116_phase8_9_teacher_rescue_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc116_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc116_phase8_9_teacher_rescue.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def select_plant_actions(
    student: np.ndarray,
    teacher: np.ndarray,
    active: np.ndarray,
    phase_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if student.shape != (base.TOTAL_AGENTS, base.ACTION_SIZE):
        raise ValueError("LC116 action batch does not match the exact paired contract")
    if teacher.shape != student.shape or active.shape != (base.TOTAL_AGENTS,):
        raise ValueError("LC116 teacher/active batch does not match")
    if phase_index.shape != active.shape:
        raise ValueError("LC116 phase batch does not match")
    teacher_mask = np.zeros(base.TOTAL_AGENTS, dtype=bool)
    selected = base.group_slice(1)
    teacher_mask[selected] = (
        active[selected]
        & (phase_index[selected] >= TEACHER_PHASE_MIN)
        & (phase_index[selected] < TEACHER_PHASE_MAX_EXCLUSIVE)
    )
    plant = student.copy()
    plant[teacher_mask] = teacher[teacher_mask]
    return plant, teacher_mask


def verify_inputs() -> None:
    base.verify_inputs()
    if sha256_path(LC115_REPORT) != LC115_REPORT_SHA256:
        raise RuntimeError("LC116 bound LC115 report changed")
    rejected = json.loads(LC115_REPORT.read_text())
    control, intervention = rejected.get("items", [{}, {}])
    if (
        rejected.get("schema") != "vq2_lc115_phase9_teacher_rescue_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("training_oracle_rescued_phase9")
        or control.get("target_passes") != 0
        or intervention.get("target_passes") != 0
        or intervention.get("offline_teacher_plant_actions") != 720
        or rejected.get("offline_teacher", {}).get("actions_below_target_phase") != 0
        or rejected.get("offline_teacher", {}).get("actions_outside_candidate_group") != 0
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC115 does not authorize the LC116 earlier horizon")


def source_identity(child_report: Path) -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        LC115_REPORT, child_report,
        ROOT / "scripts/eval_vq2_lc115_phase9_teacher_rescue.py",
        ROOT / "pufferlib/vq2_oracle.py",
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
            "numpy": np.__version__,
        },
    }


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    report_path = output / "report.json"
    child_output = output / "paired_child"
    child_report_path = child_output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()) and not child_report_path.is_file():
        raise RuntimeError("LC116 output exists without a resumable child report")
    output.mkdir(parents=True, exist_ok=True)

    original_values = (
        base.TAG, base.SCHEMA, base.TARGET_PHASE, base.DEFAULT_OUTPUT,
        base.select_plant_actions,
    )
    base.TAG = f"{TAG}_paired_child"
    base.SCHEMA = "vq2_lc116_phase8_9_teacher_rescue_child_v1"
    base.TARGET_PHASE = TEACHER_PHASE_MIN
    base.DEFAULT_OUTPUT = child_output
    base.select_plant_actions = select_plant_actions
    started = time.perf_counter()
    try:
        child = base.run(
            output=child_output, device_name=device_name,
            resume=child_report_path.is_file(),
        )
    finally:
        (
            base.TAG, base.SCHEMA, base.TARGET_PHASE, base.DEFAULT_OUTPUT,
            base.select_plant_actions,
        ) = original_values

    control = dict(child["items"][0])
    intervention = dict(child["items"][1])
    intervention.update({
        "name": "phase8_9_training_oracle",
        "plant": "puffer_except_training_oracle_at_held_phases8_9",
        "teacher_phase_min": TEACHER_PHASE_MIN,
        "teacher_phase_max_exclusive": TEACHER_PHASE_MAX_EXCLUSIVE,
    })
    diagnostic_valid = bool(
        child.get("diagnostic_valid")
        and child.get("initial_seed_groups_exact")
        and intervention.get("offline_teacher_plant_actions", 0) > 0
        and child.get("offline_teacher", {}).get("actions_below_target_phase") == 0
        and child.get("offline_teacher", {}).get("actions_outside_candidate_group") == 0
    )
    rescued = bool(
        diagnostic_valid
        and intervention["target_passes"] >= control["target_passes"] + 1
        and intervention["paired_target_gains_vs_control"] >= 1
        and intervention["paired_target_losses_vs_control"] == 0
        and intervention["pre_target_terminals"] <= control["pre_target_terminals"]
    )
    child_sha = sha256_path(child_report_path)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "diagnostic_valid": diagnostic_valid,
        "training_oracle_rescued_phase8_9": rescued,
        "numerically_admitted": False, "deployment_candidate": False,
        "teacher_phase_min": TEACHER_PHASE_MIN,
        "teacher_phase_max_exclusive": TEACHER_PHASE_MAX_EXCLUSIVE,
        "target_raw_index": base.TARGET_RAW_INDEX,
        "items": [control, intervention],
        "offline_teacher": child["offline_teacher"],
        "child_report_sha256": child_sha,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": source_identity(child_report_path),
        "safety": child["safety"],
        "next_authority": (
            "Collect exact phase-8/9 intervention records and distill a whole-Puffer candidate; no FlightSim authority."
            if rescued else
            "Reject this teacher through phase 8 and move the causal-horizon diagnosis earlier without fitting these labels."
        ),
    }
    write_json_once(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
