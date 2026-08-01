#!/usr/bin/env python3
"""Find the latest phase from which the teacher can rescue LC105 to raw 10."""

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
import scripts.eval_vq2_lc116_phase8_9_teacher_rescue as base


TAG = "vq2_lc117_teacher_rescue_horizon_ladder_001"
SCHEMA = "vq2_lc117_teacher_rescue_horizon_ladder_report_v1"
HORIZONS = (7, 6, 5)
LC116_REPORT = base.DEFAULT_OUTPUT / "report.json"
LC116_REPORT_SHA256 = "76e7e2d8783f23a71dc6d9ba426416490c2fa81ef37a2170abf3ab6ea02782b4"
PREREGISTRATION = ROOT / "docs/vq2_lc117_teacher_rescue_horizon_ladder_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc117_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc117_teacher_rescue_horizon_ladder.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    base.verify_inputs()
    if sha256_path(LC116_REPORT) != LC116_REPORT_SHA256:
        raise RuntimeError("LC117 bound LC116 report changed")
    rejected = json.loads(LC116_REPORT.read_text())
    control, intervention = rejected.get("items", [{}, {}])
    if (
        rejected.get("schema") != "vq2_lc116_phase8_9_teacher_rescue_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("training_oracle_rescued_phase8_9")
        or control.get("target_passes") != 0
        or intervention.get("target_passes") != 0
        or intervention.get("offline_teacher_plant_actions") != 3088
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC116 does not authorize the LC117 horizon ladder")


def rung_rescued(report: dict[str, Any]) -> bool:
    if len(report.get("items", [])) != 2:
        return False
    control, intervention = report["items"]
    return bool(
        report.get("diagnostic_valid")
        and report.get("training_oracle_rescued_phase8_9")
        and intervention.get("target_passes", 0) >= control.get("target_passes", 0) + 1
        and intervention.get("paired_target_gains_vs_control", 0) >= 1
        and intervention.get("paired_target_losses_vs_control") == 0
    )


def source_identity(rung_reports: list[Path]) -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        LC116_REPORT,
        ROOT / "scripts/eval_vq2_lc116_phase8_9_teacher_rescue.py",
        ROOT / "scripts/eval_vq2_lc115_phase9_teacher_rescue.py",
        *rung_reports,
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


def run_rung(
    *, horizon: int, output: Path, device_name: str
) -> dict[str, Any]:
    originals = (
        base.TAG, base.SCHEMA, base.TEACHER_PHASE_MIN,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
    )
    base.TAG = f"{TAG}_phase{horizon}_child"
    base.SCHEMA = "vq2_lc117_teacher_rescue_horizon_child_v1"
    base.TEACHER_PHASE_MIN = horizon
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = output
    try:
        return base.run(
            output=output, device_name=device_name,
            resume=(output / "report.json").is_file(),
        )
    finally:
        (
            base.TAG, base.SCHEMA, base.TEACHER_PHASE_MIN,
            base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        ) = originals


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists():
        unexpected = [
            path for path in output.iterdir()
            if not path.is_dir() or not path.name.startswith("phase")
        ]
        if unexpected:
            raise RuntimeError("LC117 output contains an unexpected partial artifact")
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rung_items: list[dict[str, Any]] = []
    rung_report_paths: list[Path] = []
    selected_horizon: int | None = None
    diagnostic_valid = True
    for horizon in HORIZONS:
        rung_output = output / f"phase{horizon}"
        rung = run_rung(
            horizon=horizon, output=rung_output, device_name=device_name
        )
        rung_path = rung_output / "report.json"
        rung_report_paths.append(rung_path)
        control, intervention = rung["items"]
        rescued = rung_rescued(rung)
        valid = bool(rung.get("diagnostic_valid"))
        diagnostic_valid &= valid
        rung_items.append({
            "teacher_phase_min": horizon,
            "teacher_phase_max_exclusive": base.TEACHER_PHASE_MAX_EXCLUSIVE,
            "diagnostic_valid": valid,
            "rescued": rescued,
            "control_target_passes": control["target_passes"],
            "intervention_target_passes": intervention["target_passes"],
            "paired_target_gains": intervention["paired_target_gains_vs_control"],
            "paired_target_losses": intervention["paired_target_losses_vs_control"],
            "control_pre_target_terminals": control["pre_target_terminals"],
            "intervention_pre_target_terminals": intervention["pre_target_terminals"],
            "intervention_maximum_raw_index_distribution": intervention[
                "maximum_raw_index_distribution"
            ],
            "offline_teacher_plant_actions": intervention[
                "offline_teacher_plant_actions"
            ],
            "trajectories_with_offline_teacher_actions": intervention[
                "trajectories_with_offline_teacher_actions"
            ],
            "rung_report_sha256": sha256_path(rung_path),
        })
        if rescued:
            selected_horizon = horizon
            break

    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "diagnostic_valid": diagnostic_valid,
        "training_oracle_rescue_horizon": selected_horizon,
        "numerically_admitted": False, "deployment_candidate": False,
        "horizons_preregistered": list(HORIZONS),
        "horizons_executed": [item["teacher_phase_min"] for item in rung_items],
        "stopped_at_first_rescue": selected_horizon is not None,
        "target_raw_index": base.base.TARGET_RAW_INDEX,
        "items": rung_items,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": source_identity(rung_report_paths),
        "safety": {
            "runtime_teacher_actions": 0,
            "offline_training_teacher_actions": sum(
                item["offline_teacher_plant_actions"] for item in rung_items
            ),
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            f"Collect exact phase-{selected_horizon}--9 intervention records and distill a whole-Puffer candidate; no FlightSim authority."
            if selected_horizon is not None else
            "Reject this oracle through phase 5 and diagnose an earlier horizon or different teacher before fitting labels."
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
