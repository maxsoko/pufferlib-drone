#!/usr/bin/env python3
"""Replay LC009 and retain each long-course Puffer head at its best epoch."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_lc009_long_course_residual_fit as lc009
import scripts.train_vq2_vg068_indexed_mlp_intervention_features as fit_core
import scripts.train_vq2_vg069_phase_independent_early_stop as replay


TAG = "vq2_lc010r_phase_independent_early_stop_001"
SCHEMA = "vq2_lc010r_phase_independent_early_stop_report_v1"
STATE_SCHEMA = "vq2_lc010r_phase_independent_early_stop_state_v1"
CHECKPOINT_SCHEMA = "vq2_lc010r_phase_independent_early_stop_checkpoint_v1"
LC009_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc009_long_course_residual_fit_001/report.json"
)
LC009_REPORT_SHA256 = (
    "6e9cee8705303fd34947737a35789d70788bb5e883c8e136c93e600758443ff7"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc010_phase_independent_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc010_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, LC009_REPORT,
        *lc009.source_paths(),
        ROOT / "scripts/train_vq2_vg069_phase_independent_early_stop.py",
    )


def verify_inputs() -> dict[str, Any]:
    lc009.verify_inputs()
    if sha256_path(LC009_REPORT) != LC009_REPORT_SHA256:
        raise RuntimeError("LC010 bound LC009 report changed")
    report = json.loads(LC009_REPORT.read_text())
    if (
        report.get("schema") != lc009.SCHEMA
        or not report.get("completed")
        or report.get("numerically_admitted")
        or report.get("best_epoch") != 5
        or report.get("checkpoint_sha256")
        != "9cd389adc87fb17d9373f52b6eeb146dfed20b377b519a1414ae0afa6558c217"
        or not report.get("base_parameters_exact")
        or not report.get("non_target_outputs_zero")
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC009 rejection is not the fixed LC010 replay source")
    return report


def source_identity() -> tuple[str, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return commit, {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths()
    }


def numerical_admission(
    metrics: dict[str, Any], *, base_exact: bool, non_target_zero: bool,
    trainable_l2: float, config: Any,
) -> bool:
    phases = metrics["phases"]
    aggregate = (
        metrics["baseline_phase_balanced_action_mse"]
        / max(metrics["phase_balanced_action_mse"], 1e-20)
    )
    return bool(
        base_exact
        and non_target_zero
        and math.isfinite(trainable_l2)
        and trainable_l2 <= config.maximum_trainable_l2
        and aggregate >= 1.10
        and all(
            phases[str(phase)]["improvement_factor"] >= 1.05
            for phase in (1, 2, 3)
        )
        and all(
            item["selected_action_mse"]
            <= item["baseline_action_mse"] * (1.0 + 1e-5)
            for item in phases.values()
        )
    )


def configure() -> None:
    lc009.configure()
    replay.TAG = TAG
    replay.SCHEMA = SCHEMA
    replay.STATE_SCHEMA = STATE_SCHEMA
    replay.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    replay.CONFIG = lc009.CONFIG
    replay.TARGET_PHASES = lc009.TARGET_PHASES
    replay.VG068_REPORT = LC009_REPORT
    replay.VG068_REPORT_SHA256 = LC009_REPORT_SHA256
    replay.PREREGISTRATION = PREREGISTRATION
    replay.RUNNER = RUNNER
    replay.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    replay.vg068 = fit_core
    replay.NEXT_AUTHORITY_ADMITTED = (
        "One source-locked teacher-free phase-local long-course screen."
    )
    replay.source_paths = source_paths
    replay.verify_inputs = verify_inputs
    replay.source_identity = source_identity
    replay.numerical_admission = numerical_admission
    replay.checkpoint_model_contract = lc009.checkpoint_model_contract


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    configure()
    return replay.fit(output=output, device_name=device_name, resume=resume)


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
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
