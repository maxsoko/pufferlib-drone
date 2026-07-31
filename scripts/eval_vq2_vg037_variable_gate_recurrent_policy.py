#!/usr/bin/env python3
"""Accelerated fresh teacher-free variable-course screen for admitted VG033."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
import scripts.eval_vq2_vg036_variable_gate_recurrent_policy as vg036


TAG = "vq2_vg037_vg033_variable_gate_recurrent_teacher_free_256"
SEEDS = {5: 429143, 8: 429146, 11: 429149, 12: 429150}
PREREGISTRATION = (
    ROOT / "docs/vq2_vg037_variable_gate_teacher_free_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg037_vast.sh"
VG036_ABORT = ROOT / "docs/vq2_vg036_infrastructure_abort_2026-07-31.json"
VG036_ABORT_SHA256 = (
    "c5dc3287c2891c746de4eadd55df8648df7762b45e2655baeff623a29af7c27e"
)
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
fixed_thread_config = vg036.fixed_thread_config


def verify_candidate() -> None:
    vg036.verify_candidate()
    if evaluator.sha256_path(VG036_ABORT) != VG036_ABORT_SHA256:
        raise RuntimeError("VG037 does not bind the VG036 infrastructure abort")
    abort = json.loads(VG036_ABORT.read_text())
    if (
        abort.get("schema") != "vq2_vg036_infrastructure_abort_v1"
        or abort.get("status") != "aborted_before_first_completed_count"
        or abort.get("completed_counts") != []
        or abort.get("count_reports_written") != 0
        or abort.get("policy_conclusion") != "none"
        or abort.get("safety", {}).get("flight_sim_packets_sent") != 0
        or abort.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("VG036 abort evidence does not authorize VG037")


def configure_evaluator() -> None:
    evaluator.TAG = TAG
    evaluator.SCHEMA = "vq2_vg037_variable_gate_teacher_free_count_screen_v1"
    evaluator.SEEDS = dict(SEEDS)
    evaluator.CHECKPOINT = vg036.CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = vg036.CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = vg036.TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = vg036.TRAIN_REPORT_SHA256
    evaluator.PREREGISTRATION = PREREGISTRATION
    evaluator.RUNNER = RUNNER
    evaluator.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        Path(vg036.__file__).resolve(),
        vg036.VG033_ADMISSION,
        vg036.VG035_ADMISSION,
        vg036.GOAL_PROMPT,
        VG036_ABORT,
    )
    evaluator.REQUIRE_ZERO_CROSSING_MARGIN = False
    evaluator.teacher_free_config = fixed_thread_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    verify_candidate()
    configure_evaluator()
    report = evaluator.run_admission(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
