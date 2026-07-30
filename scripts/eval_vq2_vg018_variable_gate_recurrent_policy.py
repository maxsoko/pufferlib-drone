#!/usr/bin/env python3
"""Fresh teacher-free variable-course screen for the admitted VG017 actor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator


TAG = "vq2_vg018_variable_gate_recurrent_teacher_free_256"
SEEDS = {5: 429085, 8: 429088, 11: 429091, 12: 429092}
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg017_variable_gate_four_source_refit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "6769476f309beb4bb502c64a83c4f992b47da2b1a8d8b6913398b4b19b3ab4a8"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "476c730649c0f127feb4975fe2ad44f66be64b0ea8dde4d48c5ada7bb0d4c86b"
)
VG017_ADMISSION = ROOT / "docs/vq2_vg017_four_source_refit_admission_2026-07-30.json"
VG017_ADMISSION_SHA256 = (
    "df664cbac28ee676cbd5e056cb1f06745bf3e622710b3faa085eca90425741b6"
)
PREREGISTRATION = (
    ROOT
    / "docs/vq2_vg018_variable_gate_teacher_free_preregistration_2026-07-30.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg018_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_candidate() -> None:
    if evaluator.sha256_path(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("VG018 candidate checkpoint hash mismatch")
    if evaluator.sha256_path(TRAIN_REPORT) != TRAIN_REPORT_SHA256:
        raise RuntimeError("VG018 candidate report hash mismatch")
    if evaluator.sha256_path(VG017_ADMISSION) != VG017_ADMISSION_SHA256:
        raise RuntimeError("VG017 admission evidence hash mismatch")
    report = json.loads(TRAIN_REPORT.read_text())
    admission = json.loads(VG017_ADMISSION.read_text())
    if (
        report.get("schema") != "vq2_variable_gate_four_source_refit_report_v1"
        or report.get("tag") != "vq2_vg017_variable_gate_four_source_refit_001"
        or not report.get("completed")
        or not report.get("numerically_admitted")
        or report.get("best_epoch") != 11
        or report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or report.get("minimum_transition_window_exposure", 0.0) < 3.0
        or not report.get("equal_source_weight_audit")
    ):
        raise RuntimeError("VG017 training report is not screen-admissible")
    if (
        admission.get("schema") != "vq2_vg017_four_source_refit_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != CHECKPOINT_SHA256
        or admission.get("artifact_sha256", {}).get("report")
        != TRAIN_REPORT_SHA256
    ):
        raise RuntimeError("VG017 admission evidence does not bind the candidate")


def configure_evaluator() -> None:
    """Bind the generic evaluator to VG018 before source identity is created."""

    evaluator.TAG = TAG
    evaluator.SCHEMA = "vq2_vg018_variable_gate_teacher_free_count_screen_v1"
    evaluator.SEEDS = dict(SEEDS)
    evaluator.CHECKPOINT = CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    evaluator.PREREGISTRATION = PREREGISTRATION
    evaluator.RUNNER = RUNNER
    evaluator.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), VG017_ADMISSION)
    evaluator.REQUIRE_ZERO_CROSSING_MARGIN = False


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
