#!/usr/bin/env python3
"""Fresh teacher-free variable-course screen for admitted VG033."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator


TAG = "vq2_vg036_vg033_variable_gate_recurrent_teacher_free_256"
SEEDS = {5: 429135, 8: 429138, 11: 429141, 12: 429142}
SCREEN_THREADS = 4
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg033_variable_gate_eight_source_refit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60"
)
VG033_ADMISSION = (
    ROOT / "docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
)
VG033_ADMISSION_SHA256 = (
    "dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44"
)
VG035_ADMISSION = (
    ROOT / "docs/vq2_vg035_paired_count11_diagnostic_admission_2026-07-31.json"
)
VG035_ADMISSION_SHA256 = (
    "7c7676b72429157c8434761bf39ec790eb00a53abb76b4733128b980d8d95d98"
)
GOAL_PROMPT = ROOT / "docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"
GOAL_PROMPT_SHA256 = (
    "03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_vg036_variable_gate_teacher_free_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg036_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_GENERIC_TEACHER_FREE_CONFIG = evaluator.teacher_free_config


def fixed_thread_config(
    pufferl_module: Any,
    *,
    num_gates: int,
) -> tuple[dict[str, Any], list[str]]:
    """Retain the parity-proven four-thread native rollout path."""

    config, overrides = _GENERIC_TEACHER_FREE_CONFIG(
        pufferl_module, num_gates=num_gates
    )
    config["vec"]["num_threads"] = SCREEN_THREADS
    return config, [*overrides, "--vec.num-threads", str(SCREEN_THREADS)]


def verify_candidate() -> None:
    expected = {
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        VG033_ADMISSION: VG033_ADMISSION_SHA256,
        VG035_ADMISSION: VG035_ADMISSION_SHA256,
        GOAL_PROMPT: GOAL_PROMPT_SHA256,
    }
    for path, digest in expected.items():
        if evaluator.sha256_path(path) != digest:
            raise RuntimeError(f"VG036 candidate evidence hash mismatch: {path}")
    report = json.loads(TRAIN_REPORT.read_text())
    admission = json.loads(VG033_ADMISSION.read_text())
    diagnostic = json.loads(VG035_ADMISSION.read_text())
    if (
        report.get("schema") != "vq2_variable_gate_eight_source_refit_report_v1"
        or report.get("tag") != "vq2_vg033_variable_gate_eight_source_refit_001"
        or not report.get("completed")
        or not report.get("numerically_admitted")
        or report.get("best_epoch") != 6
        or report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or report.get("minimum_transition_window_exposure", 0.0) < 4.0
        or not report.get("equal_source_weight_audit")
    ):
        raise RuntimeError("VG033 training report is not screen-admissible")
    if (
        admission.get("schema") != "vq2_vg033_eight_source_refit_admission_v1"
        or not admission.get("completed")
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != CHECKPOINT_SHA256
        or admission.get("artifact_sha256", {}).get("report")
        != TRAIN_REPORT_SHA256
    ):
        raise RuntimeError("VG033 admission evidence does not bind the candidate")
    if (
        diagnostic.get("schema")
        != "vq2_vg035_paired_count11_diagnostic_admission_v1"
        or not diagnostic.get("completed")
        or not diagnostic.get("qualified_for_full_screen")
        or diagnostic.get("candidate", {}).get("checkpoint_sha256")
        != CHECKPOINT_SHA256
        or diagnostic.get("safety", {}).get("flight_sim_packets_sent") != 0
        or diagnostic.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("VG035 does not authorize the VG036 offline screen")


def configure_evaluator() -> None:
    evaluator.TAG = TAG
    evaluator.SCHEMA = "vq2_vg036_variable_gate_teacher_free_count_screen_v1"
    evaluator.SEEDS = dict(SEEDS)
    evaluator.CHECKPOINT = CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    evaluator.PREREGISTRATION = PREREGISTRATION
    evaluator.RUNNER = RUNNER
    evaluator.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        VG033_ADMISSION,
        VG035_ADMISSION,
        GOAL_PROMPT,
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
