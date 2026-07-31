#!/usr/bin/env python3
"""Fresh teacher-free variable-course screen for admitted VG028."""

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


TAG = "vq2_vg031_vg028_variable_gate_recurrent_teacher_free_256"
SEEDS = {5: 429123, 8: 429126, 11: 429129, 12: 429130}
SCREEN_THREADS = 4
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg028_variable_gate_seven_source_refit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "ec1206b5dbccb2ee679fa049534956b189b419da2b927f5e3a61eeaf5a359e65"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "b51b7a9b25bc9f213a8b93a3d60fca6a2226d2511f9475c0a6da7e12cd21909e"
)
VG028_ADMISSION = (
    ROOT / "docs/vq2_vg028_seven_source_refit_admission_2026-07-31.json"
)
VG028_ADMISSION_SHA256 = (
    "3282c98fab562a9593261538f452226190bc51b10e5e64495957cd348403755b"
)
VG030_ADMISSION = (
    ROOT / "docs/vq2_vg030_paired_count11_diagnostic_admission_2026-07-31.json"
)
VG030_ADMISSION_SHA256 = (
    "d44cacc132c5aa8fb16ed6f138cbf49730f3913c64ac4b784e0acf4a38bf639f"
)
GOAL_PROMPT = ROOT / "docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"
GOAL_PROMPT_SHA256 = (
    "03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_vg031_variable_gate_teacher_free_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg031_vast.sh"
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
        VG028_ADMISSION: VG028_ADMISSION_SHA256,
        VG030_ADMISSION: VG030_ADMISSION_SHA256,
        GOAL_PROMPT: GOAL_PROMPT_SHA256,
    }
    for path, digest in expected.items():
        if evaluator.sha256_path(path) != digest:
            raise RuntimeError(f"VG031 candidate evidence hash mismatch: {path}")
    report = json.loads(TRAIN_REPORT.read_text())
    admission = json.loads(VG028_ADMISSION.read_text())
    diagnostic = json.loads(VG030_ADMISSION.read_text())
    if (
        report.get("schema") != "vq2_variable_gate_seven_source_refit_report_v1"
        or report.get("tag") != "vq2_vg028_variable_gate_seven_source_refit_001"
        or not report.get("completed")
        or not report.get("numerically_admitted")
        or report.get("best_epoch") != 6
        or report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or report.get("minimum_transition_window_exposure", 0.0) < 4.0
        or not report.get("equal_source_weight_audit")
    ):
        raise RuntimeError("VG028 training report is not screen-admissible")
    if (
        admission.get("schema") != "vq2_vg028_seven_source_refit_admission_v1"
        or not admission.get("completed")
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != CHECKPOINT_SHA256
        or admission.get("artifact_sha256", {}).get("report")
        != TRAIN_REPORT_SHA256
    ):
        raise RuntimeError("VG028 admission evidence does not bind the candidate")
    if (
        diagnostic.get("schema")
        != "vq2_vg030_paired_count11_diagnostic_admission_v1"
        or not diagnostic.get("completed")
        or not diagnostic.get("qualified_for_full_screen")
        or diagnostic.get("candidate", {}).get("checkpoint_sha256")
        != CHECKPOINT_SHA256
        or diagnostic.get("safety", {}).get("flight_sim_packets_sent") != 0
        or diagnostic.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("VG030 does not authorize the VG031 offline screen")


def configure_evaluator() -> None:
    evaluator.TAG = TAG
    evaluator.SCHEMA = "vq2_vg031_variable_gate_teacher_free_count_screen_v1"
    evaluator.SEEDS = dict(SEEDS)
    evaluator.CHECKPOINT = CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    evaluator.PREREGISTRATION = PREREGISTRATION
    evaluator.RUNNER = RUNNER
    evaluator.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        VG028_ADMISSION,
        VG030_ADMISSION,
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
