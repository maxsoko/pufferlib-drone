#!/usr/bin/env python3
"""Fresh 32-thread teacher-free variable-course screen for admitted VG025."""

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


TAG = "vq2_vg026_variable_gate_recurrent_teacher_free_256"
SEEDS = {5: 429111, 8: 429114, 11: 429117, 12: 429118}
SCREEN_THREADS = 32
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg025_variable_gate_six_source_refit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "85671c31a7c7a25cf49414cd51259357a2efd64b5d2b441fea27c51b93bf77aa"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "b9d22d2867b8518a0f8b1b8a0fbbad9d0977f3a57e189b83d1362be98f38dcb6"
)
VG025_ADMISSION = ROOT / "docs/vq2_vg025_six_source_refit_admission_2026-07-30.json"
VG025_ADMISSION_SHA256 = (
    "633a34d5b6470b271f3e376ab0d4c35ef179ad471c23cb3af8b840f090468ac8"
)
THREAD_PARITY_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg026_screen_thread_parity_001/report.json"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_vg026_variable_gate_teacher_free_preregistration_2026-07-30.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg026_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_GENERIC_TEACHER_FREE_CONFIG = evaluator.teacher_free_config


def accelerated_teacher_free_config(
    pufferl_module: Any,
    *,
    num_gates: int,
) -> tuple[dict[str, Any], list[str]]:
    """Use all 32 preregistered CPU workers for each 64-agent count."""

    config, overrides = _GENERIC_TEACHER_FREE_CONFIG(
        pufferl_module, num_gates=num_gates
    )
    config["vec"]["num_threads"] = SCREEN_THREADS
    return config, [*overrides, "--vec.num-threads", str(SCREEN_THREADS)]


def verify_candidate() -> None:
    expected = {
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        VG025_ADMISSION: VG025_ADMISSION_SHA256,
    }
    for path, digest in expected.items():
        if evaluator.sha256_path(path) != digest:
            raise RuntimeError(f"VG026 candidate evidence hash mismatch: {path}")
    report = json.loads(TRAIN_REPORT.read_text())
    admission = json.loads(VG025_ADMISSION.read_text())
    if (
        report.get("schema") != "vq2_variable_gate_six_source_refit_report_v1"
        or report.get("tag") != "vq2_vg025_variable_gate_six_source_refit_001"
        or not report.get("completed")
        or not report.get("numerically_admitted")
        or report.get("best_epoch") not in range(1, 7)
        or report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or report.get("minimum_transition_window_exposure", 0.0) < 4.0
        or not report.get("equal_source_weight_audit")
    ):
        raise RuntimeError("VG025 training report is not screen-admissible")
    if (
        admission.get("schema") != "vq2_vg025_six_source_refit_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != CHECKPOINT_SHA256
        or admission.get("artifact_sha256", {}).get("report")
        != TRAIN_REPORT_SHA256
    ):
        raise RuntimeError("VG025 admission evidence does not bind the candidate")


def verify_thread_parity() -> None:
    parity = json.loads(THREAD_PARITY_REPORT.read_text())
    if (
        parity.get("schema") != "vq2_vg026_screen_thread_parity_v1"
        or parity.get("tag") != "vq2_vg026_screen_thread_parity_001"
        or parity.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or parity.get("threads") != [4, SCREEN_THREADS]
        or parity.get("optimizer_steps") != 0
        or parity.get("state_writes") != 0
        or not parity.get("admitted")
    ):
        raise RuntimeError("VG026 4-vs-32-thread parity report is not admitted")


def configure_evaluator() -> None:
    evaluator.TAG = TAG
    evaluator.SCHEMA = "vq2_vg026_variable_gate_teacher_free_count_screen_v1"
    evaluator.SEEDS = dict(SEEDS)
    evaluator.CHECKPOINT = CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    evaluator.PREREGISTRATION = PREREGISTRATION
    evaluator.RUNNER = RUNNER
    evaluator.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        VG025_ADMISSION,
        THREAD_PARITY_REPORT,
    )
    evaluator.REQUIRE_ZERO_CROSSING_MARGIN = False
    evaluator.teacher_free_config = accelerated_teacher_free_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    verify_candidate()
    verify_thread_parity()
    configure_evaluator()
    report = evaluator.run_admission(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
