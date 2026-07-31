#!/usr/bin/env python3
"""Collect capped phase-6 labels on states owned by the LC073 Puffer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_public_phase import LONG_COURSE_GATE_CAP
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc012_index1_student_dagger_features as base


TAG = "vq2_lc074_phase6_capped_student_features_001"
SCHEMA = "vq2_lc074_phase6_capped_student_features_report_v1"
STATE_SCHEMA = "vq2_lc074_phase6_capped_student_features_state_v1"
AGENTS = EPISODES = 256
THREADS = 32
SEED = 431740
TARGET_PHASE = 6
STEP_LIMIT = 6_000
RECORD_STOP = MINIMUM_RECORDS = 20_000
MINIMUM_QUERY_AGENTS = 8
LC073_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc073_phase2_constrained_endpoint_full_course_001"
)
LC073_REPORT = LC073_DIR / "report.json"
LC073_REPORT_SHA256 = "ae68af50524b02425f9a970c7f87dccbcbb6b60a0debb9899fcf1baa167b5d75"
CHECKPOINT = LC073_DIR / "policy_selected.pt"
CHECKPOINT_SHA256 = "5614a95cd2f9b428a99ddd43ec5e324c095344cae51b8d771aea3d5f05d35be8"
PREREGISTRATION = ROOT / "docs/vq2_lc074_phase6_capped_student_features_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc074_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc074_phase6_capped_student_features.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    for path, digest in {
        LC073_REPORT: LC073_REPORT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC074 bound input changed: {path}")
    report = json.loads(LC073_REPORT.read_text())
    selected = report.get("selected_candidate", {})
    if (
        report.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_report_v1"
        or not report.get("diagnostic_valid")
        or not report.get("numerically_admitted")
        or report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or selected.get("mean_gates_passed") != 3.390625
        or selected.get("maximum_raw_index") != 6
        or selected.get("maximum_raw_index_distribution", {}).get("6") != 9
        or selected.get("crash_rate") != 0.15625
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC073 does not authorize LC074")


def predicates(report: dict[str, Any]) -> dict[str, bool]:
    metrics = report.get("metrics", {})
    phases = report.get("feature_phase_records", [])
    required_zero = (
        "out_of_order", "action_envelope_violation",
        "wire_rate_envelope_violation", "thrust_envelope_violation",
    )
    return {
        "fixed_24_gate_course": report.get("num_gates") == 24,
        "source_batch_exact": (
            report.get("agents") == AGENTS and report.get("episodes") == EPISODES
        ),
        "deterministic_record_stop": (
            report.get("record_stop") == RECORD_STOP
            and report.get("stopped_on_record_target") is True
            and RECORD_STOP <= report.get("feature_records", 0) < RECORD_STOP + AGENTS
        ),
        "minimum_query_agent_diversity": (
            report.get("query_agents", 0) >= MINIMUM_QUERY_AGENTS
        ),
        "some_terminal_evidence": report.get("completed_agents", 0) > 0,
        "zero_hard_native_fault": not any(
            metrics.get(f"env/{name}", math.inf) != 0.0 for name in required_zero
        ),
        "records_match_queries": (
            report.get("feature_records") == report.get("teacher_query_actions_recorded")
        ),
        "only_target_phase_recorded": (
            len(phases) == LONG_COURSE_GATE_CAP + 1
            and phases[TARGET_PHASE] == report.get("feature_records")
            and sum(value for index, value in enumerate(phases) if index != TARGET_PHASE) == 0
        ),
        "student_owns_every_plant_action": (
            report.get("teacher_plant_actions_executed") == 0
            and report.get("student_plant_actions_executed")
            == report.get("total_plant_actions_executed")
        ),
        "query_bounds_exact": (
            report.get("feature_query_phase_min") == TARGET_PHASE
            and report.get("feature_query_phase_max_exclusive") == TARGET_PHASE + 1
            and report.get("minimum_teacher_phase_index") == TARGET_PHASE
            and report.get("maximum_teacher_phase_index") == TARGET_PHASE
        ),
        "plant_action_history_exact": (
            report.get("executed_action_max_error", math.inf)
            <= base.base.MAX_EXECUTED_ACTION_ERROR
        ),
        "phase_transport_exact": (
            report.get("phase_changes_off_tick") == 0
            and report.get("phase_decreases") == 0
            and report.get("phase_skips") == 0
            and report.get("raw_phase_encoding_max_error", math.inf) <= 1e-6
        ),
        "finite_in_envelope_labels": (
            report.get("nonfinite_values") == 0
            and report.get("teacher_action_envelope_violations") == 0
        ),
        "training_only_authority": (
            report.get("safety", {}).get("flight_sim_packets_sent") == 0
            and report.get("safety", {}).get("submission_authorized") is False
            and report.get("safety", {}).get("runtime_teacher_authorized") is False
        ),
    }


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.AGENTS, base.EPISODES, base.SEED = AGENTS, EPISODES, SEED
    base.THREADS = THREADS
    base.STEP_LIMIT, base.MINIMUM_RECORDS = STEP_LIMIT, MINIMUM_RECORDS
    base.TARGET_PHASE = TARGET_PHASE
    base.MINIMUM_TARGET_PHASE_REACHED = 0.0
    base.CHECKPOINT, base.CHECKPOINT_SHA256 = CHECKPOINT, CHECKPOINT_SHA256
    base.CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc073_phase2_constrained_endpoint_full_course_checkpoint_v1"
    )
    base.TRAIN_REPORT, base.TRAIN_REPORT_SHA256 = LC073_REPORT, LC073_REPORT_SHA256
    base.LC011_REPORT, base.LC011_REPORT_SHA256 = LC073_REPORT, LC073_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), TEST, LC073_REPORT)
    base.verify_inputs = verify_inputs
    base.configure()
    base.base.RECORD_STOP = RECORD_STOP
    base.base.collection_predicates = predicates


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    return base.base.collect(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_dataset_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
