#!/usr/bin/env python3
"""Collect complete phase-6 trajectories with uncensored target outcomes."""

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
import scripts.collect_vq2_lc074_phase6_capped_student_features as lc074


TAG = "vq2_lc079_phase6_uncensored_outcomes_001"
SCHEMA = "vq2_lc079_phase6_uncensored_outcomes_report_v1"
STATE_SCHEMA = "vq2_lc079_phase6_uncensored_outcomes_state_v1"
AGENTS = EPISODES = 256
THREADS = 32
SEED = 431790
TARGET_PHASE = 6
STEP_LIMIT = 12_000
MINIMUM_RECORDS = 20_000
MINIMUM_QUERY_AGENTS = 8
MINIMUM_OUTCOME_CLASS_AGENTS = 2
CHECKPOINT = lc074.CHECKPOINT
CHECKPOINT_SHA256 = lc074.CHECKPOINT_SHA256
LC073_REPORT = lc074.LC073_REPORT
LC073_REPORT_SHA256 = lc074.LC073_REPORT_SHA256
LC075_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc075_phase6_horizon_capped_features_001/report.json"
)
LC075_REPORT_SHA256 = "0be7acea06bff0ab46591c67b1f78125103d82672018e5e320bfd30cb9836d76"
LC078_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc078_phase6_measured_bias_milestone_001/report.json"
)
LC078_REPORT_SHA256 = "76c54aa4ce433da86273392f2c16310cc99761d46da76292f544b8fb679513ae"
PREREGISTRATION = ROOT / "docs/vq2_lc079_phase6_uncensored_outcomes_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc079_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc079_phase6_uncensored_outcomes.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    lc074.verify_inputs()
    for path, digest in {
        LC075_REPORT: LC075_REPORT_SHA256,
        LC078_REPORT: LC078_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC079 bound rejection evidence changed: {path}")
    capped = json.loads(LC075_REPORT.read_text())
    rejected = json.loads(LC078_REPORT.read_text())
    if (
        capped.get("schema") != "vq2_lc075_phase6_horizon_capped_features_report_v1"
        or not capped.get("training_dataset_admitted")
        or capped.get("feature_records") != 20_007
        or capped.get("query_agents") != 18
        or capped.get("completed_agents") != 89
        or capped.get("stopped_on_record_target") is not True
        or rejected.get("schema") != "vq2_lc078_phase6_measured_bias_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or rejected.get("numerically_admitted")
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC075/LC078 do not authorize uncensored collection")


def predicates(report: dict[str, Any]) -> dict[str, bool]:
    metrics = report.get("metrics", {})
    phases = report.get("feature_phase_records", [])
    required_zero = (
        "out_of_order", "action_envelope_violation",
        "wire_rate_envelope_violation", "thrust_envelope_violation",
    )
    successes = report.get("query_outcome_success_agents", 0)
    failures = report.get("query_outcome_failure_agents", 0)
    return {
        "fixed_24_gate_course": report.get("num_gates") == 24,
        "source_batch_exact": (
            report.get("agents") == AGENTS and report.get("episodes") == EPISODES
        ),
        "all_trajectories_completed": (
            report.get("completed_agents") == AGENTS
            and report.get("stopped_on_record_target") is False
            and report.get("vector_steps", STEP_LIMIT + 1) <= STEP_LIMIT
        ),
        "minimum_feature_records": report.get("feature_records", 0) >= MINIMUM_RECORDS,
        "minimum_query_agent_diversity": report.get("query_agents", 0) >= MINIMUM_QUERY_AGENTS,
        "uncensored_target_outcomes": (
            report.get("query_outcome_censored_agents") == 0
            and report.get("query_outcome_complete_agents") == report.get("query_agents")
            and successes + failures == report.get("query_agents")
        ),
        "both_outcome_classes": (
            successes >= MINIMUM_OUTCOME_CLASS_AGENTS
            and failures >= MINIMUM_OUTCOME_CLASS_AGENTS
        ),
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
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), TEST, LC073_REPORT, LC075_REPORT, LC078_REPORT,
    )
    base.verify_inputs = verify_inputs
    base.configure()
    base.base.RECORD_STOP = None
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
