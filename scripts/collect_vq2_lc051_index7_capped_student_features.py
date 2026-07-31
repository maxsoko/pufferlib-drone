#!/usr/bin/env python3
"""Collect diverse phase-7 labels with a deterministic useful-record stop."""

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


TAG = "vq2_lc051_index7_capped_student_features_001"
SCHEMA = "vq2_lc051_index7_capped_student_report_v1"
STATE_SCHEMA = "vq2_lc051_index7_capped_student_state_v1"
AGENTS = EPISODES = 2048
THREADS = 128
SEED = 431510
TARGET_PHASE = 7
STEP_LIMIT = 12_000
RECORD_STOP = MINIMUM_RECORDS = 50_000
MINIMUM_QUERY_AGENTS = 8
LC048 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc048_phase6_sequential_interpolation_bracket_001"
)
LC048_REPORT = LC048 / "report.json"
LC048_REPORT_SHA256 = (
    "0f5849a41f7554b88565caff758ec9a60e25d3c359d3c9603b45646c1dceb669"
)
CHECKPOINT = LC048 / "a0p003/policy_selected.pt"
CHECKPOINT_SHA256 = (
    "5a2046a856aa29e8c8bff81789664ff68cd44d67e5ad6e6bf90ba02cf9dec084"
)
CHECKPOINT_REPORT = CHECKPOINT.parent / "report.json"
CHECKPOINT_REPORT_SHA256 = (
    "b2fb203ca9a49e74371694da71f8bd72757a1d0aab3d930ce5c2c8afdd85b3de"
)
LC050_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc050_phase7_reduced_interpolation_bracket_001/report.json"
)
LC050_REPORT_SHA256 = (
    "8bef58764fda323f9d4c61701bc21821b3c147a5f2edc12c56e440710ec2641b"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc051_index7_capped_student_features_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc051_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    expected = {
        LC048_REPORT: LC048_REPORT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
        CHECKPOINT_REPORT: CHECKPOINT_REPORT_SHA256,
        LC050_REPORT: LC050_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC051 bound input changed: {path}")
    aggregate = json.loads(LC048_REPORT.read_text())
    child = json.loads(CHECKPOINT_REPORT.read_text())
    rejected = json.loads(LC050_REPORT.read_text())
    if (
        aggregate.get("schema")
        != "vq2_lc048_phase6_sequential_interpolation_bracket_report_v1"
        or not aggregate.get("numerically_admitted")
        or aggregate.get("selected", {}).get("alpha") != 0.0025
        or aggregate.get("selected", {}).get("checkpoint_sha256")
        != CHECKPOINT_SHA256
        or child.get("schema") != "vq2_lc048_phase6_interpolation_screen_v1"
        or not child.get("numerically_admitted")
        or child.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or rejected.get("schema")
        != "vq2_lc050_phase7_reduced_interpolation_bracket_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("numerically_admitted")
        or rejected.get("selected") is not None
        or rejected.get("baseline", {}).get("mean_gates_passed") != 3.8125
        or rejected.get("baseline", {}).get("maximum_raw_index") != 8
        or rejected.get("baseline", {}).get("crash_rate") != 0.03125
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or aggregate.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC048/LC050 do not authorize phase-7 collection")


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
            and RECORD_STOP <= report.get("feature_records", 0)
            < RECORD_STOP + AGENTS
        ),
        "minimum_query_agent_diversity": (
            report.get("query_agents", 0) >= MINIMUM_QUERY_AGENTS
        ),
        "some_terminal_evidence": report.get("completed_agents", 0) > 0,
        "zero_hard_native_fault": not any(
            metrics.get(f"env/{name}", math.inf) != 0.0
            for name in required_zero
        ),
        "records_match_queries": (
            report.get("feature_records")
            == report.get("teacher_query_actions_recorded")
        ),
        "only_target_phase_recorded": (
            len(phases) == LONG_COURSE_GATE_CAP + 1
            and phases[TARGET_PHASE] == report.get("feature_records")
            and sum(
                value for index, value in enumerate(phases)
                if index != TARGET_PHASE
            ) == 0
        ),
        "student_owns_every_plant_action": (
            report.get("teacher_plant_actions_executed") == 0
            and report.get("student_plant_actions_executed")
            == report.get("total_plant_actions_executed")
        ),
        "query_bounds_exact": (
            report.get("feature_query_phase_min") == TARGET_PHASE
            and report.get("feature_query_phase_max_exclusive")
            == TARGET_PHASE + 1
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
        "vq2_lc048_phase6_interpolation_checkpoint_v1"
    )
    base.TRAIN_REPORT, base.TRAIN_REPORT_SHA256 = (
        CHECKPOINT_REPORT, CHECKPOINT_REPORT_SHA256,
    )
    base.LC011_REPORT, base.LC011_REPORT_SHA256 = (
        LC048_REPORT, LC048_REPORT_SHA256,
    )
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), LC050_REPORT)
    base.verify_inputs = verify_inputs
    base.configure()
    base.base.RECORD_STOP = RECORD_STOP
    base.base.collection_predicates = predicates


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
            resume: bool = False) -> dict[str, Any]:
    configure()
    return base.base.collect(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_dataset_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
