#!/usr/bin/env python3
"""Collect the exact successful LC117 phase-6--9 intervention window."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_public_phase import LONG_COURSE_GATE_CAP
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc106_phase8_full_batch_recapture as lineage


TAG = "vq2_lc118_phase6_9_rescue_features_001"
SCHEMA = "vq2_lc118_phase6_9_rescue_features_report_v1"
STATE_SCHEMA = "vq2_lc118_phase6_9_rescue_features_state_v1"
AGENTS = EPISODES = 256
THREADS = 32
SEED = 432_050
STEP_LIMIT = 12_000
PHASE_MIN = 6
PHASE_MAX_EXCLUSIVE = 10
EXPECTED_RECORDS = 120_164
EXPECTED_QUERY_AGENTS = 38
EXPECTED_SUCCESS_AGENTS = 8
EXPECTED_FAILURE_AGENTS = 30
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc105_phase7_endpoint_full_course_001"
)
CHECKPOINT = PARENT_DIR / "policy_selected.pt"
CHECKPOINT_SHA256 = "005e5e7929258fd282ab390fd230fda817700afaee790e78144200e67b3e10a4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c614e6929282399cac2f18465084f8337ed8770389a74e48b86ff1fcf4315e7d"
LC117_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc117_teacher_rescue_horizon_ladder_001/report.json"
)
LC117_REPORT_SHA256 = "e2897fc5f79532ccf85bc8397186dc8652980a1e242fb9e6dfe892e8e4be08e5"
PREREGISTRATION = ROOT / "docs/vq2_lc118_phase6_9_rescue_features_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc118_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc118_phase6_9_rescue_features.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    for path, digest in {
        CHECKPOINT: CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC117_REPORT: LC117_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC118 bound input changed: {path}")
    payload = __import__("torch").load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    parent = json.loads(PARENT_REPORT.read_text())
    ladder = json.loads(LC117_REPORT.read_text())
    selected = ladder.get("items", [{}, {}])[-1]
    if (
        payload.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_report_v1"
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or ladder.get("schema") != "vq2_lc117_teacher_rescue_horizon_ladder_report_v1"
        or not ladder.get("diagnostic_valid")
        or ladder.get("training_oracle_rescue_horizon") != PHASE_MIN
        or ladder.get("horizons_executed") != [7, 6]
        or not selected.get("rescued")
        or selected.get("intervention_target_passes") != 4
        or selected.get("paired_target_gains") != 4
        or selected.get("paired_target_losses") != 0
        or selected.get("offline_teacher_plant_actions") != EXPECTED_RECORDS // 2
        or selected.get("trajectories_with_offline_teacher_actions")
        != EXPECTED_QUERY_AGENTS // 2
        or ladder.get("safety", {}).get("flight_sim_packets_sent") != 0
        or ladder.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC117 do not authorize the LC118 corpus")


def select_window_plant_actions(
    student: np.ndarray,
    teacher: np.ndarray,
    active: np.ndarray,
    phase_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    student = np.asarray(student, dtype=np.float32)
    teacher = np.asarray(teacher, dtype=np.float32)
    active = np.asarray(active, dtype=bool)
    phase_index = np.asarray(phase_index, dtype=np.int32)
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("LC118 student/teacher action batches changed")
    if student.shape[0] != active.size or active.shape != phase_index.shape:
        raise ValueError("LC118 active/phase batches changed")
    teacher_mask = (
        active
        & (phase_index >= PHASE_MIN)
        & (phase_index < PHASE_MAX_EXCLUSIVE)
    )
    plant = student.copy()
    plant[teacher_mask] = teacher[teacher_mask]
    return plant, teacher_mask


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
        "all_trajectories_completed": (
            report.get("completed_agents") == AGENTS
            and not report.get("stopped_on_record_target")
            and report.get("vector_steps") == STEP_LIMIT
        ),
        "exact_feature_and_plant_count": (
            report.get("feature_records") == EXPECTED_RECORDS
            and report.get("teacher_query_actions_recorded") == EXPECTED_RECORDS
            and report.get("teacher_plant_actions_executed") == EXPECTED_RECORDS
        ),
        "source_locked_outcomes_exact": (
            report.get("query_agents") == EXPECTED_QUERY_AGENTS
            and report.get("query_outcome_complete_agents") == EXPECTED_QUERY_AGENTS
            and report.get("query_outcome_success_agents") == EXPECTED_SUCCESS_AGENTS
            and report.get("query_outcome_failure_agents") == EXPECTED_FAILURE_AGENTS
            and report.get("query_outcome_censored_agents") == 0
        ),
        "only_rescue_window_recorded": (
            len(phases) == LONG_COURSE_GATE_CAP + 1
            and sum(phases[PHASE_MIN:PHASE_MAX_EXCLUSIVE]) == EXPECTED_RECORDS
            and sum(phases[:PHASE_MIN]) == 0
            and sum(phases[PHASE_MAX_EXCLUSIVE:]) == 0
        ),
        "query_bounds_exact": (
            report.get("feature_query_phase_min") == PHASE_MIN
            and report.get("feature_query_phase_max_exclusive")
            == PHASE_MAX_EXCLUSIVE
            and report.get("minimum_teacher_phase_index") == PHASE_MIN
            and report.get("maximum_teacher_phase_index")
            == PHASE_MAX_EXCLUSIVE - 1
        ),
        "zero_hard_native_fault": not any(
            metrics.get(f"env/{name}", math.inf) != 0.0 for name in required_zero
        ),
        "plant_action_history_exact": (
            report.get("executed_action_max_error", math.inf)
            <= collector().MAX_EXECUTED_ACTION_ERROR
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


def collector() -> Any:
    return lineage.base.recapture.base.base


def configure() -> tuple[Any, Any]:
    lineage.configure()
    lc100 = lineage.base
    lc100.TAG, lc100.SCHEMA, lc100.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    lc100.SEED, lc100.TARGET_PHASE = SEED, PHASE_MIN
    lc100.MINIMUM_RECORDS = EXPECTED_RECORDS
    lc100.EXPECTED_QUERY_AGENTS = EXPECTED_QUERY_AGENTS
    lc100.EXPECTED_SUCCESS_AGENTS = EXPECTED_SUCCESS_AGENTS
    lc100.EXPECTED_FAILURE_AGENTS = EXPECTED_FAILURE_AGENTS
    lc100.PREREGISTRATION, lc100.RUNNER, lc100.TEST = PREREGISTRATION, RUNNER, TEST
    lc100.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    original_verify = lc100.verify_inputs
    lc100.verify_inputs = verify_inputs
    lc100.configure()

    lc012 = lc100.recapture.base
    lc012.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), TEST, LC117_REPORT,
        ROOT / "scripts/eval_vq2_lc117_teacher_rescue_horizon_ladder.py",
    )
    target = collector()
    target.TAG, target.SCHEMA, target.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    target.AGENTS, target.EPISODES, target.SEED = AGENTS, EPISODES, SEED
    target.STEP_LIMIT = STEP_LIMIT
    target.INTERVENTION_PHASE_MIN = PHASE_MIN
    target.FEATURE_QUERY_PHASE_MIN = PHASE_MIN
    target.FEATURE_QUERY_PHASE_MAX_EXCLUSIVE = PHASE_MAX_EXCLUSIVE
    target.TEACHER_PLANT_ENABLED = True
    target.RECORD_STOP = None
    target.ACTOR_INFERENCE_CHUNK_SIZE = None
    target.PLANT_ACTION_CONTRACT = (
        "Saved LC105 Puffer owns phases below 6 and above 9; the training-only "
        "alignment oracle owns complete plant actions only at held phases 6--9"
    )
    target.select_configured_intervention_plant_actions = select_window_plant_actions
    target.collection_predicates = predicates
    target.PREREGISTRATION, target.RUNNER, target.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, DEFAULT_OUTPUT,
    )
    return lc100, original_verify


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    lc100, original_verify = configure()
    try:
        return collector().collect(
            output=output, device_name=device_name, resume=resume
        )
    finally:
        lc100.verify_inputs = original_verify


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
