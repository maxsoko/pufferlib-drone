#!/usr/bin/env python3
"""Collect dense phase-6--23 teacher trajectories from gate-local starts."""

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


TAG = "vq2_lc095_late_phase_local_teacher_features_001"
SCHEMA = "vq2_lc095_late_phase_local_teacher_features_report_v1"
STATE_SCHEMA = "vq2_lc095_late_phase_local_teacher_features_state_v1"
AGENTS = EPISODES = 512
THREADS = 32
SEED = 431_950
PHASE_MIN = 6
PHASE_MAX_EXCLUSIVE = 24
STEP_LIMIT = 2_048
OFFSET_MIN = 2.0
OFFSET_MAX = 5.0
MINIMUM_RECORDS = 150_000
MINIMUM_RECORDS_PER_PHASE = 2_000
MAXIMUM_CRASH_RATE = 0.05
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc094_full_batch_serialized_checkpoint_001"
)
CHECKPOINT = PARENT_DIR / "policy_selected.pt"
CHECKPOINT_SHA256 = "14a5f90e8a7a80d3535ef0d86d3f9b5d75d42317c387a0554cf59ecf283e340a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "639edd98ec0d99c981d16837c802d758adc6622736314b3e6afe09a58af21139"
PREREGISTRATION = ROOT / "docs/vq2_lc095_late_phase_local_teacher_features_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc095_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc095_late_phase_local_teacher_features.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    for path, digest in {
        CHECKPOINT: CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC095 bound input changed: {path}")
    parent = json.loads(PARENT_REPORT.read_text())
    selected = parent.get("selected_candidate", {})
    if (
        parent.get("schema") != "vq2_lc094_full_batch_serialized_checkpoint_report_v1"
        or not parent.get("diagnostic_valid")
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or parent.get("mean_gate_gain") != 0.0390625
        or parent.get("promotion_target_pass_gain") != 3
        or selected.get("maximum_raw_index") != 9
        or parent.get("safety", {}).get("flight_sim_packets_sent") != 0
        or parent.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC094 does not authorize LC095")


def local_teacher_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = base.dagger_config(pufferl_module)
    config["env"].update({
        "evaluation_episode_limit": 1,
        "max_steps": STEP_LIMIT,
        "time_limit_seconds": STEP_LIMIT / 64.0,
        "gate_local_start_curriculum": 1,
        "gate_local_start_probability": 1.0,
        "gate_local_start_offset_min": OFFSET_MIN,
        "gate_local_start_offset_max": OFFSET_MAX,
        "gate_local_start_gate_min": PHASE_MIN,
        "gate_local_start_gate_max_exclusive": PHASE_MAX_EXCLUSIVE,
        "mixed_start_curriculum": 0,
        "segment_start_probability": 0.0,
        "use_custom_start": 0,
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
    })
    return config, overrides


def predicates(report: dict[str, Any]) -> dict[str, bool]:
    metrics = report.get("metrics", {})
    phases = report.get("feature_phase_records", [])
    required_zero = (
        "out_of_order", "action_envelope_violation",
        "wire_rate_envelope_violation", "thrust_envelope_violation",
    )
    phase_coverage = bool(
        len(phases) == LONG_COURSE_GATE_CAP + 1
        and all(phases[phase] >= MINIMUM_RECORDS_PER_PHASE
                for phase in range(PHASE_MIN, PHASE_MAX_EXCLUSIVE))
        and sum(phases[:PHASE_MIN]) == 0
        and sum(phases[PHASE_MAX_EXCLUSIVE:]) == 0
    )
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
        "all_resets_gate_local": (
            metrics.get("env/gate_local_start_rate") == 1.0
            and metrics.get("env/gate_local_reset_count") == 1.0
        ),
        "minimum_feature_records": report.get("feature_records", 0) >= MINIMUM_RECORDS,
        "all_late_phases_covered": phase_coverage,
        "bounded_crash_rate": metrics.get("env/crash", math.inf) <= MAXIMUM_CRASH_RATE,
        "zero_hard_native_fault": not any(
            metrics.get(f"env/{name}", math.inf) != 0.0 for name in required_zero
        ),
        "records_match_queries": (
            report.get("feature_records") == report.get("teacher_query_actions_recorded")
        ),
        "teacher_owns_every_plant_action": (
            report.get("teacher_plant_actions_executed")
            == report.get("total_plant_actions_executed")
            and report.get("student_plant_actions_executed") == 0
        ),
        "query_bounds_exact": (
            report.get("feature_query_phase_min") == PHASE_MIN
            and report.get("feature_query_phase_max_exclusive") == PHASE_MAX_EXCLUSIVE
            and report.get("minimum_teacher_phase_index") == PHASE_MIN
            and report.get("maximum_teacher_phase_index") == PHASE_MAX_EXCLUSIVE - 1
        ),
        "plant_action_history_exact": (
            report.get("executed_action_max_error", math.inf)
            <= base.base.MAX_EXECUTED_ACTION_ERROR
        ),
        "phase_transport_exact_after_local_jump": (
            report.get("phase_changes_off_tick") == 0
            and report.get("phase_decreases") == 0
            and report.get("phase_skips") == AGENTS
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
    base.TARGET_PHASE = PHASE_MIN
    base.MINIMUM_TARGET_PHASE_REACHED = 0.0
    base.CHECKPOINT, base.CHECKPOINT_SHA256 = CHECKPOINT, CHECKPOINT_SHA256
    base.CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc094_full_batch_serialized_checkpoint_checkpoint_v1"
    )
    base.TRAIN_REPORT, base.TRAIN_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.LC011_REPORT, base.LC011_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), TEST, PARENT_REPORT)
    base.verify_inputs = verify_inputs
    base.configure()
    base.base.RECORD_STOP = None
    base.base.INTERVENTION_PHASE_MIN = PHASE_MIN
    base.base.FEATURE_QUERY_PHASE_MIN = PHASE_MIN
    base.base.FEATURE_QUERY_PHASE_MAX_EXCLUSIVE = PHASE_MAX_EXCLUSIVE
    base.base.TEACHER_PLANT_ENABLED = True
    base.base.PLANT_ACTION_CONTRACT = (
        "Training-only alignment teacher owns every gate-local plant action; "
        "the Puffer provides legal recurrent features and is the sole future runtime policy"
    )
    base.base.intervention_config = local_teacher_config
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
