#!/usr/bin/env python3
"""Collect phases 4--23 behind a Puffer-warmed, training-only teacher."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_public_phase import (
    LONG_COURSE_GATE_CAP,
    OFFICIAL_PROGRESS_SCALE,
)
from pufferlib.vq2_recurrent_phase_residual import (
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_lc001_long_course_oracle import load_config
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc012_index1_student_dagger_features as student_base
import scripts.collect_vq2_vg062_warmed_teacher_intervention_features as base


TAG = "vq2_lc038_phase4plus_teacher_features_001"
SCHEMA = "vq2_lc038_phase4plus_teacher_report_v1"
STATE_SCHEMA = "vq2_lc038_phase4plus_teacher_state_v1"
AGENTS = EPISODES = 128
SEED = 431380
NUM_GATES = 24
THREADS = 32
STEP_LIMIT = 32_000
INTERVENTION_PHASE_MIN = 4
FEATURE_QUERY_PHASE_MIN = 4
FEATURE_QUERY_PHASE_MAX_EXCLUSIVE = 24
MINIMUM_RECORDS = 400_000
MINIMUM_PER_PHASE_RECORDS = 1_000
LC037 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc037_phase4_interpolation_bracket_001"
)
LC037_REPORT = LC037 / "report.json"
LC037_REPORT_SHA256 = (
    "5678e4550d2301e8befb1360d4c2b6b0a1b5bec7e984c8f09dd9aae00f6c114b"
)
PARENT_CHECKPOINT = LC037 / "a0p005/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "da1940dd6bedee11e4eb6449bcd2a7844bb6ca3cdcf238900c1c6b22cd47507d"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "73e0bcd3fa320140aade4bfae1d0393544dd816d5e460bf6eeb6be107ff4f18f"
)
ORACLE_24_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc001_long_course_oracle_001/vector_24g.json"
)
ORACLE_24_REPORT_SHA256 = (
    "b1a83357786adb6ac8169a4aef62f9add6efc6446bb6eadcc843c77df892203c"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc038_phase4plus_teacher_features_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc038_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER,
        LC037_REPORT, PARENT_CHECKPOINT, PARENT_REPORT, ORACLE_24_REPORT,
        ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc001_long_course_oracle.py",
        ROOT / "scripts/collect_vq2_lc012_index1_student_dagger_features.py",
        ROOT / "scripts/collect_vq2_vg062_warmed_teacher_intervention_features.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
    )


def verify_inputs() -> None:
    expected = {
        LC037_REPORT: LC037_REPORT_SHA256,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        ORACLE_24_REPORT: ORACLE_24_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC038 bound input changed: {path}")
    aggregate = json.loads(LC037_REPORT.read_text())
    parent = json.loads(PARENT_REPORT.read_text())
    oracle = json.loads(ORACLE_24_REPORT.read_text())
    selected = aggregate.get("selected", {})
    if (
        aggregate.get("schema")
        != "vq2_lc037_phase4_interpolation_bracket_report_v1"
        or not aggregate.get("numerically_admitted")
        or selected.get("alpha") != 0.005
        or selected.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or selected.get("mean_gates_passed") != 3.71875
        or selected.get("maximum_raw_index") != 7
        or selected.get("crash_rate") != 0.03125
        or parent.get("schema") != "vq2_lc037_phase4_interpolation_screen_v1"
        or not parent.get("numerically_admitted")
        or oracle.get("num_gates") != 24
        or not oracle.get("admitted")
        or oracle.get("metrics", {}).get("env/success_rate") != 1.0
        or oracle.get("metrics", {}).get("env/crash") != 0.0
        or oracle.get("safety", {}).get("flight_sim_packets_sent") != 0
        or oracle.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC037/oracle evidence does not authorize LC038")


def intervention_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = load_config(
        pufferl_module, num_gates=NUM_GATES, agents=AGENTS,
        episodes=EPISODES, seed=SEED, threads=THREADS,
    )
    config["env"].update({
        "evaluation_episode_limit": 1,
        "max_steps": STEP_LIMIT,
        "time_limit_seconds": STEP_LIMIT / 64.0,
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
        "observable_gate_index_denominator": OFFICIAL_PROGRESS_SCALE,
        "observable_gate_progress_unbounded": 1,
        "gate_local_start_curriculum": 0,
        "gate_local_start_probability": 0.0,
        "mixed_start_curriculum": 0,
        "segment_start_probability": 0.0,
        "use_custom_start": 0,
    })
    return config, overrides


def load_actor(
    device: torch.device,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any]]:
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != "vq2_lc037_phase4_interpolation_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or contract.get("class") != "VQ2UnboundedProgressMLPResidualActor"
    ):
        raise RuntimeError("LC038 parent Puffer contract changed")
    actor = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(payload["model_state"])
    actor.eval()
    return actor, payload


def predicates(report: dict[str, Any]) -> dict[str, bool]:
    metrics = report.get("metrics", {})
    phases = report.get("feature_phase_records", [])
    required_zero = (
        "out_of_order", "action_envelope_violation",
        "wire_rate_envelope_violation", "thrust_envelope_violation",
    )
    all_late = (
        len(phases) == LONG_COURSE_GATE_CAP + 1
        and all(
            phases[index] >= MINIMUM_PER_PHASE_RECORDS
            for index in range(FEATURE_QUERY_PHASE_MIN, NUM_GATES)
        )
        and sum(phases[:FEATURE_QUERY_PHASE_MIN]) == 0
        and sum(phases[NUM_GATES:]) == 0
    )
    return {
        "exact_episode_count": metrics.get("env/n") == float(EPISODES),
        "fixed_24_gate_course": metrics.get("env/gate_count24_episode") == 1.0,
        "student_prefix_reaches_phase4": (
            metrics.get("env/ordered_gate3_sampled", 0.0) >= 0.15
        ),
        "teacher_continuation_completes": (
            metrics.get("env/success_rate", 0.0) >= 0.10
        ),
        "bounded_crash_rate": metrics.get("env/crash", math.inf) <= 0.20,
        "zero_hard_native_fault": not any(
            metrics.get(f"env/{name}", math.inf) != 0.0
            for name in required_zero
        ),
        "minimum_feature_records": report.get("feature_records", 0)
        >= MINIMUM_RECORDS,
        "all_phase4_through_23_heads_observed": all_late,
        "records_match_teacher_plant_actions": (
            report.get("feature_records")
            == report.get("teacher_plant_actions_executed")
        ),
        "student_prefix_and_teacher_suffix_present": (
            report.get("student_plant_actions_executed", 0) > 0
            and report.get("teacher_plant_actions_executed", 0) > 0
            and report.get("student_plant_actions_executed", 0)
            + report.get("teacher_plant_actions_executed", 0)
            == report.get("total_plant_actions_executed")
        ),
        "query_bounds_exact": (
            report.get("feature_query_phase_min") == FEATURE_QUERY_PHASE_MIN
            and report.get("feature_query_phase_max_exclusive")
            == FEATURE_QUERY_PHASE_MAX_EXCLUSIVE
            and report.get("minimum_teacher_phase_index")
            == FEATURE_QUERY_PHASE_MIN
            and report.get("maximum_teacher_phase_index") == NUM_GATES - 1
        ),
        "plant_action_history_exact": report.get(
            "executed_action_max_error", math.inf
        ) <= base.MAX_EXECUTED_ACTION_ERROR,
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
    base.STEP_LIMIT, base.MINIMUM_RECORDS = STEP_LIMIT, MINIMUM_RECORDS
    base.INTERVENTION_PHASE_MIN = INTERVENTION_PHASE_MIN
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT = PARENT_REPORT
    base.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.PHASE_INDEX_SCALE = OFFICIAL_PROGRESS_SCALE
    base.FEATURE_QUERY_PHASE_MIN = FEATURE_QUERY_PHASE_MIN
    base.FEATURE_QUERY_PHASE_MAX_EXCLUSIVE = FEATURE_QUERY_PHASE_MAX_EXCLUSIVE
    base.TEACHER_PLANT_ENABLED = True
    base.PLANT_ACTION_CONTRACT = (
        "LC037 recurrent Puffer owns every plant action through held public "
        "phase 3; training-only alignment oracle owns phase 4+ while the "
        "Puffer recurrent state continues on legal observations"
    )
    base.verify_inputs = verify_inputs
    base.source_paths = source_paths
    base.intervention_config = intervention_config
    base.update_held_phase = student_base.update_held_progress
    base._load_actor = load_actor
    base.collection_predicates = predicates


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
            resume: bool = False) -> dict[str, Any]:
    configure()
    return base.collect(output=output, device_name=device_name, resume=resume)


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
