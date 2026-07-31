#!/usr/bin/env python3
"""Collect compact oracle labels on student-owned states at one public phase."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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
    PUBLIC_STATUS_INTERVAL_STEPS,
)
from pufferlib.vq2_recurrent_phase_residual import (
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_lc001_long_course_oracle import load_config
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_vg062_warmed_teacher_intervention_features as base


TAG = "vq2_lc012_index1_student_dagger_features_001"
SCHEMA = "vq2_lc012_index1_student_dagger_report_v1"
STATE_SCHEMA = "vq2_lc012_index1_student_dagger_state_v1"
AGENTS = 256
EPISODES = 256
SEED = 431120
NUM_GATES = 24
THREADS = 32
STEP_LIMIT = 5000
MINIMUM_RECORDS = 100_000
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc010s_phase_independent_early_stop_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "34747327c2f431a6153d200891bad45b8431a5a5d0fe2418108ae822aa716d4d"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "afae8fa795159824bd447472b7eb5047167f19a4848264c95593c08ab43af21f"
)
LC011_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc011_bounded_prefix_screen_001/report.json"
)
LC011_REPORT_SHA256 = (
    "ed82daf9a06babafffc7236820045a924ce3100faba5e948807aed27fdccffd6"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc012_index1_student_dagger_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc012_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
TARGET_PHASE = 1
CHECKPOINT_SCHEMA_EXPECTED = (
    "vq2_lc010s_phase_independent_early_stop_checkpoint_v1"
)
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, CHECKPOINT,
        TRAIN_REPORT, LC011_REPORT, ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc001_long_course_oracle.py",
        ROOT / "scripts/collect_vq2_vg062_warmed_teacher_intervention_features.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
        *EXTRA_SOURCE_PATHS,
    )


def verify_inputs() -> None:
    expected = {
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        LC011_REPORT: LC011_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC012 bound input changed: {path}")
    train = json.loads(TRAIN_REPORT.read_text())
    screen = json.loads(LC011_REPORT.read_text())
    if (
        not train.get("numerically_admitted")
        or train.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or not screen.get("diagnostic_valid")
        or screen.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or screen.get("components", {}).get("24", {}).get("mean_gates_passed")
        != 1.09375
        or screen.get("components", {}).get("24", {}).get("miss_rate")
        != 0.9375
        or screen.get("safety", {}).get("flight_sim_packets_sent") != 0
        or screen.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC011 does not authorize the LC012 DAgger corpus")


def source_identity() -> tuple[str, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return commit, {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths()
    }


def dagger_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
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
    })
    return config, overrides


def update_held_progress(
    raw_progress: np.ndarray, held_progress: np.ndarray, *, step: int,
) -> tuple[np.ndarray, bool, float]:
    raw = np.asarray(raw_progress, dtype=np.float32)
    held = np.asarray(held_progress, dtype=np.float32).copy()
    if raw.shape != held.shape or not np.isfinite(raw).all() or raw.min(
        initial=0.0
    ) < -1e-7:
        raise RuntimeError("LC012 native progress is invalid")
    scaled = raw.astype(np.float64) * OFFICIAL_PROGRESS_SCALE
    error = float(np.max(np.abs(scaled - np.rint(scaled)), initial=0.0))
    if error > 1e-6:
        raise RuntimeError("LC012 native progress encoding changed")
    sampled = step % PUBLIC_STATUS_INTERVAL_STEPS == 0
    if sampled:
        held[:] = raw
    return held, sampled, error


def load_actor(
    device: torch.device,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any]]:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA_EXPECTED
        or not payload.get("numerically_admitted")
        or contract.get("class") != "VQ2UnboundedProgressMLPResidualActor"
    ):
        raise RuntimeError("student-state collector Puffer contract changed")
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
    return {
        "exact_episode_count": metrics.get("env/n") == float(EPISODES),
        "fixed_24_gate_course": metrics.get("env/gate_count24_episode") == 1.0,
        "target_phase_reached": metrics.get(
            f"env/ordered_gate{TARGET_PHASE - 1}_sampled", 0.0
        ) >= 0.90,
        "zero_hard_native_fault": not any(
            metrics.get(f"env/{name}", math.inf) != 0.0
            for name in required_zero
        ),
        "minimum_feature_records": report.get("feature_records", 0) >= MINIMUM_RECORDS,
        "records_match_queries": (
            report.get("feature_records") == report.get("teacher_query_actions_recorded")
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
            <= base.MAX_EXECUTED_ACTION_ERROR
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
    base.TAG = TAG
    base.SCHEMA = SCHEMA
    base.STATE_SCHEMA = STATE_SCHEMA
    base.AGENTS = AGENTS
    base.EPISODES = EPISODES
    base.SEED = SEED
    base.STEP_LIMIT = STEP_LIMIT
    base.INTERVENTION_PHASE_MIN = TARGET_PHASE
    base.MINIMUM_RECORDS = MINIMUM_RECORDS
    base.ENGINE_GATE_CAP = LONG_COURSE_GATE_CAP
    base.PHASE_INDEX_SCALE = OFFICIAL_PROGRESS_SCALE
    base.PARENT_CHECKPOINT = CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.FEATURE_QUERY_PHASE_MIN = TARGET_PHASE
    base.FEATURE_QUERY_PHASE_MAX_EXCLUSIVE = TARGET_PHASE + 1
    base.TEACHER_PLANT_ENABLED = False
    base.PLANT_ACTION_CONTRACT = (
        "The deterministic recurrent Puffer owns every plant action; the "
        f"offline oracle labels only held public-index-{TARGET_PHASE} states"
    )
    base.verify_inputs = verify_inputs
    base.source_identity = source_identity
    base.intervention_config = dagger_config
    base.update_held_phase = update_held_progress
    base._load_actor = load_actor
    base.collection_predicates = predicates


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
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
