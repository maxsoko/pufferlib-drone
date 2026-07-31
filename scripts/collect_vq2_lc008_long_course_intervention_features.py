#!/usr/bin/env python3
"""Collect compact Gate-1-warmed teacher features across a 24-gate course."""

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


TAG = "vq2_lc008_long_course_intervention_features_001"
SCHEMA = "vq2_lc008_long_course_intervention_report_v1"
STATE_SCHEMA = "vq2_lc008_long_course_intervention_state_v1"
AGENTS = 128
EPISODES = 128
SEED = 431080
NUM_GATES = 24
THREADS = 32
STEP_LIMIT = 38400
INTERVENTION_PHASE_MIN = 1
MINIMUM_SUCCESS_RATE = 0.95
MINIMUM_RECORDS = 2_000_000
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg071_nonlinear_residual_count5_multi_offset_001/policy_selected.pt"
)
CHECKPOINT_SHA256 = (
    "60677e385cefb6d6af5c957071cb846de8396d8872880a90832d987b7494f010"
)
LC007_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc007_converted_vg071_long_course_001/report.json"
)
LC007_REPORT_SHA256 = (
    "0f55347e761631d02bb3f7fb670316fd4e3253146b2e4e3bda3997b64cb79a0b"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc008_long_course_intervention_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc008_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, CHECKPOINT,
        LC007_REPORT, ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc001_long_course_oracle.py",
        ROOT / "scripts/collect_vq2_vg062_warmed_teacher_intervention_features.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
    )


def verify_inputs() -> None:
    if sha256_path(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("LC008 VG071 checkpoint hash changed")
    if sha256_path(LC007_REPORT) != LC007_REPORT_SHA256:
        raise RuntimeError("LC008 LC007 report hash changed")
    report = json.loads(LC007_REPORT.read_text())
    if (
        report.get("schema")
        != "vq2_lc007_converted_vg071_long_course_report_v1"
        or not report.get("completed")
        or not report.get("diagnostic_valid")
        or report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or any(
            report.get("components", {}).get(str(count), {}).get("success_rate")
            != 0.0
            for count in (20, 24)
        )
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC007 does not authorize the LC008 offline corpus")
    if not PREREGISTRATION.is_file() or not RUNNER.is_file():
        raise RuntimeError("LC008 source-lock surface is incomplete")


def source_identity() -> tuple[str, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return commit, {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths()
    }


def intervention_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = load_config(
        pufferl_module,
        num_gates=NUM_GATES,
        agents=AGENTS,
        episodes=EPISODES,
        seed=SEED,
        threads=THREADS,
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
    raw_progress: np.ndarray,
    held_progress: np.ndarray,
    *,
    step: int,
) -> tuple[np.ndarray, bool, float]:
    raw = np.asarray(raw_progress, dtype=np.float32)
    held = np.asarray(held_progress, dtype=np.float32).copy()
    if raw.shape != held.shape:
        raise ValueError("LC008 progress batches do not align")
    if not np.isfinite(raw).all() or raw.min(initial=0.0) < -1e-7:
        raise RuntimeError("LC008 native progress is invalid")
    scaled = raw.astype(np.float64) * OFFICIAL_PROGRESS_SCALE
    encoding_error = float(
        np.max(np.abs(scaled - np.rint(scaled)), initial=0.0)
    )
    if encoding_error > 1e-6:
        raise RuntimeError("LC008 progress is not an integer index divided by 6")
    sampled = step % PUBLIC_STATUS_INTERVAL_STEPS == 0
    if sampled:
        held[:] = raw
    return held, sampled, encoding_error


def load_actor(
    device: torch.device,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any]]:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != "vq2_vg071_paired_synthetic_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or contract.get("class") != "VQ2IndexedPhaseMLPResidualActor"
        or contract.get("hidden_size") != 256
        or contract.get("residual_size") != 64
    ):
        raise RuntimeError("LC008 parent contract changed")
    actor = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_converted_state(payload["model_state"])
    actor.eval()
    return actor, payload


def collection_predicates(report: dict[str, Any]) -> dict[str, bool]:
    metrics = report.get("metrics", {})
    phases = report.get("feature_phase_records", [])
    required_zero = (
        "out_of_order", "action_envelope_violation",
        "wire_rate_envelope_violation", "thrust_envelope_violation",
    )
    return {
        "exact_episode_count": metrics.get("env/n") == float(EPISODES),
        "fixed_24_gate_course": (
            metrics.get("env/gate_count24_episode") == 1.0
        ),
        "minimum_success_rate": (
            metrics.get("env/success_rate", 0.0) >= MINIMUM_SUCCESS_RATE
        ),
        "zero_crash": metrics.get("env/crash", math.inf) == 0.0,
        "zero_hard_native_fault": not any(
            metrics.get(f"env/{name}", math.inf) != 0.0
            for name in required_zero
        ),
        "gate_1_student_warmup_reliable": (
            metrics.get("env/ordered_gate0_sampled", 0.0) >= 0.95
        ),
        "minimum_feature_records": (
            report.get("feature_records", 0) >= MINIMUM_RECORDS
        ),
        "feature_records_match_teacher_actions": (
            report.get("feature_records")
            == report.get("teacher_plant_actions_executed")
        ),
        "all_later_public_heads_observed": (
            len(phases) == LONG_COURSE_GATE_CAP + 1
            and all(phases[index] > 0 for index in range(1, NUM_GATES))
        ),
        "teacher_begins_only_at_public_index_1": (
            report.get("minimum_teacher_phase_index") == INTERVENTION_PHASE_MIN
        ),
        "teacher_stops_before_terminal_index": (
            report.get("maximum_teacher_phase_index") == NUM_GATES - 1
        ),
        "plant_action_history_exact": (
            report.get("executed_action_max_error", math.inf)
            <= base.MAX_EXECUTED_ACTION_ERROR
        ),
        "phase_changes_only_on_public_ticks": (
            report.get("phase_changes_off_tick") == 0
        ),
        "phase_never_decreases": report.get("phase_decreases") == 0,
        "phase_never_skips": report.get("phase_skips") == 0,
        "phase_encoding_exact": (
            report.get("raw_phase_encoding_max_error", math.inf) <= 1e-6
        ),
        "finite_actions_and_features": report.get("nonfinite_values") == 0,
        "teacher_actions_in_envelope": (
            report.get("teacher_action_envelope_violations") == 0
        ),
        "feature_file_hash_present": len(report.get("feature_sha256", "")) == 64,
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
    base.INTERVENTION_PHASE_MIN = INTERVENTION_PHASE_MIN
    base.MINIMUM_SUCCESS_RATE = MINIMUM_SUCCESS_RATE
    base.MINIMUM_RECORDS = MINIMUM_RECORDS
    base.ENGINE_GATE_CAP = LONG_COURSE_GATE_CAP
    base.PHASE_INDEX_SCALE = OFFICIAL_PROGRESS_SCALE
    base.PARENT_CHECKPOINT = CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.PLANT_ACTION_CONTRACT = (
        "converted VG071 deterministic mean through held public index 0; "
        "training-only alignment oracle query at held public indices 1+"
    )
    base.verify_inputs = verify_inputs
    base.source_identity = source_identity
    base.intervention_config = intervention_config
    base.update_held_phase = update_held_progress
    base._load_actor = load_actor
    base.collection_predicates = collection_predicates


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
