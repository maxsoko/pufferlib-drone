#!/usr/bin/env python3
"""Source-lock the N399 VQ2 Gate-2 failure and its native-screen mismatch.

This is command-free analysis.  It consumes only recorded public observations,
recorded Puffer actions, official race status, and already-produced native
screening artifacts.  It neither opens a simulator socket nor emits control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


OBSERVATIONS = 32
ACTIONS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_gate_vector(observation: np.ndarray) -> np.ndarray:
    if observation.shape != (OBSERVATIONS,):
        raise ValueError("policy observation must contain 32 values")
    clipped = np.clip(observation[11:14], -0.999999, 0.999999)
    return np.arctanh(clipped) * np.asarray((10.0, 5.0, 5.0))


def policy_samples(report: dict) -> list[dict]:
    samples = report.get("policy_trace", {}).get("samples", [])
    if not samples:
        raise ValueError("report contains no policy_trace.samples")
    for index, sample in enumerate(samples):
        observation = np.asarray(sample.get("observation"), dtype=np.float32)
        action = np.asarray(sample.get("normalized_action"), dtype=np.float32)
        if observation.shape != (OBSERVATIONS,):
            raise ValueError(f"sample {index} has an invalid observation")
        if action.shape != (ACTIONS,):
            raise ValueError(f"sample {index} has an invalid action")
    return samples


def first_phase(samples: list[dict], phase: int) -> int:
    for index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) == phase:
            return index
    raise ValueError(f"trace never reaches official phase {phase}")


def raw_pose(sample: dict) -> np.ndarray:
    pose = sample.get("gate_pose")
    if not isinstance(pose, dict):
        raise ValueError("transition sample has no raw gate pose")
    result = np.asarray(pose.get("body_vector_ned_m"), dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("transition raw gate pose is invalid")
    return result


def unique_pose_jumps(samples: list[dict]) -> list[dict]:
    records: list[dict] = []
    previous_pose: np.ndarray | None = None
    previous_time: float | None = None
    for sample in samples:
        try:
            pose = raw_pose(sample)
        except ValueError:
            continue
        now = float(sample["elapsed_s"])
        if previous_pose is not None and np.array_equal(pose, previous_pose):
            continue
        if previous_pose is not None and previous_time is not None:
            dt = now - previous_time
            jump = float(np.linalg.norm(pose - previous_pose))
            records.append(
                {
                    "elapsed_s": now,
                    "dt_s": dt,
                    "jump_m": jump,
                    "apparent_speed_m_s": jump / dt if dt > 0.0 else None,
                    "from_body_vector_ned_m": previous_pose.tolist(),
                    "to_body_vector_ned_m": pose.tolist(),
                }
            )
        previous_pose = pose
        previous_time = now
    return records


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix_checkpoint", type=Path)
    parser.add_argument("gate2_checkpoint", type=Path)
    parser.add_argument("n295_report", type=Path)
    parser.add_argument("n399_report", type=Path)
    parser.add_argument("n399_passive", type=Path)
    parser.add_argument("native_report", type=Path)
    parser.add_argument("native_trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    os.environ["PUFFER_POLICY_PREFIX_CHECKPOINT_PATH"] = str(
        args.prefix_checkpoint.resolve())
    os.environ["PUFFER_POLICY_GATE2_CHECKPOINT_PATH"] = str(
        args.gate2_checkpoint.resolve())
    os.environ["PUFFER_POLICY_INPUT_DIM"] = "32"
    os.environ["PUFFER_POLICY_LAYOUT_PRECISION_BYTES"] = "4"
    os.environ["PUFFER_POLICY_NATIVE_BF16"] = "0"
    os.environ["PUFFER_POLICY_RACE_PHASE_DENOMINATOR"] = "6"

    # Keep the reusable trace helpers importable without policy-path setup.
    # The command-line analysis itself is normally launched with scripts/ on
    # PYTHONPATH, matching every other deployment replay utility.
    try:
        import policy_callable_vq2_gate2_composite as composite
    except ModuleNotFoundError:
        from scripts import policy_callable_vq2_gate2_composite as composite

    n295 = load_json(args.n295_report)
    n399 = load_json(args.n399_report)
    passive = load_json(args.n399_passive)
    native = load_json(args.native_report)
    samples = policy_samples(n399)
    transition_index = first_phase(samples, 1)
    if transition_index == 0:
        raise ValueError("N399 has no official Gate-1 prefix")

    observations = np.asarray(
        [sample["observation"] for sample in samples], dtype=np.float32)
    recorded_actions = np.asarray(
        [sample["normalized_action"] for sample in samples], dtype=np.float32)
    phases = np.asarray(
        [sample["official_active_gate_index"] for sample in samples], dtype=np.int32)
    composite.reset()
    replayed_actions = np.asarray(
        [composite.infer(row) for row in observations], dtype=np.float32)
    errors = np.abs(replayed_actions - recorded_actions)

    before = samples[transition_index - 1]
    after = samples[transition_index]
    before_raw = raw_pose(before)
    after_raw = raw_pose(after)
    before_bearing = math.atan2(before_raw[1], max(before_raw[0], 1e-6))
    after_bearing = math.atan2(after_raw[1], max(after_raw[0], 1e-6))
    transition_dt = float(after["elapsed_s"]) - float(before["elapsed_s"])
    transition_jump = float(np.linalg.norm(after_raw - before_raw))

    native_trace = np.load(args.native_trace)
    native_observations = np.asarray(native_trace["observations"], dtype=np.float32)
    native_actions = np.asarray(native_trace["actions"], dtype=np.float32)
    if native_observations.ndim != 2 or native_observations.shape[1] != OBSERVATIONS:
        raise ValueError("native trace observation shape is invalid")
    if native_actions.ndim != 2 or native_actions.shape[1] != ACTIONS:
        raise ValueError("native trace action shape is invalid")

    live_transition_observation = observations[transition_index]
    native_transition_observation = native_observations[0]
    live_gate_vector = decode_gate_vector(live_transition_observation)
    native_gate_vector = decode_gate_vector(native_transition_observation)
    phase_counts = {
        str(int(phase)): int(np.count_nonzero(phases == phase))
        for phase in np.unique(phases)
    }
    pose_jumps = unique_pose_jumps(samples[transition_index - 1 :])
    largest_jump = max(
        pose_jumps,
        key=lambda item: float(item.get("apparent_speed_m_s") or -1.0),
    )

    passive_contract = passive.get("contract", {})
    passive_state = passive.get("latest_state", {})
    passive_zero_command = all(
        int(passive_contract.get(field, -1)) == 0
        for field in (
            "heartbeats_sent",
            "timesync_replies_sent",
            "metadata_requests_sent",
            "reset_commands_sent",
            "arm_commands_sent",
            "disarm_commands_sent",
            "setpoints_sent",
        )
    )

    sitl = n399.get("sitl", {})
    race = sitl.get("latest_telemetry", {}).get("race_status", {})
    collision_race_elapsed_s = None
    if race:
        start_ms = int(race.get("race_start_boot_time_ms", -1))
        boot_ms = int(race.get("sim_boot_time_ms", -1))
        if start_ms >= 0 and boot_ms >= start_ms:
            collision_race_elapsed_s = (boot_ms - start_ms) / 1000.0

    native_success = float(native.get("metrics", {}).get("env/success_rate", 0.0))
    native_time_limit = None
    overrides = native.get("metadata", {}).get("config_overrides", [])
    for index, value in enumerate(overrides[:-1]):
        if value == "--env.time-limit-seconds":
            native_time_limit = float(overrides[index + 1])
            break
    n295_samples = policy_samples(n295)
    n295_last_elapsed_fraction = float(n295_samples[-1]["observation"][18])

    passed = bool(
        errors.max() <= 1e-6
        and int(n399.get("official_active_gate_index", -1)) == 1
        and int(n399.get("official_race_finish_time_ns", 0)) == -1
        and bool(n399.get("crash_detected"))
        and int(sitl.get("command_rate_violations", -1)) == 0
        and 50.0 <= float(sitl.get("effective_command_hz", 0.0)) < 100.0
        and passive_zero_command
        and int(passive_state.get("base_mode", -1)) == 65
        and int(passive_state.get("system_status", -1)) == 3
        and native_success == 1.0
        and transition_jump > 3.0
        and abs(before_bearing - after_bearing) < 0.05
    )

    result = {
        "passed": passed,
        "contract": {
            "command_free": True,
            "runtime_action_source": "whole recurrent Puffer checkpoint output",
            "classical_runtime_action": False,
            "native_coordinates_deployed": False,
        },
        "artifacts": {
            "prefix_checkpoint": artifact(args.prefix_checkpoint),
            "gate2_checkpoint": artifact(args.gate2_checkpoint),
            "n295_report": artifact(args.n295_report),
            "n399_report": artifact(args.n399_report),
            "n399_passive": artifact(args.n399_passive),
            "native_report": artifact(args.native_report),
            "native_trace": artifact(args.native_trace),
            "composite_callable": artifact(Path(composite.__file__).resolve()),
        },
        "n399": {
            "phase_counts": phase_counts,
            "samples": len(samples),
            "transition_sample_index": transition_index,
            "transition_elapsed_s": float(after["elapsed_s"]),
            "official_gate1_time_s": float(n399["official_last_gate_race_time"]) / 1e9,
            "official_active_gate_index": int(n399["official_active_gate_index"]),
            "official_finish_time_ns": int(n399["official_race_finish_time_ns"]),
            "collision_id": int(
                sitl["latest_telemetry"].get("collision_id", -1)),
            "collision_impact": float(
                sitl["latest_telemetry"].get("collision_impact", 0.0)),
            "collision_race_elapsed_s": collision_race_elapsed_s,
            "commands_sent": int(sitl.get("commands_sent", 0)),
            "effective_command_hz": float(sitl.get("effective_command_hz", 0.0)),
            "command_rate_violations": int(
                sitl.get("command_rate_violations", -1)),
            "composite_replay_max_action_error": float(errors.max()),
            "phase0_replay_max_action_error": float(errors[phases == 0].max()),
            "phase1_replay_max_action_error": float(errors[phases == 1].max()),
        },
        "transition_observation_mismatch": {
            "previous_raw_body_vector_ned_m": before_raw.tolist(),
            "first_gate2_raw_body_vector_ned_m": after_raw.tolist(),
            "raw_pose_dt_s": transition_dt,
            "raw_pose_jump_m": transition_jump,
            "raw_pose_apparent_speed_m_s": transition_jump / transition_dt,
            "previous_bearing_rad": before_bearing,
            "first_gate2_bearing_rad": after_bearing,
            "bearing_change_rad": after_bearing - before_bearing,
            "live_gate_vector_from_observation_m": live_gate_vector.tolist(),
            "native_gate_vector_from_observation_m": native_gate_vector.tolist(),
            "live_elapsed_fraction": float(live_transition_observation[18]),
            "native_elapsed_fraction": float(native_transition_observation[18]),
            "n295_last_elapsed_fraction": n295_last_elapsed_fraction,
            "live_recorded_action": recorded_actions[transition_index].tolist(),
            "native_action": native_actions[0].tolist(),
            "largest_unique_raw_pose_jump": largest_jump,
            "unique_raw_pose_jumps": len(pose_jumps),
        },
        "native_screen": {
            "success_rate": native_success,
            "terminal_crossing_radial_m": float(
                native.get("metrics", {}).get("env/terminal_crossing_radial", 0.0)),
            "steps": int(native.get("metadata", {}).get("steps", 0)),
            "time_limit_seconds": native_time_limit,
            "prefix_report_duration_seconds": 12.0,
            "n399_deployment_duration_seconds": 14.0,
        },
        "passive_disarm": {
            "zero_command_contract": passive_zero_command,
            "base_mode": int(passive_state.get("base_mode", -1)),
            "system_status": int(passive_state.get("system_status", -1)),
            "race_status": passive_state.get("race_status"),
        },
        "findings": {
            "policy_execution_mismatch": False,
            "command_transport_failure": False,
            "native_screen_predicted_live_result": False,
            "legal_observation_contract_mismatch": True,
            "range_alias_at_official_transition": True,
            "elapsed_fraction_schedule_mismatch": True,
            "n399_rejected_without_unchanged_retry": True,
            "next_work": (
                "train and screen a whole-output recurrent Puffer policy using the "
                "N399 Gate-1 prefix, a continuous deployment-matched elapsed clock, "
                "and camera-range/association perturbations; no live command is authorized"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
