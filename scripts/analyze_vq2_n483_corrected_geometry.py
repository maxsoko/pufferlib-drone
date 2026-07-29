#!/usr/bin/env python3
"""Source-lock N483's rejection and the public-bearing Gate-2 correction.

This command-free analysis replays the recorded whole-Puffer actions and the
legal stationary-gate predictor.  It uses only the archived camera-derived gate
pose, vehicle quaternion, previous action, IMU-derived observation, and official
race status.  No simulator socket or native coordinate interface is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np


OBSERVATIONS = 32
ACTIONS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rotate_vector_by_quaternion(
    vector: Sequence[float], quaternion: Sequence[float]
) -> tuple[float, float, float]:
    vx, vy, vz = (float(value) for value in vector)
    qw, qx, qy, qz = (float(value) for value in quaternion)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


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


def gate_pose(sample: dict) -> tuple[np.ndarray, float]:
    pose = sample.get("gate_pose")
    if not isinstance(pose, dict):
        raise ValueError("sample has no camera-derived gate pose")
    vector = np.asarray(pose.get("body_vector_ned_m"), dtype=np.float64)
    yaw = float(pose.get("yaw_error_rad", math.nan))
    if vector.shape != (3,) or not np.isfinite(vector).all() or not math.isfinite(yaw):
        raise ValueError("sample camera-derived gate pose is invalid")
    return vector, yaw


def find_new_gate_association(samples: list[dict], phase: int = 1) -> int:
    """Find the first large stale-to-new aperture bearing transition."""
    start = first_phase(samples, phase)
    previous_vector: np.ndarray | None = None
    previous_yaw: float | None = None
    for index in range(start, len(samples)):
        try:
            vector, yaw = gate_pose(samples[index])
        except ValueError:
            continue
        if previous_vector is not None and np.array_equal(vector, previous_vector):
            continue
        if (
            previous_vector is not None
            and previous_yaw is not None
            and abs(previous_yaw) >= 0.30
            and abs(yaw) <= 0.10
            and float(np.linalg.norm(vector - previous_vector)) >= 5.0
        ):
            return index
        previous_vector = vector
        previous_yaw = yaw
    raise ValueError("trace contains no unambiguous new-gate association")


def corrected_initial_gate_position_from_bearing(
    *,
    old_initial_world: Sequence[float],
    old_current_world: Sequence[float],
    old_current_body_ned: Sequence[float],
    raw_current_body_ned: Sequence[float],
    quaternion_wxyz: Sequence[float],
) -> dict[str, np.ndarray | float]:
    """Preserve predicted range and replace only public bearing/elevation."""
    old_initial = np.asarray(old_initial_world, dtype=np.float64)
    old_current = np.asarray(old_current_world, dtype=np.float64)
    old_body = np.asarray(old_current_body_ned, dtype=np.float64)
    raw_body = np.asarray(raw_current_body_ned, dtype=np.float64)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if old_initial.shape != (3,) or old_current.shape != (3,):
        raise ValueError("world vectors must contain three values")
    if old_body.shape != (3,) or raw_body.shape != (3,):
        raise ValueError("body-NED vectors must contain three values")
    if quaternion.shape != (4,):
        raise ValueError("quaternion must contain four values")
    if not all(np.isfinite(value).all() for value in (
        old_initial, old_current, old_body, raw_body, quaternion
    )):
        raise ValueError("geometry inputs must be finite")
    if old_body[0] <= 1e-3 or raw_body[0] <= 1e-3:
        raise ValueError("both gate vectors must point forward")

    range_scale = float(old_body[0] / raw_body[0])
    corrected_body_ned = raw_body * range_scale
    corrected_current_world = np.asarray(
        rotate_vector_by_quaternion(
            (
                corrected_body_ned[0],
                corrected_body_ned[1],
                -corrected_body_ned[2],
            ),
            quaternion,
        ),
        dtype=np.float64,
    )
    predictor_displacement = old_current - old_initial
    corrected_initial_world = corrected_current_world - predictor_displacement
    return {
        "range_scale": range_scale,
        "corrected_body_ned": corrected_body_ned,
        "corrected_current_world": corrected_current_world,
        "predictor_displacement_world": predictor_displacement,
        "corrected_initial_world": corrected_initial_world,
    }


def configure_composite(
    *,
    prefix_checkpoint: Path,
    gate2_checkpoint: Path,
    fixed_prefix_report: Path,
    replay: dict,
) -> None:
    os.environ["PUFFER_POLICY_PREFIX_CHECKPOINT_PATH"] = str(
        prefix_checkpoint.resolve())
    os.environ["PUFFER_POLICY_GATE2_CHECKPOINT_PATH"] = str(
        gate2_checkpoint.resolve())
    os.environ["PUFFER_POLICY_GATE2_FIXED_PREFIX_REPORT_PATH"] = str(
        fixed_prefix_report.resolve())
    os.environ["PUFFER_POLICY_INPUT_DIM"] = "32"
    os.environ["PUFFER_POLICY_LAYOUT_PRECISION_BYTES"] = "4"
    os.environ["PUFFER_POLICY_NATIVE_BF16"] = "0"
    os.environ["PUFFER_POLICY_RACE_PHASE_DENOMINATOR"] = "6"
    os.environ["PUFFER_POLICY_RUNTIME_DURATION_SECONDS"] = str(
        replay["runtime_duration_seconds"])
    os.environ["PUFFER_POLICY_GATE2_TIME_LIMIT_SECONDS"] = str(
        replay["gate2_time_limit_seconds"])
    os.environ["PUFFER_POLICY_GATE2_WORLD_FRAME_FEATURES"] = str(
        int(replay["world_frame_gate_features"]))
    os.environ["PUFFER_POLICY_GATE2_LINEAR_GATE_FEATURES"] = str(
        int(replay["linear_gate_features"]))
    os.environ["PUFFER_POLICY_GATE2_PREDICT_GATE_KINEMATICS"] = str(
        int(replay["predict_gate_kinematics"]))
    for axis, value in zip(("X", "Y", "Z"), replay["initial_gate_rate_world"]):
        name = (
            "PUFFER_POLICY_GATE2_INITIAL_FORWARD_GATE_RATE_M_S"
            if axis == "X"
            else f"PUFFER_POLICY_GATE2_INITIAL_GATE_RATE_{axis}_M_S"
        )
        os.environ[name] = str(value)
    for axis, value in zip(
        ("X", "Y", "Z"), replay["initial_gate_position_world"]
    ):
        os.environ[f"PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_{axis}"] = str(
            value)
    os.environ["PUFFER_POLICY_GATE2_ASSOCIATION_JUMP_THRESHOLD_M"] = str(
        replay["association_jump_threshold_m"])
    os.environ["PUFFER_POLICY_GATE2_RESEED_BEARING_ON_ASSOCIATION"] = "0"
    os.environ["PUFFER_POLICY_GATE2_FORWARD_PREDICTOR_DT_SECONDS"] = str(
        replay["predictor_dt_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix_checkpoint", type=Path)
    parser.add_argument("gate2_checkpoint", type=Path)
    parser.add_argument("n483_report", type=Path)
    parser.add_argument("n483_passive", type=Path)
    parser.add_argument("n484_replay", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--registered-corrected-initial-world",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    args = parser.parse_args()

    n483 = load_json(args.n483_report)
    passive = load_json(args.n483_passive)
    replay = load_json(args.n484_replay)
    fixed_prefix_report = Path(replay["official_report"])
    samples = policy_samples(n483)
    transition_index = first_phase(samples, 1)
    association_index = find_new_gate_association(samples, 1)

    configure_composite(
        prefix_checkpoint=args.prefix_checkpoint,
        gate2_checkpoint=args.gate2_checkpoint,
        fixed_prefix_report=fixed_prefix_report,
        replay=replay,
    )
    try:
        import policy_callable_vq2_gate2_composite as composite
    except ModuleNotFoundError:
        from scripts import policy_callable_vq2_gate2_composite as composite

    composite._STATE = None
    composite._KEY = None
    composite.reset()
    replayed_actions: list[list[float]] = []
    old_current_world = None
    old_current_body_ned = None
    for index, sample in enumerate(samples):
        observation = sample["observation"]
        replayed_actions.append(composite.infer(observation))
        if index == association_index:
            state = composite._STATE
            assert state is not None and state.gate_position_world is not None
            old_current_world = np.asarray(state.gate_position_world, dtype=np.float64)
            quaternion = np.asarray(observation[6:10], dtype=np.float64)
            conjugate = quaternion.copy()
            conjugate[1:4] *= -1.0
            body_up = rotate_vector_by_quaternion(old_current_world, conjugate)
            old_current_body_ned = np.asarray(
                (body_up[0], body_up[1], -body_up[2]), dtype=np.float64)

    assert old_current_world is not None and old_current_body_ned is not None
    recorded_actions = np.asarray(
        [sample["normalized_action"] for sample in samples], dtype=np.float32)
    replayed = np.asarray(replayed_actions, dtype=np.float32)
    action_errors = np.abs(replayed - recorded_actions)
    phases = np.asarray(
        [sample["official_active_gate_index"] for sample in samples], dtype=np.int32)

    association_sample = samples[association_index]
    raw_body_ned, association_yaw = gate_pose(association_sample)
    stale_body_ned, stale_yaw = gate_pose(samples[association_index - 1])
    quaternion = np.asarray(association_sample["observation"][6:10], dtype=np.float64)
    old_initial_world = np.asarray(
        replay["initial_gate_position_world"], dtype=np.float64)
    correction = corrected_initial_gate_position_from_bearing(
        old_initial_world=old_initial_world,
        old_current_world=old_current_world,
        old_current_body_ned=old_current_body_ned,
        raw_current_body_ned=raw_body_ned,
        quaternion_wxyz=quaternion,
    )
    derived_initial = np.asarray(
        correction["corrected_initial_world"], dtype=np.float64)
    registered_initial = np.asarray(
        args.registered_corrected_initial_world, dtype=np.float64)
    registration_error = np.abs(derived_initial - registered_initial)

    sitl = n483.get("sitl", {})
    latest = sitl.get("latest_telemetry", {})
    latest_race = latest.get("race_status", {})
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
    phase_counts = {
        str(int(phase)): int(np.count_nonzero(phases == phase))
        for phase in np.unique(phases)
    }
    passed = bool(
        action_errors.max() <= 1e-6
        and transition_index == 189
        and association_index == 241
        and int(n483.get("official_active_gate_index", -1)) == 1
        and int(n483.get("official_race_finish_time_ns", 0)) == -1
        and bool(n483.get("crash_detected"))
        and bool(n483.get("invalid_run"))
        and int(latest.get("collision_id", -1)) == 1002
        and int(sitl.get("command_rate_violations", -1)) == 0
        and 50.0 <= float(sitl.get("effective_command_hz", 0.0)) < 100.0
        and abs(stale_yaw) >= 0.30
        and abs(association_yaw) <= 0.10
        and float(registration_error.max()) <= 1e-3
        and passive_zero_command
        and int(passive_state.get("base_mode", -1)) == 65
        and int(passive_state.get("system_status", -1)) == 3
    )

    report = {
        "passed": passed,
        "contract": {
            "command_free": True,
            "runtime_action_source": "whole recurrent Puffer checkpoint output",
            "classical_runtime_action": False,
            "public_inputs_only": True,
            "geometry_claim": (
                "source-calibrated stationary-gate estimate from rounded public "
                "camera bearing plus legal predictor range"
            ),
        },
        "artifacts": {
            "prefix_checkpoint": artifact(args.prefix_checkpoint),
            "gate2_checkpoint": artifact(args.gate2_checkpoint),
            "fixed_prefix_report": artifact(fixed_prefix_report),
            "n483_report": artifact(args.n483_report),
            "n483_passive": artifact(args.n483_passive),
            "n484_replay": artifact(args.n484_replay),
            "composite_callable": artifact(Path(composite.__file__).resolve()),
        },
        "n483": {
            "samples": len(samples),
            "phase_counts": phase_counts,
            "transition_sample_index": transition_index,
            "transition_elapsed_s": float(samples[transition_index]["elapsed_s"]),
            "official_gate1_time_s": (
                float(n483["official_last_gate_race_time"]) / 1e9
            ),
            "official_active_gate_index": int(n483["official_active_gate_index"]),
            "official_finish_time_ns": int(n483["official_race_finish_time_ns"]),
            "crash_detected": bool(n483["crash_detected"]),
            "invalid_run": bool(n483["invalid_run"]),
            "collision_id": int(latest.get("collision_id", -1)),
            "collision_impact": float(latest.get("collision_impact", 0.0)),
            "commands_sent": int(sitl.get("commands_sent", 0)),
            "effective_command_hz": float(sitl.get("effective_command_hz", 0.0)),
            "command_rate_violations": int(
                sitl.get("command_rate_violations", -1)),
            "full_replay_max_action_error": float(action_errors.max()),
            "phase0_replay_max_action_error": float(
                action_errors[phases == 0].max()),
            "phase1_replay_max_action_error": float(
                action_errors[phases == 1].max()),
            "race_start_boot_time_ms": int(
                latest_race.get("race_start_boot_time_ms", -1)),
            "sim_boot_time_ms": int(latest_race.get("sim_boot_time_ms", -1)),
        },
        "new_gate_association": {
            "sample_index": association_index,
            "elapsed_s": float(association_sample["elapsed_s"]),
            "stale_body_vector_ned_m": stale_body_ned.tolist(),
            "stale_yaw_error_rad": stale_yaw,
            "new_body_vector_ned_m": raw_body_ned.tolist(),
            "new_yaw_error_rad": association_yaw,
            "pose_jump_m": float(np.linalg.norm(raw_body_ned - stale_body_ned)),
        },
        "geometry_correction": {
            "old_initial_world_m": old_initial_world.tolist(),
            "old_current_world_m": old_current_world.tolist(),
            "old_current_body_ned_m": old_current_body_ned.tolist(),
            "range_scale": float(correction["range_scale"]),
            "corrected_current_body_ned_m": np.asarray(
                correction["corrected_body_ned"]).tolist(),
            "corrected_current_world_m": np.asarray(
                correction["corrected_current_world"]).tolist(),
            "predictor_displacement_world_m": np.asarray(
                correction["predictor_displacement_world"]).tolist(),
            "derived_corrected_initial_world_m": derived_initial.tolist(),
            "registered_corrected_initial_world_m": registered_initial.tolist(),
            "registration_abs_error_m": registration_error.tolist(),
            "registration_max_abs_error_m": float(registration_error.max()),
        },
        "passive_proof": {
            "zero_command": passive_zero_command,
            "contract": passive_contract,
            "base_mode": int(passive_state.get("base_mode", -1)),
            "system_status": int(passive_state.get("system_status", -1)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
