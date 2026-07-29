#!/usr/bin/env python3
"""Run the N252 policy against live v3391 telemetry without sending MAVLink.

This is a deployment-contract shadow only. It receives public race/local-pose
telemetry, reconstructs the exact native interface-3 observation, advances the
recurrent checkpoint at 60 Hz, and records the velocity command it would have
sent. It never sends heartbeat, reset, arm, setpoint, or disarm messages.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from collect_vq1_v3391_telemetry_teacher import build_observation, load_course
from drone_sitl_adapter import MavlinkSitlAdapter
from policy_callable_checkpoint import CheckpointPolicy


Vector3 = tuple[float, float, float]
EXPECTED_CHECKPOINT_SHA256 = (
    "cec95192a04ebbe0c28f61226fcd76a6851404b6548be12ca6c44165397e9d76"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def elapsed_from_race_status(status) -> float:
    if status is None:
        return 0.0
    start_ms = int(status.race_start_boot_time_ms)
    boot_ms = int(status.sim_boot_time_ms)
    if start_ms < 0 or boot_ms < start_ms:
        return 0.0
    return max(0.0, (boot_ms - start_ms) * 1e-3)


def live_observation(
    position_ned_m: Vector3,
    velocity_ned_m_s: Vector3,
    gate_index: int,
    elapsed_time_s: float,
    last_action: np.ndarray,
    native_gates: np.ndarray,
    settings: dict[str, float],
) -> np.ndarray:
    # The native course is exactly (-NED x, -NED y, -NED z).
    native_position = -np.asarray(position_ned_m, dtype=np.float64)
    native_velocity = -np.asarray(velocity_ned_m_s, dtype=np.float64)
    return build_observation(
        native_position,
        native_velocity,
        gate_index,
        elapsed_time_s,
        last_action,
        native_gates,
        settings,
    )


def action_to_ned_velocity(
    action: np.ndarray, settings: dict[str, float]
) -> np.ndarray:
    # Native desired velocity is [a0*forward, a1*lateral, -a2*vertical].
    # Negating native XYZ returns MAV_FRAME_LOCAL_NED velocity.
    return np.asarray(
        [
            -action[0] * settings["max_cmd_forward"],
            -action[1] * settings["max_cmd_lateral"],
            action[2] * settings["max_cmd_vertical"],
        ],
        dtype=np.float32,
    )


def compare_transferred_gates(state, native_gates: np.ndarray) -> dict[str, object]:
    transferred = tuple(sorted(state.track_gates, key=lambda gate: int(gate.gate_id)))
    if len(transferred) != len(native_gates):
        return {
            "available_in_shadow_window": False,
            "count": len(transferred),
            "max_position_error_m": None,
        }
    live_ned = np.asarray(
        [
            [gate.position_ned_x, gate.position_ned_y, gate.position_ned_z]
            for gate in transferred
        ],
        dtype=np.float64,
    )
    # Track transfer publishes the gate bottom/base. The deployable config stores
    # N257-proven aperture centers, one half-height NED-up from those bases.
    expected_ned = -native_gates
    expected_ned[:, 2] += np.asarray(
        [float(gate.height_m) * 0.5 for gate in transferred], dtype=np.float64
    )
    return {
        "available_in_shadow_window": True,
        "count": len(transferred),
        "comparison_kind": "transferred_base_vs_config_aperture_center_plus_half_height",
        "max_position_error_m": float(
            np.max(np.linalg.norm(live_ned - expected_ned, axis=1))
        ),
    }


def reset_ready_state(state, maximum_start_position_norm_m: float) -> bool:
    status = state.race_status
    if (
        status is None
        or state.base_mode is None
        or state.system_status is None
        or state.local_position_ned_m is None
        or state.local_velocity_ned_m_s is None
    ):
        return False
    return bool(
        int(state.base_mode) & 128
        and int(state.system_status) == 4
        and int(status.race_start_boot_time_ms) >= 0
        and int(status.active_gate_index) == 0
        and int(status.race_finish_time_ns) < 0
        and float(np.linalg.norm(state.local_position_ned_m))
        <= maximum_start_position_norm_m
    )


def run_shadow(args: argparse.Namespace) -> dict[str, object]:
    if args.duration_s <= 0.0 or args.inference_hz <= 0.0:
        raise ValueError("duration and inference rate must be positive")
    checkpoint_sha256 = _sha256(args.checkpoint)
    if args.expected_checkpoint_sha256 and (
        checkpoint_sha256 != args.expected_checkpoint_sha256
    ):
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"expected {args.expected_checkpoint_sha256}, got {checkpoint_sha256}"
        )

    native_gates, settings = load_course(args.config)
    if len(native_gates) != 6:
        raise RuntimeError("the v3391 shadow requires exactly six configured gates")
    policy = CheckpointPolicy.load(
        str(args.checkpoint),
        input_dim=32,
        hidden_dim=128,
        num_layers=2,
        num_actions=4,
        layout_precision_bytes=4,
    )
    policy.reset_state()
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.dropout_after_s)

    last_action = np.zeros(4, dtype=np.float32)
    trace: list[dict[str, object]] = []
    inference_steps = 0
    nonfinite_observations = 0
    nonfinite_actions = 0
    maximum_absolute_action = 0.0
    preflight_started_s = time.monotonic()
    preflight_deadline_s = preflight_started_s + args.preflight_timeout_s
    while time.monotonic() < preflight_deadline_s:
        adapter.drain_telemetry(timeout_s=0.01)
        if reset_ready_state(
            adapter.telemetry.state, args.maximum_start_position_norm_m
        ):
            break
    else:
        adapter.close()
        raise RuntimeError(
            "command-free shadow did not observe a fresh armed index-0 race near start"
        )
    ready_state = adapter.telemetry.state
    ready_status = ready_state.race_status
    preflight_report = {
        "elapsed_s": time.monotonic() - preflight_started_s,
        "base_mode": int(ready_state.base_mode),
        "system_status": int(ready_state.system_status),
        "sim_boot_time_ms": int(ready_status.sim_boot_time_ms),
        "race_start_boot_time_ms": int(ready_status.race_start_boot_time_ms),
        "active_gate_index": int(ready_status.active_gate_index),
        "race_finish_time_ns": int(ready_status.race_finish_time_ns),
        "position_ned_m": [float(value) for value in ready_state.local_position_ned_m],
        "velocity_ned_m_s": [
            float(value) for value in ready_state.local_velocity_ned_m_s
        ],
    }
    policy.reset_state()
    started_s = time.monotonic()
    deadline_s = started_s + args.duration_s
    next_inference_s = started_s
    next_trace_s = started_s
    period_s = 1.0 / args.inference_hz
    trace_period_s = 1.0 / args.trace_hz
    try:
        while time.monotonic() < deadline_s:
            adapter.drain_telemetry(timeout_s=min(0.005, period_s / 2.0))
            now_s = time.monotonic()
            state = adapter.telemetry.state
            status = state.race_status
            if (
                now_s < next_inference_s
                or status is None
                or state.local_position_ned_m is None
                or state.local_velocity_ned_m_s is None
            ):
                continue

            gate_index = min(max(int(status.active_gate_index), 0), 5)
            observation = live_observation(
                tuple(float(value) for value in state.local_position_ned_m),
                tuple(float(value) for value in state.local_velocity_ned_m_s),
                gate_index,
                elapsed_from_race_status(status),
                last_action,
                native_gates,
                settings,
            )
            if not np.isfinite(observation).all():
                nonfinite_observations += 1
                next_inference_s += period_s
                continue
            action = np.asarray(policy.infer(observation), dtype=np.float32)
            if not np.isfinite(action).all():
                nonfinite_actions += 1
                next_inference_s += period_s
                continue
            last_action = np.clip(action, -1.0, 1.0)
            commanded_velocity_ned = action_to_ned_velocity(last_action, settings)
            maximum_absolute_action = max(
                maximum_absolute_action, float(np.max(np.abs(last_action)))
            )
            inference_steps += 1
            if now_s >= next_trace_s:
                trace.append(
                    {
                        "elapsed_wall_s": round(now_s - started_s, 6),
                        "sim_boot_time_ms": int(status.sim_boot_time_ms),
                        "race_start_boot_time_ms": int(status.race_start_boot_time_ms),
                        "active_gate_index": int(status.active_gate_index),
                        "position_ned_m": [
                            float(value) for value in state.local_position_ned_m
                        ],
                        "velocity_ned_m_s": [
                            float(value) for value in state.local_velocity_ned_m_s
                        ],
                        "policy_action": [float(value) for value in last_action],
                        "shadow_command_velocity_ned_m_s": [
                            float(value) for value in commanded_velocity_ned
                        ],
                    }
                )
                next_trace_s += trace_period_s
            next_inference_s += period_s
            if next_inference_s <= now_s:
                next_inference_s = now_s + period_s
    finally:
        adapter.close()

    elapsed_s = time.monotonic() - started_s
    metrics = adapter.telemetry.metrics
    final_state = adapter.telemetry.state
    minimum_steps = math.floor(args.duration_s * args.inference_hz * 0.90)
    accepted = bool(
        inference_steps >= minimum_steps
        and metrics.local_positions > 0
        and metrics.race_statuses > 0
        and metrics.telemetry_dropouts == 0
        and metrics.drain_limit_hits == 0
        and nonfinite_observations == 0
        and nonfinite_actions == 0
        and maximum_absolute_action <= 1.0
    )
    return {
        "accepted": accepted,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "config": str(args.config),
        "preflight": preflight_report,
        "contract": {
            "kind": "command_free_v3391_telemetry_policy_shadow",
            "heartbeats_sent": 0,
            "reset_commands_sent": 0,
            "arm_commands_sent": 0,
            "setpoints_sent": 0,
            "disarm_commands_sent": 0,
            "inference_hz_requested": args.inference_hz,
            "inference_steps": inference_steps,
            "minimum_inference_steps": minimum_steps,
            "effective_inference_hz": inference_steps / max(elapsed_s, 1e-9),
            "nonfinite_observations": nonfinite_observations,
            "nonfinite_actions": nonfinite_actions,
            "maximum_absolute_action": maximum_absolute_action,
        },
        "elapsed_wall_s": elapsed_s,
        "track_transfer_comparison": compare_transferred_gates(final_state, native_gates),
        "telemetry": dataclasses.asdict(metrics),
        "latest_telemetry": dataclasses.asdict(final_state),
        "trace": trace,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--preflight-timeout-s", type=float, default=30.0)
    parser.add_argument("--maximum-start-position-norm-m", type=float, default=5.0)
    parser.add_argument("--inference-hz", type=float, default=60.0)
    parser.add_argument("--trace-hz", type=float, default=5.0)
    parser.add_argument("--dropout-after-s", type=float, default=2.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "checkpoints"
        / "drone_race_vq1_v3391_telemetry"
        / "n252_bc_6gate.bin",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "drone_race_vq1_v3391_telemetry.ini",
    )
    parser.add_argument("--json-path", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.trace_hz <= 0.0
        or args.preflight_timeout_s <= 0.0
        or args.maximum_start_position_norm_m <= 0.0
    ):
        parser.error("trace rate, preflight timeout, and start bound must be positive")

    report = run_shadow(args)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
