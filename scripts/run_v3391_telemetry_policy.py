#!/usr/bin/env python3
"""Run a learned telemetry policy in VQ1 v3391 with strict official safety gates."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np

from drone_sitl_adapter import AttitudeSetpoint, LocalNedSetpoint, MavlinkSitlAdapter
from policy_callable_checkpoint import CheckpointPolicy
from run_v3391_telemetry_waypoint_lap import (
    _norm,
    _reset_and_wait_for_start,
    _sub,
    gate_scheduled_speed,
    is_valid_finish,
    segment_velocity_command,
)
from shadow_v3391_telemetry_policy import (
    _sha256,
    action_to_ned_velocity,
    compare_transferred_gates,
    elapsed_from_race_status,
    live_observation,
    load_course,
)


Vector3 = tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class GovernedVelocity:
    velocity_ned_m_s: Vector3
    along_to_gate_m: float
    cross_track_error_m: float
    policy_along_speed_m_s: float
    scheduled_along_speed_m_s: float
    correction_ned_m_s: Vector3


@dataclasses.dataclass(frozen=True)
class PlaneMissGuardDecision:
    crossing_boot_time_ms: int | None
    should_abort: bool
    deferred_for_status_ack: bool
    reason: str


def gate_governed_velocity(
    position_ned_m: Vector3,
    segment_start_ned_m: Vector3,
    gate_center_ned_m: Vector3,
    policy_velocity_ned_m_s: Vector3,
    *,
    crossing_speed_m_s: float,
    maximum_along_speed_m_s: float,
    slowdown_distance_m: float,
    cross_track_gain_s_inv: float,
    maximum_cross_track_correction_m_s: float,
) -> GovernedVelocity:
    """Retain learned along-track speed while centering the public gate approach."""
    if maximum_along_speed_m_s < crossing_speed_m_s:
        raise ValueError("maximum along speed must be at least crossing speed")
    segment = _sub(gate_center_ned_m, segment_start_ned_m)
    segment_length = _norm(segment)
    if segment_length <= 1e-6:
        raise ValueError("segment start and gate center must be distinct")
    tangent = tuple(value / segment_length for value in segment)
    policy_along_speed_m_s = sum(
        policy_velocity_ned_m_s[index] * tangent[index] for index in range(3)
    )
    learned_cruise_speed_m_s = min(
        maximum_along_speed_m_s,
        max(crossing_speed_m_s, policy_along_speed_m_s),
    )
    initial = segment_velocity_command(
        position_ned_m,
        segment_start_ned_m,
        gate_center_ned_m,
        speed_m_s=learned_cruise_speed_m_s,
        cross_track_gain_s_inv=cross_track_gain_s_inv,
        max_cross_track_correction_m_s=maximum_cross_track_correction_m_s,
    )
    scheduled_speed_m_s = gate_scheduled_speed(
        initial.along_to_gate_m,
        cruise_speed_m_s=learned_cruise_speed_m_s,
        crossing_speed_m_s=crossing_speed_m_s,
        slowdown_distance_m=slowdown_distance_m,
    )
    command = segment_velocity_command(
        position_ned_m,
        segment_start_ned_m,
        gate_center_ned_m,
        speed_m_s=scheduled_speed_m_s,
        cross_track_gain_s_inv=cross_track_gain_s_inv,
        max_cross_track_correction_m_s=maximum_cross_track_correction_m_s,
    )
    return GovernedVelocity(
        velocity_ned_m_s=command.velocity_ned_m_s,
        along_to_gate_m=command.along_to_gate_m,
        cross_track_error_m=command.cross_track_error_m,
        policy_along_speed_m_s=policy_along_speed_m_s,
        scheduled_along_speed_m_s=scheduled_speed_m_s,
        correction_ned_m_s=command.correction_m_s,
    )


def plane_miss_guard_decision(
    *,
    along_to_gate_m: float,
    abort_distance_m: float,
    crossing_boot_time_ms: int | None,
    current_boot_time_ms: int | None,
    race_status_boot_time_ms: int | None,
    status_ack_grace_s: float,
) -> PlaneMissGuardDecision:
    """Fail closed after the official status stream can acknowledge a crossing."""
    if abort_distance_m <= 0.0:
        raise ValueError("abort_distance_m must be positive")
    if status_ack_grace_s < 0.0:
        raise ValueError("status_ack_grace_s must be nonnegative")
    crossing = crossing_boot_time_ms
    if crossing is None and along_to_gate_m <= 0.0 and current_boot_time_ms is not None:
        crossing = int(current_boot_time_ms)
    if along_to_gate_m >= -abort_distance_m:
        return PlaneMissGuardDecision(crossing, False, False, "before_abort_distance")
    if (
        status_ack_grace_s <= 0.0
        or crossing is None
        or current_boot_time_ms is None
        or race_status_boot_time_ms is None
    ):
        return PlaneMissGuardDecision(crossing, True, False, "legacy_distance_abort")
    if int(race_status_boot_time_ms) > crossing:
        return PlaneMissGuardDecision(
            crossing, True, False, "post_crossing_status_still_same_gate"
        )
    elapsed_since_crossing_ms = int(current_boot_time_ms) - crossing
    if elapsed_since_crossing_ms >= round(status_ack_grace_s * 1000.0):
        return PlaneMissGuardDecision(crossing, True, False, "status_ack_timeout")
    return PlaneMissGuardDecision(crossing, False, True, "awaiting_status_ack")


def run_policy(args: argparse.Namespace) -> dict[str, object]:
    if args.duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if not 2.0 <= args.heartbeat_hz:
        raise ValueError("heartbeat_hz must be at least two")
    if not 50.0 <= args.command_hz < 100.0:
        raise ValueError("command_hz must be in [50, 100)")
    if not 1 <= args.stop_after_gate_index <= 6:
        raise ValueError("stop_after_gate_index must be between one and six")
    if args.gate_governor and args.governor_plane_miss_abort_m <= 0.0:
        raise ValueError("governor_plane_miss_abort_m must be positive")
    if args.governor_plane_miss_status_ack_grace_s < 0.0:
        raise ValueError("governor_plane_miss_status_ack_grace_s must be nonnegative")

    checkpoint_sha256 = _sha256(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    native_gates, settings = load_course(args.config)
    if len(native_gates) != 6:
        raise RuntimeError("policy config must contain exactly six aperture centers")
    policy = CheckpointPolicy.load(
        str(args.checkpoint),
        input_dim=32,
        hidden_dim=128,
        num_layers=2,
        num_actions=4,
        layout_precision_bytes=4,
    )
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.dropout_after_s)
    exit_disarm_sent = False
    control_commands_sent = 0
    heartbeats_sent = 0
    collision = None
    invalid_reason = None
    launch_support_contacts: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []
    reset_report = None
    started_s = None
    initial_position = None
    try:
        reset_report, initial_position = _reset_and_wait_for_start(
            adapter, timeout_s=args.reset_timeout_s, required_gate_count=6
        )
        gate_comparison = compare_transferred_gates(adapter.telemetry.state, native_gates)
        if (
            not gate_comparison["available_in_shadow_window"]
            or float(gate_comparison["max_position_error_m"]) > args.max_gate_transfer_error_m
        ):
            raise RuntimeError(f"aperture/base gate transform mismatch: {gate_comparison}")

        policy.reset_state()
        last_action = np.zeros(4, dtype=np.float32)
        started_s = time.monotonic()
        deadline_s = started_s + args.duration_s
        next_command_s = started_s
        next_heartbeat_s = started_s
        next_trace_s = started_s
        command_period_s = 1.0 / args.command_hz
        heartbeat_period_s = 1.0 / args.heartbeat_hz
        trace_period_s = 1.0 / args.trace_hz
        observed_collisions = int(adapter.telemetry.metrics.collisions)
        last_gate_index = 0
        gate_plane_crossing_boot_ms = None
        gate_plane_crossings: list[dict[str, object]] = []
        plane_miss_ack_deferrals = 0
        maximum_race_status_lag_ms = 0
        plane_miss_abort_diagnostic = None
        ned_gate_centers = tuple(
            tuple(float(value) for value in -native_gate) for native_gate in native_gates
        )

        while time.monotonic() < deadline_s:
            adapter.drain_telemetry(timeout_s=0.005)
            now_s = time.monotonic()
            state = adapter.telemetry.state
            status = state.race_status
            if status is None or state.local_position_ned_m is None:
                continue
            gate_index = int(status.active_gate_index)
            if gate_index != last_gate_index:
                transitions.append(
                    {
                        "elapsed_s": round(now_s - started_s, 6),
                        "from_gate_index": last_gate_index,
                        "to_gate_index": gate_index,
                        "position_ned_m": [float(v) for v in state.local_position_ned_m],
                        "last_gate_race_time": int(status.last_gate_race_time),
                    }
                )
                last_gate_index = gate_index
                gate_plane_crossing_boot_ms = None

            collision_messages = int(adapter.telemetry.metrics.collisions)
            if collision_messages > observed_collisions:
                displacement_m = _norm(
                    _sub(tuple(float(v) for v in state.local_position_ned_m), initial_position)
                )
                event = {
                    "id": int(state.collision_id),
                    "threat_level": int(state.collision_threat_level or 0),
                    "impact": float(state.collision_impact or 0.0),
                    "elapsed_s": round(now_s - started_s, 6),
                    "displacement_from_start_m": displacement_m,
                    "message_count_delta": collision_messages - observed_collisions,
                }
                if (
                    now_s - started_s <= args.launch_contact_grace_s
                    and displacement_m <= args.launch_contact_radius_m
                    and event["id"] == 1002
                    and event["threat_level"] <= 1
                ):
                    launch_support_contacts.append(event)
                else:
                    collision = event
                    break
                observed_collisions = collision_messages

            if args.stop_after_gate_index < 6 and gate_index >= args.stop_after_gate_index:
                break
            if is_valid_finish(status, 6):
                break
            if now_s - started_s > 1.0 and (
                state.base_mode is None
                or not (int(state.base_mode) & 128)
                or int(state.system_status or -1) != 4
            ):
                invalid_reason = "race_became_unarmed_or_inactive"
                break

            if now_s >= next_heartbeat_s:
                adapter.send_heartbeat()
                heartbeats_sent += 1
                next_heartbeat_s += heartbeat_period_s
            if now_s < next_command_s:
                continue

            launch_active = now_s - started_s < args.attitude_launch_duration_s
            governor_command = None
            if launch_active:
                adapter.send_attitude_setpoint(
                    AttitudeSetpoint(thrust=args.attitude_launch_thrust), mode="body_rates"
                )
                command_velocity = None
            else:
                if state.local_velocity_ned_m_s is None:
                    continue
                observation = live_observation(
                    tuple(float(v) for v in state.local_position_ned_m),
                    tuple(float(v) for v in state.local_velocity_ned_m_s),
                    min(max(gate_index, 0), 5),
                    elapsed_from_race_status(status),
                    last_action,
                    native_gates,
                    settings,
                )
                action = np.asarray(policy.infer(observation), dtype=np.float32)
                if not np.isfinite(observation).all() or not np.isfinite(action).all():
                    invalid_reason = "nonfinite_policy_value"
                    break
                last_action = np.clip(action, -1.0, 1.0)
                command_velocity = action_to_ned_velocity(last_action, settings)
                if args.gate_governor:
                    active_segment = min(max(gate_index, 0), 5)
                    segment_start_ned_m = (
                        initial_position
                        if active_segment == 0
                        else ned_gate_centers[active_segment - 1]
                    )
                    governor_command = gate_governed_velocity(
                        tuple(float(v) for v in state.local_position_ned_m),
                        segment_start_ned_m,
                        ned_gate_centers[active_segment],
                        tuple(float(v) for v in command_velocity),
                        crossing_speed_m_s=args.governor_crossing_speed_m_s,
                        maximum_along_speed_m_s=args.governor_maximum_along_speed_m_s,
                        slowdown_distance_m=args.governor_slowdown_distance_m,
                        cross_track_gain_s_inv=args.governor_cross_track_gain_s_inv,
                        maximum_cross_track_correction_m_s=(
                            args.governor_maximum_cross_track_correction_m_s
                        ),
                    )
                    current_boot_time_ms = (
                        None
                        if state.attitude_time_boot_ms is None
                        else int(state.attitude_time_boot_ms)
                    )
                    race_status_boot_time_ms = int(status.sim_boot_time_ms)
                    if current_boot_time_ms is not None:
                        maximum_race_status_lag_ms = max(
                            maximum_race_status_lag_ms,
                            max(0, current_boot_time_ms - race_status_boot_time_ms),
                        )
                    previous_crossing_boot_ms = gate_plane_crossing_boot_ms
                    miss_decision = plane_miss_guard_decision(
                        along_to_gate_m=governor_command.along_to_gate_m,
                        abort_distance_m=args.governor_plane_miss_abort_m,
                        crossing_boot_time_ms=gate_plane_crossing_boot_ms,
                        current_boot_time_ms=current_boot_time_ms,
                        race_status_boot_time_ms=race_status_boot_time_ms,
                        status_ack_grace_s=(
                            args.governor_plane_miss_status_ack_grace_s
                        ),
                    )
                    gate_plane_crossing_boot_ms = miss_decision.crossing_boot_time_ms
                    if (
                        previous_crossing_boot_ms is None
                        and gate_plane_crossing_boot_ms is not None
                    ):
                        gate_plane_crossings.append(
                            {
                                "gate_index": gate_index,
                                "crossing_boot_time_ms": gate_plane_crossing_boot_ms,
                                "race_status_boot_time_ms": race_status_boot_time_ms,
                                "along_to_gate_m": governor_command.along_to_gate_m,
                            }
                        )
                    if miss_decision.deferred_for_status_ack:
                        plane_miss_ack_deferrals += 1
                    if miss_decision.should_abort:
                        invalid_reason = "gate_governor_plane_miss"
                        plane_miss_abort_diagnostic = {
                            **dataclasses.asdict(miss_decision),
                            "gate_index": gate_index,
                            "along_to_gate_m": governor_command.along_to_gate_m,
                            "current_boot_time_ms": current_boot_time_ms,
                            "race_status_boot_time_ms": race_status_boot_time_ms,
                        }
                        break
                    command_velocity = np.asarray(
                        governor_command.velocity_ned_m_s, dtype=np.float32
                    )
                adapter.send_local_ned_setpoint(
                    LocalNedSetpoint(
                        vx=float(command_velocity[0]),
                        vy=float(command_velocity[1]),
                        vz=float(command_velocity[2]),
                    ),
                    frame="local_ned",
                    yaw_mode="ignore",
                )
            control_commands_sent += 1
            next_command_s += command_period_s
            if next_command_s <= now_s:
                next_command_s = now_s + command_period_s

            if now_s >= next_trace_s:
                trace.append(
                    {
                        "elapsed_s": round(now_s - started_s, 6),
                        "active_gate_index": gate_index,
                        "position_ned_m": [float(v) for v in state.local_position_ned_m],
                        "velocity_ned_m_s": (
                            None
                            if state.local_velocity_ned_m_s is None
                            else [float(v) for v in state.local_velocity_ned_m_s]
                        ),
                        "policy_action": [float(v) for v in last_action],
                        "command_velocity_ned_m_s": (
                            None if command_velocity is None else [float(v) for v in command_velocity]
                        ),
                        "phase": "attitude_launch" if launch_active else "policy_velocity",
                        "governor": (
                            None
                            if governor_command is None
                            else dataclasses.asdict(governor_command)
                        ),
                    }
                )
                next_trace_s += trace_period_s

        adapter.drain_telemetry(timeout_s=0.05)
        final_state = adapter.telemetry.state
        final_status = final_state.race_status
        elapsed_s = time.monotonic() - started_s
        bounded_progress = bool(
            final_status is not None
            and int(final_status.active_gate_index) >= args.stop_after_gate_index
        )
        accepted = bool(
            collision is None
            and invalid_reason is None
            and (
                is_valid_finish(final_status, 6)
                if args.stop_after_gate_index == 6
                else bounded_progress
            )
        )
        adapter.send_local_ned_setpoint(LocalNedSetpoint(), frame="local_ned", yaw_mode="ignore")
        adapter.send_disarm_command()
        exit_disarm_sent = True
        return {
            "accepted": accepted,
            "acceptance_kind": (
                "official_six_gate_finish"
                if args.stop_after_gate_index == 6
                else "bounded_official_gate_progress"
            ),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "config": str(args.config),
            "reset": reset_report,
            "gate_transfer_comparison": gate_comparison,
            "contract": {
                "stop_after_gate_index": args.stop_after_gate_index,
                "command_hz": args.command_hz,
                "heartbeat_hz": args.heartbeat_hz,
                "control_commands_sent": control_commands_sent,
                "heartbeats_sent": heartbeats_sent,
                "effective_command_hz": control_commands_sent / max(elapsed_s, 1e-9),
                "exit_stop_setpoints_sent": 1,
                "exit_disarm_sent": exit_disarm_sent,
                "camera_used": False,
                "hidden_simulator_state_used": False,
                "gate_governor_enabled": args.gate_governor,
                "gate_governor_source": (
                    "public_track_transfer_and_local_position"
                    if args.gate_governor
                    else None
                ),
                "governor_plane_miss_abort_m": args.governor_plane_miss_abort_m,
                "governor_plane_miss_status_ack_grace_s": (
                    args.governor_plane_miss_status_ack_grace_s
                ),
                "plane_miss_ack_deferrals": plane_miss_ack_deferrals,
                "maximum_race_status_lag_ms": maximum_race_status_lag_ms,
            },
            "elapsed_wall_s": round(elapsed_s, 6),
            "official_active_gate_index": (
                None if final_status is None else int(final_status.active_gate_index)
            ),
            "official_race_finish_time_ns": (
                None if final_status is None else int(final_status.race_finish_time_ns)
            ),
            "collision": collision,
            "invalid_reason": invalid_reason,
            "launch_support_contacts": launch_support_contacts,
            "gate_transitions": transitions,
            "gate_plane_crossings": gate_plane_crossings,
            "plane_miss_abort_diagnostic": plane_miss_abort_diagnostic,
            "trace": trace,
            "telemetry": dataclasses.asdict(adapter.telemetry.metrics),
            "latest_telemetry": dataclasses.asdict(final_state),
        }
    finally:
        if not exit_disarm_sent:
            try:
                adapter.send_local_ned_setpoint(LocalNedSetpoint(), frame="local_ned", yaw_mode="ignore")
                adapter.send_disarm_command()
            except Exception:
                pass
        adapter.close()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument(
        "--config", type=Path, default=root / "config" / "drone_race_vq1_v3391_telemetry.ini"
    )
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--reset-timeout-s", type=float, default=12.0)
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--command-hz", type=float, default=60.0)
    parser.add_argument("--trace-hz", type=float, default=10.0)
    parser.add_argument("--dropout-after-s", type=float, default=2.0)
    parser.add_argument("--attitude-launch-duration-s", type=float, default=0.5)
    parser.add_argument("--attitude-launch-thrust", type=float, default=0.27)
    parser.add_argument("--launch-contact-grace-s", type=float, default=1.0)
    parser.add_argument("--launch-contact-radius-m", type=float, default=1.0)
    parser.add_argument("--max-gate-transfer-error-m", type=float, default=0.01)
    parser.add_argument("--stop-after-gate-index", type=int, default=1)
    parser.add_argument("--gate-governor", action="store_true")
    parser.add_argument("--governor-crossing-speed-m-s", type=float, default=4.0)
    parser.add_argument("--governor-maximum-along-speed-m-s", type=float, default=8.0)
    parser.add_argument("--governor-slowdown-distance-m", type=float, default=10.0)
    parser.add_argument("--governor-cross-track-gain-s-inv", type=float, default=1.5)
    parser.add_argument(
        "--governor-maximum-cross-track-correction-m-s", type=float, default=4.0
    )
    parser.add_argument("--governor-plane-miss-abort-m", type=float, default=1.5)
    parser.add_argument(
        "--governor-plane-miss-status-ack-grace-s", type=float, default=0.0
    )
    parser.add_argument("--json-path", type=Path, required=True)
    args = parser.parse_args()
    report = run_policy(args)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
