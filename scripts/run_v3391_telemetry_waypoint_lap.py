#!/usr/bin/env python3
"""Run a reset-isolated VQ1 v3391 lap from published pose and gate telemetry.

The telemetry-enabled legacy VQ1 build publishes LOCAL_POSITION_NED and the
six ordered gate centers during its command-31000 reset transfer.  This runner
uses those public feeds directly: each segment commands a constant along-track
velocity plus bounded cross-track correction through
SET_POSITION_TARGET_LOCAL_NED.  It does not use camera perception or hidden
simulator state.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

from drone_sitl_adapter import (
    AttitudeSetpoint,
    LocalNedSetpoint,
    MavlinkSitlAdapter,
    RaceStatus,
)


Vector3 = tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class SegmentCommand:
    velocity_ned_m_s: Vector3
    along_to_gate_m: float
    cross_track_error_m: float
    correction_m_s: Vector3


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vector3, value: float) -> Vector3:
    return (a[0] * value, a[1] * value, a[2] * value)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vector3) -> float:
    return math.sqrt(_dot(a, a))


def _bounded_norm(a: Vector3, maximum: float) -> Vector3:
    magnitude = _norm(a)
    if magnitude <= maximum or magnitude <= 1e-12:
        return a
    return _scale(a, maximum / magnitude)


def segment_velocity_command(
    position_ned_m: Vector3,
    segment_start_ned_m: Vector3,
    gate_center_ned_m: Vector3,
    *,
    speed_m_s: float,
    cross_track_gain_s_inv: float,
    max_cross_track_correction_m_s: float,
) -> SegmentCommand:
    if speed_m_s <= 0.0:
        raise ValueError("speed_m_s must be positive")
    if cross_track_gain_s_inv < 0.0:
        raise ValueError("cross_track_gain_s_inv must be non-negative")
    if max_cross_track_correction_m_s < 0.0:
        raise ValueError("max_cross_track_correction_m_s must be non-negative")

    segment = _sub(gate_center_ned_m, segment_start_ned_m)
    segment_length = _norm(segment)
    if segment_length <= 1e-6:
        raise ValueError("segment start and gate center must be distinct")
    tangent = _scale(segment, 1.0 / segment_length)

    to_gate = _sub(gate_center_ned_m, position_ned_m)
    along_to_gate_m = _dot(to_gate, tangent)
    cross_track = _sub(to_gate, _scale(tangent, along_to_gate_m))
    correction = _bounded_norm(
        _scale(cross_track, cross_track_gain_s_inv),
        max_cross_track_correction_m_s,
    )
    velocity = _add(_scale(tangent, speed_m_s), correction)
    return SegmentCommand(
        velocity_ned_m_s=velocity,
        along_to_gate_m=along_to_gate_m,
        cross_track_error_m=_norm(cross_track),
        correction_m_s=correction,
    )


def gate_scheduled_speed(
    along_to_gate_m: float,
    *,
    cruise_speed_m_s: float,
    crossing_speed_m_s: float,
    slowdown_distance_m: float,
) -> float:
    if cruise_speed_m_s <= 0.0:
        raise ValueError("cruise_speed_m_s must be positive")
    if not 0.0 < crossing_speed_m_s <= cruise_speed_m_s:
        raise ValueError("crossing_speed_m_s must be positive and no greater than cruise")
    if slowdown_distance_m <= 0.0:
        raise ValueError("slowdown_distance_m must be positive")
    alpha = max(0.0, min(1.0, along_to_gate_m / slowdown_distance_m))
    smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return crossing_speed_m_s + (cruise_speed_m_s - crossing_speed_m_s) * smooth_alpha


def is_valid_finish(status: RaceStatus | None, gate_count: int) -> bool:
    return bool(
        status is not None
        and int(status.active_gate_index) >= int(gate_count)
        and int(status.race_finish_time_ns) >= 0
    )


def is_centered_low_threat_gate_contact(
    collision_event: dict,
    position_ned_m: Vector3,
    gate_center_ned_m: Vector3,
    *,
    enabled: bool,
    radius_m: float,
) -> bool:
    """Admit only the bounded low-threat warning observed before a clean plane cross."""
    return bool(
        enabled
        and collision_event["id"] == 1001
        and collision_event["threat_level"] <= 1
        and _norm(_sub(position_ned_m, gate_center_ned_m)) <= radius_m
    )


def _gate_position(gate) -> Vector3:
    return (
        float(gate.position_ned_x),
        float(gate.position_ned_y),
        float(gate.position_ned_z),
    )


def offset_gate_position(gate_position_ned_m: Vector3, ned_z_offset_m: float) -> Vector3:
    return (
        gate_position_ned_m[0],
        gate_position_ned_m[1],
        gate_position_ned_m[2] + float(ned_z_offset_m),
    )


def _wait_for_preflight(adapter: MavlinkSitlAdapter, timeout_s: float) -> RaceStatus:
    deadline_s = time.monotonic() + timeout_s
    next_heartbeat_s = 0.0
    while time.monotonic() < deadline_s:
        now_s = time.monotonic()
        if now_s >= next_heartbeat_s:
            adapter.send_heartbeat()
            next_heartbeat_s = now_s + 0.5
        adapter.drain_telemetry(timeout_s=0.05)
        state = adapter.telemetry.state
        if (
            state.race_status is not None
            and state.local_position_ned_m is not None
            and state.local_velocity_ned_m_s is not None
            and state.base_mode is not None
            and state.system_status is not None
        ):
            return state.race_status
    raise RuntimeError("waypoint preflight did not receive VQ1 race and pose telemetry")


def _reset_and_wait_for_start(
    adapter: MavlinkSitlAdapter,
    *,
    timeout_s: float,
    required_gate_count: int,
) -> tuple[dict, Vector3]:
    pre_status = _wait_for_preflight(adapter, min(timeout_s, 3.0))
    pre_boot_ms = int(pre_status.sim_boot_time_ms)
    pre_base_mode = adapter.telemetry.state.base_mode
    pre_system_status = adapter.telemetry.state.system_status

    adapter.send_disarm_command()
    time.sleep(0.1)
    adapter.send_sim_reset_command()
    reset_command_s = time.monotonic()

    deadline_s = reset_command_s + timeout_s
    next_heartbeat_s = reset_command_s
    next_arm_s = reset_command_s
    reset_seen_s = None
    arm_commands_sent = 0
    initial_position = None
    scheduled_start_s = None
    accepted_status = None
    while time.monotonic() < deadline_s:
        now_s = time.monotonic()
        if now_s >= next_heartbeat_s:
            adapter.send_heartbeat()
            next_heartbeat_s = now_s + 0.5
        adapter.drain_telemetry(timeout_s=0.02)
        state = adapter.telemetry.state
        status = state.race_status
        if status is None:
            continue
        if reset_seen_s is None and int(status.sim_boot_time_ms) + 1000 < pre_boot_ms:
            reset_seen_s = now_s
        reset_ready = (
            reset_seen_s is not None
            and int(status.race_start_boot_time_ms) >= 0
            and int(status.active_gate_index) == 0
            and int(status.race_finish_time_ns) < 0
            and len(state.track_gates) == required_gate_count
            and state.local_position_ned_m is not None
        )
        if reset_ready and scheduled_start_s is None:
            initial_position = tuple(float(v) for v in state.local_position_ned_m)
            remaining_from_status_s = (
                int(status.race_start_boot_time_ms) - int(status.sim_boot_time_ms)
            ) * 1e-3
            scheduled_start_s = now_s + max(0.0, remaining_from_status_s)
            accepted_status = status
        if scheduled_start_s is None:
            continue

        remaining_s = scheduled_start_s - now_s
        if remaining_s <= 0.25 and now_s >= next_arm_s:
            adapter.send_arm_command()
            arm_commands_sent += 1
            next_arm_s = now_s + 0.1
        if remaining_s <= 0.04:
            return (
                {
                    "pre_reset_sim_boot_time_ms": pre_boot_ms,
                    "pre_reset_base_mode": pre_base_mode,
                    "pre_reset_system_status": pre_system_status,
                    "post_reset_sim_boot_time_ms": int(accepted_status.sim_boot_time_ms),
                    "race_start_boot_time_ms": int(accepted_status.race_start_boot_time_ms),
                    "reset_detection_s": round(reset_seen_s - reset_command_s, 6),
                    "pre_reset_disarm_commands_sent": 1,
                    "reset_commands_sent": 1,
                    "arm_commands_sent": arm_commands_sent,
                    "track_gate_count": len(state.track_gates),
                },
                initial_position,
            )
        time.sleep(0.001)

    if reset_seen_s is None:
        raise RuntimeError("v3391 command 31000 did not produce a boot rollback")
    if len(adapter.telemetry.state.track_gates) != required_gate_count:
        raise RuntimeError(
            "v3391 reset did not publish the required six-gate track transfer"
        )
    raise RuntimeError("v3391 reset countdown did not reach the control boundary")


def run_lap(args: argparse.Namespace) -> dict:
    if args.duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if not 2.0 <= args.heartbeat_hz:
        raise ValueError("heartbeat_hz must be at least 2")
    if not 0.0 < args.command_hz < 100.0:
        raise ValueError("command_hz must be between 0 and 100")
    if args.required_gate_count != 6:
        raise ValueError("the official VQ1 course requires exactly six gates")
    if args.launch_climb_m < 0.0:
        raise ValueError("launch_climb_m must be non-negative")
    if args.launch_speed_m_s <= 0.0:
        raise ValueError("launch_speed_m_s must be positive")
    if args.launch_timeout_s <= 0.0:
        raise ValueError("launch_timeout_s must be positive")
    if args.attitude_launch_duration_s < 0.0:
        raise ValueError("attitude_launch_duration_s must be non-negative")
    if not 0.0 <= args.attitude_launch_thrust <= 1.0:
        raise ValueError("attitude_launch_thrust must be between zero and one")
    if args.launch_contact_grace_s < 0.0:
        raise ValueError("launch_contact_grace_s must be non-negative")
    if args.launch_contact_radius_m < 0.0:
        raise ValueError("launch_contact_radius_m must be non-negative")
    if args.low_threat_gate_contact_radius_m <= 0.0:
        raise ValueError("low_threat_gate_contact_radius_m must be positive")
    if not 1 <= args.stop_after_gate_index <= args.required_gate_count:
        raise ValueError("stop_after_gate_index must be between one and six")
    if abs(args.gate_ned_z_offset_m) > 1.5:
        raise ValueError("gate_ned_z_offset_m magnitude must not exceed 1.5 metres")
    gate_scheduled_speed(
        args.gate_slowdown_distance_m,
        cruise_speed_m_s=args.speed_m_s,
        crossing_speed_m_s=args.gate_crossing_speed_m_s,
        slowdown_distance_m=args.gate_slowdown_distance_m,
    )

    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=2.0)
    commands_sent = 0
    exit_disarm_sent = False
    trajectory: list[dict] = []
    transitions: list[dict] = []
    launch_support_contacts: list[dict] = []
    low_threat_gate_contacts: list[dict] = []
    collision = None
    reset_report = None
    initial_position: Vector3 | None = None
    started_s = None
    last_gate_index = 0
    last_sample_s = 0.0
    try:
        reset_report, initial_position = _reset_and_wait_for_start(
            adapter,
            timeout_s=args.reset_timeout_s,
            required_gate_count=args.required_gate_count,
        )
        gates = tuple(sorted(adapter.telemetry.state.track_gates, key=lambda gate: gate.gate_id))
        if [int(gate.gate_id) for gate in gates] != list(range(args.required_gate_count)):
            raise RuntimeError("v3391 gate transfer did not contain ordered IDs 0 through 5")
        transmitted_gate_positions = tuple(_gate_position(gate) for gate in gates)
        gate_positions = tuple(
            offset_gate_position(position, args.gate_ned_z_offset_m)
            for position in transmitted_gate_positions
        )

        started_s = time.monotonic()
        deadline_s = started_s + args.duration_s
        next_command_s = started_s
        next_heartbeat_s = started_s
        command_period_s = 1.0 / args.command_hz
        heartbeat_period_s = 1.0 / args.heartbeat_hz
        last_gate_index = 0
        observed_collision_messages = int(adapter.telemetry.metrics.collisions)

        while time.monotonic() < deadline_s:
            now_s = time.monotonic()
            adapter.drain_telemetry(timeout_s=0.005)
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

            collision_messages = int(adapter.telemetry.metrics.collisions)
            if collision_messages > observed_collision_messages:
                position_ned_m = tuple(float(v) for v in state.local_position_ned_m)
                displacement_from_start_m = _norm(
                    _sub(
                        position_ned_m,
                        initial_position,
                    )
                )
                collision_event = {
                    "id": int(state.collision_id),
                    "threat_level": int(state.collision_threat_level or 0),
                    "impact": float(state.collision_impact or 0.0),
                    "elapsed_s": round(now_s - started_s, 6),
                    "message_count_delta": collision_messages - observed_collision_messages,
                    "displacement_from_start_m": displacement_from_start_m,
                }
                launch_support_contact = (
                    now_s - started_s <= args.launch_contact_grace_s
                    and displacement_from_start_m <= args.launch_contact_radius_m
                    and collision_event["id"] == 1002
                    and collision_event["threat_level"] <= 1
                )
                if launch_support_contact:
                    launch_support_contacts.append(collision_event)
                elif is_centered_low_threat_gate_contact(
                    collision_event,
                    position_ned_m,
                    gate_positions[min(max(gate_index, 0), args.required_gate_count - 1)],
                    enabled=args.allow_centered_low_threat_gate_contact,
                    radius_m=args.low_threat_gate_contact_radius_m,
                ):
                    collision_event["distance_to_active_gate_center_m"] = _norm(
                        _sub(
                            position_ned_m,
                            gate_positions[
                                min(max(gate_index, 0), args.required_gate_count - 1)
                            ],
                        )
                    )
                    low_threat_gate_contacts.append(collision_event)
                else:
                    collision = collision_event
                    break
                observed_collision_messages = collision_messages
            if (
                args.stop_after_gate_index < args.required_gate_count
                and gate_index >= args.stop_after_gate_index
            ):
                break
            if is_valid_finish(status, args.required_gate_count):
                break

            active_segment = min(max(gate_index, 0), args.required_gate_count - 1)
            segment_start = (
                initial_position if active_segment == 0 else gate_positions[active_segment - 1]
            )
            command = segment_velocity_command(
                tuple(float(v) for v in state.local_position_ned_m),
                segment_start,
                gate_positions[active_segment],
                speed_m_s=args.speed_m_s,
                cross_track_gain_s_inv=args.cross_track_gain_s_inv,
                max_cross_track_correction_m_s=args.max_cross_track_correction_m_s,
            )
            scheduled_speed_m_s = gate_scheduled_speed(
                command.along_to_gate_m,
                cruise_speed_m_s=args.speed_m_s,
                crossing_speed_m_s=args.gate_crossing_speed_m_s,
                slowdown_distance_m=args.gate_slowdown_distance_m,
            )
            if scheduled_speed_m_s != args.speed_m_s:
                command = segment_velocity_command(
                    tuple(float(v) for v in state.local_position_ned_m),
                    segment_start,
                    gate_positions[active_segment],
                    speed_m_s=scheduled_speed_m_s,
                    cross_track_gain_s_inv=args.cross_track_gain_s_inv,
                    max_cross_track_correction_m_s=args.max_cross_track_correction_m_s,
                )
            launch_climb_complete = (
                float(state.local_position_ned_m[2])
                <= float(initial_position[2]) - args.launch_climb_m
            )
            launch_active = (
                args.launch_climb_m > 0.0
                and not launch_climb_complete
                and now_s - started_s < args.launch_timeout_s
            )
            attitude_launch_active = (
                now_s - started_s < args.attitude_launch_duration_s
            )
            commanded_velocity = (
                (0.0, 0.0, -args.launch_speed_m_s)
                if launch_active and not attitude_launch_active
                else command.velocity_ned_m_s
            )

            if now_s >= next_heartbeat_s:
                adapter.send_heartbeat()
                next_heartbeat_s += heartbeat_period_s
            if now_s >= next_command_s:
                if attitude_launch_active:
                    adapter.send_attitude_setpoint(
                        AttitudeSetpoint(thrust=args.attitude_launch_thrust),
                        mode="body_rates",
                    )
                else:
                    vx, vy, vz = commanded_velocity
                    adapter.send_local_ned_setpoint(
                        LocalNedSetpoint(vx=vx, vy=vy, vz=vz),
                        frame="local_ned",
                        yaw_mode="ignore",
                    )
                commands_sent += 1
                next_command_s += command_period_s
                if next_command_s <= now_s:
                    next_command_s = now_s + command_period_s

            if now_s - last_sample_s >= 1.0 / args.trace_hz:
                trajectory.append(
                    {
                        "elapsed_s": round(now_s - started_s, 6),
                        "sim_boot_time_ms": int(status.sim_boot_time_ms),
                        "active_gate_index": gate_index,
                        "position_ned_m": [float(v) for v in state.local_position_ned_m],
                        "velocity_ned_m_s": (
                            None
                            if state.local_velocity_ned_m_s is None
                            else [float(v) for v in state.local_velocity_ned_m_s]
                        ),
                        "command_velocity_ned_m_s": [
                            float(v) for v in commanded_velocity
                        ] if not attitude_launch_active else None,
                        "attitude_launch_thrust": (
                            args.attitude_launch_thrust if attitude_launch_active else None
                        ),
                        "phase": (
                            "attitude_launch"
                            if attitude_launch_active
                            else ("launch_climb" if launch_active else "segment_track")
                        ),
                        "along_to_gate_m": command.along_to_gate_m,
                        "scheduled_speed_m_s": scheduled_speed_m_s,
                        "cross_track_error_m": command.cross_track_error_m,
                    }
                )
                last_sample_s = now_s
            time.sleep(0.0005)

        adapter.drain_telemetry(timeout_s=0.05)
        final_state = adapter.telemetry.state
        final_status = final_state.race_status
        valid_finish = is_valid_finish(final_status, args.required_gate_count) and collision is None
        bounded_progress_reached = bool(
            final_status is not None
            and int(final_status.active_gate_index) >= args.stop_after_gate_index
            and collision is None
        )
        accepted = (
            valid_finish
            if args.stop_after_gate_index == args.required_gate_count
            else bounded_progress_reached
        )
        elapsed_s = time.monotonic() - started_s
        final_position = final_state.local_position_ned_m

        adapter.send_local_ned_setpoint(
            LocalNedSetpoint(), frame="local_ned", yaw_mode="ignore"
        )
        commands_sent += 1
        adapter.send_disarm_command()
        exit_disarm_sent = True

        report = {
            "accepted": accepted,
            "acceptance_kind": (
                "official_six_gate_finish"
                if args.stop_after_gate_index == args.required_gate_count
                else "bounded_official_gate_progress"
            ),
            "bounded_progress_reached": bounded_progress_reached,
            "controller": {
                "kind": "published_track_segment_velocity",
                "speed_m_s": args.speed_m_s,
                "gate_crossing_speed_m_s": args.gate_crossing_speed_m_s,
                "gate_slowdown_distance_m": args.gate_slowdown_distance_m,
                "cross_track_gain_s_inv": args.cross_track_gain_s_inv,
                "max_cross_track_correction_m_s": args.max_cross_track_correction_m_s,
                "command_frame": "MAV_FRAME_LOCAL_NED",
                "yaw_ignored": True,
                "launch_climb_m": args.launch_climb_m,
                "launch_speed_m_s": args.launch_speed_m_s,
                "launch_timeout_s": args.launch_timeout_s,
                "attitude_launch_duration_s": args.attitude_launch_duration_s,
                "attitude_launch_thrust": args.attitude_launch_thrust,
                "launch_contact_grace_s": args.launch_contact_grace_s,
                "launch_contact_radius_m": args.launch_contact_radius_m,
                "allow_centered_low_threat_gate_contact": (
                    args.allow_centered_low_threat_gate_contact
                ),
                "low_threat_gate_contact_radius_m": (
                    args.low_threat_gate_contact_radius_m
                ),
                "gate_ned_z_offset_m": args.gate_ned_z_offset_m,
            },
            "contract": {
                "required_gate_count": args.required_gate_count,
                "stop_after_gate_index": args.stop_after_gate_index,
                "commands_sent": commands_sent,
                "command_hz": args.command_hz,
                "heartbeat_hz": args.heartbeat_hz,
                "exit_disarm_sent": exit_disarm_sent,
                "camera_used": False,
                "hidden_simulator_state_used": False,
            },
            "reset": reset_report,
            "track_gates": [dataclasses.asdict(gate) for gate in gates],
            "controller_gate_targets_ned_m": [list(position) for position in gate_positions],
            "elapsed_wall_s": round(elapsed_s, 6),
            "initial_position_ned_m": list(initial_position),
            "final_position_ned_m": (
                None if final_position is None else [float(v) for v in final_position]
            ),
            "official_active_gate_index": (
                None if final_status is None else int(final_status.active_gate_index)
            ),
            "official_race_finish_time_ns": (
                None if final_status is None else int(final_status.race_finish_time_ns)
            ),
            "official_race_finish_time_s": (
                None
                if final_status is None or int(final_status.race_finish_time_ns) < 0
                else int(final_status.race_finish_time_ns) / 1e9
            ),
            "collision": collision,
            "launch_support_contacts": launch_support_contacts,
            "low_threat_gate_contacts": low_threat_gate_contacts,
            "gate_transitions": transitions,
            "trajectory": trajectory,
            "telemetry": dataclasses.asdict(adapter.telemetry.metrics),
            "latest_telemetry": dataclasses.asdict(final_state),
        }
        return report
    finally:
        if not exit_disarm_sent:
            try:
                adapter.send_local_ned_setpoint(
                    LocalNedSetpoint(), frame="local_ned", yaw_mode="ignore"
                )
                adapter.send_disarm_command()
            except Exception:
                pass
        adapter.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a deterministic telemetry waypoint lap in VQ1 v3391"
    )
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--reset-timeout-s", type=float, default=12.0)
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--command-hz", type=float, default=60.0)
    parser.add_argument("--speed-m-s", type=float, default=5.0)
    parser.add_argument("--gate-crossing-speed-m-s", type=float, default=1.5)
    parser.add_argument("--gate-slowdown-distance-m", type=float, default=5.0)
    parser.add_argument("--cross-track-gain-s-inv", type=float, default=1.0)
    parser.add_argument("--max-cross-track-correction-m-s", type=float, default=2.0)
    parser.add_argument("--launch-climb-m", type=float, default=0.5)
    parser.add_argument("--launch-speed-m-s", type=float, default=0.5)
    parser.add_argument("--launch-timeout-s", type=float, default=2.0)
    parser.add_argument("--attitude-launch-duration-s", type=float, default=0.75)
    parser.add_argument("--attitude-launch-thrust", type=float, default=0.6)
    parser.add_argument("--launch-contact-grace-s", type=float, default=1.0)
    parser.add_argument("--launch-contact-radius-m", type=float, default=1.0)
    parser.add_argument("--allow-centered-low-threat-gate-contact", action="store_true")
    parser.add_argument("--low-threat-gate-contact-radius-m", type=float, default=0.5)
    parser.add_argument("--gate-ned-z-offset-m", type=float, default=0.0)
    parser.add_argument("--trace-hz", type=float, default=10.0)
    parser.add_argument("--required-gate-count", type=int, default=6)
    parser.add_argument("--stop-after-gate-index", type=int, default=6)
    parser.add_argument("--json-path", required=True)
    args = parser.parse_args()

    report = run_lap(args)
    output_path = Path(args.json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
