#!/usr/bin/env python3
"""One-command official TS-002 gate-1 validation runner.

Flow:
1) Probe MAVLink + camera traffic.
2) If traffic is present, run competition smoke.
3) Emit a deterministic summary JSON with blocker or pass/fail status.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import drone_sitl_competition_smoke as smoke
from drone_sitl_adapter import MavlinkSitlAdapter
import sitl_stream_probe as probe


@dataclass
class ValidationSummary:
    started_unix_s: float
    finished_unix_s: float
    elapsed_s: float
    status: str
    reset_sent: bool
    probe: dict
    race_start_check: dict | None
    smoke: dict | None
    next_commands: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _write_json(path: str, payload: dict) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _build_smoke_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        acceptance_config=args.acceptance_config,
        endpoint=args.endpoint,
        heartbeat_hz=args.heartbeat_hz,
        command_hz=args.command_hz,
        duration=args.smoke_duration,
        telemetry_timeout_s=args.telemetry_timeout_s,
        telemetry_dropout_s=args.telemetry_dropout_s,
        idle_sleep_s=args.idle_sleep_s,
        arm_on_start=args.arm_on_start,
        arm_attempts=args.arm_attempts,
        prearm_heartbeat_timeout_s=args.prearm_heartbeat_timeout_s,
        official_reset_on_start=bool(
            getattr(args, "policy_ready_reset", False)
            and getattr(args, "send_sim_reset", False)
        ),
        official_reset_start_timeout_s=float(
            getattr(args, "official_reset_start_timeout_s", 12.0)
        ),
        official_policy_lead_s=float(
            getattr(args, "official_policy_lead_s", 0.05)
        ),
        control_mode=args.control_mode,
        command_frame=args.command_frame,
        command_yaw_mode=args.command_yaw_mode,
        attitude_mode=args.attitude_mode,
        attitude_roll_rad=args.attitude_roll_rad,
        attitude_pitch_rad=args.attitude_pitch_rad,
        attitude_yaw_rad=args.attitude_yaw_rad,
        body_roll_rate_rad_s=args.body_roll_rate_rad_s,
        body_pitch_rate_rad_s=args.body_pitch_rate_rad_s,
        body_yaw_rate_rad_s=args.body_yaw_rate_rad_s,
        attitude_thrust=args.attitude_thrust,
        attitude_servo_desired_standoff_m=args.attitude_servo_desired_standoff_m,
        attitude_servo_max_pitch_rate_rad_s=args.attitude_servo_max_pitch_rate_rad_s,
        attitude_servo_max_roll_rate_rad_s=args.attitude_servo_max_roll_rate_rad_s,
        attitude_servo_max_yaw_rate_rad_s=args.attitude_servo_max_yaw_rate_rad_s,
        attitude_servo_hover_thrust=args.attitude_servo_hover_thrust,
        attitude_servo_min_thrust=args.attitude_servo_min_thrust,
        attitude_servo_max_thrust=args.attitude_servo_max_thrust,
        attitude_servo_k_pitch=args.attitude_servo_k_pitch,
        attitude_servo_k_roll=getattr(args, "attitude_servo_k_roll", 0.0),
        attitude_servo_k_image_roll=getattr(args, "attitude_servo_k_image_roll", 0.0),
        attitude_servo_k_yaw=getattr(args, "attitude_servo_k_yaw", 1.2),
        attitude_servo_k_thrust=getattr(args, "attitude_servo_k_thrust", 0.08),
        attitude_servo_search_pitch_rate_rad_s=args.attitude_servo_search_pitch_rate_rad_s,
        attitude_servo_search_yaw_rate_rad_s=args.attitude_servo_search_yaw_rate_rad_s,
        attitude_servo_search_thrust=args.attitude_servo_search_thrust,
        attitude_servo_forward_yaw_tolerance_rad=args.attitude_servo_forward_yaw_tolerance_rad,
        attitude_servo_forward_z_tolerance_m=args.attitude_servo_forward_z_tolerance_m,
        attitude_servo_uncentered_forward_scale=args.attitude_servo_uncentered_forward_scale,
        attitude_servo_min_control_size_px=getattr(args, "attitude_servo_min_control_size_px", 0.0),
        attitude_servo_max_control_range_m=getattr(args, "attitude_servo_max_control_range_m", 0.0),
        attitude_servo_min_control_confidence=getattr(args, "attitude_servo_min_control_confidence", 0.0),
        attitude_servo_final_approach=getattr(args, "attitude_servo_final_approach", False),
        attitude_servo_final_trigger_range_m=getattr(args, "attitude_servo_final_trigger_range_m", 6.5),
        attitude_servo_final_min_size_px=getattr(args, "attitude_servo_final_min_size_px", 40.0),
        attitude_servo_final_min_confidence=getattr(args, "attitude_servo_final_min_confidence", 0.45),
        attitude_servo_final_max_yaw_error_rad=getattr(args, "attitude_servo_final_max_yaw_error_rad", 0.25),
        attitude_servo_final_max_abs_z_m=getattr(args, "attitude_servo_final_max_abs_z_m", 1.0),
        attitude_servo_final_duration_s=getattr(args, "attitude_servo_final_duration_s", 1.0),
        attitude_servo_final_pitch_rate_rad_s=getattr(args, "attitude_servo_final_pitch_rate_rad_s", 0.2),
        attitude_servo_final_max_yaw_rate_rad_s=getattr(args, "attitude_servo_final_max_yaw_rate_rad_s", 0.35),
        attitude_servo_final_thrust=getattr(args, "attitude_servo_final_thrust", 0.62),
        attitude_servo_final_max_activations=getattr(args, "attitude_servo_final_max_activations", 1),
        angle_servo_k_pitch=getattr(args, "angle_servo_k_pitch", 0.02),
        angle_servo_max_pitch_rad=getattr(args, "angle_servo_max_pitch_rad", 0.12),
        angle_servo_k_roll=getattr(args, "angle_servo_k_roll", 0.0),
        angle_servo_max_roll_rad=getattr(args, "angle_servo_max_roll_rad", 0.10),
        angle_servo_k_yaw=getattr(args, "angle_servo_k_yaw", 0.8),
        angle_servo_max_yaw_step_rad=getattr(args, "angle_servo_max_yaw_step_rad", 0.02),
        angle_servo_blind_range_m=getattr(args, "angle_servo_blind_range_m", 4.0),
        angle_servo_blind_duration_s=getattr(args, "angle_servo_blind_duration_s", 2.0),
        angle_servo_blind_pitch_rad=getattr(args, "angle_servo_blind_pitch_rad", 0.10),
        angle_servo_search_yaw_rate_rad_s=getattr(args, "angle_servo_search_yaw_rate_rad_s", 0.0),
        angle_servo_k_dx=getattr(args, "angle_servo_k_dx", 0.01),
        angle_servo_k_dz=getattr(args, "angle_servo_k_dz", 0.03),
        angle_servo_search_thrust=getattr(args, "angle_servo_search_thrust", None),
        angle_servo_track_jump_gate_m=getattr(args, "angle_servo_track_jump_gate_m", 5.0),
        angle_servo_track_coast_s=getattr(args, "angle_servo_track_coast_s", 1.5),
        policy_action_json=args.policy_action_json,
        policy_callable=args.policy_callable,
        policy_state_hz=getattr(args, "policy_state_hz", 0.0),
        policy_stop_before_gate_index=getattr(
            args, "policy_stop_before_gate_index", -1
        ),
        policy_stop_forward_m=getattr(args, "policy_stop_forward_m", 0.0),
        stop_after_official_gate_index=getattr(
            args, "stop_after_official_gate_index", -1
        ),
        policy_race_phase_observation=getattr(
            args, "policy_race_phase_observation", False
        ),
        policy_race_phase_denominator=getattr(
            args, "policy_race_phase_denominator", 3
        ),
        policy_phase_adapter_observation=getattr(
            args, "policy_phase_adapter_observation", False
        ),
        policy_gate_progress_adapter_observation=getattr(
            args, "policy_gate_progress_adapter_observation", False
        ),
        policy_gate_phase_onehot_adapter_observation=getattr(
            args, "policy_gate_phase_onehot_adapter_observation", False
        ),
        policy_hybrid_prefix_confidence_observation=getattr(
            args, "policy_hybrid_prefix_confidence_observation", False
        ),
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        camera_timeout_s=args.camera_timeout_s,
        camera_max_packets_per_loop=getattr(args, "camera_max_packets_per_loop", 512),
        no_camera=False,
        debug_frame_dir=getattr(args, "debug_frame_dir", ""),
        max_detection_age_s=args.max_detection_age_s,
        visual_servo_desired_standoff_m=args.visual_servo_desired_standoff_m,
        visual_servo_max_forward_m_s=args.visual_servo_max_forward_m_s,
        visual_servo_max_lateral_m_s=args.visual_servo_max_lateral_m_s,
        visual_servo_max_vertical_m_s=args.visual_servo_max_vertical_m_s,
        visual_servo_max_yaw_rate_rad_s=args.visual_servo_max_yaw_rate_rad_s,
        visual_servo_k_forward=args.visual_servo_k_forward,
        visual_servo_k_lateral=args.visual_servo_k_lateral,
        visual_servo_k_vertical=args.visual_servo_k_vertical,
        visual_servo_k_yaw=args.visual_servo_k_yaw,
        detector_min_area_px=args.detector_min_area_px,
        detector_max_aspect_error=args.detector_max_aspect_error,
        detector_min_fill_ratio=args.detector_min_fill_ratio,
        detector_require_color=getattr(args, "detector_require_color", False),
        gate_inner_width_m=getattr(args, "gate_inner_width_m", 1.5),
        gate_outer_width_m=getattr(args, "gate_outer_width_m", 2.7),
        course_controller_config=getattr(
            args, "course_controller_config", "config/course_fsm_defaults.json"
        ),
        max_approach_diagnostic_samples=args.max_approach_diagnostic_samples,
        target_gate_count=args.target_gate_count,
        require_official_race_progress=args.require_official_race_progress,
        gate_confidence_arm_min=None,
        gate_confidence_pass_min=None,
        gate_pass_arm_range_m=None,
        gate_pass_range_m=None,
        gate_pass_rearm_range_m=None,
        gate_pass_min_consecutive_frames=None,
        gate_pass_max_missed_frames=None,
        gate_pass_cooldown_s=None,
        require_telemetry=True,
        require_camera=True,
        # The normal multigate run leaves this unset and inherits the frozen
        # acceptance config. A bounded diagnostic may explicitly lower it to
        # the same value as --target-gate-count.
        min_gate_passes=getattr(args, "min_gate_passes", None),
        max_command_rate_violations=None,
        min_telemetry_messages=None,
        min_camera_frames=None,
        max_telemetry_dropouts=None,
        json_path=args.smoke_json_path,
        csv_path=args.smoke_csv_path,
    )


def _maybe_send_sim_reset(args) -> bool:
    if not getattr(args, "send_sim_reset", False):
        return False
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    try:
        heartbeat_deadline = time.monotonic() + 2.0
        while (
            adapter.telemetry.metrics.heartbeats <= 0
            and time.monotonic() < heartbeat_deadline
        ):
            adapter.poll_telemetry(timeout_s=0.05)
        if adapter.telemetry.metrics.heartbeats <= 0:
            raise RuntimeError("cannot reset simulator before receiving a MAVLink heartbeat")
        adapter.send_heartbeat()
        adapter.send_disarm_command()
        time.sleep(0.1)
        adapter.send_sim_reset_command()
        # Reset can race the last armed setpoint retained by the flight
        # controller. Reassert disarm after respawn so the vehicle stays on the
        # pad during stream probing and stationary IMU calibration.
        time.sleep(0.5)
        for _ in range(3):
            adapter.send_heartbeat()
            adapter.send_disarm_command()
            time.sleep(0.1)
    finally:
        adapter.close()
    post_reset_sleep_s = float(getattr(args, "post_reset_sleep_s", 2.0))
    if post_reset_sleep_s > 0.0:
        time.sleep(post_reset_sleep_s)
    return True


def _check_race_started(args) -> dict:
    duration_s = float(getattr(args, "race_start_check_s", 1.0))
    if duration_s <= 0.0:
        return {"skipped": True, "duration_s": 0.0, "race_started": None, "race_status": None}

    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    started_s = time.monotonic()
    deadline_s = started_s + duration_s
    try:
        while time.monotonic() < deadline_s:
            adapter.poll_telemetry(timeout_s=min(0.05, max(0.0, deadline_s - time.monotonic())))
            if adapter.telemetry.state.race_status is not None:
                break
    finally:
        adapter.close()

    race_status = adapter.telemetry.state.race_status
    race_started = None
    if race_status is not None:
        race_started = int(race_status.race_start_boot_time_ms) >= 0
    return {
        "skipped": False,
        "duration_s": round(time.monotonic() - started_s, 6),
        "race_started": race_started,
        "race_status": None if race_status is None else asdict(race_status),
        "heartbeats_seen": int(adapter.telemetry.metrics.heartbeats),
        "messages_seen": int(adapter.telemetry.metrics.messages_seen),
    }


def _default_next_commands(args) -> list[str]:
    return [
        (
            "python scripts/sitl_stream_probe.py "
            f"--host {args.host} "
            f"--mavlink-port {args.mavlink_port} "
            f"--camera-port {args.camera_port} "
            "--duration 5 "
            "--require-mavlink --require-camera --require-ts002-header "
            f"--json-path {args.probe_json_path}"
        ),
        (
            "python scripts/drone_sitl_competition_smoke.py "
            f"--acceptance-config {args.acceptance_config} "
            f"--endpoint {args.endpoint} "
            f"--control-mode {args.control_mode} "
            f"--command-frame {args.command_frame} "
            f"--command-yaw-mode {args.command_yaw_mode} "
            f"--attitude-mode {args.attitude_mode} "
            f"--attitude-roll-rad {args.attitude_roll_rad} "
            f"--attitude-pitch-rad {args.attitude_pitch_rad} "
            f"--attitude-yaw-rad {args.attitude_yaw_rad} "
            f"--body-roll-rate-rad-s {args.body_roll_rate_rad_s} "
            f"--body-pitch-rate-rad-s {args.body_pitch_rate_rad_s} "
            f"--body-yaw-rate-rad-s {args.body_yaw_rate_rad_s} "
            f"--attitude-thrust {args.attitude_thrust} "
            f"--attitude-servo-desired-standoff-m {args.attitude_servo_desired_standoff_m} "
            f"--attitude-servo-max-pitch-rate-rad-s {args.attitude_servo_max_pitch_rate_rad_s} "
            f"--attitude-servo-max-roll-rate-rad-s {args.attitude_servo_max_roll_rate_rad_s} "
            f"--attitude-servo-max-yaw-rate-rad-s {args.attitude_servo_max_yaw_rate_rad_s} "
            f"--attitude-servo-hover-thrust {args.attitude_servo_hover_thrust} "
            f"--attitude-servo-min-thrust {args.attitude_servo_min_thrust} "
            f"--attitude-servo-max-thrust {args.attitude_servo_max_thrust} "
            f"--attitude-servo-k-pitch {args.attitude_servo_k_pitch} "
            f"--attitude-servo-k-roll {args.attitude_servo_k_roll} "
            f"--attitude-servo-k-image-roll {getattr(args, 'attitude_servo_k_image_roll', 0.0)} "
            f"--attitude-servo-k-yaw {args.attitude_servo_k_yaw} "
            f"--attitude-servo-k-thrust {args.attitude_servo_k_thrust} "
            f"--attitude-servo-search-pitch-rate-rad-s {args.attitude_servo_search_pitch_rate_rad_s} "
            f"--attitude-servo-search-yaw-rate-rad-s {args.attitude_servo_search_yaw_rate_rad_s} "
            + (
                ""
                if args.attitude_servo_search_thrust is None
                else f"--attitude-servo-search-thrust {args.attitude_servo_search_thrust} "
            )
            + (
                ""
                if args.attitude_servo_forward_yaw_tolerance_rad is None
                else (
                    "--attitude-servo-forward-yaw-tolerance-rad "
                    f"{args.attitude_servo_forward_yaw_tolerance_rad} "
                )
            )
            + (
                ""
                if args.attitude_servo_forward_z_tolerance_m is None
                else f"--attitude-servo-forward-z-tolerance-m {args.attitude_servo_forward_z_tolerance_m} "
            )
            + f"--attitude-servo-uncentered-forward-scale {args.attitude_servo_uncentered_forward_scale} "
            + f"--attitude-servo-min-control-size-px {getattr(args, 'attitude_servo_min_control_size_px', 0.0)} "
            + f"--attitude-servo-max-control-range-m {getattr(args, 'attitude_servo_max_control_range_m', 0.0)} "
            + (
                "--attitude-servo-min-control-confidence "
                f"{getattr(args, 'attitude_servo_min_control_confidence', 0.0)} "
            )
            + ("--attitude-servo-final-approach " if getattr(args, "attitude_servo_final_approach", False) else "")
            + (
                "--attitude-servo-final-trigger-range-m "
                f"{getattr(args, 'attitude_servo_final_trigger_range_m', 6.5)} "
            )
            + (
                "--attitude-servo-final-min-size-px "
                f"{getattr(args, 'attitude_servo_final_min_size_px', 40.0)} "
            )
            + (
                "--attitude-servo-final-min-confidence "
                f"{getattr(args, 'attitude_servo_final_min_confidence', 0.45)} "
            )
            + (
                "--attitude-servo-final-max-yaw-error-rad "
                f"{getattr(args, 'attitude_servo_final_max_yaw_error_rad', 0.25)} "
            )
            + (
                ""
                if getattr(args, "attitude_servo_final_max_abs_z_m", 1.0) is None
                else (
                    "--attitude-servo-final-max-abs-z-m "
                    f"{getattr(args, 'attitude_servo_final_max_abs_z_m', 1.0)} "
                )
            )
            + (
                "--attitude-servo-final-duration-s "
                f"{getattr(args, 'attitude_servo_final_duration_s', 1.0)} "
            )
            + (
                "--attitude-servo-final-pitch-rate-rad-s "
                f"{getattr(args, 'attitude_servo_final_pitch_rate_rad_s', 0.2)} "
            )
            + (
                "--attitude-servo-final-max-yaw-rate-rad-s "
                f"{getattr(args, 'attitude_servo_final_max_yaw_rate_rad_s', 0.35)} "
            )
            + f"--attitude-servo-final-thrust {getattr(args, 'attitude_servo_final_thrust', 0.62)} "
            + (
                "--attitude-servo-final-max-activations "
                f"{getattr(args, 'attitude_servo_final_max_activations', 1)} "
            )
            + f"--policy-action-json '{args.policy_action_json}' "
            + (f"--policy-callable {args.policy_callable} " if args.policy_callable else "")
            + f"--policy-state-hz {getattr(args, 'policy_state_hz', 0.0)} "
            + (
                "--policy-stop-before-gate-index "
                f"{getattr(args, 'policy_stop_before_gate_index', -1)} "
            )
            + (
                "--policy-stop-forward-m "
                f"{getattr(args, 'policy_stop_forward_m', 0.0)} "
            )
            + (
                "--policy-race-phase-observation "
                if getattr(args, "policy_race_phase_observation", False)
                else ""
            )
            + (
                "--policy-phase-adapter-observation "
                if getattr(args, "policy_phase_adapter_observation", False)
                else ""
            )
            + (
                "--policy-gate-phase-onehot-adapter-observation "
                if getattr(
                    args,
                    "policy_gate_phase_onehot_adapter_observation",
                    False,
                )
                else ""
            )
            + (
                "--policy-hybrid-prefix-confidence-observation "
                if getattr(
                    args,
                    "policy_hybrid_prefix_confidence_observation",
                    False,
                )
                else ""
            )
            + (
                "--policy-race-phase-denominator "
                f"{getattr(args, 'policy_race_phase_denominator', 3)} "
            )
            + f"--duration {args.smoke_duration} "
            + ("" if args.arm_on_start else "--no-arm-on-start ")
            + f"--arm-attempts {args.arm_attempts} "
            + f"--prearm-heartbeat-timeout-s {args.prearm_heartbeat_timeout_s} "
            + f"--camera-host {args.camera_host} "
            + f"--camera-port {args.camera_port} "
            + f"--camera-max-packets-per-loop {getattr(args, 'camera_max_packets_per_loop', 512)} "
            + (f"--debug-frame-dir {args.debug_frame_dir} " if getattr(args, "debug_frame_dir", "") else "")
            + f"--max-approach-diagnostic-samples {args.max_approach_diagnostic_samples} "
            + f"--max-detection-age-s {args.max_detection_age_s} "
            + f"--visual-servo-desired-standoff-m {args.visual_servo_desired_standoff_m} "
            + f"--visual-servo-max-forward-m-s {args.visual_servo_max_forward_m_s} "
            + f"--visual-servo-max-lateral-m-s {args.visual_servo_max_lateral_m_s} "
            + f"--visual-servo-max-vertical-m-s {args.visual_servo_max_vertical_m_s} "
            + f"--visual-servo-max-yaw-rate-rad-s {args.visual_servo_max_yaw_rate_rad_s} "
            + f"--visual-servo-k-forward {args.visual_servo_k_forward} "
            + f"--visual-servo-k-lateral {args.visual_servo_k_lateral} "
            + f"--visual-servo-k-vertical {args.visual_servo_k_vertical} "
            + f"--visual-servo-k-yaw {args.visual_servo_k_yaw} "
            + f"--detector-min-area-px {args.detector_min_area_px} "
            + f"--detector-max-aspect-error {args.detector_max_aspect_error} "
            + f"--detector-min-fill-ratio {args.detector_min_fill_ratio} "
            + ("--detector-require-color " if getattr(args, "detector_require_color", False) else "")
            + f"--target-gate-count {args.target_gate_count} "
            + (
                f"--min-gate-passes {args.min_gate_passes} "
                if getattr(args, "min_gate_passes", None) is not None
                else ""
            )
            + ("--require-official-race-progress " if args.require_official_race_progress else "")
            + f"--json-path {args.smoke_json_path} "
            + f"--csv-path {args.smoke_csv_path}"
        ),
    ]


def run_validation(args) -> ValidationSummary:
    started = time.time()
    policy_ready_reset = bool(getattr(args, "policy_ready_reset", False))
    # A full policy attempt must own the first timed flight target.  In that
    # mode the already-loaded smoke controller issues 31000 itself; the legacy
    # pre-probe reset remains available to diagnostic/FSM runners.
    reset_sent = False if policy_ready_reset else _maybe_send_sim_reset(args)
    probe_report = probe.run_probe(
        host=args.host,
        mavlink_port=args.mavlink_port,
        camera_port=args.camera_port,
        duration_s=args.probe_duration,
    )
    requirements_met, blockers = probe.evaluate_probe_requirements(
        probe_report,
        require_mavlink=True,
        require_camera=not policy_ready_reset,
        require_ts002_header=not policy_ready_reset,
    )
    probe_report.requirements_met = requirements_met
    probe_report.blockers = blockers
    probe_payload = probe_report.to_dict()
    _write_json(args.probe_json_path, probe_payload)

    smoke_payload = None
    race_start_check_payload = None
    status = "blocked_no_traffic"
    if requirements_met:
        race_start_check_payload = (
            {
                "skipped": True,
                "duration_s": 0.0,
                "race_started": None,
                "race_status": None,
                "reason": "policy_ready_reset_owns_race_start",
            }
            if policy_ready_reset
            else _check_race_started(args)
        )
        if (
            not policy_ready_reset
            and race_start_check_payload.get("race_started") is False
        ):
            status = "blocked_race_not_started"
        else:
            smoke_args = _build_smoke_args(args)
            smoke_report = smoke.run_smoke(smoke_args)
            if policy_ready_reset:
                reset_sent = bool(
                    smoke_report.control_inputs
                    .get("official_reset_start", {})
                    .get("reset_sent", False)
                )
            smoke_payload = smoke_report.to_dict()
            _write_json(args.smoke_json_path, smoke_payload)
            smoke.maybe_write_csv(args.smoke_csv_path, smoke_report)
            status = "smoke_passed" if smoke_report.acceptance_passed else "smoke_failed_acceptance"
            race_status = smoke_report.sitl.latest_telemetry.race_status
            if (
                status == "smoke_failed_acceptance"
                and race_status is not None
                and race_status.race_start_boot_time_ms is not None
                and int(race_status.race_start_boot_time_ms) < 0
            ):
                status = "blocked_race_not_started"

    finished = time.time()
    return ValidationSummary(
        started_unix_s=started,
        finished_unix_s=finished,
        elapsed_s=round(finished - started, 6),
        status=status,
        reset_sent=reset_sent,
        probe=probe_payload,
        race_start_check=race_start_check_payload,
        smoke=smoke_payload,
        next_commands=_default_next_commands(args),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official TS-002 gate-1 probe + smoke validation")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--mavlink-port", type=int, default=14540)
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--probe-duration", type=float, default=5.0)
    parser.add_argument("--race-start-check-s", type=float, default=1.0)
    parser.add_argument("--probe-json-path", default="logs/sitl/stream_probe_official.json")
    parser.add_argument("--send-sim-reset", action="store_true")
    parser.add_argument("--post-reset-sleep-s", type=float, default=2.0)
    parser.add_argument(
        "--policy-ready-reset",
        action="store_true",
        help=(
            "Delegate MAVLink 31000 to the loaded full-policy smoke runner so "
            "its first command is installed at the scheduled race start."
        ),
    )
    parser.add_argument("--official-reset-start-timeout-s", type=float, default=12.0)
    parser.add_argument("--official-policy-lead-s", type=float, default=0.05)

    parser.add_argument("--acceptance-config", default=os.path.join("config", "sitl_competition_acceptance.json"))
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14540")
    parser.add_argument(
        "--control-mode",
        choices=[
            "visual-servo",
            "policy",
            "policy-attitude",
            "attitude-rates",
            "visual-servo-attitude",
            "visual-servo-angles",
            "course-fsm",
        ],
        default="visual-servo",
    )
    parser.add_argument("--command-frame", choices=["body_ned", "local_ned"], default="local_ned")
    parser.add_argument("--command-yaw-mode", choices=["yaw_and_rate", "ignore"], default="yaw_and_rate")
    parser.add_argument(
        "--attitude-mode",
        choices=["body_rates", "attitude", "attitude_and_rates"],
        default="body_rates",
    )
    parser.add_argument("--attitude-roll-rad", type=float, default=0.0)
    parser.add_argument("--attitude-pitch-rad", type=float, default=0.0)
    parser.add_argument("--attitude-yaw-rad", type=float, default=0.0)
    parser.add_argument("--body-roll-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--body-pitch-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--body-yaw-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--attitude-thrust", type=float, default=0.5)
    parser.add_argument("--attitude-servo-desired-standoff-m", type=float, default=0.0)
    parser.add_argument("--attitude-servo-max-pitch-rate-rad-s", type=float, default=0.5)
    parser.add_argument("--attitude-servo-max-roll-rate-rad-s", type=float, default=0.4)
    parser.add_argument("--attitude-servo-max-yaw-rate-rad-s", type=float, default=0.7)
    parser.add_argument("--attitude-servo-hover-thrust", type=float, default=0.58)
    parser.add_argument("--attitude-servo-min-thrust", type=float, default=0.35)
    parser.add_argument("--attitude-servo-max-thrust", type=float, default=0.75)
    parser.add_argument("--attitude-servo-k-pitch", type=float, default=0.16)
    parser.add_argument("--attitude-servo-k-roll", type=float, default=0.0)
    parser.add_argument("--attitude-servo-k-image-roll", type=float, default=0.0)
    parser.add_argument("--attitude-servo-k-yaw", type=float, default=1.2)
    parser.add_argument("--attitude-servo-k-thrust", type=float, default=0.08)
    parser.add_argument("--attitude-servo-search-pitch-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--attitude-servo-search-yaw-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--attitude-servo-search-thrust", type=float, default=None)
    parser.add_argument("--attitude-servo-forward-yaw-tolerance-rad", type=float, default=None)
    parser.add_argument("--attitude-servo-forward-z-tolerance-m", type=float, default=None)
    parser.add_argument("--attitude-servo-uncentered-forward-scale", type=float, default=1.0)
    parser.add_argument("--attitude-servo-min-control-size-px", type=float, default=0.0)
    parser.add_argument("--attitude-servo-max-control-range-m", type=float, default=0.0)
    parser.add_argument("--attitude-servo-min-control-confidence", type=float, default=0.0)
    parser.add_argument("--attitude-servo-final-approach", action="store_true")
    parser.add_argument("--attitude-servo-final-trigger-range-m", type=float, default=6.5)
    parser.add_argument("--attitude-servo-final-min-size-px", type=float, default=40.0)
    parser.add_argument("--attitude-servo-final-min-confidence", type=float, default=0.45)
    parser.add_argument("--attitude-servo-final-max-yaw-error-rad", type=float, default=0.25)
    parser.add_argument("--attitude-servo-final-max-abs-z-m", type=float, default=1.0)
    parser.add_argument("--attitude-servo-final-duration-s", type=float, default=1.0)
    parser.add_argument("--attitude-servo-final-pitch-rate-rad-s", type=float, default=0.2)
    parser.add_argument("--attitude-servo-final-max-yaw-rate-rad-s", type=float, default=0.35)
    parser.add_argument("--attitude-servo-final-thrust", type=float, default=0.62)
    parser.add_argument("--attitude-servo-final-max-activations", type=int, default=1)
    parser.add_argument("--angle-servo-k-pitch", type=float, default=0.02)
    parser.add_argument("--angle-servo-max-pitch-rad", type=float, default=0.12)
    parser.add_argument("--angle-servo-k-roll", type=float, default=0.0)
    parser.add_argument("--angle-servo-max-roll-rad", type=float, default=0.10)
    parser.add_argument("--angle-servo-k-yaw", type=float, default=0.8)
    parser.add_argument("--angle-servo-max-yaw-step-rad", type=float, default=0.02)
    parser.add_argument("--angle-servo-blind-range-m", type=float, default=4.0)
    parser.add_argument("--angle-servo-blind-duration-s", type=float, default=2.0)
    parser.add_argument("--angle-servo-blind-pitch-rad", type=float, default=0.10)
    parser.add_argument("--angle-servo-search-yaw-rate-rad-s", type=float, default=0.0)
    parser.add_argument("--angle-servo-k-dx", type=float, default=0.01)
    parser.add_argument("--angle-servo-k-dz", type=float, default=0.03)
    parser.add_argument("--angle-servo-search-thrust", type=float, default=None)
    parser.add_argument("--angle-servo-track-jump-gate-m", type=float, default=5.0)
    parser.add_argument("--angle-servo-track-coast-s", type=float, default=1.5)
    parser.add_argument("--policy-callable", default="")
    parser.add_argument("--policy-state-hz", type=float, default=0.0)
    parser.add_argument("--policy-action-json", default="[0.0, 0.0, 0.0, 0.0]")
    parser.add_argument("--policy-stop-before-gate-index", type=int, default=-1)
    parser.add_argument("--policy-stop-forward-m", type=float, default=0.0)
    parser.add_argument("--stop-after-official-gate-index", type=int, default=-1)
    parser.add_argument("--policy-race-phase-observation", action="store_true")
    parser.add_argument("--policy-race-phase-denominator", type=int, default=3)
    parser.add_argument("--policy-phase-adapter-observation", action="store_true")
    parser.add_argument(
        "--policy-gate-progress-adapter-observation", action="store_true"
    )
    parser.add_argument(
        "--policy-gate-phase-onehot-adapter-observation", action="store_true"
    )
    parser.add_argument(
        "--policy-hybrid-prefix-confidence-observation", action="store_true"
    )
    parser.add_argument("--smoke-duration", type=float, default=60.0)
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--command-hz", type=float, default=50.0)
    parser.add_argument("--telemetry-timeout-s", type=float, default=0.0)
    parser.add_argument("--telemetry-dropout-s", type=float, default=1.0)
    parser.add_argument("--idle-sleep-s", type=float, default=0.001)
    parser.add_argument("--no-arm-on-start", dest="arm_on_start", action="store_false")
    parser.set_defaults(arm_on_start=True)
    parser.add_argument("--arm-attempts", type=int, default=3)
    parser.add_argument("--prearm-heartbeat-timeout-s", type=float, default=2.0)
    parser.add_argument("--camera-host", default="0.0.0.0")
    parser.add_argument("--camera-timeout-s", type=float, default=0.0)
    parser.add_argument("--camera-max-packets-per-loop", type=int, default=512)
    parser.add_argument("--debug-frame-dir", default="")
    parser.add_argument("--max-detection-age-s", type=float, default=0.25)
    parser.add_argument("--visual-servo-desired-standoff-m", type=float, default=1.0)
    parser.add_argument("--visual-servo-max-forward-m-s", type=float, default=1.0)
    parser.add_argument("--visual-servo-max-lateral-m-s", type=float, default=0.5)
    parser.add_argument("--visual-servo-max-vertical-m-s", type=float, default=0.4)
    parser.add_argument("--visual-servo-max-yaw-rate-rad-s", type=float, default=0.6)
    parser.add_argument("--visual-servo-k-forward", type=float, default=0.45)
    parser.add_argument("--visual-servo-k-lateral", type=float, default=0.7)
    parser.add_argument("--visual-servo-k-vertical", type=float, default=0.7)
    parser.add_argument("--visual-servo-k-yaw", type=float, default=1.2)
    parser.add_argument("--detector-min-area-px", type=float, default=1200.0)
    parser.add_argument("--detector-max-aspect-error", type=float, default=0.5)
    parser.add_argument("--detector-min-fill-ratio", type=float, default=0.15)
    parser.add_argument("--detector-require-color", action="store_true")
    parser.add_argument("--gate-inner-width-m", type=float, default=1.5)
    parser.add_argument("--gate-outer-width-m", type=float, default=2.7)
    parser.add_argument(
        "--course-controller-config",
        default="config/course_fsm_defaults.json",
    )
    parser.add_argument("--max-approach-diagnostic-samples", type=int, default=12)
    parser.add_argument("--target-gate-count", type=int, default=1)
    parser.add_argument("--min-gate-passes", type=int, default=None)
    parser.add_argument("--require-official-race-progress", action="store_true")
    parser.add_argument("--smoke-json-path", default="logs/sitl/competition_smoke_gate1_official.json")
    parser.add_argument("--smoke-csv-path", default="logs/sitl/competition_smoke_gate1_official.csv")

    parser.add_argument("--summary-json-path", default="logs/sitl/official_gate1_validation_summary.json")
    args = parser.parse_args()
    if args.min_gate_passes is not None and args.min_gate_passes <= 0:
        parser.error("--min-gate-passes must be positive")
    if args.policy_race_phase_denominator <= 0:
        parser.error("--policy-race-phase-denominator must be positive")
    if not math.isfinite(args.policy_state_hz) or args.policy_state_hz < 0.0:
        parser.error("--policy-state-hz must be finite and nonnegative")
    observation_modes = sum(
        bool(value)
        for value in (
            args.policy_race_phase_observation,
            args.policy_phase_adapter_observation,
            args.policy_gate_progress_adapter_observation,
            args.policy_gate_phase_onehot_adapter_observation,
        )
    )
    if observation_modes > 1:
        parser.error(
            "choose only one policy phase/progress observation adapter"
        )
    if (
        args.policy_hybrid_prefix_confidence_observation
        and not args.policy_gate_phase_onehot_adapter_observation
    ):
        parser.error(
            "--policy-hybrid-prefix-confidence-observation requires "
            "--policy-gate-phase-onehot-adapter-observation"
        )
    if args.policy_ready_reset and not args.send_sim_reset:
        parser.error("--policy-ready-reset requires --send-sim-reset")
    if args.policy_ready_reset and args.control_mode != "policy-attitude":
        parser.error("--policy-ready-reset requires --control-mode=policy-attitude")

    summary = run_validation(args)
    payload = summary.to_dict()
    _write_json(args.summary_json_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))

    if summary.status == "blocked_no_traffic":
        raise SystemExit("official validation blocked: no required MAVLink/camera traffic")
    if summary.status == "smoke_failed_acceptance":
        raise SystemExit("official validation failed: smoke acceptance did not pass")


if __name__ == "__main__":
    main()
