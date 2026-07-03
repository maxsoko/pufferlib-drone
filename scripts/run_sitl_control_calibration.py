#!/usr/bin/env python3
"""Run a deterministic official-SITL command calibration matrix.

The goal is to discover which MAVLink control primitive the AI-GP simulator
actually obeys before spending GPU time on a policy that cannot transfer.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import time
from types import SimpleNamespace

import drone_sitl_competition_smoke as smoke
from drone_sitl_adapter import MavlinkSitlAdapter, validate_rates
import sitl_stream_probe as probe


@dataclasses.dataclass(frozen=True)
class CalibrationCase:
    name: str
    description: str
    control_mode: str
    command_frame: str = "local_ned"
    command_yaw_mode: str = "ignore"
    policy_action_json: str = "[0.0, 0.0, 0.0, 0.0]"
    attitude_mode: str = "body_rates"
    attitude_roll_rad: float = 0.0
    attitude_pitch_rad: float = 0.0
    attitude_yaw_rad: float = 0.0
    body_roll_rate_rad_s: float = 0.0
    body_pitch_rate_rad_s: float = 0.0
    body_yaw_rate_rad_s: float = 0.0
    attitude_thrust: float = 0.5


@dataclasses.dataclass
class CaseResult:
    case: dict
    status: str
    json_path: str
    csv_path: str
    acceptance_passed: bool = False
    acceptance_blockers: list[str] = dataclasses.field(default_factory=list)
    official_active_gate_index: int | None = None
    official_last_gate_race_time: int | None = None
    ordered_gate_passes: int = 0
    closest_range_m: float | None = None
    closest_range_elapsed_s: float | None = None
    highest_confidence: float | None = None
    telemetry_messages_seen: int = 0
    camera_frames_seen: int = 0
    collisions: int = 0
    commands_sent: int = 0
    command_rate_violations: int = 0
    command_kind: str = ""
    error: str = ""


@dataclasses.dataclass
class CalibrationSummary:
    started_unix_s: float
    finished_unix_s: float
    elapsed_s: float
    run_id: str
    endpoint: str
    mavlink_port: int
    camera_port: int
    probe: dict | None
    cases_run: int
    target_gate_count: int
    official_passing_cases: list[str]
    best_by_closest_range: str | None
    best_by_highest_confidence: str | None
    case_results: list[CaseResult]
    next_commands: list[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def default_cases() -> list[CalibrationCase]:
    return [
        CalibrationCase(
            name="velocity_local_forward_ignore_yaw",
            description="Official-example local-NED forward velocity with yaw/yaw-rate ignored.",
            control_mode="policy",
            command_frame="local_ned",
            command_yaw_mode="ignore",
            policy_action_json="[1.0, 0.0, 0.0, 0.0]",
        ),
        CalibrationCase(
            name="velocity_local_forward_yaw_active",
            description="Local-NED forward velocity with yaw/yaw-rate fields active.",
            control_mode="policy",
            command_frame="local_ned",
            command_yaw_mode="yaw_and_rate",
            policy_action_json="[1.0, 0.0, 0.0, 0.0]",
        ),
        CalibrationCase(
            name="velocity_body_forward_ignore_yaw",
            description="Body-NED forward velocity sanity check with yaw/yaw-rate ignored.",
            control_mode="policy",
            command_frame="body_ned",
            command_yaw_mode="ignore",
            policy_action_json="[1.0, 0.0, 0.0, 0.0]",
        ),
        CalibrationCase(
            name="velocity_local_right_positive",
            description="Local-NED positive right/lateral command sign probe.",
            control_mode="policy",
            command_frame="local_ned",
            command_yaw_mode="ignore",
            policy_action_json="[0.0, 1.0, 0.0, 0.0]",
        ),
        CalibrationCase(
            name="velocity_local_right_negative",
            description="Local-NED negative right/lateral command sign probe.",
            control_mode="policy",
            command_frame="local_ned",
            command_yaw_mode="ignore",
            policy_action_json="[0.0, -1.0, 0.0, 0.0]",
        ),
        CalibrationCase(
            name="velocity_local_down_positive",
            description="Local-NED positive down command sign probe.",
            control_mode="policy",
            command_frame="local_ned",
            command_yaw_mode="ignore",
            policy_action_json="[0.0, 0.0, 1.0, 0.0]",
        ),
        CalibrationCase(
            name="velocity_local_down_negative",
            description="Local-NED negative down command sign probe.",
            control_mode="policy",
            command_frame="local_ned",
            command_yaw_mode="ignore",
            policy_action_json="[0.0, 0.0, -1.0, 0.0]",
        ),
        CalibrationCase(
            name="attitude_rates_pitch_negative_t060",
            description="Simulator-example attitude-rate primitive: negative pitch rate, thrust 0.60.",
            control_mode="attitude-rates",
            attitude_mode="body_rates",
            body_pitch_rate_rad_s=-0.3,
            attitude_thrust=0.60,
        ),
        CalibrationCase(
            name="attitude_rates_pitch_negative_t065",
            description="Stronger forward attitude-rate primitive: negative pitch rate, thrust 0.65.",
            control_mode="attitude-rates",
            attitude_mode="body_rates",
            body_pitch_rate_rad_s=-0.5,
            attitude_thrust=0.65,
        ),
        CalibrationCase(
            name="attitude_rates_pitch_positive_t060",
            description="Pitch sign sanity check: positive pitch rate, thrust 0.60.",
            control_mode="attitude-rates",
            attitude_mode="body_rates",
            body_pitch_rate_rad_s=0.3,
            attitude_thrust=0.60,
        ),
        CalibrationCase(
            name="attitude_rates_roll_positive_t060",
            description="Roll sign sanity check: positive roll rate, thrust 0.60.",
            control_mode="attitude-rates",
            attitude_mode="body_rates",
            body_roll_rate_rad_s=0.2,
            attitude_thrust=0.60,
        ),
        CalibrationCase(
            name="attitude_rates_roll_negative_t060",
            description="Roll sign sanity check: negative roll rate, thrust 0.60.",
            control_mode="attitude-rates",
            attitude_mode="body_rates",
            body_roll_rate_rad_s=-0.2,
            attitude_thrust=0.60,
        ),
        CalibrationCase(
            name="attitude_rates_hover_t055",
            description="Neutral body rates, thrust 0.55 hover/vertical response probe.",
            control_mode="attitude-rates",
            attitude_mode="body_rates",
            attitude_thrust=0.55,
        ),
        CalibrationCase(
            name="visual_servo_attitude_default",
            description="Camera-steered attitude-rate controller using pitch/yaw/thrust from gate pose.",
            control_mode="visual-servo-attitude",
            attitude_mode="body_rates",
            attitude_thrust=0.58,
        ),
    ]


def sanitize_case_name(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return out or "case"


def write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def maybe_send_sim_reset(args) -> bool:
    if args.no_reset_between_cases:
        return False
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    try:
        adapter.send_heartbeat()
        adapter.send_sim_reset_command()
    finally:
        adapter.close()
    if args.post_reset_sleep_s > 0.0:
        time.sleep(args.post_reset_sleep_s)
    return True


def build_smoke_args(args, case: CalibrationCase, *, json_path: str, csv_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        acceptance_config=args.acceptance_config,
        endpoint=args.endpoint,
        heartbeat_hz=args.heartbeat_hz,
        command_hz=args.command_hz,
        duration=args.case_duration,
        telemetry_timeout_s=args.telemetry_timeout_s,
        telemetry_dropout_s=args.telemetry_dropout_s,
        idle_sleep_s=args.idle_sleep_s,
        arm_on_start=args.arm_on_start,
        arm_attempts=args.arm_attempts,
        prearm_heartbeat_timeout_s=args.prearm_heartbeat_timeout_s,
        control_mode=case.control_mode,
        command_frame=case.command_frame,
        command_yaw_mode=case.command_yaw_mode,
        attitude_mode=case.attitude_mode,
        attitude_roll_rad=case.attitude_roll_rad,
        attitude_pitch_rad=case.attitude_pitch_rad,
        attitude_yaw_rad=case.attitude_yaw_rad,
        body_roll_rate_rad_s=case.body_roll_rate_rad_s,
        body_pitch_rate_rad_s=case.body_pitch_rate_rad_s,
        body_yaw_rate_rad_s=case.body_yaw_rate_rad_s,
        attitude_thrust=case.attitude_thrust,
        attitude_servo_desired_standoff_m=args.attitude_servo_desired_standoff_m,
        attitude_servo_max_pitch_rate_rad_s=args.attitude_servo_max_pitch_rate_rad_s,
        attitude_servo_max_roll_rate_rad_s=args.attitude_servo_max_roll_rate_rad_s,
        attitude_servo_max_yaw_rate_rad_s=args.attitude_servo_max_yaw_rate_rad_s,
        attitude_servo_hover_thrust=args.attitude_servo_hover_thrust,
        attitude_servo_min_thrust=args.attitude_servo_min_thrust,
        attitude_servo_max_thrust=args.attitude_servo_max_thrust,
        attitude_servo_k_pitch=getattr(args, "attitude_servo_k_pitch", 0.16),
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
        policy_action_json=case.policy_action_json,
        policy_callable="",
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        camera_timeout_s=args.camera_timeout_s,
        camera_max_packets_per_loop=args.camera_max_packets_per_loop,
        no_camera=False,
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
        min_gate_passes=1,
        max_command_rate_violations=None,
        min_telemetry_messages=None,
        min_camera_frames=None,
        max_telemetry_dropouts=None,
        json_path=json_path,
        csv_path=csv_path,
    )


def summarize_case(case: CalibrationCase, report: smoke.CompetitionSmokeReport, *, json_path: str, csv_path: str) -> CaseResult:
    approach = report.approach_diagnostics
    return CaseResult(
        case=dataclasses.asdict(case),
        status="completed",
        json_path=json_path,
        csv_path=csv_path,
        acceptance_passed=bool(report.acceptance_passed),
        acceptance_blockers=list(report.acceptance_blockers),
        official_active_gate_index=report.official_active_gate_index,
        official_last_gate_race_time=report.official_last_gate_race_time,
        ordered_gate_passes=int(report.ordered_gate_passes),
        closest_range_m=approach.closest_range_m,
        closest_range_elapsed_s=approach.closest_range_elapsed_s,
        highest_confidence=approach.highest_confidence,
        telemetry_messages_seen=int(report.sitl.telemetry.messages_seen),
        camera_frames_seen=int(report.vision.frames_seen),
        collisions=int(report.sitl.telemetry.collisions),
        commands_sent=int(report.sitl.commands_sent),
        command_rate_violations=int(report.sitl.command_rate_violations),
        command_kind=report.sitl.command_kind,
    )


def failed_case(case: CalibrationCase, *, json_path: str, csv_path: str, error: Exception) -> CaseResult:
    return CaseResult(
        case=dataclasses.asdict(case),
        status="error",
        json_path=json_path,
        csv_path=csv_path,
        acceptance_blockers=["case_error"],
        error=f"{type(error).__name__}: {error}",
    )


def choose_best_name(results: list[CaseResult], field: str, *, reverse: bool = False) -> str | None:
    candidates = [
        result for result in results
        if result.status == "completed" and getattr(result, field) is not None
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda result: getattr(result, field), reverse=reverse)
    return str(candidates[0].case["name"])


def run_calibration(args) -> CalibrationSummary:
    validate_rates(args.heartbeat_hz, args.command_hz)
    if args.case_duration <= 0.0:
        raise ValueError("case_duration must be positive")
    if args.post_reset_sleep_s < 0.0:
        raise ValueError("post_reset_sleep_s must be non-negative")

    started = time.time()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S", time.localtime(started))
    os.makedirs(args.output_dir, exist_ok=True)

    probe_payload = None
    if not args.skip_probe:
        probe_report = probe.run_probe(
            host=args.host,
            mavlink_port=args.mavlink_port,
            camera_port=args.camera_port,
            duration_s=args.probe_duration,
        )
        met, blockers = probe.evaluate_probe_requirements(
            probe_report,
            require_mavlink=True,
            require_camera=True,
            require_ts002_header=True,
        )
        probe_report.requirements_met = met
        probe_report.blockers = blockers
        probe_payload = probe_report.to_dict()
        probe_path = os.path.join(args.output_dir, f"control_calibration_{run_id}_probe.json")
        write_json(probe_path, probe_payload)
        if args.stop_on_failed_probe and not met:
            finished = time.time()
            return CalibrationSummary(
                started_unix_s=started,
                finished_unix_s=finished,
                elapsed_s=round(finished - started, 6),
                run_id=run_id,
                endpoint=args.endpoint,
                mavlink_port=args.mavlink_port,
                camera_port=args.camera_port,
                probe=probe_payload,
                cases_run=0,
                target_gate_count=args.target_gate_count,
                official_passing_cases=[],
                best_by_closest_range=None,
                best_by_highest_confidence=None,
                case_results=[],
                next_commands=default_next_commands(args, run_id=run_id),
            )

    case_filter = re.compile(args.case_filter) if args.case_filter else None
    cases = [
        case for case in default_cases()
        if case_filter is None or case_filter.search(case.name)
    ]
    if not cases:
        raise ValueError("case_filter selected no calibration cases")

    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        safe_name = sanitize_case_name(case.name)
        json_path = os.path.join(args.output_dir, f"control_calibration_{run_id}_{index:02d}_{safe_name}.json")
        csv_path = os.path.join(args.output_dir, f"control_calibration_{run_id}_{index:02d}_{safe_name}.csv")
        try:
            maybe_send_sim_reset(args)
            smoke_args = build_smoke_args(args, case, json_path=json_path, csv_path=csv_path)
            report = smoke.run_smoke(smoke_args)
            payload = report.to_dict()
            write_json(json_path, payload)
            smoke.maybe_write_csv(csv_path, report)
            results.append(summarize_case(case, report, json_path=json_path, csv_path=csv_path))
        except Exception as exc:  # Keep the matrix moving so one bad primitive does not hide the rest.
            result = failed_case(case, json_path=json_path, csv_path=csv_path, error=exc)
            write_json(json_path, dataclasses.asdict(result))
            results.append(result)

    passing = [
        str(result.case["name"]) for result in results
        if result.official_active_gate_index is not None
        and int(result.official_active_gate_index) >= args.target_gate_count
    ]
    finished = time.time()
    return CalibrationSummary(
        started_unix_s=started,
        finished_unix_s=finished,
        elapsed_s=round(finished - started, 6),
        run_id=run_id,
        endpoint=args.endpoint,
        mavlink_port=args.mavlink_port,
        camera_port=args.camera_port,
        probe=probe_payload,
        cases_run=len(results),
        target_gate_count=args.target_gate_count,
        official_passing_cases=passing,
        best_by_closest_range=choose_best_name(results, "closest_range_m"),
        best_by_highest_confidence=choose_best_name(results, "highest_confidence", reverse=True),
        case_results=results,
        next_commands=default_next_commands(args, run_id=run_id),
    )


def default_next_commands(args, *, run_id: str) -> list[str]:
    summary_path = args.summary_json_path or os.path.join(
        args.output_dir,
        f"control_calibration_{run_id}_summary.json",
    )
    return [
        (
            "python scripts/run_sitl_control_calibration.py "
            f"--endpoint {args.endpoint} "
            f"--mavlink-port {args.mavlink_port} "
            f"--camera-port {args.camera_port} "
            f"--case-duration {args.case_duration} "
            f"--run-id {run_id} "
            f"--summary-json-path {summary_path}"
        ),
        (
            "python scripts/run_official_gate1_validation.py "
            f"--endpoint {args.endpoint} "
            f"--mavlink-port {args.mavlink_port} "
            f"--camera-port {args.camera_port} "
            "--control-mode visual-servo-attitude "
            "--attitude-mode body_rates "
            "--attitude-servo-hover-thrust 0.58 "
            "--attitude-servo-k-pitch 0.16 "
            "--attitude-servo-k-yaw 1.2 "
            "--attitude-servo-k-thrust 0.08 "
            "--require-official-race-progress"
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official SITL velocity/attitude command calibration")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--mavlink-port", type=int, default=14550)
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--probe-duration", type=float, default=3.0)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--stop-on-failed-probe", action="store_true")

    parser.add_argument("--acceptance-config", default=os.path.join("config", "sitl_competition_acceptance.json"))
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--case-duration", type=float, default=18.0)
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
    parser.add_argument("--max-approach-diagnostic-samples", type=int, default=12)
    parser.add_argument("--target-gate-count", type=int, default=1)
    parser.add_argument("--require-official-race-progress", action="store_true")
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

    parser.add_argument("--no-reset-between-cases", action="store_true")
    parser.add_argument("--post-reset-sleep-s", type=float, default=2.0)
    parser.add_argument("--case-filter", default="", help="Regex filter over built-in case names")
    parser.add_argument("--output-dir", default=os.path.join("logs", "sitl"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--summary-json-path", default="")
    parser.add_argument("--require-passing-case", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_calibration(args)
    payload = summary.to_dict()
    summary_path = args.summary_json_path or os.path.join(
        args.output_dir,
        f"control_calibration_{summary.run_id}_summary.json",
    )
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_passing_case and not summary.official_passing_cases:
        raise SystemExit("control calibration found no official passing case")


if __name__ == "__main__":
    main()
