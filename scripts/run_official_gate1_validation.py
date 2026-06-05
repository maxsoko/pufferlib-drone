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
        control_mode=args.control_mode,
        policy_action_json=args.policy_action_json,
        policy_callable=args.policy_callable,
        camera_host=args.camera_host,
        camera_port=args.camera_port,
        camera_timeout_s=args.camera_timeout_s,
        camera_max_packets_per_loop=getattr(args, "camera_max_packets_per_loop", 512),
        no_camera=False,
        max_detection_age_s=args.max_detection_age_s,
        detector_min_area_px=args.detector_min_area_px,
        detector_max_aspect_error=args.detector_max_aspect_error,
        detector_min_fill_ratio=args.detector_min_fill_ratio,
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
        json_path=args.smoke_json_path,
        csv_path=args.smoke_csv_path,
    )


def _maybe_send_sim_reset(args) -> bool:
    if not getattr(args, "send_sim_reset", False):
        return False
    adapter = MavlinkSitlAdapter(args.endpoint, dropout_after_s=args.telemetry_dropout_s)
    try:
        adapter.send_heartbeat()
        adapter.send_sim_reset_command()
    finally:
        adapter.close()
    post_reset_sleep_s = float(getattr(args, "post_reset_sleep_s", 2.0))
    if post_reset_sleep_s > 0.0:
        time.sleep(post_reset_sleep_s)
    return True


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
            + (f"--policy-callable {args.policy_callable} " if args.policy_callable else "")
            + f"--duration {args.smoke_duration} "
            + f"--camera-host {args.camera_host} "
            + f"--camera-port {args.camera_port} "
            + f"--camera-max-packets-per-loop {getattr(args, 'camera_max_packets_per_loop', 512)} "
            + f"--target-gate-count {args.target_gate_count} "
            + ("--require-official-race-progress " if args.require_official_race_progress else "")
            + f"--json-path {args.smoke_json_path} "
            + f"--csv-path {args.smoke_csv_path}"
        ),
    ]


def run_validation(args) -> ValidationSummary:
    started = time.time()
    reset_sent = _maybe_send_sim_reset(args)
    probe_report = probe.run_probe(
        host=args.host,
        mavlink_port=args.mavlink_port,
        camera_port=args.camera_port,
        duration_s=args.probe_duration,
    )
    requirements_met, blockers = probe.evaluate_probe_requirements(
        probe_report,
        require_mavlink=True,
        require_camera=True,
        require_ts002_header=True,
    )
    probe_report.requirements_met = requirements_met
    probe_report.blockers = blockers
    probe_payload = probe_report.to_dict()
    _write_json(args.probe_json_path, probe_payload)

    smoke_payload = None
    status = "blocked_no_traffic"
    if requirements_met:
        smoke_args = _build_smoke_args(args)
        smoke_report = smoke.run_smoke(smoke_args)
        smoke_payload = smoke_report.to_dict()
        _write_json(args.smoke_json_path, smoke_payload)
        smoke.maybe_write_csv(args.smoke_csv_path, smoke_report)
        status = "smoke_passed" if smoke_report.acceptance_passed else "smoke_failed_acceptance"

    finished = time.time()
    return ValidationSummary(
        started_unix_s=started,
        finished_unix_s=finished,
        elapsed_s=round(finished - started, 6),
        status=status,
        reset_sent=reset_sent,
        probe=probe_payload,
        smoke=smoke_payload,
        next_commands=_default_next_commands(args),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official TS-002 gate-1 probe + smoke validation")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--mavlink-port", type=int, default=14540)
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--probe-duration", type=float, default=5.0)
    parser.add_argument("--probe-json-path", default="logs/sitl/stream_probe_official.json")
    parser.add_argument("--send-sim-reset", action="store_true")
    parser.add_argument("--post-reset-sleep-s", type=float, default=2.0)

    parser.add_argument("--acceptance-config", default=os.path.join("config", "sitl_competition_acceptance.json"))
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14540")
    parser.add_argument("--control-mode", choices=["visual-servo", "policy"], default="visual-servo")
    parser.add_argument("--policy-callable", default="")
    parser.add_argument("--policy-action-json", default="[0.0, 0.0, 0.0, 0.0]")
    parser.add_argument("--smoke-duration", type=float, default=60.0)
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--command-hz", type=float, default=50.0)
    parser.add_argument("--telemetry-timeout-s", type=float, default=0.0)
    parser.add_argument("--telemetry-dropout-s", type=float, default=1.0)
    parser.add_argument("--idle-sleep-s", type=float, default=0.001)
    parser.add_argument("--camera-host", default="0.0.0.0")
    parser.add_argument("--camera-timeout-s", type=float, default=0.0)
    parser.add_argument("--camera-max-packets-per-loop", type=int, default=512)
    parser.add_argument("--max-detection-age-s", type=float, default=0.25)
    parser.add_argument("--detector-min-area-px", type=float, default=1200.0)
    parser.add_argument("--detector-max-aspect-error", type=float, default=0.5)
    parser.add_argument("--detector-min-fill-ratio", type=float, default=0.15)
    parser.add_argument("--target-gate-count", type=int, default=1)
    parser.add_argument("--require-official-race-progress", action="store_true")
    parser.add_argument("--smoke-json-path", default="logs/sitl/competition_smoke_gate1_official.json")
    parser.add_argument("--smoke-csv-path", default="logs/sitl/competition_smoke_gate1_official.csv")

    parser.add_argument("--summary-json-path", default="logs/sitl/official_gate1_validation_summary.json")
    args = parser.parse_args()

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
