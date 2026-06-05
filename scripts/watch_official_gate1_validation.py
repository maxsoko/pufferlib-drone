#!/usr/bin/env python3
"""Watch for official TS-002 traffic and auto-run gate-1 validation."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from types import SimpleNamespace

import run_official_gate1_validation as validator


@dataclasses.dataclass
class WatchSummary:
    started_unix_s: float
    finished_unix_s: float
    elapsed_s: float
    poll_interval_s: float
    max_wait_s: float
    attempts: int
    final_status: str
    attempt_statuses: list[str]
    final_validation: dict | None
    next_commands: list[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _write_json(path: str, payload: dict) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _build_validation_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        host=args.host,
        mavlink_port=args.mavlink_port,
        camera_port=args.camera_port,
        probe_duration=args.probe_duration,
        probe_json_path=args.probe_json_path,
        send_sim_reset=args.send_sim_reset,
        post_reset_sleep_s=args.post_reset_sleep_s,
        acceptance_config=args.acceptance_config,
        endpoint=args.endpoint,
        control_mode=args.control_mode,
        policy_callable=args.policy_callable,
        policy_action_json=args.policy_action_json,
        smoke_duration=args.smoke_duration,
        heartbeat_hz=args.heartbeat_hz,
        command_hz=args.command_hz,
        telemetry_timeout_s=args.telemetry_timeout_s,
        telemetry_dropout_s=args.telemetry_dropout_s,
        idle_sleep_s=args.idle_sleep_s,
        camera_host=args.camera_host,
        camera_timeout_s=args.camera_timeout_s,
        camera_max_packets_per_loop=args.camera_max_packets_per_loop,
        max_detection_age_s=args.max_detection_age_s,
        detector_min_area_px=args.detector_min_area_px,
        detector_max_aspect_error=args.detector_max_aspect_error,
        detector_min_fill_ratio=args.detector_min_fill_ratio,
        target_gate_count=args.target_gate_count,
        require_official_race_progress=args.require_official_race_progress,
        smoke_json_path=args.smoke_json_path,
        smoke_csv_path=args.smoke_csv_path,
        summary_json_path=args.summary_json_path,
    )


def run_watch(
    args,
    *,
    run_once=validator.run_validation,
    now_fn=time.time,
    sleep_fn=time.sleep,
) -> WatchSummary:
    if args.poll_interval_s <= 0.0:
        raise ValueError("poll_interval_s must be positive")
    if args.max_wait_s == 0.0:
        raise ValueError("max_wait_s must be non-zero; use negative for unbounded watch")

    started = now_fn()
    deadline = None if args.max_wait_s < 0.0 else (started + args.max_wait_s)
    validation_args = _build_validation_args(args)
    attempt_statuses: list[str] = []
    final_validation = None
    attempt_index = 0

    while True:
        attempt_index += 1
        result = run_once(validation_args)
        final_validation = result.to_dict()
        attempt_statuses.append(result.status)
        if args.attempt_json_template:
            _write_json(args.attempt_json_template.format(attempt=attempt_index), final_validation)

        if result.status != "blocked_no_traffic":
            final_status = result.status
            break

        now = now_fn()
        if deadline is not None and now >= deadline:
            final_status = "watch_timeout_no_traffic"
            break
        sleep_fn(args.poll_interval_s)

    finished = now_fn()
    next_commands = (final_validation or {}).get("next_commands", [])
    return WatchSummary(
        started_unix_s=started,
        finished_unix_s=finished,
        elapsed_s=round(finished - started, 6),
        poll_interval_s=args.poll_interval_s,
        max_wait_s=args.max_wait_s,
        attempts=attempt_index,
        final_status=final_status,
        attempt_statuses=attempt_statuses,
        final_validation=final_validation,
        next_commands=next_commands,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch official TS-002 traffic and run gate-1 validation")

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

    parser.add_argument(
        "--attempt-json-template",
        default="logs/sitl/official_gate1_validation_attempt_{attempt:03d}.json",
        help="Per-attempt validation JSON path template with '{attempt}' placeholder",
    )
    parser.add_argument("--poll-interval-s", type=float, default=15.0)
    parser.add_argument(
        "--max-wait-s",
        type=float,
        default=900.0,
        help="Total watch budget in seconds; negative means watch indefinitely",
    )
    parser.add_argument("--watch-summary-json-path", default="logs/sitl/official_gate1_watch_summary.json")
    args = parser.parse_args()

    summary = run_watch(args)
    payload = summary.to_dict()
    _write_json(args.watch_summary_json_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))

    if summary.final_status == "watch_timeout_no_traffic":
        raise SystemExit("official watch timed out before required traffic appeared")
    if summary.final_status == "smoke_failed_acceptance":
        raise SystemExit("official watch reached traffic but smoke acceptance failed")


if __name__ == "__main__":
    main()
