#!/usr/bin/env python3
"""Deterministic replay regression runner for SITL smoke acceptance."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from types import SimpleNamespace

import drone_sitl_competition_smoke as smoke
import sitl_udp_replay as replay


def write_json(path: str, payload: dict) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_single_mode(
    *,
    mode: str,
    policy_callable: str,
    runs: int,
    events: list[replay.ReplayEvent],
    args,
    output_dir: str,
) -> dict:
    reports: list[dict] = []
    valid_runs = 0

    for run_idx in range(runs):
        replay_metrics_holder: dict = {}

        def replay_worker() -> None:
            time.sleep(args.replay_start_delay_s)
            replay_metrics_holder["metrics"] = replay.replay_events(
                events,
                mavlink_target=(args.replay_mavlink_host, args.replay_mavlink_port),
                camera_target=(args.replay_camera_host, args.replay_camera_port),
                speed=args.replay_speed,
                loops=1,
                loss_rate=args.loss_rate,
                reorder_rate=args.reorder_rate,
                latency_ms=args.latency_ms,
                jitter_ms=args.jitter_ms,
                seed=args.seed + run_idx,
            ).to_dict()

        worker = threading.Thread(target=replay_worker, daemon=True)
        worker.start()
        smoke_args = SimpleNamespace(
            acceptance_config=args.acceptance_config,
            endpoint=args.smoke_endpoint,
            heartbeat_hz=args.heartbeat_hz,
            command_hz=args.command_hz,
            duration=args.smoke_duration,
            telemetry_timeout_s=args.telemetry_timeout_s,
            telemetry_dropout_s=args.telemetry_dropout_s,
            idle_sleep_s=args.idle_sleep_s,
            control_mode=mode,
            policy_action_json=args.policy_action_json,
            policy_callable=policy_callable,
            camera_host=args.smoke_camera_host,
            camera_port=args.smoke_camera_port,
            camera_timeout_s=args.camera_timeout_s,
            no_camera=False,
            max_detection_age_s=args.max_detection_age_s,
            detector_min_area_px=args.detector_min_area_px,
            detector_max_aspect_error=args.detector_max_aspect_error,
            detector_min_fill_ratio=args.detector_min_fill_ratio,
            target_gate_count=args.target_gate_count,
            gate_confidence_arm_min=None,
            gate_confidence_pass_min=None,
            gate_pass_arm_range_m=None,
            gate_pass_range_m=None,
            gate_pass_rearm_range_m=None,
            gate_pass_min_consecutive_frames=None,
            gate_pass_max_missed_frames=None,
            gate_pass_cooldown_s=None,
            require_telemetry=args.require_telemetry,
            require_camera=args.require_camera,
            min_gate_passes=args.min_gate_passes,
            max_command_rate_violations=None,
            min_telemetry_messages=None,
            min_camera_frames=None,
            max_telemetry_dropouts=None,
            json_path="",
            csv_path="",
        )
        run_report = smoke.run_smoke(smoke_args)
        worker.join()
        if run_report.acceptance_passed:
            valid_runs += 1

        basename = f"{mode.replace('-', '_')}_run_{run_idx:02d}"
        run_json_path = os.path.join(output_dir, f"{basename}.json")
        run_csv_path = os.path.join(output_dir, f"{basename}.csv")
        write_json(run_json_path, run_report.to_dict())
        smoke.maybe_write_csv(run_csv_path, run_report)
        reports.append(
            {
                "run_index": run_idx,
                "json_path": run_json_path,
                "csv_path": run_csv_path,
                "acceptance_passed": run_report.acceptance_passed,
                "ordered_gate_passes": run_report.ordered_gate_passes,
                "completion_time_s": run_report.completion_time_s,
                "replay_metrics": replay_metrics_holder.get("metrics", {}),
            }
        )

    return {
        "mode": mode,
        "policy_callable": policy_callable,
        "runs": runs,
        "valid_runs": valid_runs,
        "valid_rate": (valid_runs / runs) if runs else 0.0,
        "reports": reports,
    }


def compare_modes(baseline: dict, policy: dict) -> dict:
    baseline_rate = float(baseline.get("valid_rate", 0.0))
    policy_rate = float(policy.get("valid_rate", 0.0))
    non_inferior = policy_rate >= baseline_rate
    return {
        "baseline_valid_rate": baseline_rate,
        "policy_valid_rate": policy_rate,
        "policy_non_inferior_to_baseline": non_inferior,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay regression runner for competition smoke")
    parser.add_argument("--input-path", required=True, help="Replay JSONL events")
    parser.add_argument("--acceptance-config", default=os.path.join("config", "sitl_competition_acceptance.json"))
    parser.add_argument("--output-dir", default=os.path.join("logs", "sitl", "replay_regression"))
    parser.add_argument("--summary-json", default=os.path.join("logs", "sitl", "replay_regression_summary.json"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--run-policy", action="store_true")
    parser.add_argument("--policy-callable", default="")
    parser.add_argument("--policy-action-json", default="[0.0, 0.0, 0.0, 0.0]")

    parser.add_argument("--smoke-endpoint", default="udpin:0.0.0.0:14540")
    parser.add_argument("--smoke-duration", type=float, default=12.0)
    parser.add_argument("--smoke-camera-host", default="0.0.0.0")
    parser.add_argument("--smoke-camera-port", type=int, default=5600)
    parser.add_argument("--telemetry-timeout-s", type=float, default=0.0)
    parser.add_argument("--telemetry-dropout-s", type=float, default=1.0)
    parser.add_argument("--camera-timeout-s", type=float, default=0.0)
    parser.add_argument("--idle-sleep-s", type=float, default=0.001)
    parser.add_argument("--heartbeat-hz", type=float, default=2.0)
    parser.add_argument("--command-hz", type=float, default=50.0)
    parser.add_argument("--max-detection-age-s", type=float, default=0.25)
    parser.add_argument("--detector-min-area-px", type=float, default=1200.0)
    parser.add_argument("--detector-max-aspect-error", type=float, default=0.5)
    parser.add_argument("--detector-min-fill-ratio", type=float, default=0.15)
    parser.add_argument("--target-gate-count", type=int, default=1)
    parser.add_argument("--require-telemetry", action="store_true")
    parser.add_argument("--require-camera", action="store_true")
    parser.add_argument("--min-gate-passes", type=int, default=None)

    parser.add_argument("--replay-mavlink-host", default="127.0.0.1")
    parser.add_argument("--replay-mavlink-port", type=int, default=14540)
    parser.add_argument("--replay-camera-host", default="127.0.0.1")
    parser.add_argument("--replay-camera-port", type=int, default=5600)
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--replay-start-delay-s", type=float, default=0.25)
    parser.add_argument("--loss-rate", type=float, default=0.0)
    parser.add_argument("--reorder-rate", type=float, default=0.0)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("runs must be positive")
    if args.run_policy and not args.policy_callable:
        raise ValueError("--run-policy requires --policy-callable")

    os.makedirs(args.output_dir, exist_ok=True)
    events = replay.load_events(args.input_path)
    baseline = run_single_mode(
        mode="visual-servo",
        policy_callable="",
        runs=args.runs,
        events=events,
        args=args,
        output_dir=args.output_dir,
    )
    output = {
        "input_path": args.input_path,
        "acceptance_config": os.path.abspath(args.acceptance_config),
        "runs": args.runs,
        "impairments": {
            "loss_rate": args.loss_rate,
            "reorder_rate": args.reorder_rate,
            "latency_ms": args.latency_ms,
            "jitter_ms": args.jitter_ms,
        },
        "baseline": baseline,
    }
    if args.run_policy:
        policy = run_single_mode(
            mode="policy",
            policy_callable=args.policy_callable,
            runs=args.runs,
            events=events,
            args=args,
            output_dir=args.output_dir,
        )
        output["policy"] = policy
        output["comparison"] = compare_modes(baseline, policy)

    write_json(args.summary_json, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
