#!/usr/bin/env python3
"""Run repeated official policy validation with MAVLink 31000 in-place resets."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_acceptance(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _official_finish_time_ns(smoke: dict) -> int:
    value = smoke.get("official_race_finish_time_ns")
    if value is None:
        value = (smoke.get("race_status") or {}).get("race_finish_time_ns", -1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _clean_gate_progress_passed(
    *, smoke: dict, official_index: int, minimum_index: int, child_exit_code: int
) -> bool:
    """Require official progress and a clean accepted child run."""
    return (
        child_exit_code == 0
        and official_index >= minimum_index
        and smoke.get("acceptance_passed") is True
        and smoke.get("crash_detected") is not True
        and smoke.get("invalid_run") is not True
    )


def run_attempt(args: argparse.Namespace, attempt: int) -> dict:
    tag = f"{args.tag}_attempt_{attempt:03d}"
    summary_path = ROOT / "logs" / "sitl" / f"official_policy_validation_{tag}.json"
    smoke_path = ROOT / "logs" / "sitl" / f"competition_smoke_policy_{tag}.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_official_gate1_validation.py"),
        "--endpoint",
        args.endpoint,
        "--host",
        args.host,
        "--mavlink-port",
        str(args.mavlink_port),
        "--camera-port",
        str(args.camera_port),
        "--camera-host",
        args.camera_host,
        "--probe-duration",
        str(args.probe_duration),
        "--smoke-duration",
        str(args.smoke_duration),
        "--command-hz",
        str(args.command_hz),
        "--acceptance-config",
        str(ROOT / args.acceptance_config),
        "--target-gate-count",
        str(args.target_gate_count),
        "--min-gate-passes",
        str(args.target_gate_count),
        "--camera-max-packets-per-loop",
        str(args.camera_max_packets_per_loop),
        "--post-reset-sleep-s",
        str(args.post_reset_sleep_s),
        "--control-mode",
        args.control_mode,
        "--policy-callable",
        args.policy_callable,
        "--policy-state-hz",
        str(args.policy_state_hz),
        "--policy-action-json",
        args.policy_action_json,
        "--policy-stop-before-gate-index",
        str(args.policy_stop_before_gate_index),
        "--policy-stop-forward-m",
        str(args.policy_stop_forward_m),
        "--stop-after-official-gate-index",
        str(args.stop_after_official_gate_index),
        "--summary-json-path",
        str(summary_path),
        "--probe-json-path",
        str(ROOT / "logs" / "sitl" / f"stream_probe_policy_{tag}.json"),
        "--smoke-json-path",
        str(smoke_path),
        "--smoke-csv-path",
        str(ROOT / "logs" / "sitl" / f"competition_smoke_policy_{tag}.csv"),
        "--debug-frame-dir",
        str(ROOT / "logs" / "sitl" / f"full_policy_{tag}_frames"),
    ]
    if args.send_sim_reset:
        cmd.append("--send-sim-reset")
    if args.policy_ready_reset:
        cmd.append("--policy-ready-reset")
    if args.require_official_race_progress:
        cmd.append("--require-official-race-progress")
    if args.policy_race_phase_observation:
        cmd.extend(
            [
                "--policy-race-phase-observation",
                "--policy-race-phase-denominator",
                str(args.policy_race_phase_denominator),
            ]
        )
    if args.policy_phase_adapter_observation:
        cmd.append("--policy-phase-adapter-observation")
    if args.policy_gate_progress_adapter_observation:
        cmd.extend(
            [
                "--policy-gate-progress-adapter-observation",
                "--policy-race-phase-denominator",
                str(args.policy_race_phase_denominator),
            ]
        )
    if args.policy_gate_phase_onehot_adapter_observation:
        cmd.extend(
            [
                "--policy-gate-phase-onehot-adapter-observation",
                "--policy-race-phase-denominator",
                str(args.policy_race_phase_denominator),
            ]
        )
    if args.policy_hybrid_prefix_confidence_observation:
        cmd.append("--policy-hybrid-prefix-confidence-observation")

    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    payload: dict = {"attempt": attempt, "exit_code": proc.returncode}
    if summary_path.exists():
        payload.update(json.loads(summary_path.read_text(encoding="utf-8")))
    official_index = 0
    smoke = payload.get("smoke") or {}
    if smoke:
        official_index = int(
            smoke.get("official_active_gate_index")
            or (smoke.get("race_status") or {}).get("active_gate_index")
            or 0
        )
    payload["official_active_gate_index"] = official_index
    payload["official_race_finish_time_ns"] = _official_finish_time_ns(smoke)
    payload["gate_progress_passed"] = _clean_gate_progress_passed(
        smoke=smoke,
        official_index=official_index,
        minimum_index=args.min_official_gate_index,
        child_exit_code=proc.returncode,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-repeats", type=int, default=0)
    parser.add_argument(
        "--min-valid-runs",
        type=int,
        default=0,
        help="Required successful attempts; defaults to the acceptance config, capped at n-repeats",
    )
    parser.add_argument("--min-official-gate-index", type=int, default=1)
    parser.add_argument("--tag", default="sitl_plant_policy")
    parser.add_argument("--acceptance-config", default="config/sitl_competition_acceptance.json")
    parser.add_argument("--send-sim-reset", action="store_true")
    parser.add_argument("--policy-ready-reset", action="store_true")
    parser.add_argument("--post-reset-sleep-s", type=float, default=5.0)
    parser.add_argument("--require-official-race-progress", action="store_true")
    parser.add_argument("--control-mode", choices=["policy", "policy-attitude"], default="policy-attitude")
    parser.add_argument("--policy-callable", default="scripts/policy_callable_checkpoint.py:infer")
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
    parser.add_argument("--stop-after-official-finish", action="store_true")
    parser.add_argument("--endpoint", default="udpin:0.0.0.0:14550")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--mavlink-port", type=int, default=14550)
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument("--camera-host", default="0.0.0.0")
    parser.add_argument("--probe-duration", type=float, default=8.0)
    parser.add_argument("--smoke-duration", type=float, default=45.0)
    parser.add_argument("--command-hz", type=float, default=60.0)
    parser.add_argument("--target-gate-count", type=int, default=1)
    parser.add_argument("--camera-max-packets-per-loop", type=int, default=512)
    args = parser.parse_args()

    if args.target_gate_count <= 0:
        parser.error("--target-gate-count must be positive")
    if args.min_official_gate_index < args.target_gate_count:
        parser.error("--min-official-gate-index must be >= --target-gate-count")
    if args.policy_race_phase_denominator <= 0:
        parser.error("--policy-race-phase-denominator must be positive")
    if not math.isfinite(args.policy_state_hz) or args.policy_state_hz < 0.0:
        parser.error("--policy-state-hz must be finite and nonnegative")
    if args.stop_after_official_gate_index < -1:
        parser.error("--stop-after-official-gate-index must be -1 or nonnegative")
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
    os.environ["PUFFER_POLICY_RACE_PHASE_DENOMINATOR"] = str(
        args.policy_race_phase_denominator
    )

    if not os.getenv("PUFFER_POLICY_CHECKPOINT_PATH", "").strip():
        print("PUFFER_POLICY_CHECKPOINT_PATH must be set", file=sys.stderr)
        return 2

    acceptance = _load_acceptance(ROOT / args.acceptance_config)
    promotion = acceptance.get("promotion", {})
    n_repeats = args.n_repeats or int(promotion.get("n_repeats", 30))
    configured_min_valid = int(
        promotion.get("normal_min_valid_runs", max(1, n_repeats - 1))
    )
    min_valid = args.min_valid_runs or min(configured_min_valid, n_repeats)
    if min_valid < 1 or min_valid > n_repeats:
        parser.error("--min-valid-runs must be between 1 and n-repeats")

    attempts: list[dict] = []
    passes = 0
    stopped_after_official_finish = False
    started = time.time()
    for attempt in range(1, n_repeats + 1):
        print(f"==> policy validation attempt {attempt}/{n_repeats}", file=sys.stderr)
        result = run_attempt(args, attempt)
        attempts.append(result)
        if result.get("gate_progress_passed"):
            passes += 1
        if (
            args.stop_after_official_finish
            and int(result.get("official_race_finish_time_ns", -1)) >= 0
        ):
            stopped_after_official_finish = True
            break

    aggregate = {
        "tag": args.tag,
        "control_mode": args.control_mode,
        "policy_callable": args.policy_callable,
        "checkpoint_path": os.getenv("PUFFER_POLICY_CHECKPOINT_PATH"),
        "checkpoint_layout_precision_bytes": os.getenv(
            "PUFFER_POLICY_LAYOUT_PRECISION_BYTES", "2"
        ),
        "policy_native_bf16": os.getenv("PUFFER_POLICY_NATIVE_BF16", "0"),
        "base_checkpoint_path": os.getenv("PUFFER_POLICY_BASE_CHECKPOINT_PATH"),
        "residual_checkpoint_path": os.getenv("PUFFER_POLICY_RESIDUAL_CHECKPOINT_PATH"),
        "final_checkpoint_path": os.getenv("PUFFER_POLICY_FINAL_CHECKPOINT_PATH"),
        "policy_race_phase_observation": args.policy_race_phase_observation,
        "policy_race_phase_denominator": args.policy_race_phase_denominator,
        "policy_phase_adapter_observation": args.policy_phase_adapter_observation,
        "policy_gate_progress_adapter_observation": (
            args.policy_gate_progress_adapter_observation
        ),
        "policy_gate_phase_onehot_adapter_observation": (
            args.policy_gate_phase_onehot_adapter_observation
        ),
        "policy_hybrid_prefix_confidence_observation": (
            args.policy_hybrid_prefix_confidence_observation
        ),
        "policy_action_bias": {
            "gate_index": os.getenv("PUFFER_POLICY_ACTION_BIAS_GATE_INDEX"),
            "progress_index": os.getenv("PUFFER_POLICY_ACTION_BIAS_PROGRESS_INDEX"),
            "denominator": os.getenv("PUFFER_POLICY_ACTION_BIAS_DENOMINATOR"),
            "pitch": os.getenv("PUFFER_POLICY_ACTION_BIAS_PITCH"),
            "roll": os.getenv("PUFFER_POLICY_ACTION_BIAS_ROLL"),
            "thrust": os.getenv("PUFFER_POLICY_ACTION_BIAS_THRUST"),
            "yaw": os.getenv("PUFFER_POLICY_ACTION_BIAS_YAW"),
            "table_json": os.getenv("PUFFER_POLICY_ACTION_BIAS_TABLE_JSON"),
        },
        "policy_stop_before_gate_index": args.policy_stop_before_gate_index,
        "policy_stop_forward_m": args.policy_stop_forward_m,
        "n_repeats": n_repeats,
        "attempts_run": len(attempts),
        "stopped_after_official_finish": stopped_after_official_finish,
        "passes": passes,
        "pass_rate": passes / max(len(attempts), 1),
        "promotion_min_valid_runs": min_valid,
        "promotion_passed": passes >= min_valid,
        "elapsed_s": round(time.time() - started, 3),
        "attempts": attempts,
    }
    out_path = ROOT / "logs" / "sitl" / f"official_policy_validation_summary_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return 0 if aggregate["promotion_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
