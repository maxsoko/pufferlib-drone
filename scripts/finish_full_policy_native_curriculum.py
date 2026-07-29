#!/usr/bin/env python3
"""Promote a completed true-aperture full-policy radius curriculum.

Wait for the retained-parent gate-3 state to reach 0.75 m, then run the exact
4096-episode deterministic native promotion evaluation with every gate fixed at
0.75 m. This script never launches the official simulator or another trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EVAL_RUNNER = ROOT / "scripts" / "eval_drone_race_checkpoint.py"
ENV_NAME = "drone_race_full_policy_stage_d_gate4"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def accepted_target_parent(state: dict, target_radius: float) -> Path | None:
    if state.get("status") != "complete":
        return None
    radius = state.get("retained_radius")
    checkpoint = state.get("retained_checkpoint")
    if (
        radius is None
        or not checkpoint
        or abs(float(radius) - target_radius) > 1e-9
    ):
        return None
    parent = Path(checkpoint).expanduser().resolve()
    return parent if parent.is_file() else None


def promotion_metrics(report: dict) -> dict[str, float]:
    metrics = report["metrics"]
    return {
        "success_rate": float(metrics.get("env/success_rate", 0.0)),
        "crash_rate": float(metrics.get("env/crash", 0.0)),
        "gates_passed": float(metrics.get("env/gates_passed", 0.0)),
        "completion_time": float(metrics.get("env/completion_time", 0.0)),
        "episodes": float(metrics.get("env/n", 0.0)),
        "out_of_order_rate": float(metrics.get("env/out_of_order", 0.0)),
    }


def promotion_passes(
    metrics: dict[str, float],
    *,
    success_threshold: float,
    crash_limit: float,
    required_episodes: int,
) -> bool:
    return (
        metrics["success_rate"] >= success_threshold
        and metrics["crash_rate"] <= crash_limit
        and metrics.get("out_of_order_rate", 0.0) == 0.0
        and metrics["episodes"] == required_episodes
    )


def build_promotion_command(
    *,
    checkpoint: Path,
    target_radius: float,
    promotion_episodes: int,
    promotion_json: Path,
    promotion_csv: Path,
) -> list[str]:
    radius_token = f"{target_radius:g}".replace(".", "p")
    return [
        sys.executable,
        str(EVAL_RUNNER),
        str(checkpoint),
        "--env-name", ENV_NAME,
        "--eval-episodes", str(promotion_episodes),
        "--horizon", "32",
        "--max-rollouts", "160",
        "--json-path", str(promotion_json),
        "--csv-path", str(promotion_csv),
        "--label", f"full_policy_true_aperture_r{radius_token}_promotion_4096",
        "--env.gate-radius", f"{target_radius:g}",
        "--env.gate2-radius", f"{target_radius:g}",
        "--env.gate3-radius", f"{target_radius:g}",
        "--require-exact-episodes",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate3-state", type=Path, required=True)
    parser.add_argument("--pipeline-state", type=Path)
    parser.add_argument("--target-radius", type=float, default=0.75)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--promotion-episodes", type=int, default=4096)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--crash-limit", type=float, default=0.10)
    args = parser.parse_args()
    if abs(args.target_radius - 0.75) > 1e-9:
        parser.error("true-aperture promotion requires --target-radius 0.75")
    if args.promotion_episodes != 4096:
        parser.error("native promotion requires exactly 4096 episodes")
    if args.success_threshold < 0.90:
        parser.error("native promotion success threshold cannot be below 0.90")
    if args.crash_limit > 0.10:
        parser.error("native promotion crash limit cannot exceed 0.10")

    output_dir = ROOT / "logs" / ENV_NAME
    gate3_state_path = args.gate3_state.expanduser().resolve()
    pipeline_state_path = (
        args.pipeline_state or output_dir / "native_true_aperture_pipeline_state.json"
    ).expanduser().resolve()
    radius_token = f"{args.target_radius:g}".replace(".", "p")
    promotion_json = output_dir / (
        f"full_policy_true_aperture_r{radius_token}_promotion_4096.json"
    )
    promotion_csv = output_dir / (
        f"full_policy_true_aperture_r{radius_token}_promotion_4096.csv"
    )

    pipeline = {
        "status": "waiting_for_gate3",
        "gate3_state": str(gate3_state_path),
        "target_radius": args.target_radius,
        "promotion_episodes": args.promotion_episodes,
    }
    write_json(pipeline_state_path, pipeline)

    started = time.monotonic()
    gate3_parent = None
    while gate3_parent is None:
        if not gate3_state_path.is_file():
            pipeline["last_error"] = "gate3 state missing"
        else:
            gate3_state = read_json(gate3_state_path)
            pipeline["gate3_status"] = gate3_state.get("status")
            pipeline["gate3_retained_radius"] = gate3_state.get("retained_radius")
            pipeline["gate3_retained_checkpoint"] = gate3_state.get("retained_checkpoint")
            gate3_parent = accepted_target_parent(gate3_state, args.target_radius)
            if gate3_state.get("status") in {"blocked", "invalid_source"}:
                pipeline["status"] = "gate3_blocked"
                pipeline["blocked_radius"] = gate3_state.get("blocked_radius")
                write_json(pipeline_state_path, pipeline)
                return 3
        write_json(pipeline_state_path, pipeline)
        if gate3_parent is not None:
            break
        if args.wait_timeout_seconds > 0 and time.monotonic() - started >= args.wait_timeout_seconds:
            pipeline["status"] = "wait_timeout"
            write_json(pipeline_state_path, pipeline)
            return 4
        time.sleep(max(args.poll_seconds, 0.1))

    pipeline["status"] = "promotion_eval"
    pipeline["target_checkpoint"] = str(gate3_parent)
    write_json(pipeline_state_path, pipeline)
    eval_command = build_promotion_command(
        checkpoint=gate3_parent,
        target_radius=args.target_radius,
        promotion_episodes=args.promotion_episodes,
        promotion_json=promotion_json,
        promotion_csv=promotion_csv,
    )
    pipeline["promotion_command"] = eval_command
    write_json(pipeline_state_path, pipeline)
    subprocess.run(eval_command, cwd=ROOT, check=True)
    metrics = promotion_metrics(read_json(promotion_json))
    pipeline["promotion_metrics"] = metrics
    pipeline["promotion_report"] = str(promotion_json)
    pipeline["status"] = (
        "native_promotion_passed"
        if promotion_passes(
            metrics,
            success_threshold=args.success_threshold,
            crash_limit=args.crash_limit,
            required_episodes=args.promotion_episodes,
        )
        else "native_promotion_failed"
    )
    write_json(pipeline_state_path, pipeline)
    return 0 if pipeline["status"] == "native_promotion_passed" else 7


if __name__ == "__main__":
    raise SystemExit(main())
