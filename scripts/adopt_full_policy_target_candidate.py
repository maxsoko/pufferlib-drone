#!/usr/bin/env python3
"""Atomically adopt a candidate backed by passing exact promotion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from finish_full_policy_native_curriculum import promotion_metrics, promotion_passes


ROOT = Path(__file__).resolve().parents[1]
ENV_NAME = "drone_race_full_policy_stage_d_gate4"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def override_map(values: list[str]) -> dict[str, str]:
    if len(values) % 2:
        raise ValueError("evaluation report has an incomplete config override")
    return dict(zip(values[::2], values[1::2]))


def validate_evidence(
    report: dict,
    checkpoint: Path,
    *,
    target_radius: float,
    required_episodes: int,
    success_threshold: float,
    crash_limit: float,
) -> dict[str, float]:
    metadata = report.get("metadata", {})
    reported_checkpoint = Path(str(metadata.get("weights", ""))).expanduser()
    if not reported_checkpoint.is_absolute():
        reported_checkpoint = ROOT / reported_checkpoint
    if reported_checkpoint.resolve() != checkpoint.resolve():
        raise ValueError("evaluation report checkpoint does not match candidate")
    if metadata.get("env_name") != ENV_NAME:
        raise ValueError("evaluation report uses the wrong environment")
    if metadata.get("action_mode") != "deterministic_mean":
        raise ValueError("promotion evidence must use deterministic mean actions")
    if metadata.get("exact_episodes_required") is not True:
        raise ValueError("promotion evidence did not require exact episodes")
    if int(metadata.get("eval_episodes_target", 0)) != required_episodes:
        raise ValueError("promotion evidence has the wrong episode target")

    overrides = override_map(list(metadata.get("config_overrides", [])))
    expected = f"{target_radius:g}"
    for key in ("--env.gate-radius", "--env.gate2-radius", "--env.gate3-radius"):
        if overrides.get(key) != expected:
            raise ValueError(f"promotion evidence does not lock {key} to {expected}")

    metrics = promotion_metrics(report)
    if not promotion_passes(
        metrics,
        success_threshold=success_threshold,
        crash_limit=crash_limit,
        required_episodes=required_episodes,
    ):
        raise ValueError("candidate does not pass native promotion thresholds")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-radius", type=float, default=0.75)
    parser.add_argument("--required-episodes", type=int, default=4096)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--crash-limit", type=float, default=0.10)
    args = parser.parse_args()

    state_path = args.state.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not checkpoint.is_file() or not report_path.is_file() or not state_path.is_file():
        parser.error("state, checkpoint, and report must all exist")
    state = read_json(state_path)
    if state.get("env_name") != ENV_NAME or int(state.get("gate_index", -1)) != 3:
        parser.error("state is not the Gate 3 full-policy curriculum")
    if abs(float(state.get("target_radius", -1.0)) - args.target_radius) > 1e-9:
        parser.error("state target radius does not match adoption radius")
    try:
        metrics = validate_evidence(
            read_json(report_path),
            checkpoint,
            target_radius=args.target_radius,
            required_episodes=args.required_episodes,
            success_threshold=args.success_threshold,
            crash_limit=args.crash_limit,
        )
    except ValueError as exc:
        parser.error(str(exc))

    state.setdefault("attempts", []).append({
        "accepted": True,
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "mode": "targeted_phase_feature_correction",
        "radius": args.target_radius,
        "report": str(report_path),
    })
    state["retained_checkpoint"] = str(checkpoint)
    state["retained_radius"] = args.target_radius
    state["status"] = "complete"
    state["completed_at_ms"] = int(time.time() * 1000)
    state.pop("blocked_radius", None)
    write_json(state_path, state)
    print(f"Adopted checkpoint: {checkpoint}")
    print(f"State: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
