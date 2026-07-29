#!/usr/bin/env python3
"""Retained-parent curriculum for native-to-v3385 full-policy transfer.

The curriculum changes only simulator/perception conditions. The recurrent
policy still owns pitch, roll, yaw, and thrust, and its observation remains the
official-observable 32-value contract. This script never starts or resets the
official simulator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval_drone_race_checkpoint.py"
DEFAULT_CURRICULUM = ROOT / "config" / "full_policy_v3385_transfer_curriculum.json"


def parse_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("overrides must use SECTION.KEY=VALUE")
    key, setting = value.split("=", 1)
    key = key.strip().lstrip("-")
    setting = setting.strip()
    if not key or not setting:
        raise argparse.ArgumentTypeError("overrides must use SECTION.KEY=VALUE")
    return key, setting


def cli_overrides(settings: dict[str, str]) -> list[str]:
    result: list[str] = []
    for key, value in settings.items():
        result.extend((f"--{key}", str(value)))
    return result


def load_curriculum(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("curriculum must contain a non-empty stages list")
    names: set[str] = set()
    previous_jitter = (0.0, 0.0, 0.0)
    result: list[dict] = []
    for index, raw in enumerate(stages):
        if not isinstance(raw, dict):
            raise ValueError(f"stage {index} must be an object")
        name = str(raw.get("name", "")).strip()
        overrides = raw.get("overrides")
        if not name or name in names:
            raise ValueError(f"stage {index} has a missing or duplicate name")
        if not isinstance(overrides, dict) or not overrides:
            raise ValueError(f"stage {name} must contain overrides")
        normalized = {str(key): str(value) for key, value in overrides.items()}
        dropout_from = int(normalized.get("env.sitl-gate-obs-dropout-from-index", 99))
        if dropout_from > 1:
            raise ValueError(f"stage {name} does not model gate-1 camera loss")
        jitter = tuple(
            abs(float(normalized.get(f"env.gate-position-jitter-{axis}", 0.0)))
            for axis in "xyz"
        )
        if any(current + 1e-12 < previous for current, previous in zip(jitter, previous_jitter)):
            raise ValueError(f"stage {name} decreases gate-position jitter")
        previous_jitter = jitter
        names.add(name)
        result.append({"name": name, "overrides": normalized})
    return result


def checkpoint_inventory(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.glob("*/*.bin")}


def select_new_checkpoint(before: set[Path], checkpoint_root: Path) -> Path:
    candidates = checkpoint_inventory(checkpoint_root) - before
    if not candidates:
        raise RuntimeError(f"training produced no checkpoint under {checkpoint_root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, int(path.stem)))


def write_checkpoint_with_log_std(
    source: Path,
    destination: Path,
    log_std: float,
    *,
    input_dim: int = 32,
    hidden_dim: int = 128,
    num_actions: int = 4,
) -> list[float]:
    if not math.isfinite(log_std):
        raise ValueError("exploration log-std must be finite")
    weights = np.fromfile(source, dtype=np.float32)
    encoder_end = (hidden_dim * input_dim + 7) & ~7
    decoder_end = (encoder_end + (num_actions + 1) * hidden_dim + 7) & ~7
    log_std_end = decoder_end + num_actions
    if log_std_end > weights.size:
        raise ValueError("checkpoint ended before the continuous-action log_std")
    before = weights[decoder_end:log_std_end].copy()
    weights[decoder_end:log_std_end] = np.float32(log_std)
    weights.tofile(destination)
    return [float(value) for value in before]


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def metric_summary(report: dict) -> dict[str, float]:
    metrics = report["metrics"]
    return {
        "episodes": float(metrics.get("env/n", 0.0)),
        "success_rate": float(metrics.get("env/success_rate", 0.0)),
        "crash_rate": float(metrics.get("env/crash", 0.0)),
        "gates_passed": float(metrics.get("env/gates_passed", 0.0)),
        "completion_time": float(metrics.get("env/completion_time", 0.0)),
        "missed_gate_rate": float(metrics.get("env/missed_gate", 0.0)),
    }


def passes(metrics: dict[str, float], success_threshold: float, crash_limit: float) -> bool:
    return metrics["success_rate"] >= success_threshold and metrics["crash_rate"] <= crash_limit


def progress_regressed(
    parent: dict[str, float],
    child: dict[str, float],
    *,
    maximum_success_drop: float,
    maximum_gates_drop: float,
) -> bool:
    """Detect a bad child even while both policies have zero completions."""
    if child["success_rate"] < parent["success_rate"] - maximum_success_drop:
        return True
    return (
        child["success_rate"] <= 0.001
        and parent["success_rate"] <= 0.001
        and child["gates_passed"] < parent["gates_passed"] - maximum_gates_drop
    )


def run_command(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def evaluate(
    *,
    python: str,
    checkpoint: Path,
    env_name: str,
    stage_name: str,
    settings: dict[str, str],
    output_dir: Path,
    attempt_id: str,
    episodes: int,
    horizon: int,
    max_rollouts: int,
    dry_run: bool,
) -> tuple[dict[str, float] | None, Path, list[str]]:
    label = f"transfer_{attempt_id}_{stage_name}"
    json_path = output_dir / f"{label}.json"
    csv_path = output_dir / f"{label}.csv"
    command = [
        python,
        str(EVAL_SCRIPT),
        str(checkpoint),
        "--env-name",
        env_name,
        "--eval-episodes",
        str(episodes),
        "--require-exact-episodes",
        "--horizon",
        str(horizon),
        "--max-rollouts",
        str(max_rollouts),
        "--json-path",
        str(json_path),
        "--csv-path",
        str(csv_path),
        "--label",
        label,
        *cli_overrides(settings),
    ]
    run_command(command, dry_run=dry_run)
    if dry_run:
        return None, json_path, command
    metrics = metric_summary(json.loads(json_path.read_text(encoding="utf-8")))
    if metrics["episodes"] != float(episodes):
        raise RuntimeError(f"exact evaluation expected {episodes} episodes, got {metrics['episodes']}")
    return metrics, json_path, command


def train(
    *,
    python: str,
    source: Path,
    env_name: str,
    settings: dict[str, str],
    checkpoint_root: Path,
    timesteps: int,
    learning_rate: float,
    reward_scale: float,
    reward_clip: float,
    exploration_log_std: float | None,
    dry_run: bool,
) -> tuple[Path | None, float, list[str]]:
    before = checkpoint_inventory(checkpoint_root)
    prepared_source = source
    temporary_source: Path | None = None
    if exploration_log_std is not None and not dry_run:
        handle = tempfile.NamedTemporaryFile(
            prefix="drone_transfer_exploration_", suffix=".bin", delete=False
        )
        handle.close()
        temporary_source = Path(handle.name)
        old = write_checkpoint_with_log_std(source, temporary_source, exploration_log_std)
        prepared_source = temporary_source
        print(f"training exploration log_std: {old} -> {[exploration_log_std] * 4}", flush=True)
    command = [
        python,
        "-m",
        "pufferlib.pufferl",
        "train",
        env_name,
        "--load-model-path",
        str(prepared_source),
        "--checkpoint-interval",
        str(timesteps),
        "--eval-episodes",
        "0",
        "--train.total-timesteps",
        str(timesteps),
        "--train.learning-rate",
        f"{learning_rate:g}",
        "--train.reward-scale",
        f"{reward_scale:g}",
        "--train.reward-clip",
        f"{reward_clip:g}",
        *cli_overrides(settings),
    ]
    started = time.monotonic()
    try:
        run_command(command, dry_run=dry_run)
    finally:
        if temporary_source is not None:
            temporary_source.unlink(missing_ok=True)
    elapsed = time.monotonic() - started
    if dry_run:
        return None, elapsed, command
    return select_new_checkpoint(before, checkpoint_root), elapsed, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-checkpoint", type=Path)
    source_group.add_argument("--resume-state", type=Path)
    parser.add_argument("--curriculum", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument("--env-name", default="drone_race_full_policy_stage_d_gate4")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--crash-limit", type=float, default=0.10)
    parser.add_argument("--train-timesteps", type=int, default=1_000_000)
    parser.add_argument("--min-train-timesteps", type=int, default=125_000)
    parser.add_argument("--max-finetunes-per-stage", type=int, default=4)
    parser.add_argument("--max-child-success-drop", type=float, default=0.05)
    parser.add_argument("--max-child-gates-drop", type=float, default=0.10)
    # A near-deterministic Gaussian (log_std=-8) made even tiny mean updates
    # produce KL values above 1.0 and destroyed already-promoted prefixes.
    # Keep PPO in the source policy's stable exploration regime and use a
    # conservative learning rate; deterministic exact screens still decide
    # whether any child is retained.
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--reward-scale", type=float, default=0.005)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--exploration-log-std", type=float, default=-4.0)
    parser.add_argument("--eval-episodes", type=int, default=1024)
    parser.add_argument("--promotion-episodes", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--max-rollouts", type=int, default=96)
    parser.add_argument(
        "--set",
        dest="overrides",
        type=parse_override,
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.success_threshold <= 1.0 or not 0.0 <= args.crash_limit <= 1.0:
        parser.error("success/crash thresholds must be in [0, 1]")
    if args.train_timesteps < 1 or args.min_train_timesteps < 1:
        parser.error("training step budgets must be positive")
    if args.min_train_timesteps > args.train_timesteps:
        parser.error("minimum training steps cannot exceed the initial budget")
    if args.max_finetunes_per_stage < 1:
        parser.error("--max-finetunes-per-stage must be positive")
    if args.eval_episodes < 1 or args.promotion_episodes < 1:
        parser.error("evaluation episode counts must be positive")

    curriculum_path = args.curriculum.expanduser().resolve()
    try:
        stages = load_curriculum(curriculum_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    global_settings = dict(args.overrides)
    output_dir = ROOT / "logs" / args.env_name
    checkpoint_root = ROOT / "checkpoints" / args.env_name
    run_id = str(int(time.time() * 1000))

    if args.resume_state is not None:
        state_path = args.resume_state.expanduser().resolve()
        if not state_path.is_file():
            parser.error(f"resume state does not exist: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("env_name") != args.env_name:
            parser.error("--env-name does not match resume state")
        if state.get("stages") != stages or state.get("global_settings") != global_settings:
            parser.error("curriculum or --set overrides do not match resume state")
        retained = Path(state["retained_checkpoint"]).expanduser().resolve()
        start_stage = int(state.get("next_stage_index", 0))
        state["status"] = "running"
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
    else:
        retained = args.source_checkpoint.expanduser().resolve()
        state_path = (
            args.state_path.expanduser().resolve()
            if args.state_path is not None
            else (output_dir / "v3385_transfer_curriculum_state.json").resolve()
        )
        start_stage = 0
        state = {
            "status": "running",
            "env_name": args.env_name,
            "curriculum": str(curriculum_path),
            "stages": stages,
            "global_settings": global_settings,
            "retained_checkpoint": str(retained),
            "next_stage_index": 0,
            "attempts": [],
            "resume_count": 0,
        }
    if not args.dry_run and not retained.is_file():
        parser.error(f"checkpoint does not exist: {retained}")
    write_state(state_path, state)

    for stage_index in range(start_stage, len(stages)):
        stage = stages[stage_index]
        settings = {**global_settings, **stage["overrides"]}
        attempt_id = f"{run_id}_s{stage_index:02d}_a{len(state['attempts']):04d}"
        baseline, report, command = evaluate(
            python=sys.executable,
            checkpoint=retained,
            env_name=args.env_name,
            stage_name=stage["name"],
            settings=settings,
            output_dir=output_dir,
            attempt_id=attempt_id,
            episodes=args.eval_episodes,
            horizon=args.horizon,
            max_rollouts=args.max_rollouts,
            dry_run=args.dry_run,
        )
        screen = {
            "stage_index": stage_index,
            "stage": stage["name"],
            "mode": "screen",
            "checkpoint": str(retained),
            "report": str(report),
            "metrics": baseline,
            "command": command,
        }
        state["attempts"].append(screen)
        write_state(state_path, state)
        if args.dry_run:
            continue
        if passes(baseline, args.success_threshold, args.crash_limit):
            screen["accepted"] = True
            state["next_stage_index"] = stage_index + 1
            write_state(state_path, state)
            continue

        parent = retained
        parent_metrics = baseline
        timesteps = args.train_timesteps
        accepted = False
        for fine_tune_index in range(1, args.max_finetunes_per_stage + 1):
            child, train_elapsed, train_command = train(
                python=sys.executable,
                source=parent,
                env_name=args.env_name,
                settings=settings,
                checkpoint_root=checkpoint_root,
                timesteps=timesteps,
                learning_rate=args.learning_rate,
                reward_scale=args.reward_scale,
                reward_clip=args.reward_clip,
                exploration_log_std=args.exploration_log_std,
                dry_run=False,
            )
            attempt_id = f"{run_id}_s{stage_index:02d}_a{len(state['attempts']):04d}"
            metrics, child_report, eval_command = evaluate(
                python=sys.executable,
                checkpoint=child,
                env_name=args.env_name,
                stage_name=stage["name"],
                settings=settings,
                output_dir=output_dir,
                attempt_id=attempt_id,
                episodes=args.eval_episodes,
                horizon=args.horizon,
                max_rollouts=args.max_rollouts,
                dry_run=False,
            )
            child_accepted = passes(metrics, args.success_threshold, args.crash_limit)
            attempt = {
                "stage_index": stage_index,
                "stage": stage["name"],
                "mode": "fine_tune",
                "fine_tune_index": fine_tune_index,
                "parent_checkpoint": str(parent),
                "checkpoint": str(child),
                "report": str(child_report),
                "metrics": metrics,
                "accepted": child_accepted,
                "requested_training_timesteps": timesteps,
                "training_steps": int(child.stem),
                "training_elapsed_seconds": round(train_elapsed, 6),
                "training_effective_sps": int(child.stem) / max(train_elapsed, 1e-9),
                "train_command": train_command,
                "eval_command": eval_command,
            }
            state["attempts"].append(attempt)
            write_state(state_path, state)
            if child_accepted:
                retained = child
                state["retained_checkpoint"] = str(retained)
                state["next_stage_index"] = stage_index + 1
                write_state(state_path, state)
                accepted = True
                break
            if progress_regressed(
                parent_metrics,
                metrics,
                maximum_success_drop=args.max_child_success_drop,
                maximum_gates_drop=args.max_child_gates_drop,
            ):
                attempt["success_regression_stop"] = True
                smaller = timesteps // 2
                if smaller < args.min_train_timesteps:
                    attempt["adaptive_backoff_exhausted"] = True
                    write_state(state_path, state)
                    break
                timesteps = smaller
                parent = retained
                parent_metrics = baseline
                attempt["next_training_timesteps"] = timesteps
                write_state(state_path, state)
            else:
                parent = child
                parent_metrics = metrics
        if not accepted:
            state["status"] = "blocked"
            state["blocked_stage_index"] = stage_index
            state["blocked_stage"] = stage["name"]
            write_state(state_path, state)
            print(f"Transfer stage {stage['name']} did not meet thresholds; retained {retained}.")
            return 3

    if args.dry_run:
        state["status"] = "dry_run"
        write_state(state_path, state)
        return 0

    target = stages[-1]
    promotion_settings = {**global_settings, **target["overrides"]}
    promotion, report, command = evaluate(
        python=sys.executable,
        checkpoint=retained,
        env_name=args.env_name,
        stage_name=f"{target['name']}_promotion",
        settings=promotion_settings,
        output_dir=output_dir,
        attempt_id=f"{run_id}_promotion",
        episodes=args.promotion_episodes,
        horizon=args.horizon,
        max_rollouts=args.max_rollouts,
        dry_run=False,
    )
    promotion_passed = passes(promotion, args.success_threshold, args.crash_limit)
    state["promotion"] = {
        "checkpoint": str(retained),
        "report": str(report),
        "metrics": promotion,
        "passed": promotion_passed,
        "command": command,
    }
    state["status"] = "complete" if promotion_passed else "promotion_failed"
    write_state(state_path, state)
    print(f"Retained checkpoint: {retained}")
    print(f"State: {state_path}")
    return 0 if promotion_passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
