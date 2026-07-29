#!/usr/bin/env python3
"""Anneal one training-only gate radius while retaining the last passing policy.

This orchestrates native PufferLib training and deterministic development
evaluation only. It never launches or resets the official simulator.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
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


def radius_schedule(initial: float, target: float, step: float) -> list[float]:
    """Return an inclusive, descending schedule without float accumulation."""
    current = Decimal(str(initial))
    target_decimal = Decimal(str(target))
    step_decimal = Decimal(str(step))
    if step_decimal <= 0:
        raise ValueError("step must be positive")
    if current < target_decimal:
        raise ValueError("initial radius must be greater than or equal to target")

    radii = [current]
    while current > target_decimal:
        current = max(target_decimal, current - step_decimal)
        radii.append(current)
    return [float(radius) for radius in radii]


def refined_radius_schedule(
    previous_radius: float,
    failed_radius: float,
    target_radius: float,
    minimum_step: float,
) -> tuple[list[float], float] | None:
    """Return a finer tail after a failed radius, or None at the step floor.

    The previous radius is the last accepted aperture. The failed radius and
    all smaller pending radii are replaced by a half-step schedule beginning
    between the accepted and failed apertures. Failed children are never used
    as the retained parent.
    """
    previous = Decimal(str(previous_radius))
    failed = Decimal(str(failed_radius))
    target = Decimal(str(target_radius))
    minimum = Decimal(str(minimum_step))
    failed_step = previous - failed
    if minimum <= 0:
        raise ValueError("minimum step must be positive")
    if failed_step <= 0:
        raise ValueError("failed radius must be below the previous radius")

    refined_step = failed_step / Decimal("2")
    if refined_step < minimum:
        return None
    schedule = radius_schedule(float(previous), float(target), float(refined_step))
    return schedule[1:], float(refined_step)


def parse_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("overrides must use SECTION.KEY=VALUE")
    key, setting = value.split("=", 1)
    key = key.strip().lstrip("-")
    if not key or not setting.strip():
        raise argparse.ArgumentTypeError("overrides must use SECTION.KEY=VALUE")
    return key, setting.strip()


def cli_overrides(settings: dict[str, str]) -> list[str]:
    result: list[str] = []
    for key, value in settings.items():
        result.extend((f"--{key}", str(value)))
    return result


def checkpoint_inventory(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.glob("*/*.bin")}


def select_new_checkpoint(before: set[Path], checkpoint_root: Path) -> Path:
    new_files = checkpoint_inventory(checkpoint_root) - before
    if not new_files:
        raise RuntimeError(f"training produced no checkpoint under {checkpoint_root}")

    def sort_key(path: Path) -> tuple[int, int, int]:
        try:
            step = int(path.stem)
        except ValueError:
            step = -1
        return path.stat().st_mtime_ns, step, path.stat().st_size

    return max(new_files, key=sort_key)


def write_checkpoint_with_log_std(
    source: Path,
    destination: Path,
    log_std: float,
    *,
    input_dim: int = 32,
    hidden_dim: int = 128,
    num_actions: int = 4,
) -> list[float]:
    """Copy a raw policy checkpoint with only Gaussian log-std replaced."""
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    weights.tofile(destination)
    return [float(value) for value in before]


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def metric_summary(report: dict) -> dict[str, float]:
    metrics = report["metrics"]
    return {
        "success_rate": float(metrics.get("env/success_rate", 0.0)),
        "crash_rate": float(metrics.get("env/crash", 0.0)),
        "gates_passed": float(metrics.get("env/gates_passed", 0.0)),
        "completion_time": float(metrics.get("env/completion_time", 0.0)),
        "closest_gate_range": float(metrics.get("env/closest_gate_range", 0.0)),
    }


def passes(metrics: dict[str, float], success_threshold: float, crash_limit: float) -> bool:
    return metrics["success_rate"] >= success_threshold and metrics["crash_rate"] <= crash_limit


def success_regressed(
    baseline_metrics: dict[str, float],
    child_metrics: dict[str, float],
    maximum_drop: float,
) -> bool:
    """Return whether a child lost too much deterministic success to chain."""
    if maximum_drop < 0.0:
        raise ValueError("maximum success drop cannot be negative")
    return (
        child_metrics["success_rate"]
        < baseline_metrics["success_rate"] - maximum_drop
    )


def radius_token(radius: float) -> str:
    return f"{radius:g}".replace("-", "m").replace(".", "p")


def evaluation_run_id(run_id: str, attempt_count: int) -> str:
    """Return an immutable report namespace for the next state attempt.

    A refined curriculum can visit the same radius more than once with a new
    parent checkpoint. Radius-only report names would overwrite the earlier
    evidence and leave old state entries pointing at a report for different
    weights. The state attempt ordinal is monotonic within one run, while a
    resumed process already receives a fresh timestamp-based ``run_id``.
    """
    if attempt_count < 0:
        raise ValueError("attempt count cannot be negative")
    return f"{run_id}_a{attempt_count:04d}"


def resume_source(state: dict) -> tuple[Path, float]:
    checkpoint = state.get("retained_checkpoint")
    radius = state.get("retained_radius")
    if not checkpoint or radius is None:
        raise ValueError("resume state has no accepted checkpoint/radius")
    return Path(checkpoint).expanduser().resolve(), float(radius)


def resume_radius_schedule(
    state: dict, target_override: float | None = None
) -> tuple[list[float], float]:
    """Recover the persisted uncompleted radius tail from an atomic state file."""
    _, retained_radius = resume_source(state)
    persisted = [float(radius) for radius in state.get("schedule", [])]
    if not persisted:
        raise ValueError("resume state has no persisted radius schedule")
    state_target = float(state.get("target_radius", persisted[-1]))
    if target_override is not None and not math.isclose(
        float(target_override), state_target, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"resume target {target_override:g} does not match persisted target "
            f"{state_target:g}"
        )
    target = state_target if target_override is None else float(target_override)

    start_radius: float | None = None
    attempts = state.get("attempts", [])
    if attempts:
        last_radius = float(attempts[-1].get("radius", retained_radius))
        if target <= last_radius < retained_radius:
            start_radius = last_radius

    refinements = state.get("schedule_refinements", [])
    if refinements:
        latest = refinements[-1]
        accepted = float(latest["accepted_radius"])
        failed = float(latest["failed_radius"])
        refined_step = float(latest["refined_step"])
        if math.isclose(retained_radius, accepted, abs_tol=1e-9):
            refined_start = accepted - refined_step
            if start_radius is None or math.isclose(start_radius, failed, abs_tol=1e-9):
                start_radius = refined_start

    retained_indices = [
        index
        for index, radius in enumerate(persisted)
        if math.isclose(radius, retained_radius, rel_tol=0.0, abs_tol=1e-9)
    ]
    if not retained_indices:
        raise ValueError("retained radius is absent from persisted schedule")
    retained_index = retained_indices[-1]
    if start_radius is None:
        tail_start = retained_index + 1
    else:
        candidates = [
            index
            for index in range(retained_index + 1, len(persisted))
            if math.isclose(
                persisted[index], start_radius, rel_tol=0.0, abs_tol=1e-9
            )
        ]
        if not candidates:
            raise ValueError("resume radius is absent from persisted schedule")
        tail_start = candidates[0]
    tail = persisted[tail_start:]
    if not tail and not math.isclose(retained_radius, target, abs_tol=1e-9):
        raise ValueError("persisted schedule has no pending radius after retained parent")
    return [retained_radius, *tail], target


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def evaluate(
    *,
    python: str,
    checkpoint: Path,
    env_name: str,
    radius: float,
    gate_index: int,
    settings: dict[str, str],
    output_dir: Path,
    run_id: str,
    eval_episodes: int,
    horizon: int,
    max_rollouts: int,
    dry_run: bool,
) -> tuple[dict[str, float] | None, Path]:
    label = f"radius_curriculum_{run_id}_gate{gate_index}_r{radius_token(radius)}"
    json_path = output_dir / f"{label}.json"
    csv_path = output_dir / f"{label}.csv"
    eval_settings = dict(settings)
    eval_settings[f"env.gate{gate_index}-radius"] = f"{radius:g}"
    command = [
        python,
        str(EVAL_SCRIPT),
        str(checkpoint),
        "--env-name",
        env_name,
        "--eval-episodes",
        str(eval_episodes),
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
        *cli_overrides(eval_settings),
    ]
    run_command(command, cwd=ROOT, dry_run=dry_run)
    if dry_run:
        return None, json_path
    return metric_summary(json.loads(json_path.read_text())), json_path


def train(
    *,
    python: str,
    source: Path,
    env_name: str,
    radius: float,
    gate_index: int,
    settings: dict[str, str],
    checkpoint_root: Path,
    timesteps: int,
    learning_rate: float,
    reward_scale: float,
    reward_clip: float,
    checkpoint_interval: int,
    trainer_eval_episodes: int,
    exploration_log_std: float | None,
    input_dim: int,
    hidden_dim: int,
    num_actions: int,
    dry_run: bool,
) -> tuple[Path | None, float]:
    train_settings = dict(settings)
    train_settings[f"env.gate{gate_index}-radius"] = f"{radius:g}"
    before = checkpoint_inventory(checkpoint_root)
    prepared_source = source
    temporary_source: Path | None = None
    if exploration_log_std is not None and not dry_run:
        handle = tempfile.NamedTemporaryFile(
            prefix="drone_train_exploration_", suffix=".bin", delete=False
        )
        handle.close()
        temporary_source = Path(handle.name)
        before_log_std = write_checkpoint_with_log_std(
            source,
            temporary_source,
            exploration_log_std,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        )
        prepared_source = temporary_source
        print(
            f"training exploration log_std: {before_log_std} -> "
            f"{[exploration_log_std] * num_actions}",
            flush=True,
        )
    command = [
        python,
        "-m",
        "pufferlib.pufferl",
        "train",
        env_name,
        "--load-model-path",
        str(prepared_source),
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--eval-episodes",
        str(trainer_eval_episodes),
        "--train.total-timesteps",
        str(timesteps),
        "--train.learning-rate",
        f"{learning_rate:g}",
        "--train.reward-scale",
        f"{reward_scale:g}",
        "--train.reward-clip",
        f"{reward_clip:g}",
        *cli_overrides(train_settings),
    ]
    started = time.monotonic()
    try:
        run_command(command, cwd=ROOT, dry_run=dry_run)
    finally:
        if temporary_source is not None:
            temporary_source.unlink(missing_ok=True)
    elapsed_seconds = time.monotonic() - started
    if dry_run:
        return None, elapsed_seconds
    return select_new_checkpoint(before, checkpoint_root), elapsed_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument(
        "--resume-state",
        type=Path,
        help="Resume from the retained checkpoint/radius in an existing state JSON.",
    )
    parser.add_argument("--env-name", default="drone_race_full_policy_stage_d_gate4")
    parser.add_argument("--gate-index", type=int, default=3)
    parser.add_argument("--initial-radius", type=float)
    parser.add_argument(
        "--target-radius",
        type=float,
        help="Final radius. On resume, defaults to and must match the persisted target.",
    )
    parser.add_argument("--radius-step", type=float, default=0.125)
    parser.add_argument(
        "--min-radius-step",
        type=float,
        help=(
            "Automatically halve a failed local radius step and retry from the "
            "last accepted parent until this positive step floor is reached."
        ),
    )
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--crash-limit", type=float, default=0.10)
    parser.add_argument(
        "--train-timesteps",
        type=int,
        default=500_000,
        help=(
            "Agent steps per bounded child. The narrow-aperture default is a "
            "0.5M correction so a near-passing parent is evaluated before PPO "
            "can overshoot it."
        ),
    )
    parser.add_argument(
        "--max-finetunes-per-radius",
        type=int,
        default=4,
        help=(
            "Maximum sequential bounded children at one radius before retaining "
            "the previous parent and refining the radius step."
        ),
    )
    parser.add_argument(
        "--min-train-timesteps",
        type=int,
        default=62_500,
        help=(
            "Smallest bounded child budget used by adaptive backoff after a "
            "child regresses deterministic success."
        ),
    )
    parser.add_argument(
        "--max-child-success-drop",
        type=float,
        default=0.03,
        help=(
            "Do not continue training from a failed child whose deterministic "
            "success falls this far below the screened parent. Retry the retained "
            "parent with half the step budget instead."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=500_000)
    parser.add_argument(
        "--trainer-eval-episodes",
        type=int,
        default=0,
        help=(
            "Internal stochastic post-training eval target. Keep at zero because "
            "the radius runner performs its own deterministic evaluation."
        ),
    )
    parser.add_argument(
        "--exploration-log-std",
        type=float,
        help=(
            "Training-only Gaussian log standard deviation applied before each "
            "bounded child. Use a lower value when exploration destroys a solved "
            "narrow-gate prefix; deterministic mean actions are unchanged."
        ),
    )
    parser.add_argument("--policy-input-dim", type=int, default=32)
    parser.add_argument("--policy-hidden-dim", type=int, default=128)
    parser.add_argument("--policy-num-actions", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--max-rollouts", type=int, default=64)
    parser.add_argument(
        "--set",
        dest="overrides",
        type=parse_override,
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Additional config override; may be repeated.",
    )
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.gate_index <= 3:
        parser.error("--gate-index must be between 0 and 3")
    if not 0 <= args.success_threshold <= 1 or not 0 <= args.crash_limit <= 1:
        parser.error("thresholds must be between 0 and 1")
    if args.max_finetunes_per_radius < 1:
        parser.error("--max-finetunes-per-radius must be at least 1")
    if args.train_timesteps < 1:
        parser.error("--train-timesteps must be positive")
    if args.min_train_timesteps < 1:
        parser.error("--min-train-timesteps must be positive")
    if args.min_train_timesteps > args.train_timesteps:
        parser.error("--min-train-timesteps cannot exceed --train-timesteps")
    if not 0 <= args.max_child_success_drop <= 1:
        parser.error("--max-child-success-drop must be between 0 and 1")
    if args.trainer_eval_episodes < 0:
        parser.error("--trainer-eval-episodes cannot be negative")
    if args.exploration_log_std is not None and not math.isfinite(args.exploration_log_std):
        parser.error("--exploration-log-std must be finite")
    if min(args.policy_input_dim, args.policy_hidden_dim, args.policy_num_actions) < 1:
        parser.error("policy dimensions must be positive")
    if args.min_radius_step is not None:
        if args.min_radius_step <= 0:
            parser.error("--min-radius-step must be positive")
        if args.min_radius_step > args.radius_step:
            parser.error("--min-radius-step cannot exceed --radius-step")

    resumed_state = None
    if args.resume_state is not None:
        resume_path = args.resume_state.expanduser().resolve()
        if not resume_path.is_file():
            parser.error(f"resume state does not exist: {resume_path}")
        resumed_state = json.loads(resume_path.read_text())
        if resumed_state.get("env_name") != args.env_name:
            parser.error("--env-name does not match resume state")
        if int(resumed_state.get("gate_index", -1)) != args.gate_index:
            parser.error("--gate-index does not match resume state")
        try:
            source, initial_radius = resume_source(resumed_state)
            schedule, target_radius = resume_radius_schedule(
                resumed_state, args.target_radius
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.source_checkpoint is not None or args.initial_radius is not None:
            parser.error("do not combine --resume-state with source/initial radius")
    else:
        if args.source_checkpoint is None or args.initial_radius is None:
            parser.error("source checkpoint and initial radius are required unless resuming")
        source = args.source_checkpoint.expanduser().resolve()
        initial_radius = args.initial_radius
        target_radius = 0.75 if args.target_radius is None else args.target_radius
        schedule = radius_schedule(initial_radius, target_radius, args.radius_step)

    if not args.dry_run and not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")

    settings = dict(args.overrides)
    if resumed_state is not None:
        persisted_settings = resumed_state.get("settings")
        if persisted_settings is not None and persisted_settings != settings:
            parser.error("--set overrides do not match the persisted resume state")
    output_dir = ROOT / "logs" / args.env_name
    checkpoint_root = ROOT / "checkpoints" / args.env_name
    state_path = args.state_path or args.resume_state or output_dir / "radius_curriculum_state.json"
    state_path = state_path.expanduser().resolve()
    run_id = str(int(time.time() * 1000))
    if resumed_state is None:
        state = {
            "status": "running",
            "env_name": args.env_name,
            "gate_index": args.gate_index,
            "schedule": schedule,
            "schedule_history": [schedule],
            "target_radius": target_radius,
            "success_threshold": args.success_threshold,
            "crash_limit": args.crash_limit,
            "retained_checkpoint": str(source),
            "retained_radius": None,
            "attempts": [],
            "resume_count": 0,
            "exploration_log_std": args.exploration_log_std,
            "trainer_eval_episodes": args.trainer_eval_episodes,
            "settings": settings,
        }
    else:
        state = resumed_state
        history = state.setdefault("schedule_history", [state.get("schedule", [])])
        history.append(schedule)
        state["schedule"] = schedule
        state["target_radius"] = target_radius
        state["status"] = "running"
        state.pop("blocked_radius", None)
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        state["resumed_at_ms"] = int(time.time() * 1000)
        state["exploration_log_std"] = args.exploration_log_std
        state["trainer_eval_episodes"] = args.trainer_eval_episodes
        state["settings"] = settings
    write_state(state_path, state)

    retained = source
    for schedule_index, radius in enumerate(schedule):
        if resumed_state is not None and schedule_index == 0 and math.isclose(
            radius, initial_radius, rel_tol=0.0, abs_tol=1e-9
        ):
            # The atomic state already contains threshold-passing deterministic
            # evidence for this exact retained parent/radius. Re-running that
            # screen adds latency but cannot change the resume decision.
            print(
                f"Resume trusts accepted parent {retained} at radius {radius:g}; "
                "continuing with the persisted pending radius.",
                flush=True,
            )
            continue
        metrics, report_path = evaluate(
            python=sys.executable,
            checkpoint=retained,
            env_name=args.env_name,
            radius=radius,
            gate_index=args.gate_index,
            settings=settings,
            output_dir=output_dir,
            run_id=evaluation_run_id(run_id, len(state["attempts"])),
            eval_episodes=args.eval_episodes,
            horizon=args.horizon,
            max_rollouts=args.max_rollouts,
            dry_run=args.dry_run,
        )
        attempt = {
            "radius": radius,
            "mode": "screen",
            "checkpoint": str(retained),
            "report": str(report_path),
            "metrics": metrics,
        }
        state["attempts"].append(attempt)

        if args.dry_run:
            continue
        # Persist the screened candidate before a potentially long fine-tune so
        # an interrupted run still explains what was in progress.
        write_state(state_path, state)
        if passes(metrics, args.success_threshold, args.crash_limit):
            attempt["accepted"] = True
            state["retained_radius"] = radius
            write_state(state_path, state)
            continue

        if schedule_index == 0:
            attempt["accepted"] = False
            state["status"] = "invalid_source"
            state["blocked_radius"] = radius
            write_state(state_path, state)
            print(f"Source checkpoint does not pass at initial radius {radius:g}; stopping.")
            return 2

        training_parent = retained
        training_timesteps = args.train_timesteps
        accepted_child = None
        for fine_tune_index in range(1, args.max_finetunes_per_radius + 1):
            trained, training_elapsed_seconds = train(
                python=sys.executable,
                source=training_parent,
                env_name=args.env_name,
                radius=radius,
                gate_index=args.gate_index,
                settings=settings,
                checkpoint_root=checkpoint_root,
                timesteps=training_timesteps,
                learning_rate=args.learning_rate,
                reward_scale=args.reward_scale,
                reward_clip=args.reward_clip,
                checkpoint_interval=args.checkpoint_interval,
                trainer_eval_episodes=args.trainer_eval_episodes,
                exploration_log_std=args.exploration_log_std,
                input_dim=args.policy_input_dim,
                hidden_dim=args.policy_hidden_dim,
                num_actions=args.policy_num_actions,
                dry_run=False,
            )
            trained_metrics, trained_report = evaluate(
                python=sys.executable,
                checkpoint=trained,
                env_name=args.env_name,
                radius=radius,
                gate_index=args.gate_index,
                settings=settings,
                output_dir=output_dir,
                run_id=evaluation_run_id(run_id, len(state["attempts"])),
                eval_episodes=args.eval_episodes,
                horizon=args.horizon,
                max_rollouts=args.max_rollouts,
                dry_run=False,
            )
            trained_attempt = {
                "radius": radius,
                "mode": "fine_tune",
                "fine_tune_index": fine_tune_index,
                "parent_checkpoint": str(training_parent),
                "checkpoint": str(trained),
                "report": str(trained_report),
                "metrics": trained_metrics,
                "accepted": passes(trained_metrics, args.success_threshold, args.crash_limit),
                "exploration_log_std": args.exploration_log_std,
                "requested_training_timesteps": training_timesteps,
                "training_steps": int(trained.stem),
                "training_elapsed_seconds": round(training_elapsed_seconds, 6),
                "training_effective_sps": (
                    int(trained.stem) / training_elapsed_seconds
                    if training_elapsed_seconds > 0.0
                    else 0.0
                ),
            }
            state["attempts"].append(trained_attempt)
            write_state(state_path, state)
            if trained_attempt["accepted"]:
                accepted_child = trained
                break
            if success_regressed(metrics, trained_metrics, args.max_child_success_drop):
                trained_attempt["success_regression_stop"] = True
                smaller_timesteps = training_timesteps // 2
                if smaller_timesteps < args.min_train_timesteps:
                    trained_attempt["adaptive_backoff_exhausted"] = True
                    write_state(state_path, state)
                    break
                trained_attempt["next_training_timesteps"] = smaller_timesteps
                # A regressed child is not a useful parent. Return to the last
                # threshold-passing checkpoint and make a smaller correction.
                training_parent = retained
                training_timesteps = smaller_timesteps
                write_state(state_path, state)
                continue
            training_parent = trained

        if accepted_child is None:
            previous_radius = state.get("retained_radius")
            if args.min_radius_step is not None and previous_radius is not None:
                refinement = refined_radius_schedule(
                    float(previous_radius),
                    radius,
                    target_radius,
                    args.min_radius_step,
                )
                if refinement is not None:
                    refined_tail, refined_step = refinement
                    schedule[schedule_index + 1 :] = refined_tail
                    state["schedule"] = schedule
                    state.setdefault("schedule_refinements", []).append(
                        {
                            "accepted_radius": float(previous_radius),
                            "failed_radius": radius,
                            "refined_step": refined_step,
                            "minimum_step": args.min_radius_step,
                        }
                    )
                    write_state(state_path, state)
                    print(
                        f"Radius {radius:g} did not meet thresholds; "
                        f"retrying from retained {float(previous_radius):g} "
                        f"with step {refined_step:g}."
                    )
                    continue
            state["status"] = "blocked"
            state["blocked_radius"] = radius
            write_state(state_path, state)
            print(f"Radius {radius:g} did not meet thresholds; retained {retained}.")
            return 3

        retained = accepted_child
        state["retained_checkpoint"] = str(retained)
        state["retained_radius"] = radius
        write_state(state_path, state)

    state["status"] = "dry_run" if args.dry_run else "complete"
    write_state(state_path, state)
    print(f"Retained checkpoint: {retained}")
    print(f"State: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
