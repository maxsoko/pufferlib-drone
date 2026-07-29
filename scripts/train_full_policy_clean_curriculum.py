#!/usr/bin/env python3
"""Train one recurrent drone policy from scratch through an exact curriculum.

Training may use a slightly easier aperture or camera-dropout condition, but
promotion is always decided by a deterministic, exact-episode evaluation under
the stage's target conditions.  A failed child never replaces the last policy
that passed a stage.  This script operates only on the fast native environment;
it never starts or resets the official simulator.
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
DEFAULT_CURRICULUM = ROOT / "config" / "full_policy_clean_curriculum.json"

DEFAULT_TRAINING = {
    "timesteps": 2_000_000,
    "max_attempts": 4,
    "learning_rate": 3e-5,
    "reward_scale": 0.005,
    "reward_clip": 1.0,
    "ent_coef": 0.001,
    "exploration_log_std": None,
}


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


def should_select_trained_child(
    *, child_accepted: bool, child_improved: bool, consolidating_screen_pass: bool
) -> bool:
    """Select a passing child, but never weaken an already passing screen."""
    return child_accepted and (
        not consolidating_screen_pass or child_improved
    )


def _training_settings(raw: object, inherited: dict) -> dict:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("training settings must be an object")
    unknown = set(raw) - set(DEFAULT_TRAINING)
    if unknown:
        raise ValueError(f"unknown training settings: {', '.join(sorted(unknown))}")
    result = {**inherited, **raw}
    result["timesteps"] = int(result["timesteps"])
    result["max_attempts"] = int(result["max_attempts"])
    for key in ("learning_rate", "reward_scale", "reward_clip", "ent_coef"):
        result[key] = float(result[key])
    exploration = result["exploration_log_std"]
    result["exploration_log_std"] = None if exploration is None else float(exploration)
    if result["timesteps"] < 1 or result["max_attempts"] < 1:
        raise ValueError("timesteps and max_attempts must be positive")
    if result["learning_rate"] <= 0.0:
        raise ValueError("learning_rate must be positive")
    if result["reward_scale"] <= 0.0 or result["reward_clip"] <= 0.0:
        raise ValueError("reward_scale and reward_clip must be positive")
    if result["ent_coef"] < 0.0:
        raise ValueError("ent_coef must be nonnegative")
    if result["exploration_log_std"] is not None and not math.isfinite(
        result["exploration_log_std"]
    ):
        raise ValueError("exploration_log_std must be finite or null")
    return result


def _stage_value_slug(value: str) -> str:
    return value.strip().replace("-", "m").replace(".", "p")


def _expand_ladders(payload: dict) -> list[dict] | None:
    """Expand compact, monotonic numeric ladders into ordinary stages."""
    ladders = payload.get("ladders")
    if ladders is None:
        return None
    if payload.get("stages") is not None:
        raise ValueError("curriculum must choose either stages or ladders")
    if not isinstance(ladders, list) or not ladders:
        raise ValueError("curriculum ladders must be a non-empty list")

    stages: list[dict] = []
    for ladder_index, ladder in enumerate(ladders):
        if not isinstance(ladder, dict):
            raise ValueError(f"ladder {ladder_index} must be an object")
        parameter = str(ladder.get("parameter", "")).strip()
        name_prefix = str(ladder.get("name_prefix", "")).strip()
        values = ladder.get("values")
        exact_base = ladder.get("exact_overrides", {})
        training_base = ladder.get("training_overrides", {})
        training = ladder.get("training", {})
        train_on_previous = ladder.get("train_on_previous_value", False)
        if not parameter or not name_prefix:
            raise ValueError(
                f"ladder {ladder_index} requires parameter and name_prefix"
            )
        if not isinstance(values, list) or not values:
            raise ValueError(f"ladder {name_prefix} requires non-empty values")
        if not isinstance(exact_base, dict) or not isinstance(training_base, dict):
            raise ValueError(
                f"ladder {name_prefix} overrides must be objects"
            )
        if not isinstance(training, dict):
            raise ValueError(f"ladder {name_prefix} training must be an object")
        if not isinstance(train_on_previous, bool):
            raise ValueError(
                f"ladder {name_prefix} train_on_previous_value must be boolean"
            )

        rendered_values = [str(value).strip() for value in values]
        if any(not value for value in rendered_values):
            raise ValueError(f"ladder {name_prefix} contains an empty value")
        if len(set(rendered_values)) != len(rendered_values):
            raise ValueError(f"ladder {name_prefix} contains duplicate values")
        for value_index, value in enumerate(rendered_values):
            exact = {str(key): str(setting) for key, setting in exact_base.items()}
            exact[parameter] = value
            train_overrides = {
                str(key): str(setting) for key, setting in training_base.items()
            }
            # Materialize the ladder parameter in every training stage.  This
            # keeps the serialized stage definition stable when switching a
            # pending ladder from previous-value to current-value training:
            # index zero is identical either way, so accepted history remains
            # immutable while later stages can be safely refreshed.
            training_value_index = (
                max(value_index - 1, 0) if train_on_previous else value_index
            )
            train_overrides[parameter] = rendered_values[training_value_index]
            stages.append(
                {
                    "name": f"{name_prefix}_{_stage_value_slug(value)}",
                    "exact_overrides": exact,
                    "training_overrides": train_overrides,
                    "training": dict(training),
                }
            )
    return stages


def load_curriculum(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = _expand_ladders(payload)
    if stages is None:
        stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("curriculum must contain a non-empty stages list")
    defaults = _training_settings(payload.get("training_defaults"), DEFAULT_TRAINING)

    names: set[str] = set()
    previous_gates = 0
    result: list[dict] = []
    for index, raw in enumerate(stages):
        if not isinstance(raw, dict):
            raise ValueError(f"stage {index} must be an object")
        name = str(raw.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"stage {index} has a missing or duplicate name")
        exact_raw = raw.get("exact_overrides")
        train_raw = raw.get("training_overrides", {})
        if not isinstance(exact_raw, dict) or not exact_raw:
            raise ValueError(f"stage {name} must contain exact_overrides")
        if not isinstance(train_raw, dict):
            raise ValueError(f"stage {name} training_overrides must be an object")
        exact = {str(key): str(value) for key, value in exact_raw.items()}
        training = {str(key): str(value) for key, value in train_raw.items()}
        effective_training = {**exact, **training}

        gates = int(exact.get("env.num-gates", previous_gates or 1))
        training_gates = int(effective_training.get("env.num-gates", gates))
        if gates < previous_gates:
            raise ValueError(f"stage {name} decreases the course prefix")
        if training_gates != gates:
            raise ValueError(f"stage {name} trains and evaluates different gate counts")
        for settings in (exact, effective_training):
            dropout_from = int(settings.get("env.sitl-gate-obs-dropout-from-index", 1))
            if gates > 1 and dropout_from > 1:
                raise ValueError(f"stage {name} does not model gate-1 camera loss")

        names.add(name)
        previous_gates = gates
        result.append(
            {
                "name": name,
                "exact_overrides": exact,
                "training_overrides": training,
                "training": _training_settings(raw.get("training"), defaults),
            }
        )

    final_exact = result[-1]["exact_overrides"]
    if int(final_exact.get("env.num-gates", 0)) != 4:
        raise ValueError("final stage must evaluate the four-gate course")
    if not math.isclose(float(final_exact.get("env.gate-radius", 0.0)), 0.75):
        raise ValueError("final stage must evaluate the 0.75 m gate radius")
    if float(final_exact.get("env.sitl-gate-obs-dropout-range-m", 0.0)) < 4.25:
        raise ValueError("final stage must include the measured 4.25 m camera dropout")
    return defaults, result


def checkpoint_inventory(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.glob("*/*.bin")}


def select_new_checkpoints(before: set[Path], checkpoint_root: Path) -> list[Path]:
    candidates = checkpoint_inventory(checkpoint_root) - before
    if not candidates:
        raise RuntimeError(f"training produced no checkpoint under {checkpoint_root}")

    def key(path: Path) -> tuple[int, int]:
        try:
            step = int(path.stem)
        except ValueError:
            step = -1
        return path.stat().st_mtime_ns, step

    return sorted(candidates, key=key)


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


def pause_after_working_screen(
    state: dict,
    *,
    checkpoint: Path,
    metrics: dict[str, float],
    stage_index: int,
) -> None:
    """Persist a screened incomplete candidate without launching PPO."""
    state["working_checkpoint"] = str(checkpoint)
    state["working_metrics"] = metrics
    state["working_stage_index"] = stage_index
    state["status"] = "paused_after_screen"
    for key in (
        "exhausted_stage",
        "exhausted_stage_index",
        "interrupted_stage",
        "interrupted_stage_index",
    ):
        state.pop(key, None)


def metric_summary(report: dict) -> dict[str, float]:
    metrics = report["metrics"]
    return {
        "episodes": float(metrics.get("env/n", 0.0)),
        "success_rate": float(metrics.get("env/success_rate", 0.0)),
        "crash_rate": float(metrics.get("env/crash", 0.0)),
        "gates_passed": float(metrics.get("env/gates_passed", 0.0)),
        "completion_time": float(metrics.get("env/completion_time", 0.0)),
        "missed_gate_rate": float(metrics.get("env/missed_gate", 0.0)),
        "closest_gate_range": float(metrics.get("env/closest_gate_range", 0.0)),
        "terminal_crossing_radial": float(
            metrics.get("env/terminal_crossing_radial", 0.0)
        ),
        "gate0_exit_vx": float(metrics.get("env/avg_gate0_exit_vx", 0.0)),
        "gate0_exit_vy": float(metrics.get("env/avg_gate0_exit_vy", 0.0)),
        "gate0_exit_vz": float(metrics.get("env/avg_gate0_exit_vz", 0.0)),
    }


def passes(metrics: dict[str, float], success_threshold: float, crash_limit: float) -> bool:
    return metrics["success_rate"] >= success_threshold and metrics["crash_rate"] <= crash_limit


def cap_training_timesteps(training: dict, maximum: int | None) -> dict:
    """Return invocation-local training settings without mutating curriculum."""
    result = dict(training)
    if maximum is not None:
        result["timesteps"] = min(int(result["timesteps"]), maximum)
    return result


def progress_score(
    metrics: dict[str, float], crash_limit: float
) -> tuple[float, float, float, float]:
    """Rank incomplete policies while penalizing crash-heavy shortcuts."""
    excess_crash = max(0.0, metrics["crash_rate"] - crash_limit)
    return (
        metrics["success_rate"] - excess_crash,
        metrics["gates_passed"] - excess_crash,
        -metrics.get("terminal_crossing_radial", math.inf),
        -metrics["missed_gate_rate"],
    )


def materially_improves(
    candidate: dict[str, float],
    baseline: dict[str, float] | None,
    crash_limit: float,
    *,
    min_radial_improvement: float,
) -> bool:
    """Return whether an incomplete child is worth using as the next parent.

    Deterministic radial changes of a few centimetres are measurable, but they
    do not justify another PPO job. Success/gate-count changes remain discrete
    promotion progress; otherwise require a configured radial improvement.
    """
    if baseline is None:
        return True
    if progress_score(candidate, crash_limit) <= progress_score(baseline, crash_limit):
        return False
    if candidate["success_rate"] > baseline["success_rate"]:
        return True
    if candidate["gates_passed"] > baseline["gates_passed"]:
        return True
    if candidate["missed_gate_rate"] < baseline["missed_gate_rate"]:
        return True
    baseline_radial = baseline.get("terminal_crossing_radial", math.inf)
    candidate_radial = candidate.get("terminal_crossing_radial", math.inf)
    return baseline_radial - candidate_radial >= min_radial_improvement


def refresh_pending_curriculum(
    *,
    state: dict,
    stages: list[dict],
    defaults: dict,
    curriculum_path: Path,
    start_stage: int,
) -> None:
    """Replace only unaccepted curriculum stages in an existing resume state."""
    previous = state.get("stages")
    if not isinstance(previous, list):
        raise ValueError("resume state does not contain a stages list")
    if len(previous) < start_stage or len(stages) < start_stage:
        raise ValueError("refreshed curriculum omits an already accepted stage")
    for index in range(start_stage):
        old_stage = previous[index]
        new_stage = stages[index]
        if (
            old_stage.get("name") != new_stage.get("name")
            or old_stage.get("exact_overrides") != new_stage.get("exact_overrides")
        ):
            raise ValueError("refreshed curriculum changes an already accepted stage")
    state.setdefault("curriculum_revisions", []).append(
        {
            "changed_at_unix_ms": int(time.time() * 1000),
            "from_stage_index": start_stage,
            "previous_pending_stages": [
                str(stage.get("name", "")) for stage in previous[start_stage:]
            ],
            "new_pending_stages": [stage["name"] for stage in stages[start_stage:]],
            "curriculum": str(curriculum_path),
        }
    )
    state["curriculum"] = str(curriculum_path)
    state["training_defaults"] = defaults
    # Accepted stages are historical evidence.  Keep their full serialized
    # definitions even when a ladder-wide training setting changes; only the
    # pending suffix is refreshed.
    state["stages"] = [*previous[:start_stage], *stages[start_stage:]]


def pending_curriculum_matches(
    previous: object, stages: list[dict], start_stage: int
) -> bool:
    """Compare only stages that remain eligible to run on resume."""
    return (
        isinstance(previous, list)
        and len(previous) >= start_stage
        and len(stages) >= start_stage
        and previous[start_stage:] == stages[start_stage:]
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
    run_id: str,
    episodes: int,
    horizon: int,
    max_rollouts: int,
    dry_run: bool,
) -> tuple[dict[str, float] | None, Path, list[str]]:
    label = f"clean_{run_id}_{stage_name}"
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
    source: Path | None,
    env_name: str,
    settings: dict[str, str],
    checkpoint_root: Path,
    training: dict,
    seed: int,
    dry_run: bool,
) -> tuple[list[Path], float, list[str]]:
    before = checkpoint_inventory(checkpoint_root)
    prepared_source = source
    temporary_source: Path | None = None
    exploration = training["exploration_log_std"]
    if source is not None and exploration is not None and not dry_run:
        handle = tempfile.NamedTemporaryFile(prefix="drone_clean_exploration_", suffix=".bin", delete=False)
        handle.close()
        temporary_source = Path(handle.name)
        old = write_checkpoint_with_log_std(source, temporary_source, exploration)
        prepared_source = temporary_source
        print(f"training exploration log_std: {old} -> {[exploration] * 4}", flush=True)

    command = [python, "-m", "pufferlib.pufferl", "train", env_name]
    if prepared_source is not None:
        command.extend(("--load-model-path", str(prepared_source)))
    command.extend(
        (
            "--checkpoint-interval",
            # PufferLib names this option like a timestep interval, but its
            # training loop applies it to the epoch counter. Five epochs is
            # roughly 164k native steps with the current 1,024x32 rollout.
            "5",
            "--eval-episodes",
            "0",
            "--train.total-timesteps",
            str(training["timesteps"]),
            "--train.learning-rate",
            f"{training['learning_rate']:g}",
            "--train.reward-scale",
            f"{training['reward_scale']:g}",
            "--train.reward-clip",
            f"{training['reward_clip']:g}",
            "--train.ent-coef",
            f"{training.get('ent_coef', DEFAULT_TRAINING['ent_coef']):g}",
            "--train.seed",
            str(seed),
            *cli_overrides(settings),
        )
    )
    started = time.monotonic()
    try:
        run_command(command, dry_run=dry_run)
    finally:
        if temporary_source is not None:
            temporary_source.unlink(missing_ok=True)
    elapsed = time.monotonic() - started
    if dry_run:
        return [], elapsed, command
    return select_new_checkpoints(before, checkpoint_root), elapsed, command


def _attempt_id(run_id: str, stage_index: int, attempt_count: int) -> str:
    return f"{run_id}_s{stage_index:02d}_a{attempt_count:04d}"


def dry_run_plan(
    *,
    args: argparse.Namespace,
    stages: list[dict],
    global_settings: dict[str, str],
    output_dir: Path,
    checkpoint_root: Path,
) -> None:
    source = args.source_checkpoint
    for index, stage in enumerate(stages):
        exact = {**stage["exact_overrides"], **global_settings}
        training_settings = {**exact, **stage["training_overrides"], **global_settings}
        _, _, command = train(
            python=sys.executable,
            source=source,
            env_name=args.env_name,
            settings=training_settings,
            checkpoint_root=checkpoint_root,
            training=cap_training_timesteps(
                stage["training"], args.max_train_timesteps
            ),
            seed=42 + index,
            dry_run=True,
        )
        placeholder = checkpoint_root / f"DRY_RUN_STAGE_{index}.bin"
        evaluate(
            python=sys.executable,
            checkpoint=placeholder,
            env_name=args.env_name,
            stage_name=stage["name"],
            settings=exact,
            output_dir=output_dir,
            run_id=f"dry_s{index:02d}",
            episodes=args.eval_episodes,
            horizon=args.horizon,
            max_rollouts=args.max_rollouts,
            dry_run=True,
        )
        source = placeholder
        if "--load-model-path" in command and index == 0 and args.source_checkpoint is None:
            raise AssertionError("fresh training unexpectedly loaded a checkpoint")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-checkpoint", type=Path)
    source.add_argument("--resume-state", type=Path)
    parser.add_argument(
        "--screen-checkpoint",
        type=Path,
        help=(
            "on resume, screen an externally calibrated policy at the pending "
            "stage; the retained checkpoint changes only if exact evaluation passes"
        ),
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help=(
            "evaluate --screen-checkpoint and persist it as the working policy "
            "without launching a training job when it does not pass"
        ),
    )
    parser.add_argument("--curriculum", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument("--env-name", default="drone_race_full_policy_stage_d_gate4")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument("--crash-limit", type=float, default=0.10)
    parser.add_argument("--eval-episodes", type=int, default=1024)
    parser.add_argument("--promotion-episodes", type=int, default=4096)
    parser.add_argument(
        "--max-stages",
        type=int,
        help="stop cleanly after this many stages are accepted in this invocation",
    )
    parser.add_argument(
        "--max-train-attempts",
        type=int,
        help="cap training jobs per stage in this invocation (screens are not counted)",
    )
    parser.add_argument(
        "--max-train-timesteps",
        type=int,
        help=(
            "cap timesteps in each training job for this invocation without "
            "changing the persisted curriculum"
        ),
    )
    parser.add_argument(
        "--min-radial-improvement",
        type=float,
        default=0.10,
        help="minimum radial-miss reduction required to chain an incomplete child",
    )
    parser.add_argument(
        "--refresh-pending-curriculum",
        action="store_true",
        help="adopt curriculum changes only from the first unaccepted stage onward",
    )
    parser.add_argument(
        "--consolidate-screen-pass",
        action="store_true",
        help=(
            "run one bounded training job even when the pending stage screen "
            "passes; retain a trained child only when it also passes and "
            "improves on the screened checkpoint"
        ),
    )
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
    if args.eval_episodes < 1 or args.promotion_episodes < 1:
        parser.error("evaluation episode counts must be positive")
    if args.max_stages is not None and args.max_stages < 1:
        parser.error("--max-stages must be positive")
    if args.max_train_attempts is not None and args.max_train_attempts < 1:
        parser.error("--max-train-attempts must be positive")
    if args.max_train_timesteps is not None and args.max_train_timesteps < 1:
        parser.error("--max-train-timesteps must be positive")
    if args.min_radial_improvement < 0.0:
        parser.error("--min-radial-improvement must be nonnegative")
    if args.screen_checkpoint is not None and args.resume_state is None:
        parser.error("--screen-checkpoint requires --resume-state")
    if args.screen_only and args.screen_checkpoint is None:
        parser.error("--screen-only requires --screen-checkpoint")

    curriculum_path = args.curriculum.expanduser().resolve()
    try:
        defaults, stages = load_curriculum(curriculum_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    global_settings = dict(args.overrides)
    output_dir = ROOT / "logs" / args.env_name
    checkpoint_root = ROOT / "checkpoints" / args.env_name
    run_id = str(int(time.time() * 1000))

    if args.dry_run:
        dry_run_plan(
            args=args,
            stages=stages,
            global_settings=global_settings,
            output_dir=output_dir,
            checkpoint_root=checkpoint_root,
        )
        return 0

    if args.resume_state is not None:
        state_path = args.resume_state.expanduser().resolve()
        if not state_path.is_file():
            parser.error(f"resume state does not exist: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("env_name") != args.env_name:
            parser.error("--env-name does not match resume state")
        start_stage = int(state.get("next_stage_index", 0))
        if state.get("global_settings") != global_settings:
            parser.error("--set overrides do not match resume state")
        if not pending_curriculum_matches(state.get("stages"), stages, start_stage):
            if not args.refresh_pending_curriculum:
                parser.error(
                    "curriculum does not match resume state; use "
                    "--refresh-pending-curriculum to replace only pending stages"
                )
            try:
                refresh_pending_curriculum(
                    state=state,
                    stages=stages,
                    defaults=defaults,
                    curriculum_path=curriculum_path,
                    start_stage=start_stage,
                )
            except ValueError as exc:
                parser.error(str(exc))
        retained_value = state.get("retained_checkpoint")
        retained = Path(retained_value).resolve() if retained_value else None
        working_value = state.get("working_checkpoint")
        working = Path(working_value).resolve() if working_value else None
        if state.get("working_stage_index") != start_stage:
            working = None
        if working is None:
            stage_attempts = [
                attempt
                for attempt in state.get("attempts", [])
                if attempt.get("stage_index") == start_stage
                and attempt.get("metrics") is not None
                and attempt.get("checkpoint")
            ]
            for attempt in stage_attempts:
                if "terminal_crossing_radial" in attempt["metrics"]:
                    continue
                report_path = Path(attempt.get("report", "")).expanduser()
                if report_path.is_file():
                    attempt["metrics"] = metric_summary(
                        json.loads(report_path.read_text(encoding="utf-8"))
                    )
            screens = [attempt for attempt in stage_attempts if attempt.get("mode") == "screen"]
            trained = [attempt for attempt in stage_attempts if attempt.get("mode") == "train"]
            if screens and trained:
                baseline = screens[-1]["metrics"]
                best_attempt = max(
                    trained,
                    key=lambda attempt: progress_score(attempt["metrics"], args.crash_limit),
                )
                recovered = Path(best_attempt["checkpoint"]).expanduser().resolve()
                if (
                    recovered.is_file()
                    and materially_improves(
                        best_attempt["metrics"],
                        baseline,
                        args.crash_limit,
                        min_radial_improvement=args.min_radial_improvement,
                    )
                ):
                    working = recovered
                    state["working_checkpoint"] = str(working)
                    state["working_metrics"] = best_attempt["metrics"]
                    state["working_stage_index"] = start_stage
        if args.screen_checkpoint is not None:
            working = args.screen_checkpoint.expanduser().resolve()
            state["working_checkpoint"] = str(working)
            state["working_metrics"] = None
            state["working_stage_index"] = start_stage
            state.setdefault("attempts", []).append(
                {
                    "stage_index": start_stage,
                    "stage": stages[start_stage]["name"],
                    "mode": "external_candidate_queued",
                    "checkpoint": str(working),
                    "accepted": False,
                }
            )
        state["status"] = "running"
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        state.setdefault("next_training_seed", 43)
    else:
        retained = args.source_checkpoint.expanduser().resolve() if args.source_checkpoint else None
        working = None
        start_stage = 0
        state_path = (
            args.state_path.expanduser().resolve()
            if args.state_path is not None
            else (output_dir / "full_policy_clean_curriculum_state.json").resolve()
        )
        state = {
            "status": "running",
            "env_name": args.env_name,
            "curriculum": str(curriculum_path),
            "training_defaults": defaults,
            "stages": stages,
            "global_settings": global_settings,
            "retained_checkpoint": str(retained) if retained else None,
            "working_checkpoint": None,
            "working_metrics": None,
            "working_stage_index": None,
            "next_stage_index": 0,
            "attempts": [],
            "resume_count": 0,
            "next_training_seed": 42,
        }
    for checkpoint in (retained, working):
        if checkpoint is not None and not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")
    write_state(state_path, state)

    stages_accepted_this_run = 0
    for stage_index in range(start_stage, len(stages)):
        stage = stages[stage_index]
        exact_settings = {**stage["exact_overrides"], **global_settings}
        training_settings = {
            **exact_settings,
            **stage["training_overrides"],
            **global_settings,
        }
        candidate = working if working is not None else retained
        best_checkpoint = candidate
        best_metrics = None
        consolidating_screen_pass = False
        passing_screen = None

        if candidate is not None:
            attempt_id = _attempt_id(run_id, stage_index, len(state["attempts"]))
            best_metrics, report, command = evaluate(
                python=sys.executable,
                checkpoint=candidate,
                env_name=args.env_name,
                stage_name=stage["name"],
                settings=exact_settings,
                output_dir=output_dir,
                run_id=attempt_id,
                episodes=args.eval_episodes,
                horizon=args.horizon,
                max_rollouts=args.max_rollouts,
                dry_run=False,
            )
            screen = {
                "stage_index": stage_index,
                "stage": stage["name"],
                "mode": "screen",
                "checkpoint": str(candidate),
                "report": str(report),
                "metrics": best_metrics,
                "accepted": passes(best_metrics, args.success_threshold, args.crash_limit),
                "consolidation_requested": args.consolidate_screen_pass,
                "selected_for_promotion": False,
                "eval_command": command,
            }
            state["attempts"].append(screen)
            write_state(state_path, state)
            if screen["accepted"]:
                if args.consolidate_screen_pass:
                    consolidating_screen_pass = True
                    passing_screen = screen
                else:
                    screen["selected_for_promotion"] = True
                    retained = candidate
                    working = None
                    state["retained_checkpoint"] = str(retained)
                    state["working_checkpoint"] = None
                    state["working_metrics"] = None
                    state["working_stage_index"] = None
                    state["next_stage_index"] = stage_index + 1
                    write_state(state_path, state)
                    stages_accepted_this_run += 1
                    if (
                        args.screen_only
                        or (
                            args.max_stages is not None
                            and stages_accepted_this_run >= args.max_stages
                        )
                    ):
                        state["status"] = "paused_after_stage"
                        write_state(state_path, state)
                        print(f"Accepted {stage['name']}; resume from {state_path}.")
                        return 0
                    continue

            if args.screen_only:
                pause_after_working_screen(
                    state,
                    checkpoint=candidate,
                    metrics=best_metrics,
                    stage_index=stage_index,
                )
                write_state(state_path, state)
                print(
                    f"Screened working candidate for {stage['name']}; "
                    f"resume from {state_path}."
                )
                return 0

        accepted = False
        attempt_limit = stage["training"]["max_attempts"]
        if args.max_train_attempts is not None:
            attempt_limit = min(attempt_limit, args.max_train_attempts)
        invocation_training = cap_training_timesteps(
            stage["training"], args.max_train_timesteps
        )
        for fine_tune_index in range(1, attempt_limit + 1):
            training_seed = int(state["next_training_seed"])
            state["next_training_seed"] = training_seed + 1
            write_state(state_path, state)
            training_parent = best_checkpoint
            try:
                children, train_elapsed, train_command = train(
                    python=sys.executable,
                    source=training_parent,
                    env_name=args.env_name,
                    settings=training_settings,
                    checkpoint_root=checkpoint_root,
                    training=invocation_training,
                    seed=training_seed,
                    dry_run=False,
                )
            except KeyboardInterrupt:
                state["status"] = "interrupted"
                state["interrupted_stage_index"] = stage_index
                state["interrupted_stage"] = stage["name"]
                write_state(state_path, state)
                return 130
            for intermediate_index, child in enumerate(children, start=1):
                attempt_id = _attempt_id(run_id, stage_index, len(state["attempts"]))
                metrics, report, eval_command = evaluate(
                    python=sys.executable,
                    checkpoint=child,
                    env_name=args.env_name,
                    stage_name=stage["name"],
                    settings=exact_settings,
                    output_dir=output_dir,
                    run_id=attempt_id,
                    episodes=args.eval_episodes,
                    horizon=args.horizon,
                    max_rollouts=args.max_rollouts,
                    dry_run=False,
                )
                child_accepted = passes(
                    metrics, args.success_threshold, args.crash_limit
                )
                child_improved = best_metrics is None or progress_score(
                    metrics, args.crash_limit
                ) > progress_score(best_metrics, args.crash_limit)
                child_material = materially_improves(
                    metrics,
                    best_metrics,
                    args.crash_limit,
                    min_radial_improvement=args.min_radial_improvement,
                )
                child_selected = should_select_trained_child(
                    child_accepted=child_accepted,
                    child_improved=child_improved,
                    consolidating_screen_pass=consolidating_screen_pass,
                )
                attempt = {
                    "stage_index": stage_index,
                    "stage": stage["name"],
                    "mode": "train",
                    "fine_tune_index": fine_tune_index,
                    "intermediate_index": intermediate_index,
                    "intermediate_count": len(children),
                    "parent_checkpoint": (
                        str(training_parent) if training_parent else None
                    ),
                    "checkpoint": str(child),
                    "report": str(report),
                    "metrics": metrics,
                    "accepted": child_accepted,
                    "selected_for_promotion": child_selected,
                    "screen_pass_consolidation": consolidating_screen_pass,
                    "best_progress": child_improved,
                    "material_progress": child_material,
                    "min_radial_improvement": args.min_radial_improvement,
                    "configured_training_timesteps": stage["training"]["timesteps"],
                    "requested_training_timesteps": invocation_training["timesteps"],
                    "training_seed": training_seed,
                    "training_steps": int(child.stem),
                    "training_elapsed_seconds": round(train_elapsed, 6),
                    "training_effective_sps": (
                        int(child.stem) / max(train_elapsed, 1e-9)
                    ),
                    "training_overrides": training_settings,
                    "exact_overrides": exact_settings,
                    "train_command": train_command,
                    "eval_command": eval_command,
                }
                state["attempts"].append(attempt)
                if child_selected:
                    if passing_screen is not None:
                        passing_screen["consolidation_improved"] = True
                    retained = child
                    working = None
                    state["retained_checkpoint"] = str(retained)
                    state["working_checkpoint"] = None
                    state["working_metrics"] = None
                    state["working_stage_index"] = None
                    state["next_stage_index"] = stage_index + 1
                    write_state(state_path, state)
                    accepted = True
                    break
                if child_material:
                    best_checkpoint = child
                    best_metrics = metrics
                    working = child
                    state["working_checkpoint"] = str(child)
                    state["working_metrics"] = metrics
                    state["working_stage_index"] = stage_index
                write_state(state_path, state)
            if accepted:
                break

        if not accepted:
            if passing_screen is not None:
                passing_screen["selected_for_promotion"] = True
                passing_screen["consolidation_improved"] = False
                retained = Path(passing_screen["checkpoint"]).resolve()
                working = None
                state["retained_checkpoint"] = str(retained)
                state["working_checkpoint"] = None
                state["working_metrics"] = None
                state["working_stage_index"] = None
                state["next_stage_index"] = stage_index + 1
                accepted = True
                write_state(state_path, state)
            else:
                state["status"] = "stage_exhausted"
                state["exhausted_stage_index"] = stage_index
                state["exhausted_stage"] = stage["name"]
                write_state(state_path, state)
                print(f"Stage {stage['name']} exhausted this run; resume from {state_path}.")
                return 3
        stages_accepted_this_run += 1
        if args.max_stages is not None and stages_accepted_this_run >= args.max_stages:
            state["status"] = "paused_after_stage"
            write_state(state_path, state)
            print(f"Accepted {stage['name']}; resume from {state_path}.")
            return 0

    final_stage = stages[-1]
    final_settings = {**final_stage["exact_overrides"], **global_settings}
    promotion, report, command = evaluate(
        python=sys.executable,
        checkpoint=retained,
        env_name=args.env_name,
        stage_name=f"{final_stage['name']}_promotion",
        settings=final_settings,
        output_dir=output_dir,
        run_id=f"{run_id}_promotion",
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
