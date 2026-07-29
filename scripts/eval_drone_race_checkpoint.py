#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _ensure_parent(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _align(index, precision_bytes):
    if precision_bytes not in (2, 4):
        raise ValueError("precision_bytes must be 2 (BF16) or 4 (FP32)")
    alignment_elements = 16 // precision_bytes
    return (index + alignment_elements - 1) & ~(alignment_elements - 1)


def _deterministic_checkpoint(
    source, *, input_dim, hidden_dim, num_actions, layout_precision_bytes=2
):
    """Return a temporary raw checkpoint with effectively zero Gaussian std."""
    weights = np.fromfile(source, dtype=np.float32)
    encoder_end = _align(hidden_dim * input_dim, layout_precision_bytes)
    decoder_end = _align(
        encoder_end + (num_actions + 1) * hidden_dim,
        layout_precision_bytes,
    )
    if decoder_end + num_actions > weights.size:
        raise ValueError("checkpoint ended before the continuous-action log_std")
    weights[decoder_end:decoder_end + num_actions] = np.float32(-20.0)
    handle = tempfile.NamedTemporaryFile(prefix="drone_eval_deterministic_", suffix=".bin", delete=False)
    handle.close()
    weights.tofile(handle.name)
    return handle.name


def _load_config(pufferl_module, env_name, overrides):
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *overrides]
        return pufferl_module.load_config(env_name)
    finally:
        sys.argv = saved_argv


def _flatten(pufferl_module, logs):
    return dict(pufferl_module.unroll_nested_dict(logs))


def _write_json(path, report):
    _ensure_parent(path)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def _write_csv(path, report):
    _ensure_parent(path)
    row = {
        **{f"meta/{key}": value for key, value in report["metadata"].items()},
        **report["metrics"],
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row))
        writer.writeheader()
        writer.writerow(row)


def _add_ratio(metrics, numerator_key, denominator_key, output_key):
    denominator = metrics.get(denominator_key, 0.0)
    metrics[output_key] = metrics.get(numerator_key, 0.0) / denominator if denominator else 0.0


def _add_derived_metrics(metrics):
    low_crash_metrics = [
        "crash_low_z",
        "crash_low_vz",
        "crash_low_progress",
        "crash_low_time",
        "crash_low_floor_margin_pre",
        "crash_low_ttf_pre",
        "crash_low_stop_margin_pre",
        "crash_low_max_up_accel_pre",
    ]
    for key in low_crash_metrics:
        _add_ratio(metrics, f"env/{key}", "env/crash_low", f"env/avg_{key}")

    _add_ratio(metrics, "env/min_floor_ttf", "env/floor_risk_sampled", "env/avg_min_floor_ttf")
    _add_ratio(
        metrics,
        "env/min_floor_stop_margin",
        "env/floor_risk_sampled",
        "env/avg_min_floor_stop_margin",
    )
    for source, destination in (
        ("env/terminal_crossing_radial", "env/avg_terminal_crossing_radial"),
        ("env/terminal_crossing_right", "env/avg_terminal_crossing_right"),
        ("env/terminal_crossing_vertical", "env/avg_terminal_crossing_vertical"),
        ("env/terminal_crossing_abs_right", "env/avg_terminal_crossing_abs_right"),
        ("env/terminal_crossing_abs_vertical", "env/avg_terminal_crossing_abs_vertical"),
    ):
        _add_ratio(
            metrics,
            source,
            "env/terminal_crossing_sampled",
            destination,
        )
    for axis in ("radial", "right", "vertical"):
        _add_ratio(
            metrics,
            f"env/gate2_terminal_crossing_{axis}",
            "env/gate2_terminal_crossing_sampled",
            f"env/avg_gate2_terminal_crossing_{axis}",
        )
    for axis in ("vx", "vy", "vz"):
        _add_ratio(
            metrics,
            f"env/gate0_exit_{axis}",
            "env/gate0_exit_sampled",
            f"env/avg_gate0_exit_{axis}",
        )
        _add_ratio(
            metrics,
            f"env/gate2_exit_{axis}",
            "env/gate2_crossing_sampled",
            f"env/avg_gate2_exit_{axis}",
        )
    for axis in ("radial", "right", "vertical"):
        _add_ratio(
            metrics,
            f"env/gate2_crossing_{axis}",
            "env/gate2_crossing_sampled",
            f"env/avg_gate2_crossing_{axis}",
        )
    for cohort in ("completion", "downstream_failure"):
        denominator = f"env/gate2_{cohort}_sampled"
        for axis in ("radial", "right", "vertical"):
            _add_ratio(
                metrics,
                f"env/gate2_{cohort}_crossing_{axis}",
                denominator,
                f"env/avg_gate2_{cohort}_crossing_{axis}",
            )
        for axis in ("vx", "vy", "vz"):
            _add_ratio(
                metrics,
                f"env/gate2_{cohort}_exit_{axis}",
                denominator,
                f"env/avg_gate2_{cohort}_exit_{axis}",
            )
    for gate, actions in (
        (0, ("pitch",)),
        (1, ("pitch", "roll", "thrust", "yaw")),
        (2, ("pitch", "roll", "thrust", "yaw")),
        (3, ("pitch", "roll", "thrust", "yaw")),
    ):
        for action in actions:
            _add_ratio(
                metrics,
                f"env/gate{gate}_action_{action}",
                f"env/gate{gate}_action_steps",
                f"env/avg_gate{gate}_action_{action}",
            )
    for action in ("pitch", "roll", "thrust", "yaw"):
        _add_ratio(
            metrics,
            f"env/gate1_early_action_{action}",
            "env/gate1_early_action_steps",
            f"env/avg_gate1_early_action_{action}",
        )
    for slot in range(4):
        _add_ratio(
            metrics,
            f"env/episode_slot{slot}_success",
            f"env/episode_slot{slot}_n",
            f"env/episode_slot{slot}_success_rate",
        )
    for gate in range(6):
        for axis in ("radial", "right", "vertical"):
            _add_ratio(
                metrics,
                f"env/terminal_gate{gate}_{axis}",
                f"env/terminal_gate{gate}_sampled",
                f"env/avg_terminal_gate{gate}_{axis}",
            )


def _validate_backend_env(compiled_env: str | None, env_name: str, backend_env_name: str) -> None:
    if compiled_env in {env_name, backend_env_name}:
        return
    raise RuntimeError(
        "pufferlib._C backend mismatch: "
        f"compiled={compiled_env!r}, env={env_name!r}, backend={backend_env_name!r}. "
        f"Rebuild with `bash build.sh {backend_env_name}`"
    )


def _configure_exact_episode_limit(
    cfg: dict, eval_episodes: int, episode_offset: int = 0
) -> int:
    """Assign an equal episode quota to every one-agent vector environment."""
    total_agents = int(cfg.get("vec", {}).get("total_agents", 0))
    num_drones = int(cfg.get("env", {}).get("num_drones", 1))
    if total_agents <= 0:
        raise ValueError("exact evaluation requires a positive vec.total_agents")
    if num_drones != 1:
        raise ValueError("exact evaluation currently requires env.num_drones == 1")
    if eval_episodes <= 0 or eval_episodes % total_agents != 0:
        raise ValueError(
            "exact evaluation episodes must be a positive multiple of "
            f"vec.total_agents ({total_agents})"
        )
    if episode_offset < 0:
        raise ValueError("exact evaluation episode offset must be nonnegative")
    per_env_limit = eval_episodes // total_agents
    cfg.setdefault("env", {})["evaluation_episode_limit"] = per_env_limit
    cfg["env"]["evaluation_episode_offset"] = episode_offset
    return per_env_limit


def _load_runtime_modules():
    from pufferlib import _C as runtime_c
    from pufferlib import pufferl as runtime_pufferl

    return runtime_c, runtime_pufferl


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a native drone_race checkpoint and write deterministic JSON/CSV artifacts."
    )
    parser.add_argument("weights", type=Path, help="Native PufferNet .bin checkpoint")
    parser.add_argument("--env-name", default="drone_race")
    parser.add_argument("--json-path", default="logs/drone_race_checkpoint_eval.json")
    parser.add_argument("--csv-path", default="logs/drone_race_checkpoint_eval.csv")
    parser.add_argument("--eval-episodes", type=int, default=4096)
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help=(
            "evaluation-only reset samples skipped per vector environment; "
            "use 2 with 2048 episodes to screen the hard half of a 4096 set"
        ),
    )
    parser.add_argument("--max-rollouts", type=int, default=256)
    parser.add_argument(
        "--horizon",
        type=int,
        default=32,
        help="Steps per eval rollout; must be long enough for episodes to finish within max-rollouts",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional human-readable label for the report, e.g. r3_best.",
    )
    parser.add_argument(
        "--stochastic-actions",
        action="store_true",
        help="Sample the learned Gaussian policy instead of evaluating its deployed mean action.",
    )
    parser.add_argument(
        "--require-exact-episodes",
        action="store_true",
        help="Freeze each one-agent vector env after an equal quota so env/n exactly matches --eval-episodes.",
    )
    parser.add_argument(
        "--checkpoint-layout-precision-bytes",
        type=int,
        choices=(2, 4),
        help=(
            "Native arena layout that wrote the checkpoint. Defaults to the "
            "loaded backend precision and must match it for native evaluation."
        ),
    )
    args, overrides = parser.parse_known_args()
    _C, pufferl_module = _load_runtime_modules()
    native_precision_bytes = int(getattr(_C, "precision_bytes", 0))
    checkpoint_layout_precision_bytes = (
        args.checkpoint_layout_precision_bytes or native_precision_bytes
    )
    if checkpoint_layout_precision_bytes != native_precision_bytes:
        parser.error(
            "checkpoint layout/native backend mismatch: "
            f"checkpoint={checkpoint_layout_precision_bytes}, "
            f"backend={native_precision_bytes}; convert the checkpoint layout "
            "or rebuild the backend"
        )

    cfg = _load_config(pufferl_module, args.env_name, overrides)
    compiled_env = getattr(_C, "env_name", None)
    backend_env_name = cfg.get("backend_env_name", args.env_name)
    _validate_backend_env(compiled_env, args.env_name, backend_env_name)

    cfg["eval_episodes"] = int(args.eval_episodes)
    cfg["load_model_path"] = str(args.weights)
    cfg["train"]["horizon"] = int(args.horizon)
    exact_episode_limit = None
    if args.require_exact_episodes:
        try:
            exact_episode_limit = _configure_exact_episode_limit(
                cfg, args.eval_episodes, args.episode_offset
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif args.episode_offset:
        parser.error("--episode-offset requires --require-exact-episodes")

    max_episode_steps = min(
        int(cfg.get("env", {}).get("max_steps", 0) or 0),
        int(
            float(cfg.get("env", {}).get("time_limit_seconds", 0.0) or 0.0)
            / max(float(cfg.get("env", {}).get("dt", 0.0) or 0.0), 1e-9)
        ),
    )
    if max_episode_steps > 0 and args.horizon * args.max_rollouts < max_episode_steps:
        parser.error(
            "--horizon * --max-rollouts is shorter than one maximum episode "
            f"({args.horizon}*{args.max_rollouts} < {max_episode_steps})"
        )

    runner = _C.create_pufferl(cfg)
    loaded_weights = str(args.weights)
    deterministic_path = None
    if not args.stochastic_actions:
        deterministic_path = _deterministic_checkpoint(
            args.weights,
            input_dim=32,
            hidden_dim=int(cfg.get("policy", {}).get("hidden_size", 128)),
            num_actions=4,
            layout_precision_bytes=checkpoint_layout_precision_bytes,
        )
        loaded_weights = deterministic_path
    _C.load_weights(runner, loaded_weights)

    flat_logs = {}
    started = time.time()
    rollouts = 0
    try:
        for rollouts in range(1, args.max_rollouts + 1):
            _C.rollouts(runner)
            flat_logs = _flatten(pufferl_module, _C.eval_log(runner))
            if flat_logs.get("env/n", 0.0) >= args.eval_episodes:
                break
    finally:
        _C.close(runner)
        if deterministic_path is not None:
            os.unlink(deterministic_path)
    _add_derived_metrics(flat_logs)
    if args.require_exact_episodes and flat_logs.get("env/n", 0.0) != args.eval_episodes:
        raise RuntimeError(
            "exact evaluation episode mismatch: "
            f"expected {args.eval_episodes}, got {flat_logs.get('env/n', 0.0)}"
        )

    metadata = {
        "env_name": args.env_name,
        "weights": str(args.weights),
        "label": args.label,
        "eval_episodes_target": args.eval_episodes,
        "max_rollouts": args.max_rollouts,
        "rollouts": rollouts,
        "horizon": args.horizon,
        "elapsed_seconds": round(time.time() - started, 6),
        "config_overrides": overrides,
        "action_mode": "stochastic" if args.stochastic_actions else "deterministic_mean",
        "exact_episodes_required": args.require_exact_episodes,
        "episodes_per_vector_env": exact_episode_limit,
        "episode_offset_per_vector_env": args.episode_offset,
        "native_precision_bytes": native_precision_bytes,
        "checkpoint_layout_precision_bytes": checkpoint_layout_precision_bytes,
    }
    report = {
        "metadata": metadata,
        "metrics": flat_logs,
    }
    _write_json(args.json_path, report)
    _write_csv(args.csv_path, report)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
