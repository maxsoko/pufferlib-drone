#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _ensure_parent(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


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


def _validate_backend_env(compiled_env: str | None, env_name: str, backend_env_name: str) -> None:
    if compiled_env in {env_name, backend_env_name}:
        return
    raise RuntimeError(
        "pufferlib._C backend mismatch: "
        f"compiled={compiled_env!r}, env={env_name!r}, backend={backend_env_name!r}. "
        f"Rebuild with `bash build.sh {backend_env_name}`"
    )


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
    parser.add_argument("--max-rollouts", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--label",
        default="",
        help="Optional human-readable label for the report, e.g. r3_best.",
    )
    args, overrides = parser.parse_known_args()
    _C, pufferl_module = _load_runtime_modules()

    cfg = _load_config(pufferl_module, args.env_name, overrides)
    compiled_env = getattr(_C, "env_name", None)
    backend_env_name = cfg.get("backend_env_name", args.env_name)
    _validate_backend_env(compiled_env, args.env_name, backend_env_name)

    cfg["eval_episodes"] = int(args.eval_episodes)
    cfg["load_model_path"] = str(args.weights)
    cfg["train"]["horizon"] = int(args.horizon)

    runner = _C.create_pufferl(cfg)
    _C.load_weights(runner, str(args.weights))

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
    _add_derived_metrics(flat_logs)

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
