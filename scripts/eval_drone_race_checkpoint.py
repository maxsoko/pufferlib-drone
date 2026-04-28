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

from pufferlib import _C
from pufferlib import pufferl


def _ensure_parent(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _load_config(env_name, overrides):
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *overrides]
        return pufferl.load_config(env_name)
    finally:
        sys.argv = saved_argv


def _flatten(logs):
    return dict(pufferl.unroll_nested_dict(logs))


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

    compiled_env = getattr(_C, "env_name", None)
    if compiled_env != args.env_name:
        raise RuntimeError(
            f"pufferlib._C was built for {compiled_env!r}; rebuild with `bash build.sh {args.env_name}`"
        )

    cfg = _load_config(args.env_name, overrides)
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
            flat_logs = _flatten(_C.eval_log(runner))
            if flat_logs.get("env/n", 0.0) >= args.eval_episodes:
                break
    finally:
        _C.close(runner)

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
