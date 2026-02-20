#!/usr/bin/env python3
import argparse
import configparser
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class LevelSpec:
    name: str
    description: str
    suite: str | None
    episodes: int
    metric: str
    operator: str
    threshold: float
    env_kwargs: dict


def _ensure_parent(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _load_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    script_path = Path(__file__).resolve().parent / "eval_drone_race.py"
    spec = importlib.util.spec_from_file_location("eval_drone_race", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_levels_config(config_path: str) -> tuple[dict, list[LevelSpec]]:
    parser = configparser.ConfigParser()
    loaded = parser.read(config_path)
    if not loaded:
        raise FileNotFoundError(f"No levels config found at: {config_path}")

    if "levels" not in parser or "order" not in parser["levels"]:
        raise ValueError("Levels config requires [levels] order")

    order = [item.strip() for item in parser["levels"]["order"].split(",") if item.strip()]
    specs: list[LevelSpec] = []
    for level_name in order:
        if level_name not in parser:
            raise ValueError(f"Missing level section: {level_name}")

        section = parser[level_name]
        try:
            episodes = int(section.get("episodes", "20"))
            threshold = float(section["threshold"])
            env_kwargs = json.loads(section.get("env_kwargs", "{}"))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid level config for {level_name}: {exc}") from exc

        specs.append(
            LevelSpec(
                name=level_name,
                description=section.get("description", ""),
                suite=section.get("suite", None),
                episodes=episodes,
                metric=section["metric"],
                operator=section.get("operator", ">="),
                threshold=threshold,
                env_kwargs=env_kwargs,
            )
        )

    meta = dict(parser["meta"]) if "meta" in parser else {}
    return meta, specs


def _max_consecutive_successes(successes: list[int]) -> int:
    best = 0
    streak = 0
    for value in successes:
        if int(value) == 1:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def compute_level_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "valid_rate": 0.0,
            "success_rate": 0.0,
            "first_gate_pass_rate": 0.0,
            "two_gate_pass_rate": 0.0,
            "mean_gates_passed": 0.0,
            "median_completion_time_valid": float("nan"),
            "max_consecutive_successes": 0,
            "mean_progress": 0.0,
        }

    successes = [int(r["success"]) for r in rows]
    valid_runs = [int(r["valid_run"]) for r in rows]
    gates_passed = [int(r["gates_passed"]) for r in rows]
    progress = [float(r["progress"]) for r in rows]
    completion_times = [
        float(r["completion_time"]) for r in rows if int(r["success"]) == 1 and np.isfinite(float(r["completion_time"]))
    ]

    first_gate = [1 if g >= 1 else 0 for g in gates_passed]
    two_gate = [1 if g >= 2 else 0 for g in gates_passed]

    return {
        "valid_rate": float(np.mean(valid_runs)),
        "success_rate": float(np.mean(successes)),
        "first_gate_pass_rate": float(np.mean(first_gate)),
        "two_gate_pass_rate": float(np.mean(two_gate)),
        "mean_gates_passed": float(np.mean(gates_passed)),
        "median_completion_time_valid": float(np.median(completion_times)) if completion_times else float("nan"),
        "max_consecutive_successes": int(_max_consecutive_successes(successes)),
        "mean_progress": float(np.mean(progress)),
    }


def evaluate_threshold(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return bool(value >= threshold)
    if operator == "<=":
        return bool(value <= threshold)
    if operator == ">":
        return bool(value > threshold)
    if operator == "<":
        return bool(value < threshold)
    if operator == "==":
        return bool(value == threshold)
    raise ValueError(f"Unsupported operator: {operator}")


def _build_level_record(level: LevelSpec, checkpoint: str, suite_used: str, metric_value: float, passed: bool, metrics: dict) -> dict:
    return {
        "level": level.name,
        "description": level.description,
        "checkpoint": checkpoint,
        "suite": suite_used,
        "metric": level.metric,
        "operator": level.operator,
        "threshold": level.threshold,
        "value": metric_value,
        "passed": int(bool(passed)),
        "valid_rate": metrics["valid_rate"],
        "success_rate": metrics["success_rate"],
        "first_gate_pass_rate": metrics["first_gate_pass_rate"],
        "two_gate_pass_rate": metrics["two_gate_pass_rate"],
        "mean_gates_passed": metrics["mean_gates_passed"],
        "median_completion_time_valid": metrics["median_completion_time_valid"],
        "max_consecutive_successes": metrics["max_consecutive_successes"],
        "mean_progress": metrics["mean_progress"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate drone_race checkpoints against L1-L9 governance gates")
    parser.add_argument("--levels-config", type=str, default="pufferlib/config/drone_race_levels.ini")
    parser.add_argument("--mode", choices=["scripted", "policy"], default="policy")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--checkpoint-label", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true", default=True)
    parser.add_argument("--output-json", type=str, default="experiments/drone_race_levels_report.json")
    parser.add_argument("--output-csv", type=str, default="experiments/drone_race_levels_report.csv")
    parser.add_argument("--episode-csv", type=str, default="experiments/drone_race_levels_episodes.csv")
    args = parser.parse_args()

    if args.mode == "policy" and not args.model_path:
        raise SystemExit("Policy mode requires --model-path")

    checkpoint = args.checkpoint_label or (args.model_path if args.model_path else "scripted_baseline")

    meta, levels = load_levels_config(args.levels_config)
    eval_module = _load_eval_module()

    all_level_records = []
    all_episode_rows = []

    for level in levels:
        seeds, suite_name = eval_module.resolve_eval_seeds(level.episodes, args.seed, level.suite)
        rows, summary = eval_module._evaluate_checkpoint(
            checkpoint_label=checkpoint,
            model_path=args.model_path,
            seeds=seeds,
            suite_name=suite_name,
            mode=args.mode,
            device=args.device,
            hidden_size=args.hidden_size,
            env_kwargs=level.env_kwargs,
            deterministic=args.deterministic,
        )

        for row in rows:
            row["level"] = level.name
        all_episode_rows.extend(rows)

        metrics = compute_level_metrics(rows)
        if level.metric not in metrics:
            raise ValueError(f"Unsupported metric '{level.metric}' in level {level.name}")
        value = metrics[level.metric]
        passed = evaluate_threshold(value, level.operator, level.threshold)

        record = _build_level_record(
            level=level,
            checkpoint=checkpoint,
            suite_used=summary["suite"],
            metric_value=value,
            passed=passed,
            metrics=metrics,
        )
        all_level_records.append(record)

        print(
            f"{level.name}: metric={level.metric}, value={value:.4f}, "
            f"target {level.operator} {level.threshold}, passed={passed}"
        )

        if args.stop_on_fail and not passed:
            print(f"Stopping at first failed level: {level.name}")
            break

    overall_pass = bool(all_level_records) and all(int(r["passed"]) == 1 for r in all_level_records)
    final_report = {
        "checkpoint": checkpoint,
        "mode": args.mode,
        "levels_config": args.levels_config,
        "meta": meta,
        "overall_pass": overall_pass,
        "levels_evaluated": len(all_level_records),
        "levels": all_level_records,
    }

    _ensure_parent(args.output_json)
    with open(args.output_json, "w") as f:
        json.dump(final_report, f, indent=2)

    if all_level_records:
        _ensure_parent(args.output_csv)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_level_records[0].keys())
            writer.writeheader()
            writer.writerows(all_level_records)

    if all_episode_rows:
        _ensure_parent(args.episode_csv)
        with open(args.episode_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_episode_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_episode_rows)

    print(f"Level report JSON: {args.output_json}")
    if all_level_records:
        print(f"Level report CSV: {args.output_csv}")
    if all_episode_rows:
        print(f"Episode CSV: {args.episode_csv}")
    print(f"overall_pass={overall_pass}")


if __name__ == "__main__":
    main()
