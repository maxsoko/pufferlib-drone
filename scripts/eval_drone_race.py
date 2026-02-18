#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os

import numpy as np
import torch

import pufferlib
import pufferlib.emulation
import pufferlib.pytorch
from pufferlib.environments.drone_race.environment import DroneRaceEnv
from pufferlib.environments.drone_race.torch import Policy


SEED_SUITES = {
    # Fast, repeatable smoke leaderboard suite.
    "drone_race_v1_quick20": list(range(42, 62)),
    # Primary parity suite for checkpoint comparison.
    "drone_race_v1_full100": list(range(1000, 1100)),
}


def list_seed_suites():
    suites = {}
    for name, seeds in SEED_SUITES.items():
        suites[name] = {
            "num_seeds": len(seeds),
            "first_seed": int(seeds[0]),
            "last_seed": int(seeds[-1]),
        }
    return suites


def resolve_eval_seeds(episodes, seed, suite):
    if suite is None:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        return [int(seed + i) for i in range(episodes)], None

    if suite not in SEED_SUITES:
        raise ValueError(f"Unknown suite: {suite}")
    return list(SEED_SUITES[suite]), suite


def _ensure_parent(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _load_policy(puffer_env, model_path, device, hidden_size):
    policy = Policy(puffer_env, hidden_size=hidden_size)
    policy.to(device)
    policy.eval()

    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    return policy


def _select_action(raw_env, obs, mode, policy, device, deterministic):
    if mode == "scripted":
        return raw_env.scripted_action()

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = policy.forward_eval(obs_t)
        if deterministic and hasattr(logits, "loc"):
            action = logits.loc
        elif deterministic and isinstance(logits, torch.Tensor):
            action = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            action, _, _ = pufferlib.pytorch.sample_logits(logits)

    action = action.detach().cpu().numpy().reshape(raw_env.action_space.shape)
    return np.clip(action, raw_env.action_space.low, raw_env.action_space.high)


def _eval_once(raw_env, seed, mode, policy, device, deterministic):
    obs, _ = raw_env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info = {}
    planner_fallback_steps = 0
    setpoint_delta_sum = 0.0
    setpoint_delta_count = 0
    prev_setpoint = None

    while True:
        action = _select_action(raw_env, obs, mode, policy, device, deterministic)
        obs, reward, terminated, truncated, info = raw_env.step(action)
        total_reward += reward
        steps += 1

        planner_fallback_steps += int(info.get("planner_fallback", 0))
        current_setpoint = np.array(
            [
                float(info.get("planner_setpoint_vx", 0.0)),
                float(info.get("planner_setpoint_vy", 0.0)),
                float(info.get("planner_setpoint_vz", 0.0)),
            ],
            dtype=np.float32,
        )
        if prev_setpoint is not None:
            setpoint_delta_sum += float(np.linalg.norm(current_setpoint - prev_setpoint))
            setpoint_delta_count += 1
        prev_setpoint = current_setpoint

        if terminated or truncated:
            break

    gates_passed = int(info.get("gates_passed", 0))
    gates_total = int(info.get("gates_total", 0))
    valid_run = int(info.get("valid_run", 0))
    success = int(valid_run == 1 and gates_passed == gates_total)

    completion_time = info.get("completion_time")
    completion_time = float(completion_time) if completion_time is not None else float("nan")

    return {
        "seed": int(seed),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
        "valid_run": valid_run,
        "success": success,
        "completion_time": completion_time,
        "elapsed_time": float(info.get("elapsed_time", 0.0)),
        "gates_passed": gates_passed,
        "gates_total": gates_total,
        "crash": int(info.get("crash", 0)),
        "out_of_order": int(info.get("out_of_order", 0)),
        "missed_gate": int(info.get("missed_gate", 0)),
        "progress": float(info.get("progress", 0.0)),
        "planner_fallback_steps": int(planner_fallback_steps),
        "planner_fallback_rate": float(planner_fallback_steps / max(steps, 1)),
        "setpoint_smoothness": float(setpoint_delta_sum / max(setpoint_delta_count, 1)),
        "planner_target_gate_index_final": int(info.get("planner_target_gate_index", -1)),
    }


def _summarize(rows, checkpoint, suite_name):
    valid_times = [r["completion_time"] for r in rows if r["success"] == 1]
    seeds = [int(r["seed"]) for r in rows]

    summary = {
        "checkpoint": checkpoint,
        "suite": suite_name if suite_name is not None else "custom",
        "episodes": len(rows),
        "seed_first": min(seeds) if seeds else -1,
        "seed_last": max(seeds) if seeds else -1,
        "valid_rate": float(np.mean([r["valid_run"] for r in rows])) if rows else 0.0,
        "success_rate": float(np.mean([r["success"] for r in rows])) if rows else 0.0,
        "mean_reward": float(np.mean([r["total_reward"] for r in rows])) if rows else 0.0,
        "mean_steps": float(np.mean([r["steps"] for r in rows])) if rows else 0.0,
        "mean_completion_time": float(np.mean(valid_times)) if valid_times else float("nan"),
        "median_completion_time": float(np.median(valid_times)) if valid_times else float("nan"),
        "crash_rate": float(np.mean([r["crash"] for r in rows])) if rows else 0.0,
        "out_of_order_rate": float(np.mean([r["out_of_order"] for r in rows])) if rows else 0.0,
        "missed_gate_rate": float(np.mean([r["missed_gate"] for r in rows])) if rows else 0.0,
        "mean_gates_passed": float(np.mean([r["gates_passed"] for r in rows])) if rows else 0.0,
        "mean_planner_fallback_rate": float(np.mean([r["planner_fallback_rate"] for r in rows])) if rows else 0.0,
        "mean_setpoint_smoothness": float(np.mean([r["setpoint_smoothness"] for r in rows])) if rows else 0.0,
    }
    return summary


def _evaluate_checkpoint(checkpoint_label, model_path, seeds, suite_name, mode, device, hidden_size, env_kwargs, deterministic):
    raw_env = DroneRaceEnv(**env_kwargs)
    policy = None
    if mode == "policy":
        puffer_env = pufferlib.emulation.GymnasiumPufferEnv(env=raw_env)
        policy = _load_policy(puffer_env, model_path=model_path, device=device, hidden_size=hidden_size)

    rows = []
    for episode, rollout_seed in enumerate(seeds):
        row = _eval_once(
            raw_env=raw_env,
            seed=rollout_seed,
            mode=mode,
            policy=policy,
            device=device,
            deterministic=deterministic,
        )
        row["episode"] = int(episode)
        row["checkpoint"] = checkpoint_label
        row["suite"] = suite_name if suite_name is not None else "custom"
        rows.append(row)

    summary = _summarize(rows, checkpoint_label, suite_name)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate drone_race scripted/policy checkpoints")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite", type=str, choices=sorted(SEED_SUITES.keys()), default=None)
    parser.add_argument("--list-suites", action="store_true", help="Print available fixed-seed suites and exit")
    parser.add_argument("--mode", choices=["scripted", "policy"], default="scripted")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-glob", type=str, default=None)
    parser.add_argument("--csv-path", type=str, default="experiments/drone_race_eval.csv")
    parser.add_argument("--summary-csv-path", type=str, default="experiments/drone_race_eval_summary.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--env-kwargs", type=str, default="{}", help="JSON dict of DroneRaceEnv kwargs")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions for policy mode")
    args = parser.parse_args()

    if args.list_suites:
        for name, meta in list_seed_suites().items():
            print(f"{name}: num_seeds={meta['num_seeds']}, first_seed={meta['first_seed']}, last_seed={meta['last_seed']}")
        return

    try:
        env_kwargs = json.loads(args.env_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --env-kwargs JSON: {exc}") from exc

    try:
        seeds, suite_name = resolve_eval_seeds(args.episodes, args.seed, args.suite)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    checkpoints = []
    if args.mode == "scripted":
        checkpoints.append(("scripted_baseline", None))
    else:
        if args.model_glob:
            model_paths = sorted(glob.glob(args.model_glob))
            if not model_paths:
                raise SystemExit(f"No models found for glob: {args.model_glob}")
            checkpoints.extend((path, path) for path in model_paths)
        elif args.model_path:
            checkpoints.append((args.model_path, args.model_path))
        else:
            raise SystemExit("Policy mode requires --model-path or --model-glob")

    episode_rows = []
    summary_rows = []

    for checkpoint_label, model_path in checkpoints:
        rows, summary = _evaluate_checkpoint(
            checkpoint_label=checkpoint_label,
            model_path=model_path,
            seeds=seeds,
            suite_name=suite_name,
            mode=args.mode,
            device=args.device,
            hidden_size=args.hidden_size,
            env_kwargs=env_kwargs,
            deterministic=args.deterministic,
        )
        episode_rows.extend(rows)
        summary_rows.append(summary)

        print(
            f"{checkpoint_label}: suite={summary['suite']}, valid_rate={summary['valid_rate']:.2%}, "
            f"success_rate={summary['success_rate']:.2%}, "
            f"mean_completion_time={summary['mean_completion_time']:.3f}"
        )

    _ensure_parent(args.csv_path)
    with open(args.csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)

    _ensure_parent(args.summary_csv_path)
    with open(args.summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Episode CSV saved to: {args.csv_path}")
    print(f"Summary CSV saved to: {args.summary_csv_path}")


if __name__ == "__main__":
    main()
