#!/usr/bin/env python3
import argparse
import ctypes
import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pufferlib import pufferl
from pufferlib import _C


def _ensure_parent(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Native v4 drone_race rollout smoke/eval")
    parser.add_argument("--env-name", default="drone_race")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--csv-path", default="logs/drone_race_native_eval.csv")
    parser.add_argument(
        "--action",
        type=float,
        default=0.0,
        help="Constant value for the first action dimension. Legacy mode treats this as motor action; competition mode treats it as forward velocity command.",
    )
    args = parser.parse_args()

    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    cfg = pufferl.load_config(args.env_name)
    sys.argv = saved_argv
    vec = _C.create_vec(cfg, gpu=0)
    vec.reset()

    actions = torch.zeros((vec.total_agents, vec.num_atns), dtype=torch.float32)
    actions[:, 0] = float(args.action)
    for _ in range(args.steps):
        vec.cpu_step(actions.data_ptr())

    log = dict(vec.log())
    if not log:
        rewards = torch.frombuffer(
            (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr),
            dtype=torch.float32,
        )
        terminals = torch.frombuffer(
            (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr),
            dtype=torch.float32,
        )
        log = {
            "steps": int(args.steps),
            "mean_reward_last_step": float(rewards.mean().item()),
            "terminal_rate_last_step": float(terminals.mean().item()),
            "episodes_logged": 0.0,
        }
    vec.close()

    _ensure_parent(args.csv_path)
    keys = sorted(log)
    with open(args.csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(log)

    print(log)


if __name__ == "__main__":
    main()
