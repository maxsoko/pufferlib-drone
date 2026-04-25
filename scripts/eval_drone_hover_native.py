#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pufferlib import _C
from pufferlib import pufferl


def _ensure_parent(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _load_config(env_name):
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        return pufferl.load_config(env_name)
    finally:
        sys.argv = saved_argv


def main():
    parser = argparse.ArgumentParser(
        description="Native v4 drone hover rollout smoke/eval for control-prior benchmarking"
    )
    parser.add_argument("--env-name", default="drone", help="Must match the env used to build pufferlib._C")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--csv-path", default="logs/drone_hover_native_eval.csv")
    parser.add_argument("--json-path", default="logs/drone_hover_native_eval.json")
    parser.add_argument(
        "--action",
        type=float,
        default=0.0,
        help="Constant motor action. In native v4 drone, 0.0 is centered hover thrust.",
    )
    args = parser.parse_args()

    compiled_env = getattr(_C, "env_name", None)
    if compiled_env != args.env_name:
        raise RuntimeError(
            f"pufferlib._C was built for {compiled_env!r}; rebuild with `bash build.sh {args.env_name} --cpu`"
        )

    cfg = _load_config(args.env_name)
    vec = _C.create_vec(cfg, gpu=0)
    vec.reset()

    actions = torch.full((vec.total_agents, vec.num_atns), float(args.action), dtype=torch.float32)
    for _ in range(args.steps):
        vec.cpu_step(actions.data_ptr())

    log = dict(vec.log())
    log.update({
        "steps": int(args.steps),
        "episodes_logged": float(log.get("n", 0.0)),
    })
    vec.close()

    _ensure_parent(args.csv_path)
    keys = sorted(log)
    with open(args.csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(log)

    _ensure_parent(args.json_path)
    with open(args.json_path, "w") as f:
        json.dump(log, f, indent=2, sort_keys=True)

    print(log)


if __name__ == "__main__":
    main()
