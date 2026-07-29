#!/usr/bin/env python3
"""Validate the Python DAgger query against native executed oracle actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.torch_pufferl import _cpu_tensor
from pufferlib.vq2_informed import ACTION_HISTORY, ENV_OBS_SIZE
from pufferlib.vq2_oracle import alignment_oracle_action
from scripts.eval_vq2_native_oracle import (
    BACKEND_ENV_NAME,
    flatten_log,
    load_fixed_config,
    oracle_passes,
)


TAG = "vq2_sf016_alignment_oracle_query_parity_64"
DEFAULT_OUTPUT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG
)
AGENTS = 64
EPISODES = 64
SEED = 42016
EPISODE_SECONDS = 180.0
TARGET_SPEED_M_S = 2.0
MAX_PARITY_ERROR = 5e-5


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parity_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    return load_fixed_config(
        pufferl_module,
        agents=AGENTS,
        episodes=EPISODES,
        seed=SEED,
        controller="governed",
        target_speed_m_s=TARGET_SPEED_M_S,
        episode_seconds=EPISODE_SECONDS,
        randomized_course=True,
    )


def parity_passes(
    metrics: dict[str, float],
    *,
    samples: int,
    maximum_error: np.ndarray,
) -> bool:
    return (
        samples > 0
        and maximum_error.shape == (4,)
        and bool(np.all(np.isfinite(maximum_error)))
        and float(maximum_error.max()) <= MAX_PARITY_ERROR
        and oracle_passes(metrics, episodes=EPISODES)
    )


def validate(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled backend is not drone_race_vision")
    config, overrides = parity_config(pufferl)
    environment = config["env"]
    if (
        float(environment["teacher_action_blend"]) != 1.0
        or int(environment["teacher_alignment_governor"]) != 1
    ):
        raise RuntimeError("parity screen requires the admitted native oracle")

    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native ABI differs from SF016")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    zero_actions = torch.zeros((AGENTS, 4), dtype=torch.float32)
    done = np.zeros(AGENTS, dtype=bool)
    maximum_error = np.zeros(4, dtype=np.float64)
    absolute_error_sum = np.zeros(4, dtype=np.float64)
    samples = 0
    vector_steps = 0
    started = time.perf_counter()
    native_log: dict[str, Any] = {}
    try:
        vector.reset()
        for step in range(int(environment["max_steps"])):
            active = ~done
            if not active.any():
                break
            current = observations.numpy().copy()
            queried = alignment_oracle_action(current)
            vector.cpu_step(zero_actions.data_ptr())
            executed = observations.numpy()[
                :, ACTION_HISTORY.start : ACTION_HISTORY.start + 4
            ]
            error = np.abs(queried[active] - executed[active]).astype(np.float64)
            maximum_error = np.maximum(maximum_error, error.max(axis=0))
            absolute_error_sum += error.sum(axis=0)
            samples += int(active.sum())
            done |= terminals.numpy() > 0.5
            vector_steps = step + 1
        native_log = dict(vector.log())
    finally:
        vector.close()
    metrics = flatten_log(pufferl, native_log)
    passed = parity_passes(
        metrics, samples=samples, maximum_error=maximum_error
    )
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "docs/vq2_sf016_alignment_oracle_query_preregistration_2026-07-28.md",
    ]
    report = {
        "schema": "vq2_alignment_oracle_query_parity_v1",
        "tag": TAG,
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "vector_steps": vector_steps,
        "samples": samples,
        "maximum_absolute_error": maximum_error.tolist(),
        "mean_absolute_error": (absolute_error_sum / max(samples, 1)).tolist(),
        "parity_threshold": MAX_PARITY_ERROR,
        "metrics": metrics,
        "parity_passed": passed,
        "loader_overrides": overrides,
        "wall_time_seconds": time.perf_counter() - started,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
        },
        "safety": {
            "persistent_labels_written": 0,
            "student_actions_executed": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = validate(args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["parity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
