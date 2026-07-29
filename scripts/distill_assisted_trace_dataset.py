#!/usr/bin/env python3
"""Convert a successful native assisted trace into recurrent BC records.

The native rollout buffer stores the raw PufferLib action, while observation
slots 19..22 on the next tick store the action that actually drove the plant.
With training-only teacher blending enabled, shifting those observable
last-action values back one tick yields the teacher target without adding any
privileged state to the student dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


OBSERVATIONS = 32
ACTIONS = 4
LAST_ACTION_START = 19
RECORD_WIDTH = OBSERVATIONS + ACTIONS + 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assisted_trace_records(
    observations: np.ndarray,
    terminals: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    observations = np.asarray(observations, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=np.float32)
    if observations.ndim != 3 or observations.shape[2] != OBSERVATIONS:
        raise ValueError("observations must have shape [time, agents, 32]")
    if terminals.shape != observations.shape[:2]:
        raise ValueError("terminals must match the trace time/agent axes")

    episodes: list[np.ndarray] = []
    lengths: list[int] = []
    for agent in range(observations.shape[1]):
        reset_ticks = np.flatnonzero(terminals[:, agent] > 0.5)
        if not reset_ticks.size:
            raise ValueError(f"agent {agent} has no terminal reset marker")
        reset_tick = int(reset_ticks[0])
        if reset_tick < 2:
            raise ValueError(f"agent {agent} episode is too short to shift actions")

        # Tick reset_tick is already the next reset observation. Keep source
        # observations through reset_tick - 2 and recover their executed
        # actions from the following observation's public last-action slots.
        episode_observations = observations[: reset_tick - 1, agent]
        executed_actions = observations[
            1:reset_tick,
            agent,
            LAST_ACTION_START:LAST_ACTION_START + ACTIONS,
        ]
        reset_flags = np.zeros((len(episode_observations), 1), dtype=np.float32)
        reset_flags[0, 0] = 1.0
        episodes.append(np.concatenate(
            (episode_observations, executed_actions, reset_flags), axis=1))
        lengths.append(len(episode_observations))

    return np.concatenate(episodes, axis=0), lengths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--require-success-rate",
        type=float,
        default=1.0,
        help="reject a trace whose recorded native success rate is below this value",
    )
    args = parser.parse_args()

    with np.load(args.trace, allow_pickle=False) as trace:
        observations = np.asarray(trace["observations"], dtype=np.float32)
        terminals = np.asarray(trace["terminals"], dtype=np.float32)
        metadata = json.loads(str(trace["metadata"]))
    success_rate = float(metadata.get("metrics", {}).get("env/success_rate", 0.0))
    if success_rate < args.require_success_rate:
        raise ValueError(
            f"trace success rate {success_rate:.6f} is below "
            f"required {args.require_success_rate:.6f}"
        )

    records, lengths = assisted_trace_records(observations, terminals)
    if records.shape[1] != RECORD_WIDTH:
        raise AssertionError("internal assisted dataset width mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records.astype(np.float32, copy=False).tofile(args.output)
    summary = {
        "trace": str(args.trace),
        "trace_sha256": sha256_file(args.trace),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "trace_success_rate": success_rate,
        "episodes": len(lengths),
        "records": int(records.shape[0]),
        "record_width": RECORD_WIDTH,
        "minimum_episode_length": min(lengths),
        "maximum_episode_length": max(lengths),
        "target_source": "next_observation_last_executed_action_19_22",
        "privileged_student_inputs": False,
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
