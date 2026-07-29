#!/usr/bin/env python3
"""Convert native teacher-intervention traces into recurrent BC episodes.

The native environment records the action actually executed by the plant in
the next observation's four ``last_action`` fields.  This lets us distill a
teacher-intervened trajectory without adding privileged labels to the deployed
32-value policy ABI.  The terminal observation is a reset observation, so the
last pre-reset action (which has no following observation) is intentionally
dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


OBSERVATIONS = 32
ACTIONS = 4
LAST_ACTION = slice(19, 23)
RECORD_WIDTH = OBSERVATIONS + ACTIONS + 1


def build_records(
    observations: np.ndarray,
    terminals: np.ndarray,
    *,
    rewards: np.ndarray | None = None,
    success_reward_threshold: float | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    observations = np.asarray(observations, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=np.float32)
    if observations.ndim != 3 or observations.shape[2] != OBSERVATIONS:
        raise ValueError("observations must have shape [time, agents, 32]")
    if terminals.shape != observations.shape[:2]:
        raise ValueError("terminal shape does not match observations")
    if success_reward_threshold is not None:
        if rewards is None:
            raise ValueError("rewards are required for successful-only filtering")
        rewards = np.asarray(rewards, dtype=np.float32)
        if rewards.shape != observations.shape[:2]:
            raise ValueError("reward shape does not match observations")

    episodes: list[np.ndarray] = []
    dropped_actions = 0
    rejected_episodes = 0
    for agent in range(observations.shape[1]):
        terminal_indices = np.flatnonzero(terminals[:, agent] > 0.5)
        stop = int(terminal_indices[0]) if terminal_indices.size else observations.shape[0]
        if success_reward_threshold is not None and (
            not terminal_indices.size
            or rewards is None
            or float(rewards[stop, agent]) < success_reward_threshold
        ):
            rejected_episodes += 1
            continue
        # t+1 carries the action executed for observation t.  Exclude the last
        # pre-reset action because a reset clears its following last_action.
        usable = max(stop - 1, 0)
        if usable == 0:
            continue
        episode = np.zeros((usable, RECORD_WIDTH), dtype=np.float32)
        episode[:, :OBSERVATIONS] = observations[:usable, agent]
        episode[:, OBSERVATIONS : OBSERVATIONS + ACTIONS] = observations[
            1 : usable + 1, agent, LAST_ACTION
        ]
        episode[0, -1] = 1.0
        episodes.append(episode)
        dropped_actions += 1

    if not episodes:
        raise ValueError("trace contains no usable intervention episodes")
    records = np.concatenate(episodes, axis=0)
    summary = {
        "episodes": len(episodes),
        "records": int(records.shape[0]),
        "dropped_unobservable_final_actions": dropped_actions,
    }
    if success_reward_threshold is not None:
        summary["rejected_unsuccessful_episodes"] = rejected_episodes
    return records, summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--json-path", type=Path)
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="Keep only episodes whose terminal reward indicates a finish.",
    )
    parser.add_argument(
        "--success-reward-threshold",
        type=float,
        default=100.0,
        help="Minimum terminal reward for --successful-only (default: 100).",
    )
    args = parser.parse_args()

    with np.load(args.trace, allow_pickle=False) as trace:
        records, summary = build_records(
            trace["observations"],
            trace["terminals"],
            rewards=trace["rewards"] if args.successful_only else None,
            success_reward_threshold=(
                args.success_reward_threshold if args.successful_only else None
            ),
        )
        source_metadata = json.loads(str(trace["metadata"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records.tofile(args.output)
    report = {
        **summary,
        "trace": str(args.trace),
        "trace_sha256": _sha256(args.trace),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "record_width": RECORD_WIDTH,
        "observation_width": OBSERVATIONS,
        "action_width": ACTIONS,
        "label_source": "next_observation_last_action",
        "successful_only": args.successful_only,
        "success_reward_threshold": (
            args.success_reward_threshold if args.successful_only else None
        ),
        "source_metadata": source_metadata,
    }
    json_path = args.json_path or args.output.with_suffix(args.output.suffix + ".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
