#!/usr/bin/env python3
"""Prefix native Gate-2 teacher episodes with the legal N295 policy trace.

The resulting records train a recurrent PufferLib policy from the state it
actually carries into VQ2 Gate 2.  The prefix contains only deployed
observations/actions.  Native privileged state remains confined to the
already-collected offline teacher action labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


OBSERVATIONS = 32
ACTIONS = 4
RECORD_WIDTH = OBSERVATIONS + ACTIONS + 1
LAST_ACTION_START = 19
PROGRESS_INDEX = 23
PHASE_ONEHOT_START = 24


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prefix(report_path: Path) -> np.ndarray:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    samples = report.get("policy_trace", {}).get("samples", [])
    if not samples:
        raise ValueError("official report contains no policy_trace.samples")
    records = np.zeros((len(samples), RECORD_WIDTH), dtype=np.float32)
    for index, sample in enumerate(samples):
        observation = np.asarray(sample.get("observation"), dtype=np.float32)
        action = np.asarray(sample.get("normalized_action"), dtype=np.float32)
        if observation.shape != (OBSERVATIONS,):
            raise ValueError(f"prefix sample {index} does not contain 32 observations")
        if action.shape != (ACTIONS,):
            raise ValueError(f"prefix sample {index} does not contain four actions")
        if int(sample.get("official_active_gate_index", -1)) != 0:
            raise ValueError("prefix must contain only the accepted Gate-1 phase")
        records[index, :OBSERVATIONS] = observation
        records[index, OBSERVATIONS:OBSERVATIONS + ACTIONS] = action
    records[0, -1] = 1.0
    return records


def load_teacher_episodes(dataset_path: Path) -> list[np.ndarray]:
    values = np.fromfile(dataset_path, dtype=np.float32)
    if values.size % RECORD_WIDTH:
        raise ValueError("teacher dataset contains a partial record")
    records = values.reshape(-1, RECORD_WIDTH)
    starts = np.flatnonzero(records[:, -1] > 0.5)
    if not len(starts) or starts[0] != 0:
        raise ValueError("teacher dataset does not begin with a reset record")
    ends = np.r_[starts[1:], len(records)]
    return [records[start:end].copy() for start, end in zip(starts, ends)]


def build_prefixed_records(
    prefix: np.ndarray,
    teacher_episodes: list[np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    prefix = np.asarray(prefix, dtype=np.float32)
    if prefix.ndim != 2 or prefix.shape[1] != RECORD_WIDTH:
        raise ValueError("prefix must have shape [samples, 37]")
    if not teacher_episodes:
        raise ValueError("at least one teacher episode is required")
    if not np.isclose(prefix[0, -1], 1.0) or np.count_nonzero(prefix[:, -1]) != 1:
        raise ValueError("prefix must contain exactly one leading reset marker")

    final_prefix_action = prefix[-1, OBSERVATIONS:OBSERVATIONS + ACTIONS]
    stitched: list[np.ndarray] = []
    lengths: list[int] = []
    for episode_index, teacher in enumerate(teacher_episodes):
        teacher = np.asarray(teacher, dtype=np.float32).copy()
        if teacher.ndim != 2 or teacher.shape[1] != RECORD_WIDTH:
            raise ValueError(f"teacher episode {episode_index} has invalid shape")
        if len(teacher) == 0:
            raise ValueError(f"teacher episode {episode_index} is empty")
        progress = teacher[:, PROGRESS_INDEX]
        if not np.allclose(progress, 1.0 / 6.0, atol=1e-5):
            raise ValueError("teacher episode is not the active Gate-2 phase")
        if not np.all(teacher[:, PHASE_ONEHOT_START + 1] > 0.5):
            raise ValueError("teacher episode lacks the active Gate-2 one-hot")

        # The custom native reset has no preceding action.  Replace that one
        # public input with the last action actually emitted in N295, matching
        # the first Gate-2 inference in the official continuation.
        teacher[0, LAST_ACTION_START:LAST_ACTION_START + ACTIONS] = final_prefix_action
        teacher[:, -1] = 0.0
        combined = np.concatenate((prefix, teacher), axis=0)
        stitched.append(combined)
        lengths.append(len(combined))
    return np.concatenate(stitched, axis=0), lengths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_report", type=Path)
    parser.add_argument("teacher_dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    prefix = load_prefix(args.official_report)
    teacher_episodes = load_teacher_episodes(args.teacher_dataset)
    records, lengths = build_prefixed_records(prefix, teacher_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records.astype(np.float32, copy=False).tofile(args.output)
    summary = {
        "official_report": str(args.official_report),
        "official_report_sha256": sha256_file(args.official_report),
        "teacher_dataset": str(args.teacher_dataset),
        "teacher_dataset_sha256": sha256_file(args.teacher_dataset),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "episodes": len(lengths),
        "prefix_records_per_episode": int(len(prefix)),
        "records": int(len(records)),
        "minimum_episode_length": min(lengths),
        "maximum_episode_length": max(lengths),
        "student_observation_width": OBSERVATIONS,
        "privileged_student_inputs": False,
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
