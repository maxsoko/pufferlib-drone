#!/usr/bin/env python3
"""Read-only chronological coverage audit for a quantized VQ2 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_event_latent_separability import _phase_split_counts
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256


AUDIT_SCHEMA = "vq2_dense_replay_chronological_coverage_v2"


def _parse_phase_values(value: str) -> tuple[int, ...]:
    phases = tuple(int(item) for item in value.split(","))
    if not phases or len(set(phases)) != len(phases):
        raise ValueError("required phases must be nonempty and unique")
    if any(phase < 0 or phase >= 6 for phase in phases):
        raise ValueError("required phases must lie in [0,5]")
    return phases


def _group_hash(seed: int, split: str, phase: int, group: int) -> bytes:
    return hashlib.sha256(f"{seed}:{split}:{phase}:{group}".encode()).digest()


def _stratified_episode_group_split(
    groups: np.ndarray,
    phases: np.ndarray,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> np.ndarray:
    """Split dense rows by episode, targeting fractions of phase-bearing groups."""

    groups = np.asarray(groups, dtype=np.int64)
    phases = np.asarray(phases, dtype=np.int64)
    if groups.ndim != 1 or phases.shape != groups.shape or groups.size == 0:
        raise ValueError("groups and phases must be aligned nonempty vectors")
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("validation/test fractions must lie in (0,1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation plus test fraction must be below one")
    unique_groups = set(int(value) for value in np.unique(groups))
    group_phases = {
        group: set(int(value) for value in phases[groups == group])
        for group in unique_groups
    }
    assignment = {group: 0 for group in unique_groups}
    remaining = set(unique_groups)
    phase_values = sorted(int(value) for value in np.unique(phases))
    phase_group_counts = {
        phase: sum(phase in values for values in group_phases.values())
        for phase in phase_values
    }
    for split_name, split_value, fraction in (
        ("test", 2, test_fraction),
        ("validation", 1, validation_fraction),
    ):
        selected: set[int] = set()
        targets = {
            phase: max(1, int(math.ceil(phase_group_counts[phase] * fraction)))
            for phase in phase_values
        }
        for phase in sorted(phase_values, key=lambda value: (phase_group_counts[value], value)):
            while sum(phase in group_phases[group] for group in selected) < targets[phase]:
                candidates = [
                    group
                    for group in remaining
                    if group not in selected and phase in group_phases[group]
                ]
                if not candidates:
                    raise RuntimeError(
                        f"cannot provide {split_name} episode coverage for phase {phase}"
                    )
                candidates.sort(key=lambda group: _group_hash(seed, split_name, phase, group))
                selected.add(candidates[0])
        for group in selected:
            assignment[group] = split_value
        remaining.difference_update(selected)
    encoded = np.asarray([assignment[int(group)] for group in groups], dtype=np.int8)
    for phase in phase_values:
        for split_value in (0, 1, 2):
            if not np.any((phases == phase) & (encoded == split_value)):
                raise RuntimeError(f"phase {phase} missing from dense split {split_value}")
    return encoded


def _chronological_indices(capacity: int, size: int, position: int) -> np.ndarray:
    if capacity <= 0 or not 0 <= size <= capacity or not 0 <= position < capacity:
        raise ValueError("invalid replay ring metadata")
    oldest = (position - size) % capacity
    return (oldest + np.arange(size, dtype=np.int64)) % capacity


def _episode_groups(
    valid: np.ndarray,
    continuation: np.ndarray,
    *,
    starts_at_known_boundary: bool,
) -> tuple[np.ndarray, dict[int, bool]]:
    """Assign clean episode IDs; discard a ring's unknown leading fragments."""

    valid = np.asarray(valid, dtype=bool)
    continuation = np.asarray(continuation, dtype=bool)
    if valid.shape != continuation.shape or valid.ndim != 2:
        raise ValueError("valid and continuation must be aligned [time, agent]")
    groups = np.full(valid.shape, -1, dtype=np.int64)
    next_group = 0
    complete: dict[int, bool] = {}
    for agent in range(valid.shape[1]):
        can_start = starts_at_known_boundary
        current = -1
        for step in range(valid.shape[0]):
            if not valid[step, agent]:
                can_start = True
                current = -1
                continue
            if current < 0:
                if not can_start:
                    continue
                current = next_group
                next_group += 1
                complete[current] = False
                can_start = False
            groups[step, agent] = current
            if not continuation[step, agent]:
                complete[current] = True
                current = -1
                can_start = True
    return groups, complete


def _temporal_candidates(
    groups: np.ndarray,
    *,
    depth: int,
    stride: int,
) -> np.ndarray:
    if depth <= 0 or stride <= 0:
        raise ValueError("candidate depth and stride must be positive")
    candidates: list[tuple[int, int]] = []
    for agent in range(groups.shape[1]):
        for step in range(depth - 1, groups.shape[0], stride):
            group = groups[step, agent]
            if group < 0:
                continue
            if (groups[step - depth + 1 : step + 1, agent] == group).all():
                candidates.append((step, agent))
    return np.asarray(candidates, dtype=np.int64).reshape(-1, 2)


def _range_bin_counts(forward_m: np.ndarray) -> dict[str, int]:
    forward_m = np.asarray(forward_m)
    return {
        "behind_below_minus_1m": int((forward_m < -1.0).sum()),
        "behind_minus_1m_to_0m": int(((forward_m >= -1.0) & (forward_m < 0.0)).sum()),
        "ahead_0m_to_0p5m": int(((forward_m >= 0.0) & (forward_m < 0.5)).sum()),
        "ahead_0p5m_to_1m": int(((forward_m >= 0.5) & (forward_m < 1.0)).sum()),
        "ahead_1m_to_2m": int(((forward_m >= 1.0) & (forward_m < 2.0)).sum()),
        "ahead_2m_to_4m": int(((forward_m >= 2.0) & (forward_m < 4.0)).sum()),
        "ahead_4m_to_8m": int(((forward_m >= 4.0) & (forward_m < 8.0)).sum()),
        "ahead_at_least_8m": int((forward_m >= 8.0).sum()),
    }


def _source_hashes(replay_dir: Path, n646_report: Path) -> dict[str, str]:
    result = {"n646_report": _sha256(n646_report)}
    for path in sorted(replay_dir.iterdir()):
        if path.is_file():
            result[f"replay/{path.name}"] = _sha256(path)
    return result


def audit(args: argparse.Namespace) -> dict[str, object]:
    required_phases = _parse_phase_values(args.required_phases)
    source_hashes_before = _source_hashes(args.replay_dir, args.n646_report)
    metadata = json.loads((args.replay_dir / "metadata.json").read_text(encoding="utf-8"))
    capacity = int(metadata["capacity"])
    agents = int(metadata["agents"])
    size = int(metadata["size"])
    position = int(metadata["position"])
    if metadata.get("schema") != "vq2_quantized_sequence_replay_v1":
        raise RuntimeError("replay schema mismatch")
    order = _chronological_indices(capacity, size, position)
    shapes = {
        "mask": (capacity, agents, MASK_SIZE),
        "tail": (capacity, agents, ENV_OBS_SIZE - MASK_SIZE),
        "action": (capacity, agents, 4),
        "reward": (capacity, agents),
        "continuation": (capacity, agents),
        "valid": (capacity, agents),
    }
    dtypes = {
        "mask": np.uint8,
        "tail": np.float16,
        "action": np.float16,
        "reward": np.float16,
        "continuation": np.uint8,
        "valid": np.uint8,
    }
    arrays = {
        name: np.memmap(
            args.replay_dir / f"{name}.dat",
            dtype=dtypes[name],
            mode="r",
            shape=shape,
        )
        for name, shape in shapes.items()
    }
    valid = np.asarray(arrays["valid"][order], dtype=bool)
    continuation = np.asarray(arrays["continuation"][order], dtype=bool)
    groups, complete = _episode_groups(
        valid,
        continuation,
        starts_at_known_boundary=size < capacity and int(order[0]) == 0,
    )
    candidates = _temporal_candidates(
        groups, depth=args.temporal_depth, stride=args.sample_stride
    )
    if not len(candidates):
        raise RuntimeError("replay contains no clean temporal candidate")
    chronological_tail = arrays["tail"][order]
    chronological_mask = arrays["mask"][order]
    chronological_reward = arrays["reward"][order]
    step = candidates[:, 0]
    agent = candidates[:, 1]
    candidate_group = groups[step, agent]
    candidate_tail = np.asarray(chronological_tail[step, agent], dtype=np.float32)
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    normalized_forward = candidate_tail[:, legal_tail_size + 3]
    forward_m = 10.0 * np.arctanh(np.clip(normalized_forward, -0.999999, 0.999999))
    phase = np.rint(candidate_tail[:, legal_tail_size + 32] * 6.0).astype(np.int64)
    new_frame = candidate_tail[:, 19]
    frame_age = candidate_tail[:, 20] * 0.25
    mask_mass = np.asarray(
        chronological_mask[step, agent], dtype=np.float32
    ).mean(1) / 255.0
    reward = np.asarray(chronological_reward[step, agent], dtype=np.float32)
    active = np.isin(phase, np.asarray(required_phases, dtype=np.int64))
    active_groups = candidate_group[active]
    active_phase = phase[active]
    split = _stratified_episode_group_split(
        active_groups,
        active_phase,
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    split_summary = _phase_split_counts(split, active_phase, active_groups)
    phase_counts = {
        str(value): int((phase == value).sum()) for value in sorted(np.unique(phase))
    }
    phase_group_counts = {
        str(value): int(np.unique(candidate_group[phase == value]).size)
        for value in sorted(np.unique(phase))
    }
    phase_range_bins = {
        str(value): _range_bin_counts(forward_m[phase == value])
        for value in sorted(np.unique(phase))
    }
    complete_count = sum(bool(value) for value in complete.values())
    source_hashes_after = _source_hashes(args.replay_dir, args.n646_report)
    process_gates = {
        "sources_exact": source_hashes_after == source_hashes_before,
        "ring_fully_ordered": len(np.unique(order)) == size,
        "all_required_phases_present": set(required_phases).issubset(
            set(int(value) for value in phase)
        ),
        "all_splits_cover_active_phases": all(
            all(int(count) > 0 for count in partition["phase_counts"].values())
            for partition in split_summary.values()
        ),
        "finite_targets": bool(np.isfinite(forward_m).all()),
        "read_only_memmaps": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "metadata": metadata,
        "chronological_oldest_physical_index": int(order[0]),
        "chronological_newest_physical_index": int(order[-1]),
        "valid_transitions": int(valid.sum()),
        "invalid_boundaries": int((~valid).sum()),
        "clean_episode_groups": len(complete),
        "complete_clean_episode_groups": complete_count,
        "incomplete_clean_episode_groups": len(complete) - complete_count,
        "discarded_unknown_leading_transitions": int(((groups < 0) & valid).sum()),
        "temporal_depth": args.temporal_depth,
        "sample_stride": args.sample_stride,
        "required_phases": list(required_phases),
        "temporal_candidates": int(len(candidates)),
        "active_phase_candidates": int(active.sum()),
        "active_phase_groups": int(np.unique(active_groups).size),
        "phase_candidate_counts": phase_counts,
        "phase_group_counts": phase_group_counts,
        "range_bin_counts": _range_bin_counts(forward_m),
        "phase_range_bin_counts": phase_range_bins,
        "near_plane_abs_le_0p5m_candidates": int((np.abs(forward_m) <= 0.5).sum()),
        "near_plane_abs_le_1m_candidates": int((np.abs(forward_m) <= 1.0).sum()),
        "near_plane_abs_le_0p5m_groups": int(
            np.unique(candidate_group[np.abs(forward_m) <= 0.5]).size
        ),
        "new_frame_fraction": float((new_frame > 0.5).mean()),
        "frame_age_seconds_mean": float(frame_age.mean()),
        "frame_age_seconds_max": float(frame_age.max()),
        "nonempty_mask_fraction": float((mask_mass > 0.0).mean()),
        "mask_mass_mean": float(mask_mass.mean()),
        "event_reward_candidates": int((reward > args.event_threshold).sum()),
        "fixed_group_split": split_summary,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "model_loaded": 0,
        "optimizer_steps": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n646-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--temporal-depth", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--required-phases", default="0,1,2,3,4,5")
    parser.add_argument("--split-seed", type=int, default=647)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if args.temporal_depth <= 0 or args.sample_stride <= 0:
        parser.error("temporal depth and sample stride must be positive")
    try:
        _parse_phase_values(args.required_phases)
    except ValueError as error:
        parser.error(str(error))
    if args.report.exists():
        parser.error("audit refuses to overwrite an existing report")
    return args


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
