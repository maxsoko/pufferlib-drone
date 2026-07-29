#!/usr/bin/env python3
"""Validate N535 and project steady-state discovery wall time."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "logs"
    / "drone_race_vq2_informed_dreamer"
    / "n535_replay_density_throughput"
)
VRAM_LIMIT_BYTES = int(7.2 * 1024**3)
AGENTS = 64
VECTOR_STEPS = 96
WORLD_RECORDS_PER_UPDATE = 16 * 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(run: Path) -> dict:
    specifications = {
        "d1": (1.0, 2),
        "d4": (4.0, 5),
        "d16": (16.0, 17),
        "d32": (32.0, 34),
    }
    records = {}
    update_counts = []
    wall_times = []
    for tag, (density, updates) in specifications.items():
        metrics_path = run / tag / "metrics.json"
        checkpoint_path = run / tag / "model.pt"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"{tag} has non-finite metric {key}")
        expected = {
            "configured_steady_world_records_per_collected_transition": density,
            "updates": updates,
            "actor_updates": updates,
            "world_optimizer_steps": updates,
            "critic_optimizer_steps": updates,
            "logical_batch_size": 16,
            "world_microbatch_size": 16,
            "world_gradient_accumulation_steps": 1,
            "replay_context": 16,
            "world_sequence_length": 64,
            "imagination_horizon": 16,
            "amp_dtype": "bfloat16",
            "skipped_updates_no_sequence": 0,
            "replay_steps": 96,
            "replay_capacity_steps": 96,
        }
        for key, value in expected.items():
            if metrics.get(key) != value:
                raise RuntimeError(
                    f"{tag} {key}: expected {value!r}, got {metrics.get(key)!r}"
                )
        peak = int(metrics["cuda_peak_memory_allocated_bytes"])
        if peak >= VRAM_LIMIT_BYTES:
            raise RuntimeError(f"{tag} exceeds allocated VRAM ceiling")
        wall = float(metrics["wall_seconds"])
        update_counts.append(updates)
        wall_times.append(wall)
        records[tag] = {
            "configured_world_density": density,
            "updates": updates,
            "wall_seconds": wall,
            "measured_agent_steps_per_second": float(
                metrics["agent_steps_per_second"]
            ),
            "peak_allocated_bytes": peak,
            "metrics_sha256": _sha256(metrics_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
        }

    # All four runs collect the same 96 vector steps. Fit their wall time as a
    # shared collection/overhead intercept plus a per-optimizer-update cost.
    design = np.column_stack(
        (np.ones(len(update_counts), dtype=np.float64), update_counts)
    )
    intercept, seconds_per_update = np.linalg.lstsq(
        design, np.asarray(wall_times, dtype=np.float64), rcond=None
    )[0]
    if intercept <= 0.0 or seconds_per_update <= 0.0:
        raise RuntimeError("density timing fit is nonphysical")
    seconds_per_collection_vector_step = intercept / VECTOR_STEPS

    targets = (125_000, 500_000, 2_000_000, 17_000_000)
    projections = {}
    qualified = []
    for tag, (density, _) in specifications.items():
        updates_per_vector_step = density * AGENTS / WORLD_RECORDS_PER_UPDATE
        steady_seconds_per_vector_step = (
            seconds_per_collection_vector_step
            + updates_per_vector_step * seconds_per_update
        )
        transitions_per_second = AGENTS / steady_seconds_per_vector_step
        hours = {
            str(target): target / transitions_per_second / 3600.0
            for target in targets
        }
        projections[tag] = {
            "configured_world_density": density,
            "updates_per_vector_step": updates_per_vector_step,
            "projected_agent_transitions_per_second": transitions_per_second,
            "projected_hours": hours,
        }
        if hours["500000"] <= 6.0:
            qualified.append(tag)

    return {
        "status": "pass",
        "fit": {
            "shared_96_vector_step_collection_wall_seconds": intercept,
            "seconds_per_optimizer_update": seconds_per_update,
            "seconds_per_collection_vector_step": (
                seconds_per_collection_vector_step
            ),
        },
        "runs": records,
        "steady_state_projections": projections,
        "densities_qualified_for_efficacy_screen": qualified,
        "qualification_rule": "projected 0.5M wall time at most six hours",
        "selection_limit": (
            "throughput qualification does not select learning efficacy"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.run)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
