#!/usr/bin/env python3
"""Validate N534 and select the largest near-fastest microbatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "logs"
    / "drone_race_vq2_informed_dreamer"
    / "n534_microbatch_throughput"
)
VRAM_LIMIT_BYTES = int(7.2 * 1024**3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(run: Path) -> dict:
    records = {}
    for microbatch in (4, 8, 16):
        path = run / f"mb{microbatch}" / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"mb{microbatch} has non-finite metric {key}")
        expected = {
            "amp_dtype": "bfloat16",
            "logical_batch_size": 16,
            "world_microbatch_size": microbatch,
            "world_gradient_accumulation_steps": 16 // microbatch,
            "replay_context": 16,
            "world_sequence_length": 128,
            "imagination_horizon": 16,
            "updates": 4,
            "actor_updates": 4,
            "world_optimizer_steps": 4,
            "critic_optimizer_steps": 4,
            "skipped_updates_no_sequence": 0,
            "configured_steady_world_records_per_collected_transition": 2.0,
        }
        for key, value in expected.items():
            if metrics.get(key) != value:
                raise RuntimeError(
                    f"mb{microbatch} {key}: expected {value!r}, got {metrics.get(key)!r}"
                )
        peak = int(metrics["cuda_peak_memory_allocated_bytes"])
        if peak >= VRAM_LIMIT_BYTES:
            raise RuntimeError(f"mb{microbatch} exceeds allocated VRAM ceiling")
        records[microbatch] = {
            "metrics_sha256": _sha256(path),
            "checkpoint_sha256": _sha256(run / f"mb{microbatch}" / "model.pt"),
            "wall_seconds": float(metrics["wall_seconds"]),
            "agent_steps_per_second": float(metrics["agent_steps_per_second"]),
            "peak_allocated_bytes": peak,
            "peak_reserved_bytes": int(metrics["cuda_peak_memory_reserved_bytes"]),
            "world_gradient_accumulation_steps": int(
                metrics["world_gradient_accumulation_steps"]
            ),
        }

    fastest_wall = min(value["wall_seconds"] for value in records.values())
    passing = [
        microbatch
        for microbatch, value in records.items()
        if value["wall_seconds"] <= 1.10 * fastest_wall
    ]
    selected = max(passing) if passing else 4
    return {
        "status": "pass",
        "selection_rule": "largest microbatch within 110% of fastest passing wall time",
        "selected_world_microbatch_size": selected,
        "selected_gradient_accumulation_steps": 16 // selected,
        "fastest_wall_seconds": fastest_wall,
        "vram_limit_bytes": VRAM_LIMIT_BYTES,
        "runs": {str(key): value for key, value in records.items()},
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
