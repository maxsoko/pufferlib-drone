#!/usr/bin/env python3
"""Validate and summarize the source-locked N532 hardware envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "logs"
    / "drone_race_vq2_informed_dreamer"
    / "n532_hardware_envelope"
)
TRAINING_SCHEMA = "vq2_dreamer_training_state_v2"
OBSERVATION_SCHEMA = "vq2_visual_ctbr_v2"
ACTION_SCHEMA = "normalized_attitude_ctbr_v1"
VRAM_LIMIT_BYTES = int(7.2 * 1024**3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metrics(path: Path) -> dict:
    values = json.loads(path.read_text(encoding="utf-8"))
    for key, value in values.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise RuntimeError(f"non-finite metric {key} in {path}")
    return values


def _require(metrics: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(
                f"{label} metric {key}: expected {value!r}, got {metrics.get(key)!r}"
            )
    peak = int(metrics["cuda_peak_memory_allocated_bytes"])
    if peak >= VRAM_LIMIT_BYTES:
        raise RuntimeError(f"{label} allocated VRAM {peak} exceeds N532 ceiling")


def _optimizer_steps(checkpoint: dict, name: str) -> list[int]:
    return sorted(
        {
            int(state["step"])
            for state in checkpoint[name]["state"].values()
            if "step" in state
        }
    )


def analyze(run: Path) -> dict:
    paths = {
        "fp32_s64": run / "fp32_s64" / "metrics.json",
        "bf16_s64_base": run / "bf16_s64_resume" / "base_metrics.json",
        "bf16_s64_resumed": run / "bf16_s64_resume" / "resumed_metrics.json",
        "bf16_s128": run / "bf16_s128" / "metrics.json",
    }
    metrics = {name: _load_metrics(path) for name, path in paths.items()}

    common = {
        "logical_batch_size": 16,
        "world_microbatch_size": 4,
        "world_gradient_accumulation_steps": 4,
        "replay_context": 16,
        "imagination_horizon": 16,
        "skipped_updates_no_sequence": 0,
    }
    _require(
        metrics["fp32_s64"],
        common
        | {
            "amp_dtype": "none",
            "world_sequence_length": 64,
            "updates": 4,
            "session_updates": 4,
            "actor_updates": 4,
            "session_actor_updates": 4,
            "world_optimizer_steps": 4,
            "critic_optimizer_steps": 4,
            "replay_steps": 128,
            "replay_capacity_steps": 128,
        },
        "FP32 sequence 64",
    )
    _require(
        metrics["bf16_s64_base"],
        common
        | {
            "amp_dtype": "bfloat16",
            "world_sequence_length": 64,
            "updates": 2,
            "session_updates": 2,
            "actor_updates": 2,
            "session_actor_updates": 2,
            "resume_sessions": 0,
            "resume_boundary_invalid_records": 0,
            "replay_steps": 96,
            "replay_capacity_steps": 128,
        },
        "BF16 sequence 64 base",
    )
    _require(
        metrics["bf16_s64_resumed"],
        common
        | {
            "amp_dtype": "bfloat16",
            "world_sequence_length": 64,
            "starting_environment_step": 96,
            "environment_steps": 128,
            "session_environment_steps": 32,
            "updates": 4,
            "session_updates": 2,
            "actor_updates": 4,
            "session_actor_updates": 2,
            "resume_sessions": 1,
            "resume_boundary_invalid_records": 64,
            "replay_steps": 128,
            "replay_capacity_steps": 128,
        },
        "BF16 sequence 64 resume",
    )
    _require(
        metrics["bf16_s128"],
        common
        | {
            "amp_dtype": "bfloat16",
            "world_sequence_length": 128,
            "updates": 4,
            "session_updates": 4,
            "actor_updates": 4,
            "session_actor_updates": 4,
            "world_optimizer_steps": 4,
            "critic_optimizer_steps": 4,
            "replay_steps": 192,
            "replay_capacity_steps": 192,
        },
        "BF16 sequence 128",
    )

    checkpoint_paths = {
        "fp32_s64": run / "fp32_s64" / "model.pt",
        "bf16_s64_base": run / "bf16_s64_resume" / "base_model.pt",
        "bf16_s64_resumed": run / "bf16_s64_resume" / "resumed_model.pt",
        "bf16_s128": run / "bf16_s128" / "model.pt",
    }
    expected_steps = {
        "fp32_s64": 4,
        "bf16_s64_base": 2,
        "bf16_s64_resumed": 4,
        "bf16_s128": 4,
    }
    checkpoint_evidence = {}
    for name, path in checkpoint_paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("training_state_schema") != TRAINING_SCHEMA:
            raise RuntimeError(f"{name} training schema mismatch")
        if checkpoint.get("observation_schema") != OBSERVATION_SCHEMA:
            raise RuntimeError(f"{name} observation schema mismatch")
        if checkpoint.get("action_schema") != ACTION_SCHEMA:
            raise RuntimeError(f"{name} action schema mismatch")
        steps = {
            optimizer: _optimizer_steps(checkpoint, optimizer)
            for optimizer in (
                "world_optimizer",
                "actor_optimizer",
                "critic_optimizer",
            )
        }
        expected = [expected_steps[name]]
        if any(value != expected for value in steps.values()):
            raise RuntimeError(f"{name} optimizer-state counters mismatch: {steps}")
        checkpoint_evidence[name] = {
            "sha256": _sha256(path),
            "environment_step": int(checkpoint["environment_step"]),
            "updates": int(checkpoint["updates"]),
            "actor_updates": int(checkpoint["actor_updates"]),
            "resume_sessions": int(checkpoint["resume_sessions"]),
            "optimizer_step_values": steps,
        }

    replay_dir = run / "bf16_s64_resume" / "replay"
    metadata = json.loads((replay_dir / "metadata.json").read_text())
    valid = np.memmap(
        replay_dir / "valid.dat",
        dtype=np.uint8,
        mode="r",
        shape=(int(metadata["capacity"]), int(metadata["agents"])),
    )
    invalid_slots = np.where(valid.sum(axis=1) == 0)[0].tolist()
    partial_slots = np.where(
        (valid.sum(axis=1) > 0) & (valid.sum(axis=1) < metadata["agents"])
    )[0].tolist()
    if invalid_slots != [96] or partial_slots:
        raise RuntimeError(
            f"resume boundary mismatch: invalid={invalid_slots}, partial={partial_slots}"
        )

    peaks = {
        name: int(value["cuda_peak_memory_allocated_bytes"])
        for name, value in metrics.items()
    }
    walls = {name: float(value["wall_seconds"]) for name, value in metrics.items()}
    return {
        "status": "pass",
        "selected_setting": {
            "amp_dtype": "bfloat16",
            "logical_batch_size": 16,
            "world_microbatch_size": 4,
            "world_gradient_accumulation_steps": 4,
            "replay_context": 16,
            "initial_world_sequence_length": 64,
            "verified_world_sequence_length": 128,
            "imagination_horizon": 16,
        },
        "vram_limit_bytes": VRAM_LIMIT_BYTES,
        "peak_allocated_bytes": peaks,
        "peak_reserved_bytes": {
            name: int(value["cuda_peak_memory_reserved_bytes"])
            for name, value in metrics.items()
        },
        "wall_seconds": walls,
        "fp32_to_bf16_s64_resume_peak_delta_bytes": (
            peaks["fp32_s64"] - peaks["bf16_s64_resumed"]
        ),
        "sequence_128_headroom_bytes": (
            VRAM_LIMIT_BYTES - peaks["bf16_s128"]
        ),
        "resume_boundary": {
            "all_agent_invalid_physical_slots": invalid_slots,
            "partially_invalid_physical_slots": partial_slots,
            "invalid_records_reported": metrics["bf16_s64_resumed"][
                "resume_boundary_invalid_records"
            ],
        },
        "checkpoint_evidence": checkpoint_evidence,
        "raw_metric_sha256": {
            name: _sha256(path) for name, path in paths.items()
        },
        "replay_metadata_sha256": _sha256(replay_dir / "metadata.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.run)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
