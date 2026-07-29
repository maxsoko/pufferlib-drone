#!/usr/bin/env python3
"""Generate one bounded antithetic ES generation in a checkpoint tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from convert_policy_checkpoint_layout import pack_layout, tensor_counts, unpack_layout
    from policy_callable_checkpoint import CheckpointPolicy
except ModuleNotFoundError:  # Imported as scripts.evolve_policy_tensor.
    from scripts.convert_policy_checkpoint_layout import (
        pack_layout,
        tensor_counts,
        unpack_layout,
    )
    from scripts.policy_callable_checkpoint import CheckpointPolicy


def _load_tensors(
    path: Path,
    *,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_actions: int,
    layout_precision_bytes: int,
) -> tuple[list[int], list[np.ndarray]]:
    CheckpointPolicy.load(
        str(path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
        layout_precision_bytes=layout_precision_bytes,
    )
    counts = tensor_counts(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )
    tensors = unpack_layout(
        np.fromfile(path, dtype=np.float32),
        counts,
        precision_bytes=layout_precision_bytes,
    )
    return counts, tensors


def _write_candidate(
    path: Path,
    *,
    counts: list[int],
    tensors: list[np.ndarray],
    layout_precision_bytes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_layout(
        tensors,
        counts,
        precision_bytes=layout_precision_bytes,
    ).tofile(path)


def generate_antithetic(
    parent_path: Path,
    output_dir: Path,
    *,
    tensor_index: int,
    radius: float,
    num_directions: int,
    seed: int,
    input_dim: int = 32,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_actions: int = 4,
    layout_precision_bytes: int = 4,
) -> list[dict[str, float | int | str]]:
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if num_directions < 1:
        raise ValueError("num_directions must be positive")
    counts, parent = _load_tensors(
        parent_path,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
        layout_precision_bytes=layout_precision_bytes,
    )
    if not 0 <= tensor_index < len(parent):
        raise ValueError("tensor index is outside the checkpoint")

    rng = np.random.default_rng(seed)
    base = parent[tensor_index]
    records: list[dict[str, float | int | str]] = []
    for direction_index in range(num_directions):
        direction = rng.standard_normal(base.size).astype(np.float64)
        direction /= np.linalg.norm(direction)
        requested_delta = np.asarray(radius * direction, dtype=np.float32)
        plus_value = np.asarray(base + requested_delta, dtype=np.float32)
        minus_value = np.asarray(base - requested_delta, dtype=np.float32)
        plus_delta = plus_value - base
        minus_delta = minus_value - base
        for sign, value, delta in (
            ("plus", plus_value, plus_delta),
            ("minus", minus_value, minus_delta),
        ):
            candidate = [tensor.copy() for tensor in parent]
            candidate[tensor_index] = value
            path = output_dir / f"direction{direction_index}_{sign}.bin"
            _write_candidate(
                path,
                counts=counts,
                tensors=candidate,
                layout_precision_bytes=layout_precision_bytes,
            )
            records.append(
                {
                    "direction": direction_index,
                    "sign": sign,
                    "path": str(path),
                    "actual_l2": float(np.linalg.norm(delta)),
                    "changed_values": int(np.count_nonzero(delta)),
                }
            )
    return records


def report_fitness(path: Path) -> float:
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    gates = float(metrics["env/gates_passed"])
    radial = float(metrics["env/avg_terminal_crossing_radial"])
    crash = float(metrics["env/crash"])
    return 100.0 * gates - radial - 100.0 * crash


def form_fitness_direction(
    parent_path: Path,
    probe_dir: Path,
    output_dir: Path,
    *,
    tensor_index: int,
    radius: float,
    num_directions: int,
    input_dim: int = 32,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_actions: int = 4,
    layout_precision_bytes: int = 4,
) -> list[dict[str, float | str]]:
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    counts, parent = _load_tensors(
        parent_path,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
        layout_precision_bytes=layout_precision_bytes,
    )
    if not 0 <= tensor_index < len(parent):
        raise ValueError("tensor index is outside the checkpoint")

    gradient = np.zeros(parent[tensor_index].size, dtype=np.float64)
    records: list[dict[str, float | str]] = []
    for direction_index in range(num_directions):
        plus_path = probe_dir / f"direction{direction_index}_plus.bin"
        minus_path = probe_dir / f"direction{direction_index}_minus.bin"
        _, plus = _load_tensors(
            plus_path,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_actions=num_actions,
            layout_precision_bytes=layout_precision_bytes,
        )
        _, minus = _load_tensors(
            minus_path,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_actions=num_actions,
            layout_precision_bytes=layout_precision_bytes,
        )
        direction = np.asarray(
            plus[tensor_index] - minus[tensor_index], dtype=np.float64
        )
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm == 0.0:
            raise ValueError(f"direction {direction_index} has no representable FP32 delta")
        direction /= direction_norm
        plus_fitness = report_fitness(
            probe_dir / f"direction{direction_index}_plus_floor6_exact1024.json"
        )
        minus_fitness = report_fitness(
            probe_dir / f"direction{direction_index}_minus_floor6_exact1024.json"
        )
        difference = plus_fitness - minus_fitness
        gradient += difference * direction
        records.append(
            {
                "direction": str(direction_index),
                "plus_fitness": plus_fitness,
                "minus_fitness": minus_fitness,
                "fitness_difference": difference,
            }
        )

    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm == 0.0:
        raise ValueError("antithetic probes produced a zero fitness direction")
    gradient /= gradient_norm
    requested_delta = np.asarray(radius * gradient, dtype=np.float32)
    base = parent[tensor_index]
    for sign, multiplier in (("plus", 1.0), ("minus", -1.0)):
        value = np.asarray(base + np.float32(multiplier) * requested_delta, dtype=np.float32)
        delta = value - base
        candidate = [tensor.copy() for tensor in parent]
        candidate[tensor_index] = value
        path = output_dir / f"fitness_direction_{sign}.bin"
        _write_candidate(
            path,
            counts=counts,
            tensors=candidate,
            layout_precision_bytes=layout_precision_bytes,
        )
        records.append(
            {
                "direction": f"fitness_{sign}",
                "path": str(path),
                "actual_l2": float(np.linalg.norm(delta)),
                "changed_values": float(np.count_nonzero(delta)),
            }
        )
    return records


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("parent", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tensor-index", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1e-6)
    parser.add_argument("--num-directions", type=int, default=4)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--layout-precision-bytes", type=int, choices=(2, 4), default=4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    _common_arguments(generate)
    generate.add_argument("--seed", type=int, default=3385)
    combine = subparsers.add_parser("combine")
    _common_arguments(combine)
    combine.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "generate":
            records = generate_antithetic(
                args.parent,
                args.output_dir,
                tensor_index=args.tensor_index,
                radius=args.radius,
                num_directions=args.num_directions,
                seed=args.seed,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_actions=args.num_actions,
                layout_precision_bytes=args.layout_precision_bytes,
            )
        else:
            records = form_fitness_direction(
                args.parent,
                args.probe_dir,
                args.output_dir,
                tensor_index=args.tensor_index,
                radius=args.radius,
                num_directions=args.num_directions,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_actions=args.num_actions,
                layout_precision_bytes=args.layout_precision_bytes,
            )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
