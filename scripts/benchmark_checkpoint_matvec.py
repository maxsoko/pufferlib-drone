#!/usr/bin/env python3
"""Benchmark exact recurrent checkpoint inference with small-matrix backends."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy, _mingru_g, _sigmoid


Matvec = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _matmul(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    return weights @ values


def _einsum(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.einsum("ij,j->i", weights, values, optimize=False)


def _reduce(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.add.reduce(weights * values[None, :], axis=1, dtype=np.float32)


BACKENDS: dict[str, Matvec] = {
    "matmul": _matmul,
    "einsum": _einsum,
    "reduce": _reduce,
}


def infer_with(
    model: CheckpointPolicy,
    observation: Sequence[float],
    matvec: Matvec,
) -> np.ndarray:
    if model.native_bf16:
        raise ValueError("benchmark currently covers the deployed FP32 path")
    values = np.asarray(observation, dtype=np.float32)
    hidden_state = matvec(model.encoder, values)
    for layer_idx, projection in enumerate(model.mingru_proj):
        projected = matvec(projection, hidden_state)
        hidden = projected[: model.hidden_dim]
        gate = projected[model.hidden_dim : 2 * model.hidden_dim]
        highway_projection = projected[2 * model.hidden_dim :]
        recurrent = model.state[layer_idx]
        output = recurrent + _sigmoid(gate) * (_mingru_g(hidden) - recurrent)
        highway = _sigmoid(highway_projection)
        hidden_state = highway * output + (np.float32(1.0) - highway) * hidden_state
        model.state[layer_idx] = output
    decoded = matvec(model.decoder, hidden_state)
    return np.clip(decoded[: model.num_actions], -1.0, 1.0)


def _load(path: str) -> CheckpointPolicy:
    return CheckpointPolicy.load(
        path,
        input_dim=23,
        hidden_dim=128,
        num_layers=3,
        num_actions=4,
        native_bf16=False,
        layout_precision_bytes=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")

    rng = np.random.default_rng(232)
    observations = rng.normal(0.0, 0.35, size=(args.steps, 23)).astype(np.float32)
    outputs: dict[str, np.ndarray] = {}
    states: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for name, backend in BACKENDS.items():
        model = _load(args.checkpoint)
        infer_with(model, observations[0], backend)
        model.reset_state()
        started = time.perf_counter()
        outputs[name] = np.stack(
            [infer_with(model, observation, backend) for observation in observations]
        )
        timings[name] = time.perf_counter() - started
        states[name] = model.state.copy()

    reference_actions = outputs["matmul"]
    reference_state = states["matmul"]
    for name in BACKENDS:
        elapsed = timings[name]
        print(
            {
                "backend": name,
                "steps": args.steps,
                "elapsed_s": elapsed,
                "inference_hz": args.steps / elapsed,
                "maximum_action_error": float(
                    np.max(np.abs(outputs[name] - reference_actions))
                ),
                "maximum_state_error": float(
                    np.max(np.abs(states[name] - reference_state))
                ),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
