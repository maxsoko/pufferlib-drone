#!/usr/bin/env python3
"""One recurrent policy with output heads conditioned on official race phase.

Observation[22] carries normalized official ``active_gate_index``.  The policy
restores the preceding yaw action before advancing one proven MinGRU trunk,
then selects the learned decoder head for that gate.  The head bank is part of
one policy computation: the runner receives one action and performs no flight
control, arbitration, or trajectory logic.

For the current four-gate native stage, optionally set a phase-1 checkpoint
with ``PUFFER_POLICY_PREVIOUS_CHECKPOINT_PATH``, a phase-2 checkpoint with
``PUFFER_POLICY_RESIDUAL_CHECKPOINT_PATH`` and a phase-3 checkpoint with
``PUFFER_POLICY_FINAL_CHECKPOINT_PATH``.  Future gates can be added without
changing code through ``PUFFER_POLICY_GATE_HEADS_JSON``, for example
``{"2": "gate2.bin", "3": "gate3.bin", "4": "gate4.bin"}``.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy


_BASE: CheckpointPolicy | None = None
_HEAD_DECODERS: dict[int, np.ndarray] | None = None
_LOW_CONFIDENCE_HEAD_DECODERS: dict[int, np.ndarray] | None = None
_KEY: tuple | None = None
_LAST_YAW_ACTION = 0.0


def _head_paths() -> dict[int, str]:
    encoded = os.getenv("PUFFER_POLICY_GATE_HEADS_JSON", "").strip()
    if encoded:
        payload = json.loads(encoded)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("PUFFER_POLICY_GATE_HEADS_JSON must be a nonempty object")
        paths = {int(gate): os.path.abspath(str(path)) for gate, path in payload.items()}
    else:
        previous = os.getenv("PUFFER_POLICY_PREVIOUS_CHECKPOINT_PATH", "").strip()
        residual = os.getenv("PUFFER_POLICY_RESIDUAL_CHECKPOINT_PATH", "").strip()
        final = os.getenv("PUFFER_POLICY_FINAL_CHECKPOINT_PATH", "").strip()
        if not residual and not previous:
            raise RuntimeError(
                "a named policy head checkpoint or "
                "PUFFER_POLICY_GATE_HEADS_JSON is required"
            )
        # With both checkpoints, residual steers toward gate 3 and final steers
        # toward gate 4.  A lone residual retains the legacy final-gate use.
        paths: dict[int, str] = {}
        if previous:
            paths[1] = os.path.abspath(previous)
        if residual:
            paths[2 if final else 3] = os.path.abspath(residual)
        if final and residual:
            paths[3] = os.path.abspath(final)
    if any(gate < 0 for gate in paths):
        raise ValueError("gate-head indices must be nonnegative")
    return paths


def _low_confidence_head_paths() -> dict[int, str]:
    encoded = os.getenv(
        "PUFFER_POLICY_GATE_LOW_CONFIDENCE_HEADS_JSON", ""
    ).strip()
    if encoded:
        payload = json.loads(encoded)
        if not isinstance(payload, dict):
            raise ValueError(
                "PUFFER_POLICY_GATE_LOW_CONFIDENCE_HEADS_JSON must be an object"
            )
        return {
            int(gate): os.path.abspath(str(path)) for gate, path in payload.items()
        }
    residual = os.getenv(
        "PUFFER_POLICY_RESIDUAL_LOW_CONFIDENCE_CHECKPOINT_PATH", ""
    ).strip()
    final = os.getenv(
        "PUFFER_POLICY_FINAL_LOW_CONFIDENCE_CHECKPOINT_PATH", ""
    ).strip()
    paths: dict[int, str] = {}
    if residual:
        paths[2 if final else 3] = os.path.abspath(residual)
    if final:
        paths[3] = os.path.abspath(final)
    return paths


def _head_min_confidence() -> dict[int, float]:
    encoded = os.getenv("PUFFER_POLICY_GATE_HEAD_MIN_CONFIDENCE_JSON", "").strip()
    if not encoded:
        return {}
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError(
            "PUFFER_POLICY_GATE_HEAD_MIN_CONFIDENCE_JSON must be an object"
        )
    thresholds = {int(gate): float(value) for gate, value in payload.items()}
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("gate-head confidence thresholds must be in [0, 1]")
    return thresholds


def _resolve() -> tuple[
    CheckpointPolicy, dict[int, np.ndarray], dict[int, np.ndarray]
]:
    global _BASE, _HEAD_DECODERS, _LOW_CONFIDENCE_HEAD_DECODERS, _KEY
    base_path = os.getenv("PUFFER_POLICY_BASE_CHECKPOINT_PATH", "").strip()
    if not base_path:
        raise RuntimeError("PUFFER_POLICY_BASE_CHECKPOINT_PATH is required")
    head_paths = _head_paths()
    low_confidence_paths = _low_confidence_head_paths()
    dimensions = (
        int(os.getenv("PUFFER_POLICY_INPUT_DIM", "23")),
        int(os.getenv("PUFFER_POLICY_HIDDEN_DIM", "128")),
        int(os.getenv("PUFFER_POLICY_NUM_LAYERS", "3")),
        int(os.getenv("PUFFER_POLICY_NUM_ACTIONS", "4")),
    )
    key = (
        os.path.abspath(base_path),
        tuple(sorted(head_paths.items())),
        tuple(sorted(low_confidence_paths.items())),
        *dimensions,
    )
    if (
        _BASE is None
        or _HEAD_DECODERS is None
        or _LOW_CONFIDENCE_HEAD_DECODERS is None
        or key != _KEY
    ):
        load_dimensions = {
            "input_dim": dimensions[0],
            "hidden_dim": dimensions[1],
            "num_layers": dimensions[2],
            "num_actions": dimensions[3],
        }
        base = CheckpointPolicy.load(key[0], **load_dimensions)

        def load_decoders(paths: dict[int, str], label: str) -> dict[int, np.ndarray]:
            decoders: dict[int, np.ndarray] = {}
            for gate, path in paths.items():
                head = CheckpointPolicy.load(path, **load_dimensions)
                if not np.array_equal(base.encoder, head.encoder):
                    raise ValueError(
                        f"gate {gate} {label} checkpoint changes the recurrent encoder"
                    )
                for layer, (base_weight, head_weight) in enumerate(
                    zip(base.mingru_proj, head.mingru_proj)
                ):
                    if not np.array_equal(base_weight, head_weight):
                        raise ValueError(
                            f"gate {gate} {label} checkpoint changes MinGRU layer {layer}"
                        )
                decoders[gate] = head.decoder.copy()
            return decoders

        decoders = load_decoders(head_paths, "head")
        low_confidence_decoders = load_decoders(
            low_confidence_paths, "low-confidence head"
        )
        _BASE = base
        _HEAD_DECODERS = decoders
        _LOW_CONFIDENCE_HEAD_DECODERS = low_confidence_decoders
        _KEY = key
    return _BASE, _HEAD_DECODERS, _LOW_CONFIDENCE_HEAD_DECODERS


def reset() -> None:
    global _LAST_YAW_ACTION
    base, _heads, _low_confidence_heads = _resolve()
    base.reset_state()
    _LAST_YAW_ACTION = 0.0


def infer(observation: Sequence[float]) -> list[float]:
    global _LAST_YAW_ACTION
    base, head_decoders, low_confidence_head_decoders = _resolve()
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (base.input_dim,):
        raise ValueError(f"expected observation shape ({base.input_dim},), got {values.shape}")

    denominator = int(os.getenv("PUFFER_POLICY_RACE_PHASE_DENOMINATOR", "3"))
    if denominator <= 0:
        raise ValueError("PUFFER_POLICY_RACE_PHASE_DENOMINATOR must be positive")
    phase = float(np.clip(values[22], 0.0, 1.0))
    gate_index = int(round(phase * denominator))

    recurrent_observation = values.copy()
    recurrent_observation[22] = np.float32(_LAST_YAW_ACTION)
    hidden = base.hidden_features(recurrent_observation)

    # Evaluating the complete selected decoder preserves the same numerical
    # path as CheckpointPolicy.infer.  Unlisted gates use the proven base head.
    min_confidence = _head_min_confidence().get(gate_index, 0.0)
    if float(values[17]) >= min_confidence:
        decoder = head_decoders.get(gate_index, base.decoder)
    else:
        decoder = low_confidence_head_decoders.get(gate_index, base.decoder)
    decoded = decoder @ hidden
    actions = np.clip(decoded[: base.num_actions], -1.0, 1.0).astype(np.float32)
    _LAST_YAW_ACTION = float(actions[3])
    return [float(value) for value in actions]
