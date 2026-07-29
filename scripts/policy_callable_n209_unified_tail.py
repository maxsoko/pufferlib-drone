#!/usr/bin/env python3
"""N210: official-proven H12 prefix plus one recurrent N209 late tail.

Official Gates 1--3 are delegated unchanged to the frozen H12/N203 prefix
chain.  At official Gate 4, the N209 six-gate checkpoint starts from zero
recurrent state and remains the sole controller through Gates 4--6.  The live
one-hot and prefix-confidence fields occupy reserved training features, so
they are cleared before N209 inference while six-gate progress at index 23 is
preserved.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate4_terminal_crossing_n206 as h12
import policy_callable_six_gate_composite as six_gate
from policy_callable_checkpoint import CheckpointPolicy


EXPECTED_H12_SOURCE_SHA256 = (
    "b4ccbcb90f30ce866c8a9f827c98d457708ca925345cd1847a6d3ef3ebf80f63"
)
EXPECTED_N209_CHECKPOINT_SHA256 = (
    "efad0967b46d10bd30eebf03c85a6bc2cecdfb9646bab9852a0d5c701357aefb"
)

_H12_PATH = Path(h12.__file__).resolve()
if hashlib.sha256(_H12_PATH.read_bytes()).hexdigest() != EXPECTED_H12_SOURCE_SHA256:
    raise RuntimeError("H12/N203 prefix source hash mismatch")

_TAIL: CheckpointPolicy | None = None
_TAIL_PATH: str | None = None
_TAIL_ACTIVE = False


def _required_tail_path() -> str:
    value = os.getenv("PUFFER_POLICY_GATE4_CHECKPOINT_PATH", "").strip()
    if not value:
        raise RuntimeError("PUFFER_POLICY_GATE4_CHECKPOINT_PATH is required")
    path = os.path.abspath(value)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _resolve_tail() -> CheckpointPolicy:
    global _TAIL, _TAIL_PATH
    path = _required_tail_path()
    if _TAIL is None or _TAIL_PATH != path:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != EXPECTED_N209_CHECKPOINT_SHA256:
            raise RuntimeError(
                "N209 tail checkpoint hash mismatch: "
                f"expected {EXPECTED_N209_CHECKPOINT_SHA256}, got {digest}"
            )
        _TAIL = CheckpointPolicy.load(
            path,
            input_dim=32,
            hidden_dim=128,
            num_layers=3,
            num_actions=4,
            native_bf16=False,
            layout_precision_bytes=4,
        )
        _TAIL_PATH = path
    return _TAIL


def _n209_observation(values: np.ndarray) -> np.ndarray:
    """Restore the zero-reserved-feature ABI used for N209 training."""

    current = np.asarray(values, dtype=np.float32).copy()
    current[24:32] = np.float32(0.0)
    return current


def reset() -> None:
    global _TAIL_ACTIVE
    h12.reset()
    if _TAIL is not None:
        _TAIL.reset_state()
    _TAIL_ACTIVE = False


def infer(observation: Sequence[float]) -> list[float]:
    global _TAIL_ACTIVE
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")

    gate = six_gate._gate_index(values)
    if gate <= 2:
        if _TAIL_ACTIVE and _TAIL is not None:
            _TAIL.reset_state()
        _TAIL_ACTIVE = False
        return h12.infer(values)

    tail = _resolve_tail()
    if not _TAIL_ACTIVE:
        tail.reset_state()
        _TAIL_ACTIVE = True
    return tail.infer(_n209_observation(values))


def controller_snapshot() -> dict:
    return {
        "n210_unified_tail": {
            "tail_active": _TAIL_ACTIVE,
            "tail_loaded": _TAIL is not None,
            "parameters": {
                "prefix_gate_indices": [0, 1, 2],
                "n209_tail_gate_indices": [3, 4, 5],
                "n209_zero_state_at_gate4": True,
                "n209_reserved_features_zeroed": list(range(24, 32)),
                "n209_progress_feature_index": 23,
                "manual_late_gate_action_wrappers": False,
                "course_coordinates_used": False,
            },
        },
        "h12": h12.controller_snapshot(),
    }
