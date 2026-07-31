"""Count-agnostic public race-progress encoding for VQ2 policies.

The official runtime publishes ``active_gate_index`` at 4 Hz. New variable-
gate actors append exactly one held scalar, normalized by the native engine cap
rather than by an assumed course length. No total gate count is observed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


ENGINE_GATE_CAP = 16
# Checkpoint-continuity scale used by the official visual-suffix lineage. It is
# deliberately not a course length and values above one remain valid.
OFFICIAL_PROGRESS_SCALE = 6.0
LONG_COURSE_GATE_CAP = 32
PUBLIC_STATUS_HZ = 4
POLICY_HZ = 64
PUBLIC_STATUS_INTERVAL_STEPS = POLICY_HZ // PUBLIC_STATUS_HZ


def encode_public_gate_index(active_gate_index: int) -> np.float32:
    """Encode one official index as ``clamp(index, 0, 16) / 16``."""

    if isinstance(active_gate_index, bool) or not isinstance(
        active_gate_index, (int, np.integer)
    ):
        raise TypeError("active_gate_index must be an integer")
    bounded = min(max(int(active_gate_index), 0), ENGINE_GATE_CAP)
    return np.float32(bounded / ENGINE_GATE_CAP)


def held_public_phase(
    active_gate_indices: Iterable[int],
    *,
    interval_steps: int = PUBLIC_STATUS_INTERVAL_STEPS,
) -> np.ndarray:
    """Sample raw native progress at public-status ticks and hold between them."""

    if interval_steps <= 0:
        raise ValueError("interval_steps must be positive")
    raw = np.asarray(list(active_gate_indices))
    if raw.ndim != 1:
        raise ValueError("active_gate_indices must be one-dimensional")
    if raw.size == 0:
        return np.empty(0, dtype=np.float32)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("active_gate_indices must contain integers")

    encoded = np.empty(raw.size, dtype=np.float32)
    held = encode_public_gate_index(int(raw[0]))
    for step, gate_index in enumerate(raw):
        if step % interval_steps == 0:
            held = encode_public_gate_index(int(gate_index))
        encoded[step] = held
    return encoded


def encode_unbounded_gate_progress(active_gate_index: int) -> np.float32:
    """Encode public progress as ``active_gate_index / 6`` without saturation."""

    if isinstance(active_gate_index, bool) or not isinstance(
        active_gate_index, (int, np.integer)
    ):
        raise TypeError("active_gate_index must be an integer")
    if int(active_gate_index) < 0:
        raise ValueError("active_gate_index must be nonnegative")
    return np.float32(int(active_gate_index) / OFFICIAL_PROGRESS_SCALE)


def held_unbounded_gate_progress(
    active_gate_indices: Iterable[int],
    *,
    interval_steps: int = PUBLIC_STATUS_INTERVAL_STEPS,
) -> np.ndarray:
    """Sample and hold the unsaturated official progress feature."""

    if interval_steps <= 0:
        raise ValueError("interval_steps must be positive")
    raw = np.asarray(list(active_gate_indices))
    if raw.ndim != 1:
        raise ValueError("active_gate_indices must be one-dimensional")
    if raw.size == 0:
        return np.empty(0, dtype=np.float32)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("active_gate_indices must contain integers")
    if bool((raw < 0).any()):
        raise ValueError("active_gate_indices must be nonnegative")
    encoded = np.empty(raw.size, dtype=np.float32)
    held = encode_unbounded_gate_progress(int(raw[0]))
    for step, gate_index in enumerate(raw):
        if step % interval_steps == 0:
            held = encode_unbounded_gate_progress(int(gate_index))
        encoded[step] = held
    return encoded


__all__ = [
    "ENGINE_GATE_CAP",
    "LONG_COURSE_GATE_CAP",
    "OFFICIAL_PROGRESS_SCALE",
    "POLICY_HZ",
    "PUBLIC_STATUS_HZ",
    "PUBLIC_STATUS_INTERVAL_STEPS",
    "encode_public_gate_index",
    "encode_unbounded_gate_progress",
    "held_public_phase",
    "held_unbounded_gate_progress",
]
