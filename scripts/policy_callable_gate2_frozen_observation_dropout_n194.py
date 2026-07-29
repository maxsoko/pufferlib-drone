#!/usr/bin/env python3
"""N195: exact N194 plus a Gate-2 stale-visibility sanitizer."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate3_frozen_observation_recovery_n193 as n194
import policy_callable_six_gate_composite as tail


EXPECTED_N194_SHA256 = (
    "a68c07a2a55f61fc1b9ad6522a27e3f6b0ba206dab6dd554392f8c5da417a6a7"
)
_N194_PATH = Path(n194.__file__).resolve()
if hashlib.sha256(_N194_PATH.read_bytes()).hexdigest() != EXPECTED_N194_SHA256:
    raise RuntimeError("N194 policy source hash mismatch")


FEATURE_INDICES = (0, 1, 2, 11, 12, 13, 14)
MAX_FORWARD_M = 6.0
MINIMUM_CLOSING_M_S = 1.0
FROZEN_TRIGGER_SAMPLES = 8
FEATURE_EQUALITY_ATOL = 1.0e-7


class Gate2FrozenObservationDropout:
    """Convert a demonstrably frozen Gate-2 pose into ordinary vision dropout."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._clear_watchdog()
        self.activation_count = 0
        self.active_samples = 0
        self.sanitized_samples = 0

    def _clear_watchdog(self) -> None:
        self.active = False
        self.frozen_samples = 0
        self.last_features: np.ndarray | None = None

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "activation_count": self.activation_count,
            "active_samples": self.active_samples,
            "sanitized_samples": self.sanitized_samples,
            "frozen_samples": self.frozen_samples,
            "parameters": {
                "feature_indices": FEATURE_INDICES,
                "max_forward_m": MAX_FORWARD_M,
                "minimum_closing_m_s": MINIMUM_CLOSING_M_S,
                "frozen_trigger_samples": FROZEN_TRIGGER_SAMPLES,
                "feature_equality_atol": FEATURE_EQUALITY_ATOL,
                "sanitized_field": "gate_visible",
                "sanitized_value": 0.0,
                "release_on_fresh_motion": True,
            },
        }

    def sanitize(self, observation: Sequence[float]) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (32,):
            raise ValueError(f"expected observation shape (32,), got {values.shape}")
        if tail._gate_index(values) != 1 or float(values[10]) < 0.5:
            self._clear_watchdog()
            return values
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        if not (
            0.0 < forward_m <= MAX_FORWARD_M
            and closing_m_s >= MINIMUM_CLOSING_M_S
        ):
            self._clear_watchdog()
            return values
        features = values[list(FEATURE_INDICES)]
        unchanged = bool(
            self.last_features is not None
            and np.max(np.abs(features - self.last_features))
            <= FEATURE_EQUALITY_ATOL
        )
        self.last_features = features.copy()
        if unchanged:
            self.frozen_samples += 1
        else:
            self.active = False
            self.frozen_samples = 1
        if not self.active and self.frozen_samples >= FROZEN_TRIGGER_SAMPLES:
            self.active = True
            self.activation_count += 1
        if not self.active:
            return values

        self.active_samples += 1
        sanitized = values.copy()
        sanitized[10] = 0.0
        self.sanitized_samples += 1
        return sanitized


_SANITIZER = Gate2FrozenObservationDropout()


def reset() -> None:
    n194.reset()
    _SANITIZER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    sanitized = _SANITIZER.sanitize(observation)
    return n194.infer(sanitized)


def controller_snapshot() -> dict:
    return {
        "gate2_frozen_observation_dropout": _SANITIZER.snapshot(),
        "n194": n194.controller_snapshot(),
    }
