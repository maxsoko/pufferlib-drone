#!/usr/bin/env python3
"""N206: exact H12/N203 prefix with the frozen learned Gate-4 tail direct.

The manual Gate-4 handoff overrides the N112 checkpoint selected by the N142
phase-reset composite.  This wrapper retains N203 exactly for Gates 1--3 and
uses that learned checkpoint directly while official Gate 4 is active.  Gates
5--6 continue through the same composite.  No course coordinate or privileged
state enters inference.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate3_counter_hysteresis_n202 as n203
import policy_callable_six_gate_composite as tail
import policy_callable_six_gate_hybrid as hybrid


EXPECTED_N203_SHA256 = (
    "b357536767a62beb85ba707a297b67ef5191998a425f6e0912a9882e07ee5cbe"
)
_N203_PATH = Path(n203.__file__).resolve()
if hashlib.sha256(_N203_PATH.read_bytes()).hexdigest() != EXPECTED_N203_SHA256:
    raise RuntimeError("N203 policy source hash mismatch")


class Gate4LearnedTailDirect:
    """Record the strictly Gate-4 direct learned-tail activation scope."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active_samples = 0
        self.last_action: list[float] | None = None

    def record(self, actions: Sequence[float]) -> list[float]:
        bounded = [tail._clamp(float(value)) for value in actions]
        if len(bounded) != 4:
            raise ValueError("expected four learned-tail actions")
        self.active_samples += 1
        self.last_action = bounded
        return bounded

    def snapshot(self) -> dict:
        return {
            "active_samples": self.active_samples,
            "last_action": self.last_action,
            "parameters": {
                "gate_index": 3,
                "manual_gate4_handoff_bypassed": True,
                "gate4_checkpoint": "N112_step_294912",
                "phase_reset_owned_by_n142_composite": True,
                "gate1_through_gate3_parent": "H12_N203",
                "course_coordinates_used": False,
            },
        }


_CONTROLLER = Gate4LearnedTailDirect()


def reset() -> None:
    n203.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    if tail._gate_index(values) != 3:
        return n203.infer(values)

    # N203's Gate-3 hysteresis must not remain latched after the phase change.
    # Mirror the exact phase cleanup the parent hybrid performs before its
    # manual Gate-4 handoff, then call the already-resetting N142 composite once.
    n203._controller()._clear_active()
    hybrid._clear_gate2_state()
    hybrid._clear_gate3_state()
    hybrid._clear_gate4_state()
    learned_actions = tail.infer(hybrid._tail_observation(values))
    return _CONTROLLER.record(learned_actions)


def controller_snapshot() -> dict:
    return {
        "gate4_learned_tail_direct": _CONTROLLER.snapshot(),
        "n203": n203.controller_snapshot(),
    }
