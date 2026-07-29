#!/usr/bin/env python3
"""N189: N184 with evidence-separated far and close Gate-3 control.

The frozen N188 severe-acquisition guard owns far-offset Gate-3 approaches.
When that guard is inactive, the frozen N187 projected-path PD may replace
roll inside 12 m.  The base recurrent policy is evaluated exactly once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate3_projected_path_pd as projected
import policy_callable_gate3_severe_acquisition as severe
import policy_callable_six_gate_composite as tail
import policy_callable_six_gate_hybrid as base


EXPECTED_BASE_SHA256 = (
    "134337cc8f91b6869edc921e90f55ce4d06c5664591d50a5723d7f14af544514"
)
EXPECTED_PROJECTED_SHA256 = (
    "9f63f47ee3132c76dfb0a987fb97e9d697a50ce990cdb39a6df3ff69a9bbb883"
)
EXPECTED_SEVERE_SHA256 = (
    "fbbf16d8fc61610f180e914658dac3fd504ef856e4a456b2ba64e9dc394097bd"
)


def _assert_source_hash(module, expected: str, name: str) -> None:
    path = Path(module.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{name} policy source hash mismatch")


_assert_source_hash(base, EXPECTED_BASE_SHA256, "N184 base")
_assert_source_hash(projected, EXPECTED_PROJECTED_SHA256, "N187 projected")
_assert_source_hash(severe, EXPECTED_SEVERE_SHA256, "N188 severe")


class Gate3CompositeAcquisitionPD:
    """Compose frozen controllers with explicit severe-branch precedence."""

    def __init__(self) -> None:
        self.severe = severe.Gate3SevereAcquisition()
        self.projected = projected.Gate3ProjectedPathPD()
        self.reset()

    def reset(self) -> None:
        self.severe.reset()
        self.projected.reset()
        self.branch = "base"
        self.base_steps = 0
        self.severe_steps = 0
        self.severe_release_steps = 0
        self.projected_steps = 0
        self.projected_visibility_resets = 0

    def snapshot(self) -> dict:
        return {
            "branch": self.branch,
            "base_steps": self.base_steps,
            "severe_steps": self.severe_steps,
            "severe_release_steps": self.severe_release_steps,
            "projected_steps": self.projected_steps,
            "projected_visibility_resets": self.projected_visibility_resets,
            "severe": self.severe.snapshot(),
            "projected": self.projected.snapshot(),
        }

    def apply(
        self,
        observation: Sequence[float],
        base_actions: Sequence[float],
        *,
        dt_s: float | None = None,
    ) -> list[float]:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (32,):
            raise ValueError(f"expected observation shape (32,), got {values.shape}")
        if len(base_actions) != 4:
            raise ValueError("expected four base actions")
        actions = [tail._clamp(float(value)) for value in base_actions]

        if tail._gate_index(values) != 2:
            self.severe.apply(values, actions)
            self.projected.reset()
            self.branch = "base"
            self.base_steps += 1
            return actions

        was_severe_active = self.severe.active
        severe_actions = self.severe.apply(values, actions)
        severe_changed = any(
            abs(left - right) > 1e-12
            for left, right in zip(severe_actions, actions)
        )
        if was_severe_active or self.severe.active or severe_changed:
            self.projected.reset()
            if self.severe.active or severe_changed:
                self.branch = "severe"
                self.severe_steps += 1
            else:
                # A release delegates this sample exactly; close PD may admit
                # on the following visible sample with a clean anchor.
                self.branch = "severe_release"
                self.severe_release_steps += 1
            return severe_actions

        visible = float(values[10]) >= 0.5
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        if not visible or forward_m <= 0.0:
            if self.projected.active:
                self.projected_visibility_resets += 1
            self.projected.reset()
            self.branch = "base"
            self.base_steps += 1
            return actions

        governed = self.projected.apply(values, actions, dt_s=dt_s)
        if self.projected.active:
            self.branch = "projected"
            self.projected_steps += 1
        else:
            self.branch = "base"
            self.base_steps += 1
        return governed


_CONTROLLER = Gate3CompositeAcquisitionPD()


def reset() -> None:
    base.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    actions = base.infer(observation)
    return _CONTROLLER.apply(observation, actions)


def controller_snapshot() -> dict:
    return _CONTROLLER.snapshot()
