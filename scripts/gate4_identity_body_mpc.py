#!/usr/bin/env python3
"""Robust lateral MPC for the N239 observable Gate-4 controller.

The dynamics artifact is trained only from actions and camera-relative states
recorded during official VQ1/R1 attempts.  Whole-run folds agree on the sign
and magnitude of roll -> right acceleration, but not on the forward or vertical
action channels.  This module therefore optimizes exactly one identified axis.
It never consumes simulator truth, global position, or track geometry.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np


MODEL_SCHEMA = "gate4_identity_body_dynamics_v2"
DEPLOYABLE_AXIS = "roll_to_right_acceleration"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


@dataclasses.dataclass(frozen=True)
class LateralDynamicsEnsemble:
    """Continuous right-axis models ``dv/dt = a*v + b*roll + c``."""

    rate_gain_per_s: np.ndarray
    roll_gain_m_s2: np.ndarray
    bias_m_s2: np.ndarray
    source_path: str
    source_sha256: str

    @property
    def members(self) -> int:
        return int(self.rate_gain_per_s.size)

    @classmethod
    def load(cls, path: str | Path) -> "LateralDynamicsEnsemble":
        resolved = Path(path).resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("schema") != MODEL_SCHEMA:
            raise ValueError("unsupported Gate-4 identity dynamics schema")
        if payload.get("passed") is not True or payload.get("blockers"):
            raise ValueError("Gate-4 identity dynamics artifact did not pass")
        if DEPLOYABLE_AXIS not in payload.get("deployable_action_axes", []):
            raise ValueError("roll-to-right action axis is not deployable")
        contract = payload.get("contract") or {}
        required_contract = (
            "locally_coherent_family_segments",
            "whole_run_cross_validation",
            "cross_family_transitions_excluded",
            "official_trace_actions_only",
            "deployment_is_lateral_axis_only",
        )
        if not all(contract.get(key) is True for key in required_contract):
            raise ValueError("dynamics artifact contract is incomplete")
        if contract.get("runtime_privileged_state") is not False:
            raise ValueError("privileged runtime state is forbidden")

        raw_models = payload.get("structured_ensemble") or []
        if len(raw_models) < 3:
            raise ValueError("at least three whole-run fold models are required")
        coefficients = np.asarray(
            [member["coefficients"][1] for member in raw_models],
            dtype=np.float64,
        )
        if coefficients.shape != (len(raw_models), 3):
            raise ValueError("invalid right-axis coefficient shape")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("non-finite right-axis coefficient")
        if np.any(coefficients[:, 1] <= 1.0):
            raise ValueError("roll response is not identified in every fold")
        return cls(
            rate_gain_per_s=coefficients[:, 0].copy(),
            roll_gain_m_s2=coefficients[:, 1].copy(),
            bias_m_s2=coefficients[:, 2].copy(),
            source_path=str(resolved),
            source_sha256=_sha256(resolved),
        )


@dataclasses.dataclass(frozen=True)
class LateralMPCConfig:
    minimum_horizon_s: float = 0.60
    maximum_horizon_s: float = 5.00
    minimum_closing_m_s: float = 4.00
    integration_dt_s: float = 0.05
    control_knots: int = 5
    candidates: int = 384
    iterations: int = 4
    elite_fraction: float = 0.10
    minimum_roll_norm: float = -1.00
    maximum_roll_norm: float = 0.85
    initial_standard_deviation: float = 0.60
    minimum_standard_deviation: float = 0.08
    terminal_center_limit_m: float = 1.00
    terminal_position_weight: float = 30.0
    terminal_rate_weight: float = 1.5
    running_position_weight: float = 0.10
    control_effort_weight: float = 0.03
    control_slew_weight: float = 0.08
    worst_case_weight: float = 1.5

    def __post_init__(self) -> None:
        if self.minimum_horizon_s <= 0.0:
            raise ValueError("minimum horizon must be positive")
        if self.maximum_horizon_s < self.minimum_horizon_s:
            raise ValueError("maximum horizon is below minimum horizon")
        if self.integration_dt_s <= 0.0:
            raise ValueError("integration dt must be positive")
        if self.control_knots < 2 or self.candidates < 16 or self.iterations < 1:
            raise ValueError("optimizer population is too small")
        if not 0.0 < self.elite_fraction < 0.5:
            raise ValueError("elite fraction must be in (0, 0.5)")


@dataclasses.dataclass(frozen=True)
class LateralPlan:
    roll_norm: float
    accepted: bool
    horizon_s: float
    objective: float
    worst_abs_terminal_right_m: float
    terminal_right_m: tuple[float, ...]
    terminal_right_rate_m_s: tuple[float, ...]
    control_knots: tuple[float, ...]


def propagate_lateral(
    right_m: np.ndarray,
    right_rate_m_s: np.ndarray,
    roll_norm: np.ndarray,
    ensemble: LateralDynamicsEnsemble,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate broadcast-compatible candidate/member arrays exactly."""

    dt = float(dt_s)
    a = ensemble.rate_gain_per_s
    b = ensemble.roll_gain_m_s2
    c = ensemble.bias_m_s2
    drive = b * roll_norm + c
    exponential = np.exp(a * dt)
    small = np.abs(a) < 1e-8
    rate_next = np.where(
        small,
        right_rate_m_s + drive * dt,
        exponential * right_rate_m_s + (exponential - 1.0) * drive / a,
    )
    position_delta = np.where(
        small,
        right_rate_m_s * dt + 0.5 * drive * dt * dt,
        (right_rate_m_s + drive / a) * (exponential - 1.0) / a
        - (drive / a) * dt,
    )
    return right_m + position_delta, rate_next


class Gate4IdentityLateralMPC:
    """Deterministic risk-sensitive CEM over a one-axis fold ensemble."""

    def __init__(
        self,
        ensemble: LateralDynamicsEnsemble,
        config: LateralMPCConfig | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.config = config or LateralMPCConfig()
        rng = np.random.default_rng(239)
        self._standard_samples = rng.standard_normal(
            (
                self.config.iterations,
                self.config.candidates,
                self.config.control_knots,
            )
        )
        self.plan_count = 0

    def reset(self) -> None:
        self.plan_count = 0

    def horizon_s(self, forward_m: float, forward_rate_m_s: float) -> float:
        closing_m_s = max(-float(forward_rate_m_s), self.config.minimum_closing_m_s)
        raw = max(float(forward_m), 0.0) / closing_m_s
        return clamp(
            raw,
            self.config.minimum_horizon_s,
            self.config.maximum_horizon_s,
        )

    def _initial_mean(self, right_m: float, right_rate_m_s: float) -> np.ndarray:
        a = float(np.mean(self.ensemble.rate_gain_per_s))
        b = float(np.mean(self.ensemble.roll_gain_m_s2))
        c = float(np.mean(self.ensemble.bias_m_s2))
        desired_acceleration = -0.75 * float(right_m) - 1.25 * float(right_rate_m_s)
        initial = clamp(
            (desired_acceleration - a * right_rate_m_s - c) / b,
            self.config.minimum_roll_norm,
            self.config.maximum_roll_norm,
        )
        # Front-load the counterbank, then give the receding optimizer room to
        # release it before the aperture plane.
        return np.linspace(initial, 0.0, self.config.control_knots)

    def _simulate(
        self,
        controls: np.ndarray,
        *,
        right_m: float,
        right_rate_m_s: float,
        horizon_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        candidates = int(controls.shape[0])
        members = self.ensemble.members
        position = np.full((candidates, members), float(right_m), dtype=np.float64)
        rate = np.full(
            (candidates, members), float(right_rate_m_s), dtype=np.float64
        )
        steps = max(1, int(math.ceil(horizon_s / cfg.integration_dt_s)))
        dt = horizon_s / steps
        running_cost = np.zeros((candidates, members), dtype=np.float64)
        for step in range(steps):
            knot_position = step * (cfg.control_knots - 1) / max(steps - 1, 1)
            left = min(int(math.floor(knot_position)), cfg.control_knots - 1)
            right = min(left + 1, cfg.control_knots - 1)
            fraction = knot_position - left
            roll = (
                controls[:, left] * (1.0 - fraction)
                + controls[:, right] * fraction
            )[:, None]
            position, rate = propagate_lateral(
                position,
                rate,
                roll,
                self.ensemble,
                dt,
            )
            running_cost += cfg.running_position_weight * np.square(position) * dt
        member_cost = (
            cfg.terminal_position_weight * np.square(position)
            + cfg.terminal_rate_weight * np.square(rate)
            + running_cost
        )
        effort = cfg.control_effort_weight * np.mean(np.square(controls), axis=1)
        slew = cfg.control_slew_weight * np.mean(
            np.square(np.diff(controls, axis=1)), axis=1
        )
        objective = (
            np.mean(member_cost, axis=1)
            + cfg.worst_case_weight * np.max(member_cost, axis=1)
            + effort
            + slew
        )
        return objective, position, rate

    def plan(
        self,
        *,
        forward_m: float,
        forward_rate_m_s: float,
        right_m: float,
        right_rate_m_s: float,
    ) -> LateralPlan:
        values = np.asarray(
            [forward_m, forward_rate_m_s, right_m, right_rate_m_s],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or forward_m <= 0.0:
            raise ValueError("invalid observable lateral MPC state")
        cfg = self.config
        horizon = self.horizon_s(forward_m, forward_rate_m_s)
        mean = self._initial_mean(right_m, right_rate_m_s)
        standard_deviation = np.full(
            cfg.control_knots, cfg.initial_standard_deviation, dtype=np.float64
        )
        elite_count = max(2, int(math.ceil(cfg.candidates * cfg.elite_fraction)))
        best_controls: np.ndarray | None = None
        best_objective = math.inf
        best_position: np.ndarray | None = None
        best_rate: np.ndarray | None = None
        for iteration in range(cfg.iterations):
            controls = mean + standard_deviation * self._standard_samples[iteration]
            controls = np.clip(
                controls, cfg.minimum_roll_norm, cfg.maximum_roll_norm
            )
            deterministic = (
                mean,
                np.full(cfg.control_knots, cfg.minimum_roll_norm),
                np.full(cfg.control_knots, cfg.maximum_roll_norm),
                np.zeros(cfg.control_knots),
                np.linspace(cfg.minimum_roll_norm, 0.0, cfg.control_knots),
            )
            for index, candidate in enumerate(deterministic):
                controls[index] = candidate
            objective, terminal_position, terminal_rate = self._simulate(
                controls,
                right_m=right_m,
                right_rate_m_s=right_rate_m_s,
                horizon_s=horizon,
            )
            order = np.argsort(objective, kind="stable")
            elite = controls[order[:elite_count]]
            mean = np.mean(elite, axis=0)
            standard_deviation = np.maximum(
                np.std(elite, axis=0), cfg.minimum_standard_deviation
            )
            winner = int(order[0])
            if float(objective[winner]) < best_objective:
                best_objective = float(objective[winner])
                best_controls = controls[winner].copy()
                best_position = terminal_position[winner].copy()
                best_rate = terminal_rate[winner].copy()
        assert best_controls is not None
        assert best_position is not None
        assert best_rate is not None
        self.plan_count += 1
        worst_abs_right = float(np.max(np.abs(best_position)))
        return LateralPlan(
            roll_norm=clamp(
                best_controls[0], cfg.minimum_roll_norm, cfg.maximum_roll_norm
            ),
            accepted=worst_abs_right <= cfg.terminal_center_limit_m,
            horizon_s=horizon,
            objective=best_objective,
            worst_abs_terminal_right_m=worst_abs_right,
            terminal_right_m=tuple(float(value) for value in best_position),
            terminal_right_rate_m_s=tuple(float(value) for value in best_rate),
            control_knots=tuple(float(value) for value in best_controls),
        )
