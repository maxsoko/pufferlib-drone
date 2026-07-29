#!/usr/bin/env python3
"""Bounded offline search for a smooth Gate-3 projected-path roll controller.

The search uses only fields available to the live controller.  It fits no
per-trace action table: every trajectory is controlled by the same six named
parameters.  Candidate 045 can be held out from parameter selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _inverse_tanh(value: float, scale: float) -> float:
    return float(np.arctanh(np.clip(value, -0.999999, 0.999999)) * scale)


def _roll_from_observation(observation: list[float]) -> float:
    w, x, y, z = (float(value) for value in observation[6:10])
    return math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


@dataclass(frozen=True)
class Model:
    roll_tau_s: float
    accel_per_tan_roll_m_s2: float
    right_rate_drag_per_s: float
    bias_m_s2: float


@dataclass(frozen=True)
class Parameters:
    path_slope: float
    terminal_right_m: float
    position_gain: float
    rate_gain: float
    max_roll_norm: float
    max_roll_slew_norm_s: float


@dataclass
class Trace:
    path: str
    official_result: int
    collision_impact_m_s: float | None
    times_s: np.ndarray
    closing_m_s: np.ndarray
    recorded_roll_norm: np.ndarray
    initial_forward_m: float
    initial_right_m: float
    initial_right_rate_m_s: float
    initial_roll_rad: float
    initial_roll_norm: float
    observed_terminal_right_m: float


def _load_trace(path: Path, *, admission_forward_m: float) -> Trace | None:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for sample in report.get("policy_trace", {}).get("samples", []):
        observation = sample.get("observation") or []
        pose = sample.get("gate_pose")
        if (
            int(sample.get("official_active_gate_index") or 0) != 2
            or len(observation) < 23
            or pose is None
            or float(observation[10]) < 0.5
        ):
            continue
        forward_m = _inverse_tanh(float(observation[11]), 10.0)
        if not 0.0 < forward_m <= admission_forward_m:
            continue
        closing_m_s = -_inverse_tanh(float(observation[0]), 5.0)
        if closing_m_s < 1.0:
            continue
        rows.append((sample, forward_m, closing_m_s))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda item: float(item[0]["elapsed_s"]))
    first, initial_forward_m, _initial_closing = rows[0]
    first_obs = first["observation"]
    times = np.asarray(
        [float(item[0]["elapsed_s"]) - float(first["elapsed_s"]) for item in rows]
    )
    closing = np.asarray([item[2] for item in rows])
    recorded_roll = np.asarray(
        [float(item[0]["normalized_action"][1]) for item in rows]
    )
    last, last_forward_m, last_closing = rows[-1]
    last_obs = last["observation"]
    last_right = _inverse_tanh(float(last_obs[12]), 5.0)
    last_right_rate = _inverse_tanh(float(last_obs[1]), 3.0)
    observed_terminal_right = last_right + last_right_rate * (
        last_forward_m / max(last_closing, 1.0)
    )
    latest = report.get("sitl", {}).get("latest_telemetry", {})
    return Trace(
        path=str(path),
        official_result=int(report.get("official_active_gate_index") or 0),
        collision_impact_m_s=latest.get("collision_impact"),
        times_s=times,
        closing_m_s=closing,
        recorded_roll_norm=recorded_roll,
        initial_forward_m=initial_forward_m,
        initial_right_m=_inverse_tanh(float(first_obs[12]), 5.0),
        initial_right_rate_m_s=_inverse_tanh(float(first_obs[1]), 3.0),
        initial_roll_rad=_roll_from_observation(first_obs),
        initial_roll_norm=float(first["normalized_action"][1]),
        observed_terminal_right_m=observed_terminal_right,
    )


def _profile(trace: Trace, elapsed_s: float) -> tuple[float, float]:
    index = int(np.searchsorted(trace.times_s, elapsed_s, side="right") - 1)
    index = max(0, min(index, len(trace.times_s) - 1))
    return float(trace.closing_m_s[index]), float(trace.recorded_roll_norm[index])


def _simulate(
    trace: Trace,
    model: Model,
    *,
    parameters: Parameters | None,
    dt_s: float = 1.0 / 120.0,
) -> dict:
    forward_m = trace.initial_forward_m
    right_m = trace.initial_right_m
    right_rate_m_s = trace.initial_right_rate_m_s
    roll_rad = trace.initial_roll_rad
    roll_norm = trace.initial_roll_norm
    elapsed_s = 0.0
    max_abs_roll_norm = abs(roll_norm)
    max_abs_roll_rad = abs(roll_rad)
    max_abs_slew_norm_s = 0.0
    while forward_m > 0.0 and elapsed_s < 4.0:
        closing_m_s, recorded_roll_norm = _profile(trace, elapsed_s)
        if parameters is None:
            requested_roll_norm = recorded_roll_norm
        else:
            desired_right_m = (
                parameters.terminal_right_m
                - parameters.path_slope * forward_m
            )
            desired_right_rate_m_s = parameters.path_slope * closing_m_s
            requested_roll_norm = (
                parameters.position_gain * (desired_right_m - right_m)
                + parameters.rate_gain
                * (desired_right_rate_m_s - right_rate_m_s)
            )
            requested_roll_norm = float(
                np.clip(
                    requested_roll_norm,
                    -parameters.max_roll_norm,
                    parameters.max_roll_norm,
                )
            )
            max_delta = parameters.max_roll_slew_norm_s * dt_s
            requested_roll_norm = float(
                np.clip(requested_roll_norm, roll_norm - max_delta, roll_norm + max_delta)
            )
        slew = abs(requested_roll_norm - roll_norm) / dt_s
        max_abs_slew_norm_s = max(max_abs_slew_norm_s, slew)
        roll_norm = requested_roll_norm
        desired_roll_rad = 0.5 * roll_norm
        roll_rad += (desired_roll_rad - roll_rad) * dt_s / model.roll_tau_s
        lateral_accel = (
            model.accel_per_tan_roll_m_s2 * math.tan(roll_rad)
            - model.right_rate_drag_per_s * right_rate_m_s
            + model.bias_m_s2
        )
        right_rate_m_s += lateral_accel * dt_s
        right_m += right_rate_m_s * dt_s
        forward_m -= closing_m_s * dt_s
        elapsed_s += dt_s
        max_abs_roll_norm = max(max_abs_roll_norm, abs(roll_norm))
        max_abs_roll_rad = max(max_abs_roll_rad, abs(roll_rad))
    return {
        "terminal_right_m": right_m,
        "terminal_right_rate_m_s": right_rate_m_s,
        "terminal_roll_rad": roll_rad,
        "duration_s": elapsed_s,
        "max_abs_roll_norm": max_abs_roll_norm,
        "max_abs_roll_rad": max_abs_roll_rad,
        "max_abs_slew_norm_s": max_abs_slew_norm_s,
    }


def _score(results: list[dict]) -> tuple[float, ...]:
    terminal = np.asarray([abs(row["terminal_right_m"]) for row in results])
    terminal_rate = np.asarray(
        [abs(row["terminal_right_rate_m_s"]) for row in results]
    )
    terminal_roll = np.asarray([abs(row["terminal_roll_rad"]) for row in results])
    return (
        float(np.max(terminal)),
        float(np.quantile(terminal, 0.9)),
        float(np.mean(terminal)),
        float(np.quantile(terminal_rate, 0.9)),
        float(np.quantile(terminal_roll, 0.9)),
    )


def _simulate_grid(
    trace: Trace,
    model: Model,
    parameters: list[Parameters],
    *,
    dt_s: float = 1.0 / 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate every preregistered parameter vector in one NumPy batch."""

    path_slope = np.asarray([row.path_slope for row in parameters])
    terminal_right = np.asarray([row.terminal_right_m for row in parameters])
    position_gain = np.asarray([row.position_gain for row in parameters])
    rate_gain = np.asarray([row.rate_gain for row in parameters])
    max_roll = np.asarray([row.max_roll_norm for row in parameters])
    max_slew = np.asarray([row.max_roll_slew_norm_s for row in parameters])
    forward_m = trace.initial_forward_m
    right_m = np.full(len(parameters), trace.initial_right_m)
    right_rate = np.full(len(parameters), trace.initial_right_rate_m_s)
    roll_rad = np.full(len(parameters), trace.initial_roll_rad)
    roll_norm = np.full(len(parameters), trace.initial_roll_norm)
    elapsed_s = 0.0
    while forward_m > 0.0 and elapsed_s < 4.0:
        closing_m_s, _recorded_roll = _profile(trace, elapsed_s)
        desired_right = terminal_right - path_slope * forward_m
        desired_rate = path_slope * closing_m_s
        requested = (
            position_gain * (desired_right - right_m)
            + rate_gain * (desired_rate - right_rate)
        )
        requested = np.clip(requested, -max_roll, max_roll)
        max_delta = max_slew * dt_s
        requested = np.clip(requested, roll_norm - max_delta, roll_norm + max_delta)
        roll_norm = requested
        desired_roll = 0.5 * roll_norm
        roll_rad += (desired_roll - roll_rad) * dt_s / model.roll_tau_s
        lateral_accel = (
            model.accel_per_tan_roll_m_s2 * np.tan(roll_rad)
            - model.right_rate_drag_per_s * right_rate
            + model.bias_m_s2
        )
        right_rate += lateral_accel * dt_s
        right_m += right_rate * dt_s
        forward_m -= closing_m_s * dt_s
        elapsed_s += dt_s
    return right_m, right_rate, roll_rad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--dynamics-fit", required=True, type=Path)
    parser.add_argument("--holdout-pattern", default="bounded_045")
    parser.add_argument("--admission-forward-m", type=float, default=12.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fit = json.loads(args.dynamics_fit.read_text(encoding="utf-8"))
    lateral = fit["gate3_lateral_fit"]
    model = Model(
        roll_tau_s=float(fit["attitude_tau_s"]["roll"]),
        accel_per_tan_roll_m_s2=float(lateral["accel_per_tan_roll_m_s2"]),
        right_rate_drag_per_s=float(lateral["right_rate_drag_per_s"]),
        bias_m_s2=float(lateral["bias_m_s2"]),
    )
    traces = [
        trace
        for path in args.trace
        if (trace := _load_trace(path, admission_forward_m=args.admission_forward_m))
        is not None
    ]
    development = [t for t in traces if args.holdout_pattern not in t.path]
    holdout = [t for t in traces if args.holdout_pattern in t.path]
    if not development or not holdout:
        raise ValueError("development and holdout trace sets must both be non-empty")

    replay = [_simulate(trace, model, parameters=None) for trace in traces]
    replay_error = np.asarray(
        [
            predicted["terminal_right_m"] - trace.observed_terminal_right_m
            for trace, predicted in zip(traces, replay)
        ]
    )

    search_space = [Parameters(*values) for values in itertools.product(
        (0.15, 0.20, 0.25, 0.30),
        (-0.50, -0.40, -0.30, -0.20, -0.10, 0.0),
        (0.2, 0.4, 0.6, 0.8, 1.0),
        (0.1, 0.2, 0.35, 0.5, 0.7),
        (0.5, 0.7, 0.9),
        (2.0, 4.0, 8.0),
    )]
    simulated = [_simulate_grid(trace, model, search_space) for trace in development]
    terminal = np.stack([row[0] for row in simulated], axis=1)
    terminal_rate = np.stack([row[1] for row in simulated], axis=1)
    terminal_roll = np.stack([row[2] for row in simulated], axis=1)
    abs_terminal = np.abs(terminal)
    score_columns = (
        np.max(abs_terminal, axis=1),
        np.quantile(abs_terminal, 0.9, axis=1),
        np.mean(abs_terminal, axis=1),
        np.quantile(np.abs(terminal_rate), 0.9, axis=1),
        np.quantile(np.abs(terminal_roll), 0.9, axis=1),
    )
    best_index = int(np.lexsort(tuple(reversed(score_columns)))[0])
    best_parameters = search_space[best_index]
    best_results = [
        _simulate(trace, model, parameters=best_parameters) for trace in development
    ]
    best_score = _score(best_results)
    holdout_results = [
        _simulate(trace, model, parameters=best_parameters) for trace in holdout
    ]
    result = {
        "passed": bool(
            np.median(np.abs(replay_error)) <= 0.35
            and max(abs(row["terminal_right_m"]) for row in holdout_results) <= 0.35
            and best_score[0] <= 0.55
        ),
        "model": asdict(model),
        "admission_forward_m": args.admission_forward_m,
        "trace_count": len(traces),
        "development_trace_count": len(development),
        "holdout_trace_count": len(holdout),
        "holdout_pattern": args.holdout_pattern,
        "surrogate_replay": {
            "median_abs_terminal_right_error_m": float(np.median(np.abs(replay_error))),
            "p90_abs_terminal_right_error_m": float(np.quantile(np.abs(replay_error), 0.9)),
            "max_abs_terminal_right_error_m": float(np.max(np.abs(replay_error))),
        },
        "search": {
            "evaluated_parameter_vectors": len(search_space),
            "selection": "lexicographic max/p90/mean terminal right, then p90 terminal rate/roll",
            "best_parameters": asdict(best_parameters),
            "development_score": list(best_score),
        },
        "development": [
            {
                "trace": trace.path,
                "official_result": trace.official_result,
                **simulation,
            }
            for trace, simulation in zip(development, best_results)
        ],
        "holdout": [
            {
                "trace": trace.path,
                "official_result": trace.official_result,
                "observed_terminal_right_m": trace.observed_terminal_right_m,
                **simulation,
            }
            for trace, simulation in zip(holdout, holdout_results)
        ],
        "blockers": [],
    }
    if np.median(np.abs(replay_error)) > 0.35:
        result["blockers"].append("surrogate_replay_median_error_above_0p35m")
    if best_score[0] > 0.55:
        result["blockers"].append("development_worst_terminal_right_above_0p55m")
    if max(abs(row["terminal_right_m"]) for row in holdout_results) > 0.35:
        result["blockers"].append("holdout_terminal_right_above_0p35m")
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
