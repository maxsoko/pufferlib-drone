#!/usr/bin/env python3
"""Reconstruct N264 losses and sweep telemetry-governor parameters offline.

The counterfactual model is intentionally small and inspectable.  It reuses the
N264 policy-along-speed trace as a function of remaining segment distance and
fits a per-axis first-order response directly from N264 commanded/actual
velocity pairs.  Absolute model residuals are retained in the report; candidate
times are corrected by the corresponding N264 per-gate residual so rankings are
based on modeled deltas rather than a claim of simulator equivalence.
"""

from __future__ import annotations

import argparse
import configparser
import dataclasses
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


EXPECTED_N264_SHA256 = (
    "3752249b0560a165f870fc5d222d111694f10b8290b2816de644c92ae3ce6d90"
)
EXPECTED_N259_SHA256 = (
    "f13d4b8d571d70be16731068c88dd105e81fe7bd00a86eae6f13f81031f00fc0"
)
BASELINE_PARAMETERS = {
    "crossing_speed_m_s": 4.0,
    "maximum_along_speed_m_s": 8.0,
    "slowdown_distance_m": 10.0,
    "cross_track_gain_s_inv": 1.5,
    "maximum_cross_track_correction_m_s": 4.0,
}


@dataclasses.dataclass(frozen=True)
class GovernorParameters:
    crossing_speed_m_s: float
    maximum_along_speed_m_s: float
    slowdown_distance_m: float
    cross_track_gain_s_inv: float
    maximum_cross_track_correction_m_s: float

    def validate(self) -> None:
        if self.crossing_speed_m_s <= 0.0:
            raise ValueError("crossing speed must be positive")
        if self.maximum_along_speed_m_s < self.crossing_speed_m_s:
            raise ValueError("maximum along speed must be at least crossing speed")
        if self.slowdown_distance_m <= 0.0:
            raise ValueError("slowdown distance must be positive")
        if self.cross_track_gain_s_inv < 0.0:
            raise ValueError("cross-track gain must be nonnegative")
        if self.maximum_cross_track_correction_m_s < 0.0:
            raise ValueError("cross-track correction cap must be nonnegative")


@dataclasses.dataclass(frozen=True)
class ResponseAxis:
    command_gain_s_inv: float
    velocity_decay_s_inv: float
    steady_state_gain: float
    time_constant_s: float
    rmse_m_s2: float
    r_squared: float
    samples: int


@dataclasses.dataclass(frozen=True)
class ReplayGate:
    gate_index: int
    raw_crossing_time_s: float
    crossing_error_m: float
    crossing_position_ned_m: tuple[float, float, float]
    crossing_velocity_ned_m_s: tuple[float, float, float]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def bounded(vector: np.ndarray, maximum: float) -> np.ndarray:
    magnitude = norm(vector)
    if magnitude <= maximum or magnitude <= 1e-12:
        return vector
    return vector * (maximum / magnitude)


def scheduled_speed(
    along_to_gate_m: float,
    *,
    cruise_speed_m_s: float,
    crossing_speed_m_s: float,
    slowdown_distance_m: float,
) -> float:
    alpha = float(np.clip(along_to_gate_m / slowdown_distance_m, 0.0, 1.0))
    smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return crossing_speed_m_s + (
        cruise_speed_m_s - crossing_speed_m_s
    ) * smooth_alpha


def load_ned_gate_centers(config_path: Path) -> tuple[np.ndarray, dict[str, float]]:
    parser = configparser.ConfigParser()
    if not parser.read(config_path):
        raise FileNotFoundError(config_path)
    env = parser["env"]
    gate_count = env.getint("num_gates")
    native_gates = np.asarray(
        [
            [
                env.getfloat(f"gate{gate_index}_x"),
                env.getfloat(f"gate{gate_index}_y"),
                env.getfloat(f"gate{gate_index}_z"),
            ]
            for gate_index in range(gate_count)
        ],
        dtype=np.float64,
    )
    settings = {
        "max_cmd_forward": env.getfloat("max_cmd_forward"),
        "max_cmd_lateral": env.getfloat("max_cmd_lateral"),
        "max_cmd_vertical": env.getfloat("max_cmd_vertical"),
        "gate_radius": env.getfloat("gate_radius"),
    }
    return -native_gates, settings


def fit_response_axes(trace: Sequence[dict[str, object]]) -> tuple[ResponseAxis, ...]:
    usable = [
        row
        for row in trace
        if row.get("command_velocity_ned_m_s") is not None
        and row.get("velocity_ned_m_s") is not None
    ]
    axes: list[ResponseAxis] = []
    for axis in range(3):
        predictors: list[tuple[float, float]] = []
        targets: list[float] = []
        for previous, current in zip(usable, usable[1:]):
            if previous["active_gate_index"] != current["active_gate_index"]:
                continue
            dt = float(current["elapsed_s"]) - float(previous["elapsed_s"])
            if not 0.05 <= dt <= 0.2:
                continue
            command = float(previous["command_velocity_ned_m_s"][axis])
            velocity = float(previous["velocity_ned_m_s"][axis])
            next_velocity = float(current["velocity_ned_m_s"][axis])
            predictors.append((command, -velocity))
            targets.append((next_velocity - velocity) / dt)
        x = np.asarray(predictors, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if len(y) < 3:
            raise RuntimeError(f"not enough samples to fit response axis {axis}")
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        command_gain, velocity_decay = (float(value) for value in coefficients)
        if command_gain <= 0.0 or velocity_decay <= 0.0:
            raise RuntimeError(f"nonphysical fitted response axis {axis}: {coefficients}")
        prediction = x @ coefficients
        residual = y - prediction
        denominator = float(np.sum((y - float(np.mean(y))) ** 2))
        r_squared = 1.0 - float(np.sum(residual**2)) / max(denominator, 1e-12)
        axes.append(
            ResponseAxis(
                command_gain_s_inv=command_gain,
                velocity_decay_s_inv=velocity_decay,
                steady_state_gain=command_gain / velocity_decay,
                time_constant_s=1.0 / velocity_decay,
                rmse_m_s2=float(np.sqrt(np.mean(residual**2))),
                r_squared=r_squared,
                samples=len(y),
            )
        )
    return tuple(axes)


def segment_geometry(
    gate_centers_ned_m: np.ndarray, initial_position_ned_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.vstack((initial_position_ned_m, gate_centers_ned_m[:-1]))
    segments = gate_centers_ned_m - starts
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-6):
        raise ValueError("every gate segment must have positive length")
    return starts, segments / lengths[:, None], lengths


def policy_along_curves(
    trace: Sequence[dict[str, object]], gate_count: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for gate_index in range(gate_count):
        rows = [
            row
            for row in trace
            if int(row["active_gate_index"]) == gate_index
            and row.get("governor") is not None
        ]
        if not rows:
            raise RuntimeError(f"N264 trace has no governed rows for gate {gate_index + 1}")
        remaining = np.asarray(
            [float(row["governor"]["along_to_gate_m"]) for row in rows],
            dtype=np.float64,
        )
        policy_along = np.asarray(
            [float(row["governor"]["policy_along_speed_m_s"]) for row in rows],
            dtype=np.float64,
        )
        order = np.argsort(remaining)
        curves.append((remaining[order], policy_along[order]))
    return tuple(curves)


def replay_governor(
    trace: Sequence[dict[str, object]],
    gate_centers_ned_m: np.ndarray,
    response_axes: Sequence[ResponseAxis],
    parameters_by_gate: Sequence[GovernorParameters],
    *,
    dt_s: float = 1.0 / 60.0,
    timeout_s: float = 45.0,
) -> tuple[ReplayGate, ...]:
    if len(parameters_by_gate) != len(gate_centers_ned_m):
        raise ValueError("one parameter set is required per gate")
    for parameters in parameters_by_gate:
        parameters.validate()
    first_policy_row = next(row for row in trace if row.get("governor") is not None)
    initial_position = np.asarray(trace[0]["position_ned_m"], dtype=np.float64)
    _, tangents, _ = segment_geometry(gate_centers_ned_m, initial_position)
    curves = policy_along_curves(trace, len(gate_centers_ned_m))
    steady_gain = np.asarray(
        [axis.steady_state_gain for axis in response_axes], dtype=np.float64
    )
    time_constant = np.asarray(
        [axis.time_constant_s for axis in response_axes], dtype=np.float64
    )
    decay = np.exp(-dt_s / time_constant)

    elapsed_s = float(first_policy_row["elapsed_s"])
    position = np.asarray(first_policy_row["position_ned_m"], dtype=np.float64)
    velocity = np.asarray(first_policy_row["velocity_ned_m_s"], dtype=np.float64)
    gate_index = 0
    crossings: list[ReplayGate] = []
    while elapsed_s < timeout_s and gate_index < len(gate_centers_ned_m):
        parameters = parameters_by_gate[gate_index]
        tangent = tangents[gate_index]
        to_gate = gate_centers_ned_m[gate_index] - position
        along_to_gate = float(np.dot(to_gate, tangent))
        cross_track = to_gate - tangent * along_to_gate
        correction = bounded(
            cross_track * parameters.cross_track_gain_s_inv,
            parameters.maximum_cross_track_correction_m_s,
        )
        curve_x, curve_y = curves[gate_index]
        policy_along = float(np.interp(along_to_gate, curve_x, curve_y))
        learned_cruise = float(
            np.clip(
                policy_along,
                parameters.crossing_speed_m_s,
                parameters.maximum_along_speed_m_s,
            )
        )
        along_command = scheduled_speed(
            along_to_gate,
            cruise_speed_m_s=learned_cruise,
            crossing_speed_m_s=parameters.crossing_speed_m_s,
            slowdown_distance_m=parameters.slowdown_distance_m,
        )
        command = tangent * along_command + correction
        target_velocity = steady_gain * command
        next_velocity = target_velocity + (velocity - target_velocity) * decay
        next_position = position + 0.5 * (velocity + next_velocity) * dt_s
        next_along = float(
            np.dot(gate_centers_ned_m[gate_index] - next_position, tangent)
        )
        if along_to_gate > 0.0 and next_along <= 0.0:
            fraction = along_to_gate / max(along_to_gate - next_along, 1e-12)
            hit_position = position + (next_position - position) * fraction
            hit_velocity = velocity + (next_velocity - velocity) * fraction
            residual = gate_centers_ned_m[gate_index] - hit_position
            cross_residual = residual - tangent * float(np.dot(residual, tangent))
            crossings.append(
                ReplayGate(
                    gate_index=gate_index,
                    raw_crossing_time_s=elapsed_s + fraction * dt_s,
                    crossing_error_m=norm(cross_residual),
                    crossing_position_ned_m=tuple(float(value) for value in hit_position),
                    crossing_velocity_ned_m_s=tuple(float(value) for value in hit_velocity),
                )
            )
            gate_index += 1
        position = next_position
        velocity = next_velocity
        elapsed_s += dt_s
    return tuple(crossings)


def official_gate_times(report: dict[str, object]) -> np.ndarray:
    transitions = report["gate_transitions"]
    if len(transitions) != 6:
        raise RuntimeError("N264 must contain six official transitions")
    return np.asarray(
        [int(item["last_gate_race_time"]) * 1e-9 for item in transitions],
        dtype=np.float64,
    )


def corrected_replay_summary(
    replay: Sequence[ReplayGate], baseline_residual_s: np.ndarray
) -> dict[str, object]:
    if len(replay) != len(baseline_residual_s):
        return {
            "completed_gate_count": len(replay),
            "predicted_finish_time_s": None,
            "maximum_crossing_error_m": None,
            "gates": [],
        }
    gates = []
    for item, residual in zip(replay, baseline_residual_s):
        gates.append(
            {
                **dataclasses.asdict(item),
                "residual_corrected_crossing_time_s": (
                    item.raw_crossing_time_s + float(residual)
                ),
            }
        )
    return {
        "completed_gate_count": len(replay),
        "predicted_finish_time_s": gates[-1]["residual_corrected_crossing_time_s"],
        "maximum_crossing_error_m": max(item.crossing_error_m for item in replay),
        "gates": gates,
    }


def action_to_raw_policy_velocity(
    action: Sequence[float], settings: dict[str, float]
) -> np.ndarray:
    return np.asarray(
        [
            -float(action[0]) * settings["max_cmd_forward"],
            -float(action[1]) * settings["max_cmd_lateral"],
            float(action[2]) * settings["max_cmd_vertical"],
        ],
        dtype=np.float64,
    )


def summarize_segments(
    report: dict[str, object],
    gate_centers_ned_m: np.ndarray,
    settings: dict[str, float],
    baseline: GovernorParameters,
) -> list[dict[str, object]]:
    trace = report["trace"]
    official_times = official_gate_times(report)
    initial_position = np.asarray(trace[0]["position_ned_m"], dtype=np.float64)
    _, tangents, lengths = segment_geometry(gate_centers_ned_m, initial_position)
    summaries: list[dict[str, object]] = []
    for gate_index in range(len(gate_centers_ned_m)):
        rows = [
            row
            for row in trace
            if int(row["active_gate_index"]) == gate_index
            and row.get("governor") is not None
        ]
        tangent = tangents[gate_index]
        policy_along = np.asarray(
            [float(row["governor"]["policy_along_speed_m_s"]) for row in rows]
        )
        scheduled_along = np.asarray(
            [float(row["governor"]["scheduled_along_speed_m_s"]) for row in rows]
        )
        actual_along = np.asarray(
            [float(np.dot(row["velocity_ned_m_s"], tangent)) for row in rows]
        )
        command_along = np.asarray(
            [float(np.dot(row["command_velocity_ned_m_s"], tangent)) for row in rows]
        )
        cross_errors = np.asarray(
            [float(row["governor"]["cross_track_error_m"]) for row in rows]
        )
        corrections = np.asarray(
            [norm(np.asarray(row["governor"]["correction_ned_m_s"])) for row in rows]
        )
        remaining = np.asarray(
            [float(row["governor"]["along_to_gate_m"]) for row in rows]
        )
        learned_cruise = np.clip(
            policy_along,
            baseline.crossing_speed_m_s,
            baseline.maximum_along_speed_m_s,
        )
        raw_velocities = np.asarray(
            [action_to_raw_policy_velocity(row["policy_action"], settings) for row in rows]
        )
        governed_velocities = np.asarray(
            [row["command_velocity_ned_m_s"] for row in rows], dtype=np.float64
        )
        interventions = np.linalg.norm(governed_velocities - raw_velocities, axis=1)
        accelerations: list[float] = []
        for previous, current in zip(rows, rows[1:]):
            dt = float(current["elapsed_s"]) - float(previous["elapsed_s"])
            if not 0.05 <= dt <= 0.2:
                continue
            before = float(np.dot(previous["velocity_ned_m_s"], tangent))
            after = float(np.dot(current["velocity_ned_m_s"], tangent))
            accelerations.append((after - before) / dt)
        lag = command_along - actual_along
        previous_official_s = 0.0 if gate_index == 0 else float(official_times[gate_index - 1])
        summaries.append(
            {
                "gate": gate_index + 1,
                "segment_length_m": float(lengths[gate_index]),
                "official_crossing_time_s": float(official_times[gate_index]),
                "official_segment_time_s": float(
                    official_times[gate_index] - previous_official_s
                ),
                "transition_observed_wall_s": float(
                    report["gate_transitions"][gate_index]["elapsed_s"]
                ),
                "trace_samples": len(rows),
                "last_sample_cross_track_error_m": float(cross_errors[-1]),
                "maximum_cross_track_error_m": float(np.max(cross_errors)),
                "policy_along_speed_m_s": {
                    "minimum": float(np.min(policy_along)),
                    "mean": float(np.mean(policy_along)),
                    "maximum": float(np.max(policy_along)),
                },
                "scheduled_along_speed_m_s": {
                    "minimum": float(np.min(scheduled_along)),
                    "mean": float(np.mean(scheduled_along)),
                    "maximum": float(np.max(scheduled_along)),
                },
                "actual_along_speed_m_s": {
                    "minimum": float(np.min(actual_along)),
                    "mean": float(np.mean(actual_along)),
                    "maximum": float(np.max(actual_along)),
                },
                "command_minus_actual_along_m_s": {
                    "mean": float(np.mean(lag)),
                    "rmse": float(np.sqrt(np.mean(lag**2))),
                    "maximum": float(np.max(lag)),
                },
                "observed_along_acceleration_m_s2": {
                    "minimum": float(min(accelerations)),
                    "mean": float(np.mean(accelerations)),
                    "maximum": float(max(accelerations)),
                },
                "governor_intervention": {
                    "mean_velocity_delta_m_s": float(np.mean(interventions)),
                    "maximum_velocity_delta_m_s": float(np.max(interventions)),
                    "crossing_floor_sample_fraction": float(
                        np.mean(policy_along < baseline.crossing_speed_m_s)
                    ),
                    "maximum_along_cap_sample_fraction": float(
                        np.mean(policy_along > baseline.maximum_along_speed_m_s)
                    ),
                    "slowdown_sample_fraction": float(
                        np.mean(
                            (remaining < baseline.slowdown_distance_m)
                            & (scheduled_along + 1e-9 < learned_cruise)
                        )
                    ),
                    "cross_track_cap_sample_fraction": float(
                        np.mean(
                            baseline.cross_track_gain_s_inv * cross_errors
                            >= baseline.maximum_cross_track_correction_m_s - 1e-9
                        )
                    ),
                    "mean_cross_track_correction_m_s": float(np.mean(corrections)),
                    "maximum_cross_track_correction_m_s": float(np.max(corrections)),
                },
            }
        )
    return summaries


def parameter_product(
    crossing_speeds: Iterable[float],
    maximum_along_speeds: Iterable[float],
    slowdown_distances: Iterable[float],
    cross_track_gains: Iterable[float],
    correction_caps: Iterable[float],
) -> Iterable[GovernorParameters]:
    for values in itertools.product(
        crossing_speeds,
        maximum_along_speeds,
        slowdown_distances,
        cross_track_gains,
        correction_caps,
    ):
        parameters = GovernorParameters(*values)
        if parameters.maximum_along_speed_m_s < parameters.crossing_speed_m_s:
            continue
        yield parameters


def parameter_change_count(
    candidate: GovernorParameters, baseline: GovernorParameters
) -> int:
    return sum(
        not math.isclose(getattr(candidate, field.name), getattr(baseline, field.name))
        for field in dataclasses.fields(GovernorParameters)
    )


def sweep_parameters(
    trace: Sequence[dict[str, object]],
    gate_centers_ned_m: np.ndarray,
    response_axes: Sequence[ResponseAxis],
    official_times_s: np.ndarray,
    baseline_replay: Sequence[ReplayGate],
    baseline: GovernorParameters,
    *,
    crossing_speeds: Sequence[float] = (4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0),
    maximum_along_speeds: Sequence[float] = (8.0, 8.5),
    slowdown_distances: Sequence[float] = (4.0, 6.0, 8.0, 10.0, 12.0),
    cross_track_gains: Sequence[float] = (1.25, 1.5, 1.75, 2.0),
    correction_caps: Sequence[float] = (1.0, 2.0, 4.0),
    crossing_error_limit_m: float = 0.45,
) -> list[dict[str, object]]:
    baseline_raw_times = np.asarray(
        [item.raw_crossing_time_s for item in baseline_replay], dtype=np.float64
    )
    residual = official_times_s - baseline_raw_times
    results: list[dict[str, object]] = []
    for parameters in parameter_product(
        crossing_speeds,
        maximum_along_speeds,
        slowdown_distances,
        cross_track_gains,
        correction_caps,
    ):
        replay = replay_governor(
            trace,
            gate_centers_ned_m,
            response_axes,
            (parameters,) * len(gate_centers_ned_m),
        )
        summary = corrected_replay_summary(replay, residual)
        predicted_finish = summary["predicted_finish_time_s"]
        maximum_error = summary["maximum_crossing_error_m"]
        safe = bool(
            predicted_finish is not None
            and maximum_error is not None
            and maximum_error <= crossing_error_limit_m
        )
        results.append(
            {
                "parameters": dataclasses.asdict(parameters),
                "parameter_change_count": parameter_change_count(parameters, baseline),
                "safe_offline": safe,
                "predicted_finish_time_s": predicted_finish,
                "predicted_time_saved_vs_n264_s": (
                    None
                    if predicted_finish is None
                    else float(official_times_s[-1] - predicted_finish)
                ),
                "maximum_crossing_error_m": maximum_error,
                "predicted_gate_crossing_times_s": [
                    gate["residual_corrected_crossing_time_s"]
                    for gate in summary["gates"]
                ],
                "predicted_gate_crossing_errors_m": [
                    gate["crossing_error_m"] for gate in summary["gates"]
                ],
            }
        )
    return sorted(
        results,
        key=lambda item: (
            not item["safe_offline"],
            math.inf
            if item["predicted_finish_time_s"] is None
            else item["predicted_finish_time_s"],
            item["parameter_change_count"],
            tuple(item["parameters"].values()),
        ),
    )


def choose_minimal_under_target(
    sweep: Sequence[dict[str, object]], target_s: float
) -> dict[str, object] | None:
    eligible = [
        item
        for item in sweep
        if item["safe_offline"]
        and item["predicted_finish_time_s"] is not None
        and item["predicted_finish_time_s"] < target_s
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item["parameter_change_count"],
            -item["predicted_finish_time_s"],
            item["maximum_crossing_error_m"],
            tuple(item["parameters"].values()),
        ),
    )


def coordinate_per_gate_sweep(
    trace: Sequence[dict[str, object]],
    gate_centers_ned_m: np.ndarray,
    response_axes: Sequence[ResponseAxis],
    official_times_s: np.ndarray,
    baseline_replay: Sequence[ReplayGate],
    seed: GovernorParameters,
    *,
    crossing_values: Sequence[float] = (6.5, 7.0, 7.5, 8.0),
    crossing_error_limit_m: float = 0.45,
) -> dict[str, object]:
    residual = official_times_s - np.asarray(
        [item.raw_crossing_time_s for item in baseline_replay], dtype=np.float64
    )
    schedule = [seed] * len(gate_centers_ned_m)

    def score(candidate_schedule: Sequence[GovernorParameters]):
        replay = replay_governor(
            trace, gate_centers_ned_m, response_axes, candidate_schedule
        )
        summary = corrected_replay_summary(replay, residual)
        finish = summary["predicted_finish_time_s"]
        error = summary["maximum_crossing_error_m"]
        if finish is None or error is None or error > crossing_error_limit_m:
            return (math.inf, math.inf), summary
        return (finish, error), summary

    for _pass in range(2):
        changed = False
        for gate_index in range(len(schedule)):
            best = schedule[gate_index]
            best_score, _ = score(schedule)
            for crossing in crossing_values:
                if crossing > seed.maximum_along_speed_m_s:
                    continue
                candidate = dataclasses.replace(seed, crossing_speed_m_s=crossing)
                candidate_schedule = list(schedule)
                candidate_schedule[gate_index] = candidate
                candidate_score, _ = score(candidate_schedule)
                if candidate_score < best_score:
                    best = candidate
                    best_score = candidate_score
            if best != schedule[gate_index]:
                schedule[gate_index] = best
                changed = True
        if not changed:
            break
    _, summary = score(schedule)
    return {
        "method": "two_pass_deterministic_coordinate_descent_crossing_speed",
        "parameters_by_gate": [dataclasses.asdict(item) for item in schedule],
        "result": summary,
    }


def build_report(
    source_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    *,
    target_s: float = 27.0,
) -> dict[str, object]:
    source_hash = sha256(source_path)
    checkpoint_hash = sha256(checkpoint_path)
    if source_hash != EXPECTED_N264_SHA256:
        raise RuntimeError(
            f"N264 SHA-256 mismatch: expected {EXPECTED_N264_SHA256}, got {source_hash}"
        )
    if checkpoint_hash != EXPECTED_N259_SHA256:
        raise RuntimeError(
            f"N259 SHA-256 mismatch: expected {EXPECTED_N259_SHA256}, got {checkpoint_hash}"
        )
    report = json.loads(source_path.read_text())
    if not report.get("accepted") or int(report["official_active_gate_index"]) != 6:
        raise RuntimeError("source is not the accepted six-gate N264 artifact")
    if int(report["official_race_finish_time_ns"]) < 0:
        raise RuntimeError("source does not contain an official finish time")
    if report.get("collision") is not None or report.get("invalid_reason") is not None:
        raise RuntimeError("source contains a collision or invalid result")

    gate_centers, settings = load_ned_gate_centers(config_path)
    if len(gate_centers) != 6:
        raise RuntimeError("offline sweep requires exactly six gates")
    baseline = GovernorParameters(**BASELINE_PARAMETERS)
    response_axes = fit_response_axes(report["trace"])
    baseline_replay = replay_governor(
        report["trace"],
        gate_centers,
        response_axes,
        (baseline,) * len(gate_centers),
    )
    if len(baseline_replay) != 6:
        raise RuntimeError("calibrated replay did not complete all six N264 gates")
    official_times = official_gate_times(report)
    raw_times = np.asarray(
        [item.raw_crossing_time_s for item in baseline_replay], dtype=np.float64
    )
    residuals = official_times - raw_times
    sweep = sweep_parameters(
        report["trace"],
        gate_centers,
        response_axes,
        official_times,
        baseline_replay,
        baseline,
    )
    recommendation = choose_minimal_under_target(sweep, target_s)
    if recommendation is None:
        raise RuntimeError(f"offline sweep found no safe candidate below {target_s} s")
    recommended_parameters = GovernorParameters(**recommendation["parameters"])
    per_gate = coordinate_per_gate_sweep(
        report["trace"],
        gate_centers,
        response_axes,
        official_times,
        baseline_replay,
        recommended_parameters,
    )
    fastest_safe = next(item for item in sweep if item["safe_offline"])
    per_gate_finish = per_gate["result"]["predicted_finish_time_s"]
    global_finish = fastest_safe["predicted_finish_time_s"]
    per_gate["globally_shared_settings_suboptimal"] = bool(
        per_gate_finish is not None
        and global_finish is not None
        and per_gate_finish < global_finish - 0.02
    )

    return {
        "experiment": "N265",
        "evidence_kind": "deterministic_offline_reconstruction_and_governor_sweep",
        "source_artifacts": {
            "n264": str(source_path),
            "n264_sha256": source_hash,
            "n259_checkpoint": str(checkpoint_path),
            "n259_checkpoint_sha256": checkpoint_hash,
            "config": str(config_path),
            "config_sha256": sha256(config_path),
        },
        "immutable_baseline": {
            "official_finish_time_s": float(official_times[-1]),
            "official_gate_times_s": official_times.tolist(),
            "parameters": dataclasses.asdict(baseline),
            "accepted": True,
            "collision": None,
            "invalid_reason": None,
        },
        "segment_reconstruction": summarize_segments(
            report, gate_centers, settings, baseline
        ),
        "fitted_velocity_response": {
            "model": "per_axis_dv_dt=command_gain*command-velocity_decay*velocity",
            "axes_ned_xyz": [dataclasses.asdict(axis) for axis in response_axes],
        },
        "baseline_replay_validation": {
            "integration_hz": 60.0,
            "policy_source": "N264 recorded N259 policy along speed interpolated by remaining segment distance",
            "raw_replay": corrected_replay_summary(
                baseline_replay, np.zeros(len(baseline_replay), dtype=np.float64)
            ),
            "official_minus_raw_replay_gate_residual_s": residuals.tolist(),
            "raw_finish_error_s": float(raw_times[-1] - official_times[-1]),
            "maximum_absolute_gate_time_error_s": float(np.max(np.abs(residuals))),
            "counterfactual_method": (
                "add the corresponding N264 official-minus-raw residual to each "
                "candidate gate time; retain raw crossing errors"
            ),
        },
        "sweep_contract": {
            "crossing_speeds_m_s": [4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0],
            "maximum_along_speeds_m_s": [8.0, 8.5],
            "slowdown_distances_m": [4.0, 6.0, 8.0, 10.0, 12.0],
            "cross_track_gains_s_inv": [1.25, 1.5, 1.75, 2.0],
            "maximum_cross_track_corrections_m_s": [1.0, 2.0, 4.0],
            "candidate_count": len(sweep),
            "crossing_error_limit_m": 0.45,
            "native_target_radius_m": settings["gate_radius"],
            "target_finish_time_s": target_s,
        },
        "top_safe_global_candidates": [
            item for item in sweep if item["safe_offline"]
        ][:25],
        "minimal_under_target_recommendation": {
            **recommendation,
            "selection_rule": (
                "fewest scalar changes that predict a sub-target safe lap; among "
                "ties retain the slowest candidate to avoid silently "
                "selecting an unnecessarily aggressive rung"
            ),
            "live_promotion": (
                "N266 bounded Gate-3 first; retain N259 inference and every N264 "
                "reset, collision, telemetry, plane-miss, final-stop, and disarm guard"
            ),
        },
        "fastest_safe_global_candidate": fastest_safe,
        "per_gate_schedule_sweep": per_gate,
        "limitations": [
            "This is offline counterfactual evidence, not an official lap.",
            "The replay reuses N264 policy-along-speed versus distance and does not claim exact recurrent-policy counterfactual equivalence.",
            "The first-order response fit is empirical and its full baseline residual is reported above.",
            "Only official v3391 telemetry can prove a faster valid finish.",
        ],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "logs" / "sitl" / "n264_v3391_n259_governed_full_lap_001.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "checkpoints"
        / "drone_race_vq1_v3391_telemetry"
        / "n259_aperture_fullphase_bc_6gate.bin",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "drone_race_vq1_v3391_telemetry.ini",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "logs" / "sitl" / "n265_n264_governor_offline_sweep.json",
    )
    parser.add_argument("--target-s", type=float, default=27.0)
    args = parser.parse_args()
    if args.target_s <= 0.0:
        parser.error("--target-s must be positive")
    report = build_report(
        args.source, args.checkpoint, args.config, target_s=args.target_s
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "experiment": report["experiment"],
                "output": str(args.output),
                "recommendation": report["minimal_under_target_recommendation"],
                "fastest_safe": report["fastest_safe_global_candidate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
