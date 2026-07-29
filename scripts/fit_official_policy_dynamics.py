#!/usr/bin/env python3
"""Fit the compact attitude plant from official policy trace observations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _inverse_tanh(value: float, scale: float) -> float:
    return float(np.arctanh(np.clip(value, -0.999999, 0.999999)) * scale)


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = quaternion[0]
    xyz = quaternion[1:]
    return vector + 2.0 * (
        w * np.cross(xyz, vector) + np.cross(xyz, np.cross(xyz, vector))
    )


def _quat_to_roll_pitch(quaternion: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def _command_thrust(normalized: float) -> float:
    hover, minimum, maximum = 0.27, 0.18, 0.42
    span = maximum - hover if normalized >= 0.0 else hover - minimum
    return hover + normalized * span


def _segments(samples: list[dict]) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current: list[dict] = []
    last_gate: int | None = None
    last_time: float | None = None
    for sample in sorted(samples, key=lambda row: float(row["elapsed_s"])):
        observation = sample.get("observation") or []
        gate = int(sample.get("official_active_gate_index") or 0)
        elapsed = float(sample["elapsed_s"])
        # Historical live traces use the 23-field recurrent ABI. Current
        # six-gate traces append official progress, phase flags, confidence,
        # and elapsed fraction, while preserving those first 23 fields exactly.
        usable = (
            len(observation) >= 23
            and float(observation[10]) > 0.5
            and sample.get("gate_pose") is not None
        )
        continuous = (
            usable
            and gate == last_gate
            and last_time is not None
            and 0.04 <= elapsed - last_time <= 0.25
        )
        if not continuous and current:
            if len(current) >= 5:
                segments.append(current)
            current = []
        if usable:
            current.append(sample)
        last_gate = gate if usable else None
        last_time = elapsed if usable else None
    if len(current) >= 5:
        segments.append(current)
    return segments


def _robust_lstsq(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    mask = np.ones(len(target), dtype=bool)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(4):
        coefficients, *_ = np.linalg.lstsq(design[mask], target[mask], rcond=None)
        residual = np.abs(target - design @ coefficients)
        cutoff = np.quantile(residual[mask], 0.95)
        mask &= residual <= max(float(cutoff), 1e-6)
    rmse = float(np.sqrt(np.mean((target[mask] - design[mask] @ coefficients) ** 2)))
    return coefficients, rmse


def fit(paths: list[Path], *, window: int = 7) -> dict:
    acceleration_rows: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
    attitude_rows: list[tuple[float, float, float, float, float]] = []
    gate3_lateral_rows: list[tuple[float, float, float]] = []
    source_segments = 0
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        samples = report.get("policy_trace", {}).get("samples", [])
        for segment in _segments(samples):
            source_segments += 1
            half = window // 2
            for center in range(half, len(segment) - half):
                rows = segment[center - half:center + half + 1]
                times = np.asarray([float(row["elapsed_s"]) for row in rows])
                times -= times[half]
                relative_world = []
                for row in rows:
                    observation = np.asarray(row["observation"], dtype=np.float64)
                    relative_body = np.asarray(
                        [
                            _inverse_tanh(observation[11], 10.0),
                            _inverse_tanh(observation[12], 5.0),
                            -_inverse_tanh(observation[13], 5.0),
                        ]
                    )
                    relative_world.append(
                        _quat_rotate(observation[6:10], relative_body)
                    )
                relative_world_array = np.asarray(relative_world)
                velocity = np.zeros(3)
                acceleration = np.zeros(3)
                for axis in range(3):
                    quadratic = np.polyfit(times, relative_world_array[:, axis], 2)
                    velocity[axis] = -quadratic[1]
                    acceleration[axis] = -2.0 * quadratic[0]
                center_row = rows[half]
                observation = np.asarray(center_row["observation"], dtype=np.float64)
                action = np.asarray(center_row["normalized_action"], dtype=np.float64)
                body_up_world = _quat_rotate(observation[6:10], np.asarray([0.0, 0.0, 1.0]))
                acceleration_rows.append(
                    (acceleration, velocity, _command_thrust(float(action[2])), body_up_world)
                )
                if int(center_row.get("official_active_gate_index") or 0) == 2:
                    relative_right_rates = np.asarray(
                        [
                            _inverse_tanh(float(row["observation"][1]), 3.0)
                            for row in rows
                        ],
                        dtype=np.float64,
                    )
                    lateral_accel = float(
                        np.polyfit(times, relative_right_rates, 1)[0]
                    )
                    roll, _pitch = _quat_to_roll_pitch(observation[6:10])
                    gate3_lateral_rows.append(
                        (
                            lateral_accel,
                            math.tan(roll),
                            float(relative_right_rates[half]),
                        )
                    )

            for previous, current in zip(segment, segment[1:]):
                dt = float(current["elapsed_s"]) - float(previous["elapsed_s"])
                if not 0.04 <= dt <= 0.25:
                    continue
                previous_obs = np.asarray(previous["observation"], dtype=np.float64)
                current_obs = np.asarray(current["observation"], dtype=np.float64)
                previous_action = np.asarray(previous["normalized_action"], dtype=np.float64)
                roll0, pitch0 = _quat_to_roll_pitch(previous_obs[6:10])
                roll1, pitch1 = _quat_to_roll_pitch(current_obs[6:10])
                attitude_rows.append(
                    (dt, roll0, roll1, float(previous_action[1]) * 0.5, 0.0)
                )
                attitude_rows.append(
                    (dt, pitch0, pitch1, float(previous_action[0]) * 0.5, 1.0)
                )

    if not acceleration_rows:
        raise ValueError("no usable same-gate policy-trace windows")

    acceleration = np.asarray([row[0] for row in acceleration_rows])
    velocity = np.asarray([row[1] for row in acceleration_rows])
    thrust_vector = np.asarray([row[2] * row[3] for row in acceleration_rows])
    axis_fits = []
    for axis in range(3):
        columns = [thrust_vector[:, axis], -velocity[:, axis]]
        if axis == 2:
            columns.append(np.ones(len(acceleration)))
        design = np.stack(columns, axis=1)
        coefficients, rmse = _robust_lstsq(design, acceleration[:, axis])
        axis_fits.append(
            {
                "axis": "xyz"[axis],
                "thrust_accel_per_unit": float(coefficients[0]),
                "linear_drag_per_s": float(coefficients[1]),
                "gravity_intercept_m_s2": (
                    float(coefficients[2]) if axis == 2 else 0.0
                ),
                "rmse_m_s2": rmse,
            }
        )

    tau_by_axis: dict[str, list[float]] = {"roll": [], "pitch": []}
    for dt, angle0, angle1, desired, axis_index in attitude_rows:
        denominator = desired - angle0
        if abs(denominator) < 0.03:
            continue
        alpha = (angle1 - angle0) / denominator
        if 0.01 < alpha < 0.95:
            tau = -dt / math.log(1.0 - alpha)
            if 0.01 <= tau <= 2.0:
                tau_by_axis["pitch" if axis_index else "roll"].append(tau)

    gate3_lateral_fit = None
    if gate3_lateral_rows:
        lateral_accel = np.asarray([row[0] for row in gate3_lateral_rows])
        tan_roll = np.asarray([row[1] for row in gate3_lateral_rows])
        right_rate = np.asarray([row[2] for row in gate3_lateral_rows])
        design = np.stack(
            [tan_roll, -right_rate, np.ones(len(gate3_lateral_rows))], axis=1
        )
        coefficients, rmse = _robust_lstsq(design, lateral_accel)
        gate3_lateral_fit = {
            "samples": len(gate3_lateral_rows),
            "accel_per_tan_roll_m_s2": float(coefficients[0]),
            "right_rate_drag_per_s": float(coefficients[1]),
            "bias_m_s2": float(coefficients[2]),
            "rmse_m_s2": rmse,
        }

    return {
        "sources": [str(path) for path in paths],
        "segments": source_segments,
        "acceleration_windows": len(acceleration_rows),
        "axis_fits": axis_fits,
        "attitude_tau_s": {
            axis: (float(np.median(values)) if values else None)
            for axis, values in tau_by_axis.items()
        },
        "gate3_lateral_fit": gate3_lateral_fit,
        "notes": [
            "Fits use only official-observable quaternion and relative gate pose.",
            "Gravity intercept is expected to be negative in the native world-z convention.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.window < 5 or args.window % 2 == 0:
        parser.error("window must be an odd integer of at least 5")
    result = fit(args.trace, window=args.window)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
