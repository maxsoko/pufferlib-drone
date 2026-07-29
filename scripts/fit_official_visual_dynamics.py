#!/usr/bin/env python3
"""Fit and cross-validate a bounded visual plant from official race traces.

Unlike ``fit_official_policy_dynamics.py``, this fitter never differentiates
the policy's filtered/predicted observation.  It reconstructs an inertial
gate vector from each current raw aperture pose and the observable attitude,
then fits a compact causal attitude/translational plant.  Whole reports are
held out so repeated frames from one flight cannot leak into validation.

The emitted JSON is deliberately deployment-oriented: it contains every
source hash, one leave-one-fold-out model per fold, held-out acceleration and
short-horizon rollout errors, and conservative coefficient envelopes.  A
runtime optimizer may use the ensemble only while its prediction disagreement
stays inside a separately preregistered uncertainty bound.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


RAW_SOURCE = "aperture"
OBSERVATION_SIZE = 23
GYRO_SCALE_RAD_S = 20.0
MAX_ROLL_RAD = 0.5
MAX_PITCH_RAD = 0.5
MAX_YAW_RAD = math.pi
HOVER_THRUST = 0.27
MIN_THRUST = 0.18
MAX_THRUST = 0.42


@dataclasses.dataclass(frozen=True)
class TraceSample:
    report: str
    report_fold: int
    elapsed_s: float
    gate_index: int
    body_ned_m: np.ndarray
    relative_world_m: np.ndarray
    quaternion_wxyz: np.ndarray
    euler_rpy_rad: np.ndarray
    gyro_rpy_rad_s: np.ndarray
    normalized_action: np.ndarray


@dataclasses.dataclass(frozen=True)
class DerivativeRow:
    report: str
    report_fold: int
    segment_index: int
    sample_index: int
    elapsed_s: float
    position_world_m: np.ndarray
    velocity_world_m_s: np.ndarray
    acceleration_world_m_s2: np.ndarray
    quaternion_wxyz: np.ndarray
    normalized_action: np.ndarray
    local_position_fit_rmse_m: float
    maximum_observed_step_speed_m_s: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fold_for_report(path: Path, folds: int) -> int:
    digest = hashlib.sha256(path.name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % folds


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _quaternion_rotate(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    xyz = q[1:]
    return v + 2.0 * (
        q[0] * np.cross(xyz, v) + np.cross(xyz, np.cross(xyz, v))
    )


def _quaternion_to_euler(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    return np.asarray(
        [
            math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            math.asin(_clamp(2.0 * (w * y - z * x), -1.0, 1.0)),
            math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ],
        dtype=np.float64,
    )


def _euler_to_quaternion(euler_rpy_rad: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in euler_rpy_rad)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _command_thrust(normalized: float) -> float:
    value = _clamp(normalized, -1.0, 1.0)
    span = MAX_THRUST - HOVER_THRUST if value >= 0.0 else HOVER_THRUST - MIN_THRUST
    return HOVER_THRUST + value * span


def _desired_euler(action: Sequence[float]) -> np.ndarray:
    return np.asarray(
        [
            _clamp(action[1], -1.0, 1.0) * MAX_ROLL_RAD,
            _clamp(action[0], -1.0, 1.0) * MAX_PITCH_RAD,
            _clamp(action[3], -1.0, 1.0) * MAX_YAW_RAD,
        ],
        dtype=np.float64,
    )


def _percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _robust_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    *,
    trim_iterations: int = 5,
    mad_scale: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    if len(target) < design.shape[1] + 2:
        raise ValueError("not enough rows for regression")
    keep = np.ones(len(target), dtype=bool)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(trim_iterations):
        coefficients, *_ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
        residual = np.abs(target - design @ coefficients)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        cutoff = median + mad_scale * 1.4826 * max(mad, 1e-6)
        updated = residual <= max(cutoff, 1e-6)
        if int(updated.sum()) < design.shape[1] + 2 or np.array_equal(updated, keep):
            break
        keep = updated
    coefficients, *_ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
    return coefficients, keep


def _load_segments(
    paths: Sequence[Path],
    *,
    folds: int,
    maximum_range_m: float,
    maximum_gap_s: float,
    minimum_segment_samples: int,
) -> tuple[list[list[TraceSample]], list[dict]]:
    segments: list[list[TraceSample]] = []
    sources: list[dict] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        samples = report.get("policy_trace", {}).get("samples", [])
        fold = _fold_for_report(path, folds)
        current: list[TraceSample] = []
        last_gate: int | None = None
        last_time: float | None = None
        usable_count = 0

        def flush() -> None:
            nonlocal current
            if len(current) >= minimum_segment_samples:
                segments.append(current)
            current = []

        for row in samples:
            observation = row.get("observation") or []
            pose = row.get("gate_pose")
            action = row.get("normalized_action") or []
            elapsed_s = float(row.get("elapsed_s", 0.0))
            gate_index = int(row.get("official_active_gate_index") or 0)
            usable = (
                len(observation) >= OBSERVATION_SIZE
                and len(action) == 4
                and pose is not None
                and pose.get("detection_source") == RAW_SOURCE
                and 0.0 < float(pose.get("range_camera_m", math.inf)) <= maximum_range_m
            )
            continuous = (
                usable
                and last_gate == gate_index
                and last_time is not None
                and 0.04 <= elapsed_s - last_time <= maximum_gap_s
            )
            if not continuous:
                flush()
            if usable:
                body = np.asarray(pose["body_vector_ned_m"], dtype=np.float64)
                quaternion = np.asarray(observation[6:10], dtype=np.float64)
                relative_world = _quaternion_rotate(
                    quaternion, (body[0], body[1], -body[2])
                )
                current.append(
                    TraceSample(
                        report=path.name,
                        report_fold=fold,
                        elapsed_s=elapsed_s,
                        gate_index=gate_index,
                        body_ned_m=body,
                        relative_world_m=relative_world,
                        quaternion_wxyz=quaternion,
                        euler_rpy_rad=_quaternion_to_euler(quaternion),
                        gyro_rpy_rad_s=np.asarray(observation[3:6], dtype=np.float64)
                        * GYRO_SCALE_RAD_S,
                        normalized_action=np.asarray(action, dtype=np.float64),
                    )
                )
                usable_count += 1
                last_gate = gate_index
                last_time = elapsed_s
            else:
                last_gate = None
                last_time = None
        flush()
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "fold": fold,
                "policy_trace_samples": len(samples),
                "usable_raw_aperture_samples": usable_count,
            }
        )
    return segments, sources


def _derivative_rows(
    segments: Sequence[Sequence[TraceSample]],
    *,
    window: int,
    maximum_local_fit_rmse_m: float,
    maximum_observed_step_speed_m_s: float,
    maximum_acceleration_norm_m_s2: float,
) -> tuple[
    list[DerivativeRow],
    dict[tuple[int, int], DerivativeRow],
    dict[str, int],
]:
    if window < 5 or window % 2 == 0:
        raise ValueError("derivative window must be an odd integer >= 5")
    half = window // 2
    rows: list[DerivativeRow] = []
    by_location: dict[tuple[int, int], DerivativeRow] = {}
    quality = {
        "candidate_derivative_rows": 0,
        "rejected_local_fit_rows": 0,
        "rejected_step_speed_rows": 0,
        "rejected_acceleration_rows": 0,
    }
    for segment_index, segment in enumerate(segments):
        for sample_index in range(half, len(segment) - half):
            local = segment[sample_index - half : sample_index + half + 1]
            times = np.asarray([sample.elapsed_s for sample in local], dtype=np.float64)
            times -= times[half]
            positions = np.asarray(
                [sample.relative_world_m for sample in local], dtype=np.float64
            )
            position = np.zeros(3, dtype=np.float64)
            velocity = np.zeros(3, dtype=np.float64)
            acceleration = np.zeros(3, dtype=np.float64)
            reconstructed = np.zeros_like(positions)
            for axis in range(3):
                quadratic = np.polyfit(times, positions[:, axis], 2)
                # Relative gate position is gate - vehicle.
                position[axis] = quadratic[2]
                velocity[axis] = -quadratic[1]
                acceleration[axis] = -2.0 * quadratic[0]
                reconstructed[:, axis] = np.polyval(quadratic, times)
            fit_rmse_m = float(
                np.sqrt(np.mean(np.sum(np.square(positions - reconstructed), axis=1)))
            )
            step_speeds = [
                float(
                    np.linalg.norm(positions[index + 1] - positions[index])
                    / max(times[index + 1] - times[index], 1e-9)
                )
                for index in range(len(local) - 1)
            ]
            maximum_step_speed = max(step_speeds)
            quality["candidate_derivative_rows"] += 1
            if fit_rmse_m > maximum_local_fit_rmse_m:
                quality["rejected_local_fit_rows"] += 1
                continue
            if maximum_step_speed > maximum_observed_step_speed_m_s:
                quality["rejected_step_speed_rows"] += 1
                continue
            if float(np.linalg.norm(acceleration)) > maximum_acceleration_norm_m_s2:
                quality["rejected_acceleration_rows"] += 1
                continue
            center = local[half]
            row = DerivativeRow(
                report=center.report,
                report_fold=center.report_fold,
                segment_index=segment_index,
                sample_index=sample_index,
                elapsed_s=center.elapsed_s,
                position_world_m=position,
                velocity_world_m_s=velocity,
                acceleration_world_m_s2=acceleration,
                quaternion_wxyz=center.quaternion_wxyz,
                normalized_action=center.normalized_action,
                local_position_fit_rmse_m=fit_rmse_m,
                maximum_observed_step_speed_m_s=maximum_step_speed,
            )
            rows.append(row)
            by_location[(segment_index, sample_index)] = row
    quality["retained_derivative_rows"] = len(rows)
    return rows, by_location, quality


def _fit_translation(rows: Sequence[DerivativeRow]) -> dict:
    if not rows:
        raise ValueError("no translation rows")
    velocity = np.asarray([row.velocity_world_m_s for row in rows])
    acceleration = np.asarray([row.acceleration_world_m_s2 for row in rows])
    thrust_axis = []
    for row in rows:
        up_world = _quaternion_rotate(row.quaternion_wxyz, (0.0, 0.0, 1.0))
        thrust_axis.append(up_world * _command_thrust(row.normalized_action[2]))
    thrust_axis_array = np.asarray(thrust_axis, dtype=np.float64)
    # The official quaternion convention needs the empirically established X
    # inversion; Y/Z use the ordinary rotated thrust axis.
    thrust_axis_array[:, 0] *= -1.0
    axes = []
    for axis, name in enumerate(("x", "y", "z")):
        design = np.column_stack(
            (thrust_axis_array[:, axis], -velocity[:, axis], np.ones(len(rows)))
        )
        if axis == 0:
            # Forward racing data occupies a narrow speed/action manifold, so
            # a free intercept is almost perfectly collinear with drag and
            # thrust.  It produced a non-causal +15 m/s^2 bias that could make
            # an optimizer believe it cannot brake.  The physical forward
            # class has no propulsion-independent acceleration: fit only
            # rotated thrust and drag, then append the fixed zero bias.
            reduced_design = np.column_stack(
                (thrust_axis_array[:, axis], -velocity[:, axis])
            )
            reduced, keep = _robust_lstsq(
                reduced_design, acceleration[:, axis]
            )
            coefficients = np.asarray((reduced[0], reduced[1], 0.0))
        else:
            coefficients, keep = _robust_lstsq(design, acceleration[:, axis])
        # Negative thrust authority or drag would make a short-horizon
        # counterfactual actively unstable.  They are outside the physical
        # model class rather than evidence for reverse damping, so project
        # those two coefficients onto the closed non-negative domain and keep
        # the measured bias unchanged.  Held-out metrics below use this exact
        # deployment model, not the unprojected regression.
        unconstrained = coefficients.copy()
        coefficients[0] = max(float(coefficients[0]), 0.0)
        coefficients[1] = max(float(coefficients[1]), 0.0)
        predicted = design @ coefficients
        residual = acceleration[:, axis] - predicted
        axes.append(
            {
                "axis": name,
                "thrust_gain": float(coefficients[0]),
                "linear_drag_per_s": float(coefficients[1]),
                "bias_m_s2": float(coefficients[2]),
                "unconstrained_thrust_gain": float(unconstrained[0]),
                "unconstrained_linear_drag_per_s": float(unconstrained[1]),
                "nonnegative_projection_applied": bool(
                    unconstrained[0] < 0.0 or unconstrained[1] < 0.0
                ),
                "forward_zero_bias_constraint": axis == 0,
                "fit_rows": len(rows),
                "retained_rows": int(keep.sum()),
                "retained_rmse_m_s2": float(
                    np.sqrt(np.mean(np.square(residual[keep])))
                ),
                "all_residual_abs_m_s2": _percentiles(np.abs(residual)),
            }
        )
    return {"axes": axes}


def _attitude_pairs(segments: Sequence[Sequence[TraceSample]]) -> list[dict]:
    rows: list[dict] = []
    for segment in segments:
        for current, following in zip(segment, segment[1:]):
            dt = following.elapsed_s - current.elapsed_s
            if not 0.04 <= dt <= 0.25:
                continue
            desired = _desired_euler(current.normalized_action)
            error = np.asarray(
                [_wrap_angle(desired[i] - current.euler_rpy_rad[i]) for i in range(3)]
            )
            omega_dot = (following.gyro_rpy_rad_s - current.gyro_rpy_rad_s) / dt
            rows.append(
                {
                    "report": current.report,
                    "fold": current.report_fold,
                    "error": error,
                    "omega": current.gyro_rpy_rad_s,
                    "omega_dot": omega_dot,
                }
            )
    return rows


def _fit_attitude(rows: Sequence[dict]) -> dict:
    axes = []
    for axis, name in enumerate(("roll", "pitch", "yaw")):
        design = np.asarray(
            [[row["error"][axis], row["omega"][axis], 1.0] for row in rows],
            dtype=np.float64,
        )
        target = np.asarray([row["omega_dot"][axis] for row in rows], dtype=np.float64)
        coefficients, keep = _robust_lstsq(design, target)
        damping = -float(coefficients[1])
        residual = target - design @ coefficients
        axes.append(
            {
                "axis": name,
                "angle_error_gain_per_s2": float(coefficients[0]),
                "omega_damping_per_s": damping,
                "bias_rad_s2": float(coefficients[2]),
                "derived_rate_lag_tau_s": None if damping <= 0.0 else 1.0 / damping,
                "derived_attitude_kp": None
                if damping <= 0.0
                else float(coefficients[0]) / damping,
                "fit_rows": len(rows),
                "retained_rows": int(keep.sum()),
                "retained_rmse_rad_s2": float(
                    np.sqrt(np.mean(np.square(residual[keep])))
                ),
                "all_residual_abs_rad_s2": _percentiles(np.abs(residual)),
            }
        )
    return {"axes": axes}


def _model_arrays(model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    translation = model["translation"]["axes"]
    attitude = model["attitude"]["axes"]
    return (
        np.asarray([axis["thrust_gain"] for axis in translation]),
        np.asarray([axis["linear_drag_per_s"] for axis in translation]),
        np.asarray([axis["bias_m_s2"] for axis in translation]),
        np.asarray([axis["angle_error_gain_per_s2"] for axis in attitude]),
        np.asarray([axis["omega_damping_per_s"] for axis in attitude]),
        np.asarray([axis["bias_rad_s2"] for axis in attitude]),
    )


def _rollout_step(
    position: np.ndarray,
    velocity: np.ndarray,
    euler: np.ndarray,
    omega: np.ndarray,
    action: np.ndarray,
    model: dict,
    dt: float,
) -> None:
    thrust_gain, drag, accel_bias, angle_gain, damping, omega_bias = _model_arrays(model)
    desired = _desired_euler(action)
    error = np.asarray([_wrap_angle(desired[i] - euler[i]) for i in range(3)])
    omega += (angle_gain * error - damping * omega + omega_bias) * dt
    euler += omega * dt
    euler[2] = _wrap_angle(euler[2])
    up = _quaternion_rotate(_euler_to_quaternion(euler), (0.0, 0.0, 1.0))
    thrust_axis = up * _command_thrust(action[2])
    thrust_axis[0] *= -1.0
    acceleration = thrust_gain * thrust_axis - drag * velocity + accel_bias
    velocity += acceleration * dt
    # Relative gate vector is gate - vehicle.
    position -= velocity * dt


def _rollout_errors(
    segments: Sequence[Sequence[TraceSample]],
    derivatives: dict[tuple[int, int], DerivativeRow],
    model: dict,
    *,
    holdout_fold: int,
    horizon_s: float,
    integration_dt_s: float,
) -> dict:
    errors = {"x": [], "y": [], "z": [], "lateral_vertical": []}
    horizons = []
    for segment_index, segment in enumerate(segments):
        if not segment or segment[0].report_fold != holdout_fold:
            continue
        centers = sorted(
            index
            for (candidate_segment, index) in derivatives
            if candidate_segment == segment_index
        )
        if len(centers) < 2:
            continue
        center_set = set(centers)
        # Spacing starts avoids counting nearly identical overlapping windows
        # as independent validation cases.
        for start_index in centers[::3]:
            start = derivatives[(segment_index, start_index)]
            position = start.position_world_m.copy()
            velocity = start.velocity_world_m_s.copy()
            sample = segment[start_index]
            euler = sample.euler_rpy_rad.copy()
            omega = sample.gyro_rpy_rad_s.copy()
            end_index = start_index
            while end_index + 1 < len(segment):
                current = segment[end_index]
                following = segment[end_index + 1]
                interval = following.elapsed_s - current.elapsed_s
                if interval <= 0.0:
                    break
                remaining = min(interval, horizon_s - (current.elapsed_s - start.elapsed_s))
                while remaining > 1e-9:
                    step = min(integration_dt_s, remaining)
                    _rollout_step(
                        position,
                        velocity,
                        euler,
                        omega,
                        current.normalized_action,
                        model,
                        step,
                    )
                    remaining -= step
                end_index += 1
                elapsed = segment[end_index].elapsed_s - start.elapsed_s
                if elapsed + 1e-9 >= horizon_s:
                    break
            if end_index not in center_set:
                continue
            elapsed = segment[end_index].elapsed_s - start.elapsed_s
            if elapsed < 0.75 * horizon_s:
                continue
            target = derivatives[(segment_index, end_index)].position_world_m
            error = position - target
            errors["x"].append(abs(float(error[0])))
            errors["y"].append(abs(float(error[1])))
            errors["z"].append(abs(float(error[2])))
            errors["lateral_vertical"].append(float(np.linalg.norm(error[1:3])))
            horizons.append(elapsed)
    return {
        "requested_horizon_s": horizon_s,
        "actual_horizon_s": _percentiles(horizons),
        "absolute_position_error_m": {
            name: _percentiles(values) for name, values in errors.items()
        },
    }


def _heldout_residuals(rows: Sequence[DerivativeRow], model: dict, fold: int) -> dict:
    heldout = [row for row in rows if row.report_fold == fold]
    if not heldout:
        return {"count": 0, "axes": {}}
    thrust_gain, drag, bias, *_ = _model_arrays(model)
    errors = [[] for _ in range(3)]
    for row in heldout:
        up = _quaternion_rotate(row.quaternion_wxyz, (0.0, 0.0, 1.0))
        thrust_axis = up * _command_thrust(row.normalized_action[2])
        thrust_axis[0] *= -1.0
        predicted = thrust_gain * thrust_axis - drag * row.velocity_world_m_s + bias
        for axis in range(3):
            errors[axis].append(abs(float(predicted[axis] - row.acceleration_world_m_s2[axis])))
    return {
        "count": len(heldout),
        "axes": {
            name: _percentiles(errors[index])
            for index, name in enumerate(("x", "y", "z"))
        },
    }


def _coefficient_envelopes(models: Sequence[dict]) -> dict:
    output: dict[str, dict[str, dict[str, float]]] = {"translation": {}, "attitude": {}}
    for section, keys in (
        ("translation", ("thrust_gain", "linear_drag_per_s", "bias_m_s2")),
        (
            "attitude",
            ("angle_error_gain_per_s2", "omega_damping_per_s", "bias_rad_s2"),
        ),
    ):
        names = [axis["axis"] for axis in models[0][section]["axes"]]
        for axis_index, name in enumerate(names):
            output[section][name] = {}
            for key in keys:
                values = np.asarray(
                    [model[section]["axes"][axis_index][key] for model in models],
                    dtype=np.float64,
                )
                output[section][name][key] = {
                    "minimum": float(np.min(values)),
                    "median": float(np.median(values)),
                    "maximum": float(np.max(values)),
                }
    return output


def fit(
    paths: Sequence[Path],
    *,
    folds: int = 5,
    maximum_range_m: float = 35.0,
    maximum_gap_s: float = 0.22,
    derivative_window: int = 9,
    maximum_local_fit_rmse_m: float = 0.75,
    maximum_observed_step_speed_m_s: float = 30.0,
    maximum_acceleration_norm_m_s2: float = 30.0,
    rollout_horizon_s: float = 0.8,
    integration_dt_s: float = 0.02,
) -> dict:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    minimum_segment_samples = max(derivative_window, 5)
    segments, sources = _load_segments(
        paths,
        folds=folds,
        maximum_range_m=maximum_range_m,
        maximum_gap_s=maximum_gap_s,
        minimum_segment_samples=minimum_segment_samples,
    )
    derivatives, derivative_lookup, derivative_quality = _derivative_rows(
        segments,
        window=derivative_window,
        maximum_local_fit_rmse_m=maximum_local_fit_rmse_m,
        maximum_observed_step_speed_m_s=maximum_observed_step_speed_m_s,
        maximum_acceleration_norm_m_s2=maximum_acceleration_norm_m_s2,
    )
    attitude_rows = _attitude_pairs(segments)
    if not derivatives or not attitude_rows:
        raise ValueError("official traces did not yield enough physical rows")

    models = []
    validation = []
    for fold in range(folds):
        train_derivatives = [row for row in derivatives if row.report_fold != fold]
        train_attitude = [row for row in attitude_rows if row["fold"] != fold]
        model = {
            "excluded_fold": fold,
            "translation": _fit_translation(train_derivatives),
            "attitude": _fit_attitude(train_attitude),
        }
        models.append(model)
        validation.append(
            {
                "fold": fold,
                "heldout_acceleration": _heldout_residuals(derivatives, model, fold),
                "heldout_rollout": _rollout_errors(
                    segments,
                    derivative_lookup,
                    model,
                    holdout_fold=fold,
                    horizon_s=rollout_horizon_s,
                    integration_dt_s=integration_dt_s,
                ),
            }
        )

    return {
        "schema": "official_visual_dynamics_v1",
        "observable_contract": {
            "raw_pose_source": RAW_SOURCE,
            "uses_policy_filtered_or_predicted_pose": False,
            "uses_privileged_position_or_velocity": False,
            "quaternion_source": "observation[6:10]",
            "gyro_source": "observation[3:6] * 20 rad/s",
            "action_contract": "desired_pitch_roll_thrust_yaw",
            "body_ned_to_world_z_up": "quat_rotate([forward,right,-down])",
            "forward_thrust_axis_inverted": True,
        },
        "fit_config": {
            "folds": folds,
            "maximum_raw_range_m": maximum_range_m,
            "maximum_sample_gap_s": maximum_gap_s,
            "derivative_window_samples": derivative_window,
            "maximum_local_position_fit_rmse_m": maximum_local_fit_rmse_m,
            "maximum_observed_step_speed_m_s": maximum_observed_step_speed_m_s,
            "maximum_acceleration_norm_m_s2": maximum_acceleration_norm_m_s2,
            "rollout_horizon_s": rollout_horizon_s,
            "integration_dt_s": integration_dt_s,
            "hover_thrust": HOVER_THRUST,
            "minimum_thrust": MIN_THRUST,
            "maximum_thrust": MAX_THRUST,
        },
        "sources": sources,
        "corpus": {
            "reports": len(paths),
            "segments": len(segments),
            "raw_aperture_samples_in_segments": int(sum(len(segment) for segment in segments)),
            "derivative_rows": len(derivatives),
            "derivative_quality": derivative_quality,
            "attitude_transition_rows": len(attitude_rows),
            "gate_counts": {
                str(gate): int(sum(segment[0].gate_index == gate for segment in segments))
                for gate in sorted({segment[0].gate_index for segment in segments})
            },
        },
        "fold_models": models,
        "coefficient_envelopes": _coefficient_envelopes(models),
        "heldout_validation": validation,
        "deployment_rule": {
            "ensemble_members": folds,
            "execute_only_first_control_then_replan": True,
            "prediction_must_stop_at_gate_plane_time_or_uncertainty_bound": True,
            "coefficient_extrapolation_permitted": False,
        },
    }


def _paths_from_args(values: Iterable[str]) -> list[Path]:
    paths = [Path(value).resolve() for value in values]
    if not paths:
        paths = sorted(Path("logs/sitl").resolve().glob("competition_smoke_policy_*.json"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="official smoke JSON reports")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--maximum-range-m", type=float, default=35.0)
    parser.add_argument("--maximum-gap-s", type=float, default=0.22)
    parser.add_argument("--derivative-window", type=int, default=9)
    parser.add_argument("--maximum-local-fit-rmse-m", type=float, default=0.75)
    parser.add_argument("--maximum-observed-step-speed-m-s", type=float, default=30.0)
    parser.add_argument("--maximum-acceleration-norm-m-s2", type=float, default=30.0)
    parser.add_argument("--rollout-horizon-s", type=float, default=0.8)
    parser.add_argument("--integration-dt-s", type=float, default=0.02)
    args = parser.parse_args()
    paths = _paths_from_args(args.inputs)
    report = fit(
        paths,
        folds=args.folds,
        maximum_range_m=args.maximum_range_m,
        maximum_gap_s=args.maximum_gap_s,
        derivative_window=args.derivative_window,
        maximum_local_fit_rmse_m=args.maximum_local_fit_rmse_m,
        maximum_observed_step_speed_m_s=args.maximum_observed_step_speed_m_s,
        maximum_acceleration_norm_m_s2=args.maximum_acceleration_norm_m_s2,
        rollout_horizon_s=args.rollout_horizon_s,
        integration_dt_s=args.integration_dt_s,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "corpus": report["corpus"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
