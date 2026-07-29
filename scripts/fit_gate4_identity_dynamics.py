#!/usr/bin/env python3
"""Fit body-relative Gate-4 dynamics from identity-associated live traces.

The fit treats every locally coherent family as a separate segment. Transitions
across closer-family re-associations are excluded because N238 proved that
those discontinuities can represent a different gate. Whole runs, rather than
individual rows, form the cross-validation folds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate4_identity_locked_intercept_n238 as n238


RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
MAXIMUM_TRANSITION_DT_S = 0.35
MINIMUM_TRANSITION_DT_S = 0.05
MAXIMUM_ABS_FILTERED_RATE_M_S = 18.0


def _extract_run(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    samples = list(payload.get("policy_trace", {}).get("samples", []))
    if not samples:
        raise RuntimeError(f"trace has no policy samples: {path}")
    controller = n238.Gate4IdentityLockedInterceptController()
    events: list[dict] = []
    accepted_before = 0
    for sample in samples:
        values = np.asarray(sample["observation"], dtype=np.float64)
        elapsed_s = float(sample["elapsed_s"])
        actual_action = np.asarray(
            sample.get("normalized_action", (0.0, 0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        controller.apply(values, actual_action, now_s=elapsed_s)
        if controller.accepted_fresh == accepted_before:
            continue
        accepted_before = controller.accepted_fresh
        assert controller.associated_pose is not None
        events.append(
            {
                "elapsed_s": elapsed_s,
                "pose": controller.associated_pose.copy(),
                "rate": controller.associated_rate_m_s.copy(),
                "action": actual_action[:3].copy(),
                "segment": controller.closer_reacquisitions,
            }
        )

    rows: list[dict] = []
    segments = sorted({int(event["segment"]) for event in events})
    for segment in segments:
        segment_events = [
            event for event in events if int(event["segment"]) == segment
        ]
        # Each acquisition begins with a deliberately zero rate. Skip its
        # first transition; later transitions use the deployed rate filter.
        for left, right in zip(
            segment_events[1:-1], segment_events[2:], strict=True
        ):
            dt = float(right["elapsed_s"] - left["elapsed_s"])
            if not (MINIMUM_TRANSITION_DT_S <= dt <= MAXIMUM_TRANSITION_DT_S):
                continue
            rate = np.asarray(left["rate"], dtype=np.float64)
            next_rate = np.asarray(right["rate"], dtype=np.float64)
            if (
                np.max(np.abs(rate)) > MAXIMUM_ABS_FILTERED_RATE_M_S
                or np.max(np.abs(next_rate)) > MAXIMUM_ABS_FILTERED_RATE_M_S
            ):
                continue
            acceleration = (next_rate - rate) / dt
            feature = np.concatenate(
                [
                    rate,
                    np.asarray(left["action"], dtype=np.float64),
                    np.asarray([1.0], dtype=np.float64),
                ]
            )
            rows.append(
                {
                    "feature": feature,
                    "acceleration": acceleration,
                    "dt": dt,
                    "rate": rate,
                    "next_rate": next_rate,
                }
            )
    return {
        "path": str(path.resolve()),
        "events": events,
        "rows": rows,
    }


def _fit(rows: list[dict], ridge: float) -> np.ndarray:
    x = np.stack([row["feature"] for row in rows])
    y = np.stack([row["acceleration"] for row in rows])
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _score(rows: list[dict], coefficients: np.ndarray) -> dict:
    x = np.stack([row["feature"] for row in rows])
    acceleration = np.stack([row["acceleration"] for row in rows])
    predicted_acceleration = x @ coefficients
    acceleration_error = predicted_acceleration - acceleration
    predicted_next_rate = np.stack([row["rate"] for row in rows]) + (
        predicted_acceleration
        * np.asarray([row["dt"] for row in rows], dtype=np.float64)[:, None]
    )
    next_rate = np.stack([row["next_rate"] for row in rows])
    rate_error = predicted_next_rate - next_rate
    return {
        "rows": len(rows),
        "acceleration_rmse_m_s2": np.sqrt(
            np.mean(np.square(acceleration_error), axis=0)
        ).tolist(),
        "next_rate_rmse_m_s": np.sqrt(
            np.mean(np.square(rate_error), axis=0)
        ).tolist(),
        "next_rate_mae_m_s": np.mean(np.abs(rate_error), axis=0).tolist(),
    }


def _fit_structured(rows: list[dict], ridge: float) -> np.ndarray:
    coefficients = np.empty((3, 3), dtype=np.float64)
    for axis in range(3):
        x = np.asarray(
            [
                [row["rate"][axis], row["feature"][3 + axis], 1.0]
                for row in rows
            ],
            dtype=np.float64,
        )
        y = np.asarray(
            [row["acceleration"][axis] for row in rows], dtype=np.float64
        )
        penalty = np.eye(3, dtype=np.float64) * float(ridge)
        penalty[-1, -1] = 0.0
        coefficients[axis] = np.linalg.solve(
            x.T @ x + penalty, x.T @ y
        )
    return coefficients


def _score_structured(rows: list[dict], coefficients: np.ndarray) -> dict:
    acceleration = np.stack([row["acceleration"] for row in rows])
    predicted_acceleration = np.empty_like(acceleration)
    for row_index, row in enumerate(rows):
        for axis in range(3):
            predicted_acceleration[row_index, axis] = np.dot(
                coefficients[axis],
                (row["rate"][axis], row["feature"][3 + axis], 1.0),
            )
    dt = np.asarray([row["dt"] for row in rows], dtype=np.float64)[:, None]
    rate = np.stack([row["rate"] for row in rows])
    next_rate = np.stack([row["next_rate"] for row in rows])
    predicted_next_rate = rate + predicted_acceleration * dt
    rate_error = predicted_next_rate - next_rate
    return {
        "rows": len(rows),
        "acceleration_rmse_m_s2": np.sqrt(
            np.mean(np.square(predicted_acceleration - acceleration), axis=0)
        ).tolist(),
        "next_rate_rmse_m_s": np.sqrt(
            np.mean(np.square(rate_error), axis=0)
        ).tolist(),
        "next_rate_mae_m_s": np.mean(np.abs(rate_error), axis=0).tolist(),
    }


def fit(traces: list[Path]) -> dict:
    runs = [_extract_run(path) for path in traces]
    if len(runs) < 3:
        raise RuntimeError("at least three whole-run folds are required")
    if any(len(run["rows"]) < 3 for run in runs):
        raise RuntimeError("every trace must provide at least three transitions")

    ridge_results: list[dict] = []
    for ridge in RIDGE_GRID:
        folds: list[dict] = []
        for holdout_index, holdout in enumerate(runs):
            training_rows = [
                row
                for run_index, run in enumerate(runs)
                if run_index != holdout_index
                for row in run["rows"]
            ]
            coefficients = _fit(training_rows, ridge)
            fold_score = _score(holdout["rows"], coefficients)
            fold_score["trace"] = holdout["path"]
            folds.append(fold_score)
        mean_next_rate_rmse = np.mean(
            [fold["next_rate_rmse_m_s"] for fold in folds], axis=0
        )
        ridge_results.append(
            {
                "ridge": ridge,
                "folds": folds,
                "mean_next_rate_rmse_m_s": mean_next_rate_rmse.tolist(),
                "selection_score": float(np.mean(mean_next_rate_rmse)),
            }
        )
    selected = min(ridge_results, key=lambda item: item["selection_score"])
    selected_ridge = float(selected["ridge"])
    all_rows = [row for run in runs for row in run["rows"]]
    full_coefficients = _fit(all_rows, selected_ridge)
    ensemble = []
    for holdout_index, holdout in enumerate(runs):
        training_rows = [
            row
            for run_index, run in enumerate(runs)
            if run_index != holdout_index
            for row in run["rows"]
        ]
        ensemble.append(
            {
                "held_out_trace": holdout["path"],
                "coefficients": _fit(training_rows, selected_ridge).tolist(),
            }
        )

    # Coefficients map [vx, vy, vz, pitch, roll, thrust, 1] to body-relative
    # acceleration. Report action signs explicitly so a bad/degenerate fit is
    # visible before any controller consumes the artifact.
    action_gain = full_coefficients[3:6, :]
    structured_ridge_results: list[dict] = []
    for ridge in RIDGE_GRID:
        folds: list[dict] = []
        for holdout_index, holdout in enumerate(runs):
            training_rows = [
                row
                for run_index, run in enumerate(runs)
                if run_index != holdout_index
                for row in run["rows"]
            ]
            coefficients = _fit_structured(training_rows, ridge)
            fold_score = _score_structured(holdout["rows"], coefficients)
            fold_score["trace"] = holdout["path"]
            folds.append(fold_score)
        mean_next_rate_rmse = np.mean(
            [fold["next_rate_rmse_m_s"] for fold in folds], axis=0
        )
        structured_ridge_results.append(
            {
                "ridge": ridge,
                "folds": folds,
                "mean_next_rate_rmse_m_s": mean_next_rate_rmse.tolist(),
                "selection_score": float(np.mean(mean_next_rate_rmse)),
            }
        )
    selected_structured = min(
        structured_ridge_results, key=lambda item: item["selection_score"]
    )
    structured_coefficients = _fit_structured(
        all_rows, float(selected_structured["ridge"])
    )
    structured_ensemble = []
    for holdout_index, holdout in enumerate(runs):
        training_rows = [
            row
            for run_index, run in enumerate(runs)
            if run_index != holdout_index
            for row in run["rows"]
        ]
        structured_ensemble.append(
            {
                "held_out_trace": holdout["path"],
                "coefficients": _fit_structured(
                    training_rows, float(selected_structured["ridge"])
                ).tolist(),
            }
        )
    structured_action_gain_ensemble = np.asarray(
        [
            np.asarray(member["coefficients"], dtype=np.float64)[:, 1]
            for member in structured_ensemble
        ],
        dtype=np.float64,
    )
    structured_action_gain_range = np.stack(
        (
            np.min(structured_action_gain_ensemble, axis=0),
            np.max(structured_action_gain_ensemble, axis=0),
        ),
        axis=1,
    )
    # Only roll -> right acceleration is consistently identified in every
    # whole-run fold.  The other action axes remain useful diagnostics, but a
    # deployment controller must not optimize against their sign-ambiguous
    # fold models.
    deployable_action_axes = []
    if np.all(structured_action_gain_ensemble[:, 1] > 1.0):
        deployable_action_axes.append("roll_to_right_acceleration")

    blockers: list[str] = []
    if len(all_rows) < 30:
        blockers.append("fewer than 30 clean associated transitions")
    if not np.all(np.isfinite(full_coefficients)):
        blockers.append("non-finite fitted coefficient")
    if selected["selection_score"] > 3.0:
        blockers.append("whole-run next-rate RMSE exceeds 3 m/s")
    if np.max(np.abs(action_gain)) < 0.10:
        blockers.append("action response is not identified")
    if selected_structured["selection_score"] > 3.0:
        blockers.append("structured whole-run next-rate RMSE exceeds 3 m/s")
    if np.any(structured_coefficients[:, 1] <= 0.0):
        blockers.append("structured action response sign is inconsistent")
    if "roll_to_right_acceleration" not in deployable_action_axes:
        blockers.append("roll-to-right response is not identified in every fold")

    return {
        "schema": "gate4_identity_body_dynamics_v2",
        "feature_order": [
            "forward_rate_m_s",
            "right_rate_m_s",
            "down_rate_m_s",
            "pitch_norm",
            "roll_norm",
            "thrust_norm",
            "bias",
        ],
        "target_order": [
            "forward_acceleration_m_s2",
            "right_acceleration_m_s2",
            "down_acceleration_m_s2",
        ],
        "selected_ridge": selected_ridge,
        "selection_score": selected["selection_score"],
        "coefficients": full_coefficients.tolist(),
        "action_gain": action_gain.tolist(),
        "structured_feature_order": ["axis_rate_m_s", "axis_action_norm", "bias"],
        "structured_coefficients": structured_coefficients.tolist(),
        "structured_full_fit": _score_structured(
            all_rows, structured_coefficients
        ),
        "structured_cross_validation": selected_structured,
        "structured_ridge_search": structured_ridge_results,
        "structured_ensemble": structured_ensemble,
        "structured_action_gain_ensemble": (
            structured_action_gain_ensemble.tolist()
        ),
        "structured_action_gain_range": structured_action_gain_range.tolist(),
        "deployable_action_axes": deployable_action_axes,
        "full_fit": _score(all_rows, full_coefficients),
        "cross_validation": selected,
        "ridge_search": ridge_results,
        "ensemble": ensemble,
        "runs": [
            {
                "trace": run["path"],
                "associated_family_events": len(run["events"]),
                "transition_rows": len(run["rows"]),
            }
            for run in runs
        ],
        "transition_rows": len(all_rows),
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "locally_coherent_family_segments": True,
            "whole_run_cross_validation": True,
            "cross_family_transitions_excluded": True,
            "official_trace_actions_only": True,
            "runtime_privileged_state": False,
            "deployment_is_lateral_axis_only": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit(args.trace)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
