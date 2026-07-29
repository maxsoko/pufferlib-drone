#!/usr/bin/env python3
"""Validate N237 from the first three consecutive coherent Gate-4 frames.

Unlike the original N232 admission evaluator, this extracts the state at the
same three-fresh-frame boundary used by N234-N237, including accumulated
position uncertainty.  Each dynamics fold is excluded from planning and used
as the counterfactual plant.  Rejected state/prediction plans execute the
optimizer's projected intercept action, matching N237; any other rejection
fails closed with the staged brake action.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from gate4_receding_horizon import (
    Gate4RecedingHorizonOptimizer,
    OptimizerConfig,
    VisualDynamicsEnsemble,
    VisualState,
    propagate_member,
)
from policy_callable_gate4_fresh_projected_intercept_n237 import (
    Gate4FreshProjectedInterceptController,
)


REQUIRED_COHERENT_FRESH_SAMPLES = 3
MAXIMUM_INITIAL_RANGE_M = 75.0
TRACK_STALE_RESET_S = 0.30
ALLOWED_REJECTION_REASONS = frozenset(
    {"state_uncertain", "prediction_uncertain"}
)
STAGED_BRAKE_ACTION = (0.25, 0.0, 0.0, 0.0)
OBSERVER_UNCERTAINTY_LIMIT_M = 2.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without_member(
    ensemble: VisualDynamicsEnsemble, excluded_member: int
) -> VisualDynamicsEnsemble:
    keep = np.asarray(
        [index != excluded_member for index in range(ensemble.members)], dtype=bool
    )
    return VisualDynamicsEnsemble(
        thrust_gain=ensemble.thrust_gain[keep].copy(),
        linear_drag_per_s=ensemble.linear_drag_per_s[keep].copy(),
        acceleration_bias_m_s2=ensemble.acceleration_bias_m_s2[keep].copy(),
        angle_error_gain_per_s2=ensemble.angle_error_gain_per_s2[keep].copy(),
        omega_damping_per_s=ensemble.omega_damping_per_s[keep].copy(),
        omega_bias_rad_s2=ensemble.omega_bias_rad_s2[keep].copy(),
        source_path=ensemble.source_path,
        schema=ensemble.schema,
    )


def _entry_from_report_with_ensemble(
    path: Path, ensemble: VisualDynamicsEnsemble
) -> tuple[VisualState, dict]:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    samples = report.get("policy_trace", {}).get("samples", [])
    controller = Gate4FreshProjectedInterceptController(ensemble)
    last_gate3_elapsed_s: float | None = None
    calls = 0
    for sample in samples:
        gate = int(sample.get("official_active_gate_index") or 0)
        observation = sample.get("observation") or []
        if len(observation) != 32:
            continue
        if gate == 2:
            last_gate3_elapsed_s = float(sample["elapsed_s"])
        elapsed_s = float(sample["elapsed_s"])
        target_calls = int(math.floor(elapsed_s * 60.0 + 1e-9)) + 1
        base_actions = sample.get("normalized_action") or [0.0, 0.0, 0.0, 0.0]
        while calls < target_calls:
            controller.apply(
                observation,
                base_actions,
                now_s=calls / 60.0,
            )
            calls += 1
        plans = (
            controller.optimized_plans
            + controller.fresh_projected_intercept_plans
        )
        if gate == 3 and plans > 0 and controller.last_visual_state is not None:
            state = controller.last_visual_state.copy()
            return state, {
                "entry_elapsed_s": elapsed_s,
                "last_gate3_elapsed_s": last_gate3_elapsed_s,
                "logical_calls": calls,
                "coherent_fresh_samples": controller.coherent_fresh_samples,
                "entry_body_ned_m": state.body_ned_m.tolist(),
                "entry_reference_ned_m": state.reference_ned_m.tolist(),
                "entry_velocity_world_m_s": state.velocity_world_m_s.tolist(),
                "entry_position_uncertainty_m": state.position_uncertainty_m,
                "tracker": controller.tracker.snapshot(),
                "observer": controller.snapshot()["action_propagated_observer"],
                "first_plan": dataclasses.asdict(controller.last_plan),
            }
    raise ValueError(f"no three-frame coherent Gate-4 entry in {path}")


def _evaluate_member(
    entry: VisualState,
    ensemble: VisualDynamicsEnsemble,
    *,
    plant_member: int,
    control_dt_s: float,
    maximum_duration_s: float,
) -> dict:
    optimizer = Gate4RecedingHorizonOptimizer(
        _without_member(ensemble, plant_member),
        OptimizerConfig(uncertainty_limit_m=OBSERVER_UNCERTAINTY_LIMIT_M),
    )
    state = entry.copy()
    yaw_target_rad = float(state.euler_rpy_rad[2])
    maximum_steps = int(math.ceil(maximum_duration_s / control_dt_s))
    reasons: dict[str, int] = {}
    projected_intercepts = 0
    neutralizations = 0
    first_action = None
    crossing = False
    for step in range(maximum_steps):
        plan = optimizer.plan(state, yaw_target_rad=yaw_target_rad)
        reasons[plan.reason] = reasons.get(plan.reason, 0) + 1
        if plan.accepted or plan.reason in ALLOWED_REJECTION_REASONS:
            action = plan.action
            projected_intercepts += int(not plan.accepted)
        else:
            action = STAGED_BRAKE_ACTION
            neutralizations += 1
        if first_action is None:
            first_action = [float(value) for value in action]
        previous_forward = float(state.reference_ned_m[0])
        state = propagate_member(
            state,
            action,
            ensemble,
            member=plant_member,
            duration_s=control_dt_s,
        )
        if previous_forward > optimizer.config.plane_forward_m and float(
            state.reference_ned_m[0]
        ) <= optimizer.config.plane_forward_m:
            crossing = True
            break
        if (
            not np.all(np.isfinite(state.relative_world_m))
            or float(np.linalg.norm(state.relative_world_m)) > 100.0
        ):
            break

    terminal = state.reference_ned_m
    centered = bool(
        crossing
        and abs(float(terminal[1])) <= optimizer.config.terminal_center_limit_m
        and abs(float(terminal[2])) <= optimizer.config.terminal_center_limit_m
    )
    return {
        "plant_member": plant_member,
        "success": centered,
        "crossing": crossing,
        "elapsed_s": (step + 1) * control_dt_s,
        "terminal_reference_ned_m": terminal.tolist(),
        "terminal_lateral_vertical_radius_m": float(np.linalg.norm(terminal[1:3])),
        "plan_reasons": reasons,
        "projected_intercept_plans": projected_intercepts,
        "rejected_plan_neutralizations": neutralizations,
        "first_action": first_action,
    }


def evaluate(
    paths: Sequence[Path],
    model_path: Path,
    *,
    control_dt_s: float = 0.10,
    maximum_duration_s: float = 15.0,
) -> dict:
    ensemble = VisualDynamicsEnsemble.load(model_path)
    reports = []
    for path in paths:
        entry, evidence = _entry_from_report_with_ensemble(path, ensemble)
        members = [
            _evaluate_member(
                entry,
                ensemble,
                plant_member=member,
                control_dt_s=control_dt_s,
                maximum_duration_s=maximum_duration_s,
            )
            for member in range(ensemble.members)
        ]
        reports.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "entry": evidence,
                "members": members,
                "passed_members": sum(bool(member["success"]) for member in members),
                "all_members_passed": all(bool(member["success"]) for member in members),
            }
        )
    outcomes = [member for report in reports for member in report["members"]]
    return {
        "schema": "gate4_fresh_projected_intercept_counterfactual_v1",
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "contract": {
            "official_entry_state_only": True,
            "required_consecutive_coherent_fresh_samples": REQUIRED_COHERENT_FRESH_SAMPLES,
            "maximum_initial_range_m": MAXIMUM_INITIAL_RANGE_M,
            "track_stale_reset_s": TRACK_STALE_RESET_S,
            "whole_fold_left_out_of_planner": True,
            "excluded_fold_used_as_counterfactual_plant": True,
            "allowed_rejection_reasons": sorted(ALLOWED_REJECTION_REASONS),
            "observer_uncertainty_limit_m": OBSERVER_UNCERTAINTY_LIMIT_M,
            "control_dt_s": control_dt_s,
            "maximum_duration_s": maximum_duration_s,
            "runtime_privileged_state": False,
        },
        "reports": reports,
        "summary": {
            "reports": len(reports),
            "member_rollouts": len(outcomes),
            "successful_member_rollouts": sum(bool(row["success"]) for row in outcomes),
            "all_passed": bool(outcomes) and all(bool(row["success"]) for row in outcomes),
            "maximum_abs_terminal_right_m": max(
                (abs(float(row["terminal_reference_ned_m"][1])) for row in outcomes),
                default=None,
            ),
            "maximum_abs_terminal_down_m": max(
                (abs(float(row["terminal_reference_ned_m"][2])) for row in outcomes),
                default=None,
            ),
            "maximum_elapsed_s": max(
                (float(row["elapsed_s"]) for row in outcomes), default=None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-dt-s", type=float, default=0.10)
    parser.add_argument("--maximum-duration-s", type=float, default=15.0)
    args = parser.parse_args()
    report = evaluate(
        [path.resolve() for path in args.inputs],
        args.model.resolve(),
        control_dt_s=args.control_dt_s,
        maximum_duration_s=args.maximum_duration_s,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
