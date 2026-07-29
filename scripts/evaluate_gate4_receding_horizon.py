#!/usr/bin/env python3
"""Counterfactually validate Gate-4 MPC from recorded official entries.

Each supplied official smoke report contributes the first coherent Gate-4
state plus the observable velocity carried from its final Gate-3 observation.
For every dynamics fold, the optimizer is rebuilt without that fold and the
excluded member is used as the counterfactual plant.  This prevents a member
from validating a controller that was allowed to plan with the same member.
Success requires a directed plane crossing inside the centered terminal box;
timeouts and model divergence fail closed.
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
    Gate4VisualStateTracker,
    OptimizerConfig,
    VisualDynamicsEnsemble,
    VisualState,
    propagate_member,
)


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


def _entry_from_report(path: Path) -> tuple[VisualState, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = report.get("policy_trace", {}).get("samples", [])
    tracker = Gate4VisualStateTracker()
    last_gate3_elapsed_s: float | None = None
    for sample in samples:
        gate = int(sample.get("official_active_gate_index") or 0)
        observation = sample.get("observation") or []
        if len(observation) != 32:
            continue
        if gate == 2:
            tracker.observe_prefix(observation, gate_index=gate)
            last_gate3_elapsed_s = float(sample["elapsed_s"])
        elif gate == 3:
            state, fresh = tracker.update_gate4(
                observation, now_s=float(sample["elapsed_s"])
            )
            if fresh and state is not None:
                return state, {
                    "entry_elapsed_s": float(sample["elapsed_s"]),
                    "last_gate3_elapsed_s": last_gate3_elapsed_s,
                    "entry_body_ned_m": state.body_ned_m.tolist(),
                    "entry_reference_ned_m": state.reference_ned_m.tolist(),
                    "entry_velocity_world_m_s": state.velocity_world_m_s.tolist(),
                    "entry_position_uncertainty_m": state.position_uncertainty_m,
                    "tracker": tracker.snapshot(),
                }
    raise ValueError(f"no coherent Gate-4 entry in {path}")


def _evaluate_member(
    entry: VisualState,
    ensemble: VisualDynamicsEnsemble,
    *,
    plant_member: int,
    control_dt_s: float,
    maximum_duration_s: float,
    optimizer_config: OptimizerConfig,
) -> dict:
    planning_ensemble = _without_member(ensemble, plant_member)
    optimizer = Gate4RecedingHorizonOptimizer(
        planning_ensemble, optimizer_config
    )
    state = entry.copy()
    yaw_target_rad = float(state.euler_rpy_rad[2])
    maximum_steps = int(math.ceil(maximum_duration_s / control_dt_s))
    reasons: dict[str, int] = {}
    actions: list[list[float]] = []
    plans = []
    crossing = False
    for step in range(maximum_steps):
        plan = optimizer.plan(state, yaw_target_rad=yaw_target_rad)
        reasons[plan.reason] = reasons.get(plan.reason, 0) + 1
        actions.append([float(value) for value in plan.action])
        if len(plans) < 12 or step % 10 == 0:
            plans.append(
                {
                    "step": step,
                    "reference_ned_m": state.reference_ned_m.tolist(),
                    "action": actions[-1],
                    "accepted": plan.accepted,
                    "reason": plan.reason,
                    "horizon_s": plan.horizon_s,
                    "ensemble_disagreement_m": plan.ensemble_disagreement_m,
                    "predicted_terminal_body_ned_m": plan.predicted_terminal_body_ned_m,
                }
            )
        previous_forward = float(state.reference_ned_m[0])
        state = propagate_member(
            state,
            plan.action,
            ensemble,
            member=plant_member,
            duration_s=control_dt_s,
        )
        if previous_forward > optimizer_config.plane_forward_m and float(
            state.reference_ned_m[0]
        ) <= optimizer_config.plane_forward_m:
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
        and abs(float(terminal[1])) <= optimizer_config.terminal_center_limit_m
        and abs(float(terminal[2])) <= optimizer_config.terminal_center_limit_m
    )
    return {
        "plant_member": plant_member,
        "planning_members": planning_ensemble.members,
        "success": centered,
        "crossing": crossing,
        "elapsed_s": (step + 1) * control_dt_s,
        "steps": step + 1,
        "terminal_reference_ned_m": terminal.tolist(),
        "terminal_lateral_vertical_radius_m": float(np.linalg.norm(terminal[1:3])),
        "plan_reasons": reasons,
        "first_action": None if not actions else actions[0],
        "last_action": None if not actions else actions[-1],
        "sampled_plan_trace": plans,
    }


def evaluate(
    paths: Sequence[Path],
    model_path: Path,
    *,
    control_dt_s: float = 0.10,
    maximum_duration_s: float = 15.0,
    optimizer_config: OptimizerConfig | None = None,
) -> dict:
    ensemble = VisualDynamicsEnsemble.load(model_path)
    config = optimizer_config or OptimizerConfig()
    reports = []
    for path in paths:
        entry, entry_evidence = _entry_from_report(path)
        members = [
            _evaluate_member(
                entry,
                ensemble,
                plant_member=member,
                control_dt_s=control_dt_s,
                maximum_duration_s=maximum_duration_s,
                optimizer_config=config,
            )
            for member in range(ensemble.members)
        ]
        reports.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "entry": entry_evidence,
                "members": members,
                "passed_members": sum(bool(member["success"]) for member in members),
                "all_members_passed": all(bool(member["success"]) for member in members),
            }
        )
    outcomes = [member for report in reports for member in report["members"]]
    return {
        "schema": "gate4_receding_horizon_counterfactual_v1",
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "controller_source_sha256": _sha256(
            Path(__file__).resolve().with_name("gate4_receding_horizon.py")
        ),
        "contract": {
            "official_entry_state_only": True,
            "whole_fold_left_out_of_planner": True,
            "excluded_fold_used_as_counterfactual_plant": True,
            "control_dt_s": control_dt_s,
            "maximum_duration_s": maximum_duration_s,
            "terminal_center_limit_m": config.terminal_center_limit_m,
            "runtime_privileged_state": False,
        },
        "optimizer_config": dataclasses.asdict(config),
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
    parser.add_argument("inputs", nargs="+", help="official smoke report JSON files")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-dt-s", type=float, default=0.10)
    parser.add_argument("--maximum-duration-s", type=float, default=15.0)
    args = parser.parse_args()
    report = evaluate(
        [Path(value).resolve() for value in args.inputs],
        Path(args.model).resolve(),
        control_dt_s=args.control_dt_s,
        maximum_duration_s=args.maximum_duration_s,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
