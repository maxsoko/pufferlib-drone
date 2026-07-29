#!/usr/bin/env python3
"""Offline admission audit for N240 on all three official N239 traces.

Retained legal observations are replayed through the real N240 association
object at their recorded timestamps.  The first continuously associated
near-family transition supplies the lateral state.  Each fitted dynamics member
is then held out of the planner and used as the counterfactual plant during a
0.1 s receding-horizon rollout.  This is a lateral controller admission test;
it makes no claim about unmodelled forward or vertical plant response.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

import evaluate_gate4_identity_body_mpc as evaluate_n239
import gate4_identity_body_mpc as body_mpc
import policy_callable_gate4_continuous_near_mpc_n240 as n240


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(path: Path, ensemble: body_mpc.LateralDynamicsEnsemble) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    controller = n240.Gate4ContinuousNearMPCController(ensemble)
    gate4_samples = [
        sample
        for sample in payload.get("policy_trace", {}).get("samples", [])
        if int(sample.get("official_active_gate_index", -1))
        == n240.n239.n238.GATE4_INDEX
    ]
    trigger = None
    for sample in gate4_samples:
        previous = controller.continuous_near_family_triggers
        action = controller.apply(
            np.asarray(sample["observation"], dtype=np.float64),
            [0.0, 0.0, 0.0, 0.0],
            now_s=float(sample["elapsed_s"]),
        )
        if controller.continuous_near_family_triggers > previous:
            assert controller.associated_pose is not None
            assert controller.last_lateral_plan is not None
            trigger = {
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": float(controller.associated_pose[0]),
                "forward_rate_m_s": float(controller.associated_rate_m_s[0]),
                "right_m": float(controller.associated_pose[1]),
                "right_rate_m_s": float(controller.associated_rate_m_s[1]),
                "down_m": float(controller.associated_pose[2]),
                "down_rate_m_s": float(controller.associated_rate_m_s[2]),
                "first_action": [float(value) for value in action],
                "first_plan": dataclasses.asdict(controller.last_lateral_plan),
            }
            break
    if trigger is None:
        raise RuntimeError(f"N240 did not unlock on retained trace {path}")
    trigger["retained_gate4_samples"] = len(gate4_samples)
    return trigger


def _rollout(
    entry: dict,
    ensemble: body_mpc.LateralDynamicsEnsemble,
    *,
    plant_member: int,
    dt_s: float,
) -> dict:
    planner = body_mpc.Gate4IdentityLateralMPC(
        evaluate_n239._without_member(ensemble, plant_member)
    )
    plant = evaluate_n239._one_member(ensemble, plant_member)
    forward_m = float(entry["forward_m"])
    forward_rate_m_s = float(entry["forward_rate_m_s"])
    closing_m_s = max(
        -forward_rate_m_s, planner.config.minimum_closing_m_s
    )
    total_s = planner.horizon_s(forward_m, forward_rate_m_s)
    right_m = float(entry["right_m"])
    right_rate_m_s = float(entry["right_rate_m_s"])
    actions = []
    predicted_worst = []
    elapsed_s = 0.0
    while elapsed_s < total_s - 1e-12:
        step_s = min(dt_s, total_s - elapsed_s)
        scheduled_forward_m = max(0.01, forward_m - closing_m_s * elapsed_s)
        plan = planner.plan(
            forward_m=scheduled_forward_m,
            forward_rate_m_s=-closing_m_s,
            right_m=right_m,
            right_rate_m_s=right_rate_m_s,
        )
        actions.append(float(plan.roll_norm))
        predicted_worst.append(float(plan.worst_abs_terminal_right_m))
        position, rate = body_mpc.propagate_lateral(
            np.asarray([[right_m]], dtype=np.float64),
            np.asarray([[right_rate_m_s]], dtype=np.float64),
            np.asarray([[plan.roll_norm]], dtype=np.float64),
            plant,
            step_s,
        )
        right_m = float(position[0, 0])
        right_rate_m_s = float(rate[0, 0])
        elapsed_s += step_s
    success = abs(right_m) <= planner.config.terminal_center_limit_m
    return {
        "plant_member": plant_member,
        "planning_members": planner.ensemble.members,
        "success": success,
        "horizon_s": total_s,
        "terminal_right_m": right_m,
        "terminal_right_rate_m_s": right_rate_m_s,
        "roll_min": min(actions),
        "roll_max": max(actions),
        "first_rolls": actions[:8],
        "last_rolls": actions[-8:],
        "maximum_predicted_worst_abs_terminal_right_m": max(predicted_worst),
        "plans": len(actions),
    }


def evaluate(
    paths: Sequence[Path],
    model_path: Path,
    *,
    control_dt_s: float = 0.1,
) -> dict:
    os.environ[n240.n239.MODEL_ENVIRONMENT_VARIABLE] = str(model_path.resolve())
    ensemble = body_mpc.LateralDynamicsEnsemble.load(model_path)
    reports = []
    outcomes = []
    for path in paths:
        entry = _entry(path, ensemble)
        members = [
            _rollout(
                entry,
                ensemble,
                plant_member=member,
                dt_s=control_dt_s,
            )
            for member in range(ensemble.members)
        ]
        outcomes.extend(members)
        reports.append(
            {
                "trace": str(path.resolve()),
                "trace_sha256": _sha256(path),
                "entry": entry,
                "members": members,
                "all_members_passed": all(row["success"] for row in members),
            }
        )
    return {
        "schema": "gate4_continuous_near_mpc_n240_admission_v1",
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "policy_source_sha256": _sha256(
            Path(__file__).resolve().with_name(
                "policy_callable_gate4_continuous_near_mpc_n240.py"
            )
        ),
        "optimizer_source_sha256": _sha256(
            Path(__file__).resolve().with_name("gate4_identity_body_mpc.py")
        ),
        "contract": {
            "official_n239_retained_observations_only": True,
            "continuous_accepted_association_transition_required": True,
            "threshold_forward_m": n240.TRUSTED_FAMILY_MAXIMUM_FORWARD_M,
            "forward_boundary_tolerance_m": n240.FORWARD_BOUNDARY_TOLERANCE_M,
            "nominal_threshold_changed_from_n239": False,
            "rejected_pose_action_authority": False,
            "whole_run_member_left_out_of_planner": True,
            "excluded_member_used_as_counterfactual_plant": True,
            "lateral_axis_only": True,
            "forward_schedule_is_not_a_plant_claim": True,
            "control_dt_s": control_dt_s,
            "runtime_privileged_state": False,
        },
        "reports": reports,
        "summary": {
            "traces": len(reports),
            "continuous_unlocks": len(reports),
            "member_rollouts": len(outcomes),
            "successful_member_rollouts": sum(row["success"] for row in outcomes),
            "all_passed": bool(outcomes) and all(row["success"] for row in outcomes),
            "maximum_abs_terminal_right_m": max(
                abs(float(row["terminal_right_m"])) for row in outcomes
            ),
            "maximum_abs_terminal_right_rate_m_s": max(
                abs(float(row["terminal_right_rate_m_s"])) for row in outcomes
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-dt-s", type=float, default=0.1)
    args = parser.parse_args()
    result = evaluate(
        [path.resolve() for path in args.inputs],
        args.model.resolve(),
        control_dt_s=args.control_dt_s,
    )
    args.output.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
