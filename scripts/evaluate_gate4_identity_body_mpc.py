#!/usr/bin/env python3
"""Held-out counterfactual admission test for N239 lateral MPC.

Each N238 collision trace supplies every identity-associated family entry and
the measured time remaining until the Gate-4 frame impact.  For every entry,
each whole-run dynamics model is treated as the counterfactual plant while the
planner is rebuilt without that member.  Forward range is only a measured time
schedule; no simulator truth or map state is introduced.  Admission requires
every rollout to reach the measured plane time within the lateral center box.
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

import fit_gate4_identity_dynamics as fit_dynamics
from gate4_identity_body_mpc import (
    Gate4IdentityLateralMPC,
    LateralDynamicsEnsemble,
    LateralMPCConfig,
    propagate_lateral,
)


TRUSTED_FAMILY_MAXIMUM_ACQUISITION_FORWARD_M = 16.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without_member(
    ensemble: LateralDynamicsEnsemble, excluded_member: int
) -> LateralDynamicsEnsemble:
    keep = np.asarray(
        [index != excluded_member for index in range(ensemble.members)],
        dtype=bool,
    )
    return LateralDynamicsEnsemble(
        rate_gain_per_s=ensemble.rate_gain_per_s[keep].copy(),
        roll_gain_m_s2=ensemble.roll_gain_m_s2[keep].copy(),
        bias_m_s2=ensemble.bias_m_s2[keep].copy(),
        source_path=ensemble.source_path,
        source_sha256=ensemble.source_sha256,
    )


def _one_member(
    ensemble: LateralDynamicsEnsemble, member: int
) -> LateralDynamicsEnsemble:
    selection = np.asarray([member], dtype=np.int64)
    return LateralDynamicsEnsemble(
        rate_gain_per_s=ensemble.rate_gain_per_s[selection].copy(),
        roll_gain_m_s2=ensemble.roll_gain_m_s2[selection].copy(),
        bias_m_s2=ensemble.bias_m_s2[selection].copy(),
        source_path=ensemble.source_path,
        source_sha256=ensemble.source_sha256,
    )


def _entry_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    duration_s = float(payload["sitl"]["duration_s"])
    collision_id = int(payload["sitl"]["latest_telemetry"]["collision_id"])
    if collision_id != 1002:
        raise ValueError(f"trace does not end at the Gate-4 frame: {path}")
    run = fit_dynamics._extract_run(path)
    cases = []
    for segment in sorted({int(event["segment"]) for event in run["events"]}):
        events = [
            event for event in run["events"] if int(event["segment"]) == segment
        ]
        entry = events[0]
        remaining_s = duration_s - float(entry["elapsed_s"])
        if float(entry["pose"][0]) > TRUSTED_FAMILY_MAXIMUM_ACQUISITION_FORWARD_M:
            continue
        if remaining_s < 0.50 or remaining_s > 8.0:
            continue
        cases.append(
            {
                "segment": segment,
                "entry_elapsed_s": float(entry["elapsed_s"]),
                "measured_plane_budget_s": remaining_s,
                "forward_m": float(entry["pose"][0]),
                "right_m": float(entry["pose"][1]),
                "right_rate_m_s": float(entry["rate"][1]),
                "associated_events": len(events),
            }
        )
    if not cases:
        raise ValueError(f"trace has no eligible associated entry: {path}")
    return cases


def _evaluate_case(
    case: dict,
    ensemble: LateralDynamicsEnsemble,
    *,
    plant_member: int,
    control_dt_s: float,
    config: LateralMPCConfig,
) -> dict:
    planner = Gate4IdentityLateralMPC(
        _without_member(ensemble, plant_member), config
    )
    plant = _one_member(ensemble, plant_member)
    total_s = float(case["measured_plane_budget_s"])
    initial_forward_m = float(case["forward_m"])
    forward_rate_m_s = -initial_forward_m / total_s
    right_m = float(case["right_m"])
    right_rate_m_s = float(case["right_rate_m_s"])
    steps = max(1, int(math.ceil(total_s / control_dt_s)))
    actions: list[float] = []
    worst_predicted: list[float] = []
    elapsed_s = 0.0
    for step in range(steps):
        dt_s = min(control_dt_s, total_s - elapsed_s)
        if dt_s <= 0.0:
            break
        forward_m = max(
            0.01,
            initial_forward_m * (1.0 - elapsed_s / total_s),
        )
        plan = planner.plan(
            forward_m=forward_m,
            forward_rate_m_s=forward_rate_m_s,
            right_m=right_m,
            right_rate_m_s=right_rate_m_s,
        )
        actions.append(float(plan.roll_norm))
        worst_predicted.append(float(plan.worst_abs_terminal_right_m))
        position, rate = propagate_lateral(
            np.asarray([[right_m]], dtype=np.float64),
            np.asarray([[right_rate_m_s]], dtype=np.float64),
            np.asarray([[plan.roll_norm]], dtype=np.float64),
            plant,
            dt_s,
        )
        right_m = float(position[0, 0])
        right_rate_m_s = float(rate[0, 0])
        elapsed_s += dt_s
    success = abs(right_m) <= config.terminal_center_limit_m
    return {
        "plant_member": plant_member,
        "planning_members": planner.ensemble.members,
        "success": success,
        "terminal_right_m": right_m,
        "terminal_right_rate_m_s": right_rate_m_s,
        "roll_min": min(actions),
        "roll_max": max(actions),
        "first_rolls": actions[:8],
        "last_rolls": actions[-8:],
        "maximum_predicted_worst_abs_terminal_right_m": max(worst_predicted),
        "plans": len(actions),
    }


def evaluate(
    paths: Sequence[Path],
    model_path: Path,
    *,
    control_dt_s: float = 0.10,
    config: LateralMPCConfig | None = None,
) -> dict:
    ensemble = LateralDynamicsEnsemble.load(model_path)
    optimizer_config = config or LateralMPCConfig()
    reports = []
    outcomes = []
    for path in paths:
        cases = []
        for entry in _entry_cases(path):
            members = [
                _evaluate_case(
                    entry,
                    ensemble,
                    plant_member=member,
                    control_dt_s=control_dt_s,
                    config=optimizer_config,
                )
                for member in range(ensemble.members)
            ]
            outcomes.extend(members)
            cases.append(
                {
                    "entry": entry,
                    "members": members,
                    "passed_members": sum(bool(row["success"]) for row in members),
                    "all_members_passed": all(bool(row["success"]) for row in members),
                }
            )
        reports.append(
            {
                "trace": str(path.resolve()),
                "trace_sha256": _sha256(path),
                "cases": cases,
                "all_cases_passed": all(row["all_members_passed"] for row in cases),
            }
        )
    return {
        "schema": "gate4_identity_body_lateral_mpc_counterfactual_v1",
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "optimizer_source_sha256": _sha256(
            Path(__file__).resolve().with_name("gate4_identity_body_mpc.py")
        ),
        "fit_source_sha256": _sha256(
            Path(__file__).resolve().with_name("fit_gate4_identity_dynamics.py")
        ),
        "contract": {
            "official_n238_associated_entries_only": True,
            "far_associated_families_quarantined": True,
            "trusted_family_maximum_acquisition_forward_m": (
                TRUSTED_FAMILY_MAXIMUM_ACQUISITION_FORWARD_M
            ),
            "measured_collision_plane_time_budget": True,
            "whole_run_member_left_out_of_planner": True,
            "excluded_member_used_as_counterfactual_plant": True,
            "lateral_axis_only": True,
            "control_dt_s": control_dt_s,
            "terminal_center_limit_m": optimizer_config.terminal_center_limit_m,
            "runtime_privileged_state": False,
        },
        "optimizer_config": dataclasses.asdict(optimizer_config),
        "reports": reports,
        "summary": {
            "traces": len(reports),
            "entry_cases": sum(len(report["cases"]) for report in reports),
            "member_rollouts": len(outcomes),
            "successful_member_rollouts": sum(
                bool(row["success"]) for row in outcomes
            ),
            "all_passed": bool(outcomes)
            and all(bool(row["success"]) for row in outcomes),
            "maximum_abs_terminal_right_m": max(
                (abs(float(row["terminal_right_m"])) for row in outcomes),
                default=None,
            ),
            "maximum_abs_terminal_right_rate_m_s": max(
                (abs(float(row["terminal_right_rate_m_s"])) for row in outcomes),
                default=None,
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-dt-s", type=float, default=0.10)
    args = parser.parse_args()
    report = evaluate(
        [path.resolve() for path in args.inputs],
        args.model.resolve(),
        control_dt_s=args.control_dt_s,
    )
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
