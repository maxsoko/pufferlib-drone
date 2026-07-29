#!/usr/bin/env python3
"""Scan N239 retained associations for states reachable by its lateral model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import gate4_identity_body_mpc as body_mpc
import policy_callable_gate4_identity_body_mpc_n239 as n239


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan(path: Path, ensemble: body_mpc.LateralDynamicsEnsemble) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    controller = n239.Gate4IdentityBodyMPCController(ensemble)
    states = []
    previous_accepted = 0
    for sample in payload.get("policy_trace", {}).get("samples", []):
        if int(sample.get("official_active_gate_index", -1)) != n239.n238.GATE4_INDEX:
            continue
        controller.apply(
            np.asarray(sample["observation"], dtype=np.float64),
            [0.0, 0.0, 0.0, 0.0],
            now_s=float(sample["elapsed_s"]),
        )
        if controller.accepted_fresh == previous_accepted:
            continue
        previous_accepted = controller.accepted_fresh
        assert controller.associated_pose is not None
        forward_m, right_m = (
            float(controller.associated_pose[index]) for index in (0, 1)
        )
        forward_rate_m_s, right_rate_m_s = (
            float(controller.associated_rate_m_s[index]) for index in (0, 1)
        )
        plan = controller.optimizer.plan(
            forward_m=forward_m,
            forward_rate_m_s=forward_rate_m_s,
            right_m=right_m,
            right_rate_m_s=right_rate_m_s,
        )
        states.append(
            {
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": forward_m,
                "forward_rate_m_s": forward_rate_m_s,
                "right_m": right_m,
                "right_rate_m_s": right_rate_m_s,
                "down_m": float(controller.associated_pose[2]),
                "down_rate_m_s": float(controller.associated_rate_m_s[2]),
                "plan_accepted": plan.accepted,
                "roll_norm": plan.roll_norm,
                "horizon_s": plan.horizon_s,
                "worst_abs_terminal_right_m": plan.worst_abs_terminal_right_m,
            }
        )
    near = [state for state in states if state["forward_m"] <= 16.001]
    feasible = [state for state in near if state["plan_accepted"]]
    return {
        "trace": str(path.resolve()),
        "trace_sha256": _sha256(path),
        "accepted_states": len(states),
        "near_states": len(near),
        "feasible_near_states": len(feasible),
        "minimum_worst_abs_terminal_right_m": min(
            (state["worst_abs_terminal_right_m"] for state in near),
            default=None,
        ),
        "first_feasible_near_state": None if not feasible else feasible[0],
        "near_state_scan": near,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = args.model.resolve()
    os.environ[n239.MODEL_ENVIRONMENT_VARIABLE] = str(model)
    ensemble = body_mpc.LateralDynamicsEnsemble.load(model)
    runs = [_scan(path.resolve(), ensemble) for path in args.inputs]
    result = {
        "schema": "gate4_near_reachability_n240_v1",
        "model": str(model),
        "model_sha256": _sha256(model),
        "contract": {
            "retained_legal_observations_only": True,
            "accepted_association_states_only": True,
            "plant_counterfactual_claim": False,
            "runtime_privileged_state": False,
        },
        "runs": runs,
        "all_runs_have_feasible_near_state": all(
            run["feasible_near_states"] > 0 for run in runs
        ),
    }
    args.output.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "all_runs_have_feasible_near_state": result[
                    "all_runs_have_feasible_near_state"
                ],
                "runs": [
                    {
                        key: run[key]
                        for key in (
                            "near_states",
                            "feasible_near_states",
                            "minimum_worst_abs_terminal_right_m",
                        )
                    }
                    for run in runs
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
