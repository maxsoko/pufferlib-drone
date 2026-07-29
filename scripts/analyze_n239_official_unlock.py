#!/usr/bin/env python3
"""Audit N239's near-family admission on retained official observations.

The live policy trace retains observations at 10 Hz while the policy runs at
60 Hz.  This diagnostic deliberately replays only those retained observations
at their recorded timestamps through N239's association/controller object.  It
is therefore an admission audit, not an action-parity claim or a plant replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analyze(trace_path: Path, model_path: Path) -> dict:
    import numpy as np

    import gate4_identity_body_mpc as body_mpc
    import policy_callable_gate4_identity_body_mpc_n239 as n239

    payload = json.loads(trace_path.read_text(encoding="utf-8-sig"))
    ensemble = body_mpc.LateralDynamicsEnsemble.load(model_path)
    controller = n239.Gate4IdentityBodyMPCController(ensemble)
    gate4_samples = [
        sample
        for sample in payload.get("policy_trace", {}).get("samples", [])
        if int(sample.get("official_active_gate_index", -1)) == n239.n238.GATE4_INDEX
    ]
    if not gate4_samples:
        raise RuntimeError(f"trace has no retained Gate-4 observations: {trace_path}")

    accepted_forward_m: list[float] = []
    first_unlock = None
    previous_unlock = False
    nonzero_roll_calls = 0
    for sample in gate4_samples:
        action = controller.apply(
            np.asarray(sample["observation"], dtype=np.float64),
            [0.0, 0.0, 0.0, 0.0],
            now_s=float(sample["elapsed_s"]),
        )
        if controller.associated_pose is not None:
            accepted_forward_m.append(float(controller.associated_pose[0]))
        if abs(float(action[1])) > 1e-9:
            nonzero_roll_calls += 1
        if controller.trusted_near_family and not previous_unlock:
            first_unlock = {
                "elapsed_s": float(sample["elapsed_s"]),
                "associated_pose_body_ned_m": controller.associated_pose.tolist(),
                "associated_rate_body_ned_m_s": (
                    controller.associated_rate_m_s.tolist()
                ),
                "action": [float(value) for value in action],
            }
        previous_unlock = controller.trusted_near_family

    snapshot = controller.snapshot()
    lateral = snapshot["identity_body_lateral_mpc"]
    return {
        "trace": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "retained_gate4_samples": len(gate4_samples),
        "accepted_association_samples": snapshot["accepted_fresh"],
        "initial_acquisitions": snapshot["initial_acquisitions"],
        "closer_reacquisitions": snapshot["closer_reacquisitions"],
        "rejected_aliases": snapshot["rejected_aliases"],
        "minimum_accepted_forward_m": (
            None if not accepted_forward_m else min(accepted_forward_m)
        ),
        "maximum_accepted_forward_m": (
            None if not accepted_forward_m else max(accepted_forward_m)
        ),
        "trusted_near_family": lateral["trusted_near_family"],
        "trusted_family_triggers": lateral["trusted_family_triggers"],
        "optimized_lateral_plans": lateral["optimized_lateral_plans"],
        "quarantined_associated_calls": lateral["quarantined_associated_calls"],
        "nonzero_roll_calls": nonzero_roll_calls,
        "first_unlock": first_unlock,
        "final_associated_pose_body_ned_m": snapshot[
            "associated_pose_body_ned_m"
        ],
        "final_associated_rate_body_ned_m_s": snapshot[
            "associated_rate_body_ned_m_s"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model.resolve()
    os.environ[
        "PUFFER_POLICY_GATE4_IDENTITY_DYNAMICS_PATH"
    ] = str(model_path)
    result = {
        "schema": "n239_official_near_family_admission_audit_v1",
        "contract": {
            "retained_observation_rate_hz": 10.0,
            "association_admission_audit_only": True,
            "action_parity_claim": False,
            "plant_counterfactual_claim": False,
            "runtime_privileged_state": False,
        },
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "runs": [_analyze(path.resolve(), model_path) for path in args.trace],
    }
    # The pose encoder/decoder rounds the nominal 16 m boundary by a few
    # micrometres, so report a diagnostic boundary tolerance explicitly.  This
    # does not change the deployed policy's exact comparison.
    result["threshold_forward_m"] = 16.0
    result["diagnostic_boundary_tolerance_m"] = 1e-3
    result["all_runs_reach_threshold_boundary_without_unlock"] = all(
        run["minimum_accepted_forward_m"] is not None
        and run["minimum_accepted_forward_m"]
        <= result["threshold_forward_m"]
        + result["diagnostic_boundary_tolerance_m"]
        and run["trusted_family_triggers"] == 0
        and run["optimized_lateral_plans"] == 0
        for run in result["runs"]
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
