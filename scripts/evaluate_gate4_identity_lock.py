#!/usr/bin/env python3
"""Replay retained official traces through N238's target-family contract."""

from __future__ import annotations

import argparse
import copy
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


def evaluate(trace: Path) -> dict:
    payload = json.loads(trace.read_text(encoding="utf-8-sig"))
    samples = list(payload.get("policy_trace", {}).get("samples", []))
    if not samples:
        raise RuntimeError("trace contains no retained policy samples")

    controller = n238.Gate4IdentityLockedInterceptController()
    maximum_abs_action = 0.0
    maximum_rejected_alias_action_error = 0.0
    rejected_alias_parity_checks = 0
    accepted_events: list[dict] = []
    prior_accepted = 0
    prior_rejected = 0
    prior_forward: float | None = None
    maximum_forward_regression_m = 0.0
    for sample in samples:
        values = np.asarray(sample["observation"], dtype=np.float64)
        elapsed_s = float(sample["elapsed_s"])
        base_action = sample.get("normalized_action", [0.0, 0.0, 0.0, 0.0])

        before = copy.deepcopy(controller)
        action = controller.apply(values, base_action, now_s=elapsed_s)
        maximum_abs_action = max(
            maximum_abs_action, *(abs(float(value)) for value in action)
        )

        rejected_now = controller.rejected_aliases > prior_rejected
        accepted_now = controller.accepted_fresh > prior_accepted
        if rejected_now and not accepted_now:
            missing_values = values.copy()
            missing_values[10:15] = 0.0
            missing_controller = copy.deepcopy(before)
            missing_action = missing_controller.apply(
                missing_values, base_action, now_s=elapsed_s
            )
            error = max(
                abs(float(left) - float(right))
                for left, right in zip(action, missing_action, strict=True)
            )
            maximum_rejected_alias_action_error = max(
                maximum_rejected_alias_action_error, error
            )
            rejected_alias_parity_checks += 1

        if accepted_now:
            pose = controller.associated_pose
            assert pose is not None
            forward = float(pose[0])
            if prior_forward is not None:
                maximum_forward_regression_m = max(
                    maximum_forward_regression_m, forward - prior_forward
                )
            prior_forward = forward
            accepted_events.append(
                {
                    "elapsed_s": elapsed_s,
                    "pose_body_ned_m": pose.tolist(),
                    "rate_body_ned_m_s": controller.associated_rate_m_s.tolist(),
                    "action": [float(value) for value in action],
                    "reacquisitions": controller.closer_reacquisitions,
                }
            )
        prior_accepted = controller.accepted_fresh
        prior_rejected = controller.rejected_aliases

    snapshot = controller.snapshot()
    blockers: list[str] = []
    if snapshot["initial_acquisitions"] != 1:
        blockers.append("expected exactly one initial target-family acquisition")
    if snapshot["accepted_fresh"] < 3:
        blockers.append("insufficient coherent associated measurements")
    if snapshot["rejected_aliases"] < 1:
        blockers.append("trace did not exercise alias rejection")
    if rejected_alias_parity_checks < 1:
        blockers.append("trace did not prove rejected-alias action parity")
    if maximum_rejected_alias_action_error > 1e-12:
        blockers.append("a rejected alias changed the commanded action")
    if maximum_forward_regression_m > n238.MAXIMUM_FORWARD_REGRESSION_M + 1e-9:
        blockers.append("associated family exceeded forward-regression bound")
    if maximum_abs_action > 1.0 + 1e-12:
        blockers.append("action exceeded normalized bounds")

    return {
        "schema": "n238_gate4_identity_lock_trace_eval_v1",
        "trace": str(trace.resolve()),
        "passed": not blockers,
        "blockers": blockers,
        "retained_samples": len(samples),
        "maximum_abs_action": maximum_abs_action,
        "rejected_alias_parity_checks": rejected_alias_parity_checks,
        "maximum_rejected_alias_action_error": (
            maximum_rejected_alias_action_error
        ),
        "maximum_forward_regression_m": maximum_forward_regression_m,
        "accepted_events": accepted_events,
        "controller_snapshot": snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.trace)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
