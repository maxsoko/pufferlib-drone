#!/usr/bin/env python3
"""Verify N194's isolated frozen-observation Gate-3 recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

import policy_callable_gate3_frozen_observation_recovery_n193 as policy
import policy_callable_six_gate_composite as tail


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt", path.name)
    if match is None:
        raise ValueError(f"cannot identify candidate from {path}")
    return match.group(1)


def analyze(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    controller = policy.Gate3FrozenObservationRecovery()
    rows = []
    changed_non_gate3 = 0
    max_pitch_thrust_error = 0.0
    invalid_outputs = 0
    for sample_index, sample in enumerate(sorted(
        (payload.get("policy_trace") or {}).get("samples") or [],
        key=lambda item: float(item.get("elapsed_s") or 0.0),
    )):
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            continue
        governed = np.asarray(controller.apply(values, recorded), dtype=float)
        changed = bool(np.max(np.abs(governed - recorded)) > 1e-7)
        gate = tail._gate_index(values)
        if changed and gate != 2:
            changed_non_gate3 += 1
        if changed:
            max_pitch_thrust_error = max(
                max_pitch_thrust_error,
                abs(governed[0] - recorded[0]),
                abs(governed[2] - recorded[2]),
            )
        if not np.all(np.isfinite(governed)) or np.max(np.abs(governed)) > 1.0:
            invalid_outputs += 1
        if controller.active:
            rows.append({
                "sample": sample_index,
                "elapsed_s": float(sample.get("elapsed_s") or 0.0),
                "frozen_samples": controller.frozen_samples,
                "forward_m": tail._inverse_tanh_norm(float(values[11]), 10.0),
                "right_m": tail._inverse_tanh_norm(float(values[12]), 5.0),
                "down_m": tail._inverse_tanh_norm(float(values[13]), 5.0),
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
            })
    snapshot = controller.snapshot()
    latest = ((payload.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": payload.get("official_active_gate_index"),
        "collision_id": latest.get("collision_id"),
        "activation_count": snapshot["activation_count"],
        "active_samples": snapshot["active_samples"],
        "changed_samples": snapshot["changed_samples"],
        "changed_non_gate3": changed_non_gate3,
        "max_pitch_thrust_error": max_pitch_thrust_error,
        "invalid_outputs": invalid_outputs,
        "first_activation": rows[0] if rows else None,
        "last_activation": rows[-1] if rows else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers = []
    expected_inactive = {"016", "018", "020", "021", "022"}
    expected_inactive.update(f"{value:03d}" for value in range(37, 53))
    for candidate in sorted(expected_inactive):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif report["activation_count"] != 0:
            blockers.append(f"candidate{candidate}_unexpected_activation")
    candidate53 = by_candidate.get("053")
    if candidate53 is None:
        blockers.append("candidate053_missing")
    else:
        if candidate53["activation_count"] != 1:
            blockers.append("candidate053_activation_count_not_one")
        # The detected frozen run is 271 samples long. The first four samples
        # establish the five-sample watchdog, so recovery is active for 267.
        if candidate53["active_samples"] != 267:
            blockers.append("candidate053_active_sample_count_not_267")
        if candidate53["changed_samples"] != candidate53["active_samples"]:
            blockers.append("candidate053_recovery_not_applied")
        first = candidate53["first_activation"]
        if first is None or not (15.5 <= first["elapsed_s"] <= 15.7):
            blockers.append("candidate053_activation_time_changed")
    for report in reports:
        if report["changed_non_gate3"]:
            blockers.append(f"candidate{report['candidate']}_changed_non_gate3")
        if report["max_pitch_thrust_error"] != 0.0:
            blockers.append(f"candidate{report['candidate']}_changed_pitch_thrust")
        if report["invalid_outputs"]:
            blockers.append(f"candidate{report['candidate']}_invalid_output")
    policy_path = Path(policy.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "authorization": {
            "passive_zero_command_shadow": not blockers,
            "live_flight": False,
            "reason": "offline separation authorizes only a pinned passive shadow",
        },
        "contract": {
            "base_policy": "exact N193",
            "scope": "official Gate 3 frozen near-plane observations",
            "parameters": policy.Gate3FrozenObservationRecovery().snapshot()["parameters"],
            "preserve_pitch_thrust": True,
            "activation_separation": "C053 activates once; protected traces and C037-C052 remain inactive",
        },
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
        "n193_policy_sha256": _sha256(Path(policy.n193.__file__).resolve()),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
