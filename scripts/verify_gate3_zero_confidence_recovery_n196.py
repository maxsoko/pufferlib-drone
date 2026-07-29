#!/usr/bin/env python3
"""Verify N197's persistent zero-confidence Gate-3 recovery separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import policy_callable_gate3_zero_confidence_recovery_n196 as policy
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
    controller = policy.Gate3ZeroConfidenceRecovery()
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
        changed = bool(np.max(np.abs(governed - recorded)) > 1.0e-7)
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
                "zero_confidence_samples": controller.zero_confidence_samples,
                "forward_m": tail._inverse_tanh_norm(float(values[11]), 10.0),
                "closing_m_s": -tail._inverse_tanh_norm(float(values[0]), 5.0),
                "right_m": tail._inverse_tanh_norm(float(values[12]), 5.0),
                "down_m": tail._inverse_tanh_norm(float(values[13]), 5.0),
                "raw_confidence": float(values[policy.CONFIDENCE_FIELD_INDEX]),
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--poststop", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers = []

    expected_active = {
        "034": (259, 16.437),
        "053": (272, 15.047),
        "057": (277, 14.750),
    }
    for candidate in (f"{value:03d}" for value in range(16, 58)):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
            continue
        if candidate not in expected_active and report["activation_count"] != 0:
            blockers.append(f"candidate{candidate}_unexpected_activation")
    for candidate, (expected_samples, expected_elapsed_s) in expected_active.items():
        report = by_candidate.get(candidate)
        if report is None:
            continue
        if report["activation_count"] != 1:
            blockers.append(f"candidate{candidate}_activation_count_not_one")
        if report["active_samples"] != expected_samples:
            blockers.append(f"candidate{candidate}_active_sample_count_changed")
        if report["changed_samples"] != report["active_samples"]:
            blockers.append(f"candidate{candidate}_recovery_not_applied")
        first = report["first_activation"]
        if first is None or abs(first["elapsed_s"] - expected_elapsed_s) > 0.002:
            blockers.append(f"candidate{candidate}_activation_time_changed")

    for report in reports:
        if report["changed_non_gate3"]:
            blockers.append(f"candidate{report['candidate']}_changed_non_gate3")
        if report["max_pitch_thrust_error"] != 0.0:
            blockers.append(f"candidate{report['candidate']}_changed_pitch_thrust")
        if report["invalid_outputs"]:
            blockers.append(f"candidate{report['candidate']}_invalid_output")

    poststop = None
    if args.poststop is not None:
        poststop_payload = json.loads(args.poststop.read_text(encoding="utf-8"))
        state = poststop_payload.get("latest_telemetry") or {}
        race = state.get("race_status") or {}
        poststop = {
            "path": str(args.poststop),
            "sha256": _sha256(args.poststop),
            "reset_sent": poststop_payload.get("reset_sent"),
            "base_mode": state.get("base_mode"),
            "system_status": state.get("system_status"),
            "active_gate_index": race.get("active_gate_index"),
            "race_finish_time_ns": race.get("race_finish_time_ns"),
            "collision_id": state.get("collision_id"),
        }
        if poststop["reset_sent"] is not False:
            blockers.append("candidate057_poststop_sent_reset")
        if poststop["active_gate_index"] != 2 or poststop["race_finish_time_ns"] != -1:
            blockers.append("candidate057_poststop_official_state_changed")
        if poststop["collision_id"] is not None:
            blockers.append("candidate057_poststop_collision_present")

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
            "base_policy": "exact N196",
            "scope": "official Gate 3 persistent zero raw confidence only",
            "parameters": policy.Gate3ZeroConfidenceRecovery().snapshot()["parameters"],
            "preserve_pitch_thrust": True,
            "activation_separation": (
                "C034/C053/C057 activate once; every other C016-C057 trace remains inactive"
            ),
        },
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
        "n196_policy_sha256": _sha256(Path(policy.n196.__file__).resolve()),
        "poststop": poststop,
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
