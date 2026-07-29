#!/usr/bin/env python3
"""Verify N199 weak-confidence Gate-3 latch separation on official traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import policy_callable_gate3_low_confidence_latched_recovery_n198 as policy
import policy_callable_gate3_zero_confidence_recovery_n196 as old_policy
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
    old = old_policy.Gate3ZeroConfidenceRecovery()
    new = policy.Gate3LowConfidenceLatchedRecovery()
    old_rows = []
    new_rows = []
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
        old_governed = np.asarray(old.apply(values, recorded), dtype=float)
        governed = np.asarray(new.apply(values, recorded), dtype=float)
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
        row = {
            "sample": sample_index,
            "elapsed_s": float(sample.get("elapsed_s") or 0.0),
            "forward_m": tail._inverse_tanh_norm(float(values[11]), 10.0),
            "closing_m_s": -tail._inverse_tanh_norm(float(values[0]), 5.0),
            "right_m": tail._inverse_tanh_norm(float(values[12]), 5.0),
            "down_m": tail._inverse_tanh_norm(float(values[13]), 5.0),
            "raw_confidence": float(values[policy.CONFIDENCE_FIELD_INDEX]),
            "recorded_action": recorded.tolist(),
            "governed_action": governed.tolist(),
        }
        if old.active:
            old_rows.append(row)
        if new.active:
            new_rows.append(row)
    old_snapshot = old.snapshot()
    new_snapshot = new.snapshot()
    camera_passes = (payload.get("gate_pass_summary") or {}).get("events") or []
    third_camera_pass = next(
        (event for event in camera_passes if event.get("pass_index") == 3), None
    )
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": payload.get("official_active_gate_index"),
        "third_camera_pass": third_camera_pass,
        "old_activation_count": old_snapshot["activation_count"],
        "old_active_samples": old_snapshot["active_samples"],
        "new_activation_count": new_snapshot["activation_count"],
        "new_active_samples": new_snapshot["active_samples"],
        "new_changed_samples": new_snapshot["changed_samples"],
        "changed_non_gate3": changed_non_gate3,
        "max_pitch_thrust_error": max_pitch_thrust_error,
        "invalid_outputs": invalid_outputs,
        "first_old_activation": old_rows[0] if old_rows else None,
        "first_new_activation": new_rows[0] if new_rows else None,
        "last_new_activation": new_rows[-1] if new_rows else None,
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

    expected_new = {
        "034": (305, 11.375),
        "053": (308, 11.031),
        "057": (309, 11.047),
        "061": (299, 11.344),
    }
    for candidate in (f"{value:03d}" for value in range(16, 62)):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
            continue
        expected = expected_new.get(candidate)
        if expected is None:
            if report["new_activation_count"] != 0:
                blockers.append(f"candidate{candidate}_unexpected_activation")
            continue
        expected_samples, expected_elapsed_s = expected
        if report["new_activation_count"] != 1:
            blockers.append(f"candidate{candidate}_activation_count_not_one")
        if report["new_active_samples"] != expected_samples:
            blockers.append(f"candidate{candidate}_active_sample_count_changed")
        first = report["first_new_activation"]
        if first is None or abs(first["elapsed_s"] - expected_elapsed_s) > 0.002:
            blockers.append(f"candidate{candidate}_activation_time_changed")

    for candidate in ("016", "018", "020", "021", "022", "055", "058", "060"):
        report = by_candidate.get(candidate)
        if report is not None and (
            report["new_activation_count"] != 0 or report["new_changed_samples"] != 0
        ):
            blockers.append(f"candidate{candidate}_protected_path_changed")

    candidate61 = by_candidate.get("061")
    if candidate61 is not None:
        old_first = candidate61["first_old_activation"]
        new_first = candidate61["first_new_activation"]
        camera_pass = candidate61["third_camera_pass"]
        if candidate61["old_activation_count"] != 13:
            blockers.append("candidate061_old_activation_count_changed")
        if candidate61["old_active_samples"] != 86:
            blockers.append("candidate061_old_active_sample_count_changed")
        if old_first is None or abs(old_first["elapsed_s"] - 15.266) > 0.002:
            blockers.append("candidate061_old_activation_time_changed")
        if camera_pass is None or abs(camera_pass["event_time_s"] - 10.453) > 0.002:
            blockers.append("candidate061_camera_pass_time_changed")
        if new_first is None or camera_pass is None or not (
            camera_pass["event_time_s"] < new_first["elapsed_s"] < old_first["elapsed_s"]
        ):
            blockers.append("candidate061_new_activation_not_between_pass_and_old")

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
            "race_start_boot_time_ms": race.get("race_start_boot_time_ms"),
            "race_finish_time_ns": race.get("race_finish_time_ns"),
            "collision_id": state.get("collision_id"),
        }
        if poststop["reset_sent"] is not False:
            blockers.append("candidate061_poststop_sent_reset")
        if poststop["race_finish_time_ns"] != -1 or poststop["collision_id"] is not None:
            blockers.append("candidate061_poststop_invalid_state")

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
            "base_policy": "exact N198",
            "scope": "official Gate 3 persistent weak raw confidence only",
            "parameters": policy.Gate3LowConfidenceLatchedRecovery().snapshot()["parameters"],
            "preserve_pitch_thrust": True,
            "activation_separation": (
                "only failed C034/C053/C057/C061 activate; all protected clean traces stay exact"
            ),
        },
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
        "n198_policy_sha256": _sha256(Path(policy.n198.__file__).resolve()),
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
