#!/usr/bin/env python3
"""Verify N201 Gate-1 high-down thrust-ceiling trace separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import policy_callable_gate1_high_down_thrust_ceiling_n200 as policy
import policy_callable_six_gate_composite as tail


EXPECTED_ACTIVATIONS = {"026": 6, "036": 1, "063": 5}
PROTECTED_CANDIDATES = {"055", "058", "060"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt", path.name)
    if match is None:
        raise ValueError(f"cannot identify candidate from {path}")
    return match.group(1)


def analyze(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    controller = policy.Gate1HighDownThrustCeiling()
    rows = []
    changed_samples = 0
    non_thrust_changes = 0
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
        if changed:
            changed_samples += 1
            if np.max(np.abs(governed[[0, 1, 3]] - recorded[[0, 1, 3]])) > 0.0:
                non_thrust_changes += 1
        if not np.all(np.isfinite(governed)) or np.max(np.abs(governed)) > 1.0:
            invalid_outputs += 1
        snapshot = controller.snapshot()
        if snapshot["active"]:
            rows.append({
                "sample": sample_index,
                "elapsed_s": float(sample.get("elapsed_s") or 0.0),
                "forward_m": snapshot["last_forward_m"],
                "down_m": snapshot["last_down_m"],
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
            })
    snapshot = controller.snapshot()
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "reported_official_active_gate_index": payload.get(
            "official_active_gate_index"
        ),
        "reported_crash_detected": payload.get("crash_detected"),
        "active_samples": snapshot["active_samples"],
        "changed_samples": snapshot["changed_samples"],
        "recorded_action_changed_samples": changed_samples,
        "non_thrust_changes": non_thrust_changes,
        "invalid_outputs": invalid_outputs,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--poststop", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers = []
    for candidate in (f"{value:03d}" for value in range(16, 64)):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
            continue
        expected = EXPECTED_ACTIVATIONS.get(candidate, 0)
        if report["active_samples"] != expected:
            blockers.append(f"candidate{candidate}_activation_count_changed")
    for candidate in PROTECTED_CANDIDATES:
        report = by_candidate.get(candidate)
        if report is not None and report["recorded_action_changed_samples"] != 0:
            blockers.append(f"candidate{candidate}_protected_path_changed")
    candidate063 = by_candidate.get("063")
    if candidate063 is not None:
        if candidate063["active_samples"] != 5:
            blockers.append("candidate063_activation_missing")
        if candidate063["changed_samples"] != 5:
            blockers.append("candidate063_thrust_ceiling_not_applied")
        first = candidate063["rows"][0] if candidate063["rows"] else None
        if first is None or abs(first["elapsed_s"] - 4.0) > 0.002:
            blockers.append("candidate063_first_activation_time_changed")
    for report in reports:
        if report["non_thrust_changes"]:
            blockers.append(f"candidate{report['candidate']}_non_thrust_change")
        if report["invalid_outputs"]:
            blockers.append(f"candidate{report['candidate']}_invalid_output")

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
        "last_gate_race_time": race.get("last_gate_race_time"),
        "race_finish_time_ns": race.get("race_finish_time_ns"),
        "collision_id": state.get("collision_id"),
    }
    if poststop["reset_sent"] is not False:
        blockers.append("candidate063_poststop_sent_reset")
    if poststop["active_gate_index"] != 1:
        blockers.append("candidate063_poststop_gate1_not_official")
    if poststop["race_finish_time_ns"] != -1:
        blockers.append("candidate063_poststop_unexpected_finish")

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
            "base_policy": "exact N200",
            "scope": "official Gate 1 thrust only",
            "parameters": policy.Gate1HighDownThrustCeiling().snapshot()[
                "parameters"
            ],
            "preserve_pitch_roll_yaw": True,
            "activation_separation": (
                "C063 activates from 4.000 s; protected C055/C058/C060 remain exact"
            ),
        },
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
        "n200_policy_sha256": _sha256(Path(policy.n200.__file__).resolve()),
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
