#!/usr/bin/env python3
"""Verify N202 tighter restored Gate-3 terminal-level separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import policy_callable_gate3_restored_terminal_level_n199 as old_policy
import policy_callable_gate3_tight_terminal_level_n201 as policy
import policy_callable_six_gate_composite as tail


PROTECTED_CANDIDATES = {"055", "058", "060"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt", path.name)
    if match is None:
        raise ValueError(f"cannot identify candidate from {path}")
    return match.group(1)


def _projection(values: np.ndarray) -> dict:
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
    horizon_s = (
        max(forward_m - policy.TERMINAL_PLANE_FORWARD_M, 0.0) / closing_m_s
        if closing_m_s >= policy.MINIMUM_CLOSING_M_S
        else 0.0
    )
    right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
    down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
    right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
    down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
    return {
        "forward_m": forward_m,
        "closing_m_s": closing_m_s,
        "projected_right_m": right_m + right_rate_m_s * horizon_s,
        "projected_down_m": down_m + down_rate_m_s * horizon_s,
    }


def analyze(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    old = old_policy.n196_module.Gate3LaterTerminalSafeLevel()
    tight = policy.Gate3TightTerminalSafeLevel()
    rows = []
    changed_samples = 0
    non_roll_changes = 0
    invalid_outputs = 0
    tight_without_old = 0
    for sample_index, sample in enumerate(sorted(
        (payload.get("policy_trace") or {}).get("samples") or [],
        key=lambda item: float(item.get("elapsed_s") or 0.0),
    )):
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            continue
        old_actions = np.asarray(old.apply(values, recorded), dtype=float)
        tight_actions = np.asarray(tight.apply(values, recorded), dtype=float)
        if tight.active and not old.active:
            tight_without_old += 1
        if np.max(np.abs(tight_actions - recorded)) > 1.0e-7:
            changed_samples += 1
        if np.max(np.abs(tight_actions[[0, 2, 3]] - recorded[[0, 2, 3]])) > 0.0:
            non_roll_changes += 1
        if not np.all(np.isfinite(tight_actions)) or np.max(np.abs(tight_actions)) > 1.0:
            invalid_outputs += 1
        if tail._gate_index(values) == 2 and (old.active or tight.active):
            rows.append({
                "sample": sample_index,
                "elapsed_s": float(sample.get("elapsed_s") or 0.0),
                **_projection(values),
                "old_active": bool(old.active),
                "tight_active": bool(tight.active),
                "previous_roll_norm": float(values[20]),
                "recorded_roll_norm": float(recorded[1]),
                "tight_roll_norm": float(tight_actions[1]),
            })
    old_snapshot = old.snapshot()
    tight_snapshot = tight.snapshot()
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "reported_official_active_gate_index": payload.get(
            "official_active_gate_index"
        ),
        "reported_crash_detected": payload.get("crash_detected"),
        "old_active_samples": old_snapshot["active_samples"],
        "tight_active_samples": tight_snapshot["active_samples"],
        "recorded_action_changed_samples": changed_samples,
        "non_roll_changes": non_roll_changes,
        "invalid_outputs": invalid_outputs,
        "tight_without_old": tight_without_old,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--poststop", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers = []
    for candidate in (f"{value:03d}" for value in range(16, 65)):
        if candidate not in by_candidate:
            blockers.append(f"candidate{candidate}_missing")
    for report in reports:
        if report["tight_without_old"]:
            blockers.append(f"candidate{report['candidate']}_tight_not_subset")
        if report["non_roll_changes"]:
            blockers.append(f"candidate{report['candidate']}_non_roll_change")
        if report["invalid_outputs"]:
            blockers.append(f"candidate{report['candidate']}_invalid_output")
    for candidate in PROTECTED_CANDIDATES:
        report = by_candidate.get(candidate)
        if report is not None and report["recorded_action_changed_samples"] != 0:
            blockers.append(f"candidate{candidate}_protected_path_changed")

    expected = {
        "061": (308, 308, 10.297),
        "062": (2, 2, 10.422),
        "064": (2, 0, None),
    }
    for candidate, (old_count, tight_count, first_time) in expected.items():
        report = by_candidate.get(candidate)
        if report is None:
            continue
        if report["old_active_samples"] != old_count:
            blockers.append(f"candidate{candidate}_old_count_changed")
        if report["tight_active_samples"] != tight_count:
            blockers.append(f"candidate{candidate}_tight_count_changed")
        active_rows = [row for row in report["rows"] if row["tight_active"]]
        if first_time is not None and (
            not active_rows or abs(active_rows[0]["elapsed_s"] - first_time) > 0.002
        ):
            blockers.append(f"candidate{candidate}_first_tight_activation_changed")
    candidate064 = by_candidate.get("064")
    if candidate064 is not None:
        old_only = [
            row for row in candidate064["rows"]
            if row["old_active"] and not row["tight_active"]
        ]
        if not old_only:
            blockers.append("candidate064_old_only_admission_missing")
        else:
            first = old_only[0]
            if abs(first["projected_right_m"] - (-0.3406284021673234)) > 1.0e-6:
                blockers.append("candidate064_projection_changed")

    attempt_payload = json.loads(args.attempt.read_text(encoding="utf-8"))
    smoke = attempt_payload.get("smoke") or {}
    fail_closed = {
        "path": str(args.attempt),
        "sha256": _sha256(args.attempt),
        "child_exit_code": attempt_payload.get("exit_code"),
        "status": attempt_payload.get("status"),
        "acceptance_passed": smoke.get("acceptance_passed"),
        "crash_detected": smoke.get("crash_detected"),
        "invalid_run": smoke.get("invalid_run"),
        "old_gate_progress_passed": attempt_payload.get("gate_progress_passed"),
        "corrected_gate_progress_passed": False,
    }
    if not (
        fail_closed["child_exit_code"] == 1
        and fail_closed["acceptance_passed"] is False
        and fail_closed["crash_detected"] is True
        and fail_closed["invalid_run"] is True
    ):
        blockers.append("candidate064_fail_closed_fixture_changed")

    poststop_payload = json.loads(args.poststop.read_text(encoding="utf-8"))
    state = poststop_payload.get("latest_telemetry") or {}
    poststop = {
        "path": str(args.poststop),
        "sha256": _sha256(args.poststop),
        "reset_sent": poststop_payload.get("reset_sent"),
        "base_mode": state.get("base_mode"),
        "system_status": state.get("system_status"),
        "collision_id": state.get("collision_id"),
    }
    if poststop["reset_sent"] is not False or poststop["collision_id"] != 1001:
        blockers.append("candidate064_poststop_fixture_changed")

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
            "base_policy": "exact N199 plus N201 Gate-1 ceiling",
            "scope": "restored early Gate-3 roll leveling only",
            "old_max_abs_projected_right_m": old_policy.n196_module.MAX_ABS_PROJECTED_RIGHT_M,
            "tight_parameters": policy.Gate3TightTerminalSafeLevel().snapshot()[
                "parameters"
            ],
            "preserve_pitch_thrust_yaw": True,
            "separation": (
                "C061/C062 admitted at +0.257/+0.093 m; C064 rejected at -0.341 m"
            ),
        },
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
        "n201_policy_sha256": _sha256(Path(policy.n201_module.__file__).resolve()),
        "validator_fail_closed_fixture": fail_closed,
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
