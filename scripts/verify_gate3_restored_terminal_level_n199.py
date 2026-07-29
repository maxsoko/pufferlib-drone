#!/usr/bin/env python3
"""Verify N200 restored 3.7 m Gate-3 terminal leveling separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import policy_callable_gate3_final_terminal_level_n197 as old_policy
import policy_callable_gate3_restored_terminal_level_n199 as policy
import policy_callable_six_gate_composite as tail


ARCHIVED_CANDIDATE59_COLLISION_ELAPSED_S = 9.77643871307373


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt", path.name)
    if match is None:
        raise ValueError(f"cannot identify candidate from {path}")
    return match.group(1)


def _projection(values: np.ndarray) -> dict:
    controller = policy.n196_module
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
    horizon_s = (
        max(forward_m - controller.TERMINAL_PLANE_FORWARD_M, 0.0) / closing_m_s
        if closing_m_s >= controller.MINIMUM_CLOSING_M_S
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
    old = old_policy.Gate3FinalTerminalSafeLevel()
    restored = policy.n196_module.Gate3LaterTerminalSafeLevel()
    rows = []
    changed_samples = 0
    invalid_outputs = 0
    non_roll_changes = 0
    for sample_index, sample in enumerate(sorted(
        (payload.get("policy_trace") or {}).get("samples") or [],
        key=lambda item: float(item.get("elapsed_s") or 0.0),
    )):
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            continue
        old_actions = np.asarray(old.apply(values, recorded), dtype=float)
        restored_actions = np.asarray(restored.apply(values, recorded), dtype=float)
        if not np.all(np.isfinite(restored_actions)) or np.max(np.abs(restored_actions)) > 1.0:
            invalid_outputs += 1
        if np.max(np.abs(restored_actions[[0, 2, 3]] - recorded[[0, 2, 3]])) > 0.0:
            non_roll_changes += 1
        if np.max(np.abs(restored_actions - recorded)) > 1.0e-7:
            changed_samples += 1
        if tail._gate_index(values) == 2 and restored.active:
            rows.append({
                "sample": sample_index,
                "elapsed_s": float(sample.get("elapsed_s") or 0.0),
                **_projection(values),
                "old_2m_active": bool(old.active),
                "recorded_roll": float(recorded[1]),
                "restored_roll": float(restored_actions[1]),
            })
    old_snapshot = old.snapshot()
    restored_snapshot = restored.snapshot()
    third_camera_pass = next(
        (
            event for event in (payload.get("gate_pass_summary") or {}).get("events") or []
            if event.get("pass_index") == 3
        ),
        None,
    )
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": payload.get("official_active_gate_index"),
        "third_camera_pass": third_camera_pass,
        "old_2m_active_samples": old_snapshot["active_samples"],
        "restored_active_samples": restored_snapshot["active_samples"],
        "restored_changed_samples": restored_snapshot["changed_samples"],
        "recorded_action_changed_samples": changed_samples,
        "first_restored_activation": rows[0] if rows else None,
        "invalid_outputs": invalid_outputs,
        "non_roll_changes": non_roll_changes,
        "rows": rows,
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

    expected_restored_active = {
        "019": 2, "021": 3, "024": 2, "025": 3, "026": 3,
        "027": 2, "028": 2, "030": 2, "031": 3, "037": 2,
        "038": 2, "039": 3, "040": 3, "041": 3, "042": 2,
        "043": 2, "044": 3, "045": 2, "048": 2, "051": 3,
        "056": 2, "059": 2, "060": 1, "061": 308, "062": 2,
    }
    for candidate in (f"{value:03d}" for value in range(16, 63)):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
            continue
        expected = expected_restored_active.get(candidate, 0)
        if report["restored_active_samples"] != expected:
            blockers.append(f"candidate{candidate}_restored_active_count_changed")

    for candidate in ("016", "018", "020", "021", "022", "055", "058", "060"):
        report = by_candidate.get(candidate)
        if report is not None and report["recorded_action_changed_samples"] != 0:
            blockers.append(f"candidate{candidate}_protected_path_changed")

    candidate59 = by_candidate.get("059")
    if candidate59 is not None:
        first = candidate59["first_restored_activation"]
        if first is None or first["elapsed_s"] <= ARCHIVED_CANDIDATE59_COLLISION_ELAPSED_S:
            blockers.append("candidate059_activation_not_after_collision")

    candidate61 = by_candidate.get("061")
    if candidate61 is not None:
        first = candidate61["first_restored_activation"]
        camera_pass = candidate61["third_camera_pass"]
        if first is None or abs(first["elapsed_s"] - 10.297) > 0.002:
            blockers.append("candidate061_restored_activation_time_changed")
        if camera_pass is None or first is None or first["elapsed_s"] >= camera_pass["event_time_s"]:
            blockers.append("candidate061_restored_activation_not_before_camera_pass")

    candidate62 = by_candidate.get("062")
    if candidate62 is not None:
        first = candidate62["first_restored_activation"]
        camera_pass = candidate62["third_camera_pass"]
        if first is None or abs(first["elapsed_s"] - 10.422) > 0.002:
            blockers.append("candidate062_restored_activation_time_changed")
        if camera_pass is None or first is None or first["elapsed_s"] <= camera_pass["event_time_s"]:
            blockers.append("candidate062_restored_activation_not_after_camera_pass")
        if candidate62["recorded_action_changed_samples"] != 2:
            blockers.append("candidate062_restored_change_count_not_two")

    for report in reports:
        if report["invalid_outputs"]:
            blockers.append(f"candidate{report['candidate']}_invalid_output")
        if report["non_roll_changes"]:
            blockers.append(f"candidate{report['candidate']}_non_roll_change")

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
            blockers.append("candidate062_poststop_sent_reset")
        if poststop["race_finish_time_ns"] != -1 or poststop["collision_id"] is not None:
            blockers.append("candidate062_poststop_invalid_state")

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
            "base_policy": "exact N199",
            "scope": "official Gate 3 roll only",
            "old_max_forward_m": old_policy.MAX_FORWARD_M,
            "restored_parameters": policy.n196_module.Gate3LaterTerminalSafeLevel().snapshot()["parameters"],
            "preserve_pitch_thrust_yaw": True,
            "corrected_causality": {
                "candidate059_collision_elapsed_s": ARCHIVED_CANDIDATE59_COLLISION_ELAPSED_S,
                "candidate059_first_restored_activation_elapsed_s": (
                    None if by_candidate.get("059") is None
                    or by_candidate["059"]["first_restored_activation"] is None
                    else by_candidate["059"]["first_restored_activation"]["elapsed_s"]
                ),
            },
        },
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
        "n199_policy_sha256": _sha256(Path(policy.n199.__file__).resolve()),
        "terminal_controller_source_sha256": _sha256(Path(policy.n196_module.__file__).resolve()),
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
