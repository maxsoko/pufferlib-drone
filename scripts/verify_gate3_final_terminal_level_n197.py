#!/usr/bin/env python3
"""Verify N198's final-only Gate-3 terminal leveling on official traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import policy_callable_gate3_final_terminal_level_n197 as policy
import policy_callable_gate3_later_terminal_level_n195 as old_policy
import policy_callable_six_gate_composite as tail


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt", path.name)
    if match is None:
        raise ValueError(f"cannot identify candidate from {path}")
    return match.group(1)


def _projection(values: np.ndarray, terminal_plane_forward_m: float) -> dict:
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
    horizon_s = (
        max(forward_m - terminal_plane_forward_m, 0.0) / closing_m_s
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
        "right_m": right_m,
        "down_m": down_m,
        "projected_right_m": right_m + right_rate_m_s * horizon_s,
        "projected_down_m": down_m + down_rate_m_s * horizon_s,
    }


def analyze(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    old = old_policy.Gate3LaterTerminalSafeLevel()
    new = policy.Gate3FinalTerminalSafeLevel()
    rows = []
    invalid_outputs = 0
    non_roll_changes = 0
    changed_samples = 0
    for sample_index, sample in enumerate(sorted(
        (payload.get("policy_trace") or {}).get("samples") or [],
        key=lambda item: float(item.get("elapsed_s") or 0.0),
    )):
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            continue
        old_actions = np.asarray(old.apply(values, recorded), dtype=float)
        new_actions = np.asarray(new.apply(values, recorded), dtype=float)
        if not np.all(np.isfinite(new_actions)) or np.max(np.abs(new_actions)) > 1.0:
            invalid_outputs += 1
        if np.max(np.abs(new_actions[[0, 2, 3]] - recorded[[0, 2, 3]])) > 0.0:
            non_roll_changes += 1
        if np.max(np.abs(new_actions - recorded)) > 1.0e-7:
            changed_samples += 1
        if tail._gate_index(values) != 2:
            continue
        if old.active or new.active:
            rows.append({
                "sample": sample_index,
                "elapsed_s": float(sample.get("elapsed_s") or 0.0),
                **_projection(values, policy.TERMINAL_PLANE_FORWARD_M),
                "old_active": bool(old.active),
                "new_active": bool(new.active),
                "retired_early_admission": bool(old.active and not new.active),
                "recorded_roll": float(recorded[1]),
                "old_roll": float(old_actions[1]),
                "new_roll": float(new_actions[1]),
            })
    old_snapshot = old.snapshot()
    new_snapshot = new.snapshot()
    latest = ((payload.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": payload.get("official_active_gate_index"),
        "collision_id": latest.get("collision_id"),
        "old_active_samples": old_snapshot["active_samples"],
        "new_active_samples": new_snapshot["active_samples"],
        "old_changed_samples": old_snapshot["changed_samples"],
        "new_changed_samples": new_snapshot["changed_samples"],
        "recorded_action_changed_samples": changed_samples,
        "retired_early_samples": sum(row["retired_early_admission"] for row in rows),
        "first_old_activation": next((row for row in rows if row["old_active"]), None),
        "first_new_activation": next((row for row in rows if row["new_active"]), None),
        "invalid_outputs": invalid_outputs,
        "non_roll_changes": non_roll_changes,
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

    expected_new_active = {
        "021": 3,
        "025": 2,
        "031": 2,
        "048": 1,
    }
    for candidate in (f"{value:03d}" for value in range(16, 60)):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
            continue
        expected = expected_new_active.get(candidate, 0)
        if report["new_active_samples"] != expected:
            blockers.append(f"candidate{candidate}_new_active_count_changed")
        if report["new_active_samples"] > report["old_active_samples"]:
            blockers.append(f"candidate{candidate}_new_not_subset_of_old")

    for candidate in ("016", "018", "020", "021", "022", "055", "058"):
        report = by_candidate.get(candidate)
        if report is not None and report["recorded_action_changed_samples"] != 0:
            blockers.append(f"candidate{candidate}_protected_path_changed")

    candidate21 = by_candidate.get("021")
    if candidate21 is not None:
        first = candidate21["first_new_activation"]
        if first is None or abs(first["forward_m"] - 1.771) > 0.01:
            blockers.append("candidate021_clean_activation_distance_changed")

    for candidate, expected_old in (("051", 3), ("056", 2), ("059", 2)):
        report = by_candidate.get(candidate)
        if report is None:
            continue
        if report["old_active_samples"] != expected_old:
            blockers.append(f"candidate{candidate}_old_active_count_changed")
        if report["new_active_samples"] != 0:
            blockers.append(f"candidate{candidate}_early_admission_not_retired")
        if report["retired_early_samples"] != expected_old:
            blockers.append(f"candidate{candidate}_retired_count_changed")

    for candidate in ("055", "058"):
        report = by_candidate.get(candidate)
        if report is not None and (
            report["old_active_samples"] != 0 or report["new_active_samples"] != 0
        ):
            blockers.append(f"candidate{candidate}_clean_zero_activation_changed")

    for report in reports:
        if report["invalid_outputs"]:
            blockers.append(f"candidate{report['candidate']}_invalid_output")
        if report["non_roll_changes"]:
            blockers.append(f"candidate{report['candidate']}_non_roll_change")

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
            "base_policy": "exact N197 composition rebuilt from pinned components",
            "scope": "official Gate 3 roll only",
            "old_max_forward_m": old_policy.MAX_FORWARD_M,
            "new_parameters": policy.Gate3FinalTerminalSafeLevel().snapshot()["parameters"],
            "preserve_pitch_thrust_yaw": True,
            "latch_until_official_transition": True,
            "protected_clean_candidates": ["016", "018", "020", "021", "022", "055", "058"],
        },
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
        "source_hashes": {
            name: _sha256(Path(module.__file__).resolve())
            for name, (module, _expected) in policy.EXPECTED_SOURCE_SHA256.items()
        },
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
