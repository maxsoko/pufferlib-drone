#!/usr/bin/env python3
"""Verify N196's later Gate-3 terminal-level admission on official traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import policy_callable_gate3_later_terminal_level_n195 as policy
import policy_callable_gate3_terminal_level_n191 as old_policy
import policy_callable_six_gate_composite as tail


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
    old = old_policy.Gate3TerminalSafeLevel()
    new = policy.Gate3LaterTerminalSafeLevel()
    rows = []
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
        old_before = old.snapshot()["active"]
        new_before = new.snapshot()["active"]
        old_actions = np.asarray(old.apply(values, recorded), dtype=float)
        new_actions = np.asarray(new.apply(values, recorded), dtype=float)
        if not np.all(np.isfinite(new_actions)) or np.max(np.abs(new_actions)) > 1.0:
            invalid_outputs += 1
        if np.max(np.abs(new_actions[[0, 2, 3]] - recorded[[0, 2, 3]])) > 0.0:
            non_roll_changes += 1
        if tail._gate_index(values) != 2:
            continue
        old_after = old.snapshot()["active"]
        new_after = new.snapshot()["active"]
        if old_before or old_after or new_before or new_after:
            rows.append({
                "sample": sample_index,
                "elapsed_s": float(sample.get("elapsed_s") or 0.0),
                **_projection(values),
                "old_active": bool(old_after),
                "new_active": bool(new_after),
                "retired_early_admission": bool(old_after and not new_after),
                "recorded_roll": float(recorded[1]),
                "old_roll": float(old_actions[1]),
                "new_roll": float(new_actions[1]),
            })
    old_snapshot = old.snapshot()
    new_snapshot = new.snapshot()
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": payload.get("official_active_gate_index"),
        "old_active_samples": old_snapshot["active_samples"],
        "new_active_samples": new_snapshot["active_samples"],
        "old_changed_samples": old_snapshot["changed_samples"],
        "new_changed_samples": new_snapshot["changed_samples"],
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

    for candidate in ("016", "018", "020", "021", "022", "045", "048"):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif report["new_changed_samples"] != 0:
            blockers.append(f"candidate{candidate}_protected_path_changed")

    candidate51 = by_candidate.get("051")
    if candidate51 is None:
        blockers.append("candidate051_missing")
    else:
        if candidate51["new_active_samples"] != 3:
            blockers.append("candidate051_active_count_not_three")
        if candidate51["new_changed_samples"] != 3:
            blockers.append("candidate051_change_count_not_three")
        first = candidate51["first_new_activation"]
        if first is None or not (3.5 <= first["forward_m"] <= policy.MAX_FORWARD_M):
            blockers.append("candidate051_activation_range_changed")

    candidate55 = by_candidate.get("055")
    if candidate55 is None:
        blockers.append("candidate055_missing")
    elif candidate55["new_active_samples"] != 0:
        blockers.append("candidate055_clean_path_activated")

    candidate56 = by_candidate.get("056")
    if candidate56 is None:
        blockers.append("candidate056_missing")
    else:
        if candidate56["old_active_samples"] != 5:
            blockers.append("candidate056_old_active_count_changed")
        if candidate56["new_active_samples"] >= candidate56["old_active_samples"]:
            blockers.append("candidate056_early_admission_not_reduced")
        if candidate56["retired_early_samples"] < 1:
            blockers.append("candidate056_no_early_samples_retired")
        first = candidate56["first_new_activation"]
        if first is not None and first["forward_m"] > policy.MAX_FORWARD_M:
            blockers.append("candidate056_new_activation_too_early")

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
            "base_policy": "exact N195 composition rebuilt from pinned components",
            "scope": "official Gate 3 roll only",
            "old_max_forward_m": old_policy.base.GATE3_TERMINAL_LEVEL_FORWARD_M,
            "new_parameters": policy.Gate3LaterTerminalSafeLevel().snapshot()["parameters"],
            "preserve_pitch_thrust_yaw": True,
            "latch_until_official_transition": True,
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
