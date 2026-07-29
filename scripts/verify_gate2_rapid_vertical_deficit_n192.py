#!/usr/bin/env python3
"""Verify N193's isolated Gate-2 rapid vertical-deficit latch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import policy_callable_gate2_rapid_vertical_deficit_n192 as policy
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
    controller = policy.Gate2RapidVerticalDeficit()
    rows = []
    changed_non_gate2 = 0
    max_non_thrust_error = 0.0
    invalid_outputs = 0
    for sample_index, sample in enumerate(sorted(
        (payload.get("policy_trace") or {}).get("samples") or [],
        key=lambda item: float(item.get("elapsed_s") or 0.0),
    )):
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            continue
        active_before = controller.active
        governed = np.asarray(controller.apply(values, recorded), dtype=float)
        active_after = controller.active
        changed = bool(np.max(np.abs(governed - recorded)) > 1e-7)
        gate = tail._gate_index(values)
        if changed and gate != 1:
            changed_non_gate2 += 1
        if changed:
            max_non_thrust_error = max(
                max_non_thrust_error,
                max(abs(governed[index] - recorded[index]) for index in (0, 1, 3)),
            )
        if not np.all(np.isfinite(governed)) or np.max(np.abs(governed)) > 1.0:
            invalid_outputs += 1
        if active_before or active_after:
            forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
            down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
            rows.append({
                "sample": sample_index,
                "elapsed_s": float(sample.get("elapsed_s") or 0.0),
                "forward_m": forward_m,
                "down_rate_m_s": down_rate_m_s,
                "projected_down_m": controller.last_projected_down_m,
                "active_before": active_before,
                "active_after": active_after,
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
            })
    snapshot = controller.snapshot()
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": payload.get("official_active_gate_index"),
        "activation_count": snapshot["activation_count"],
        "active_samples": snapshot["active_samples"],
        "changed_samples": snapshot["changed_samples"],
        "changed_non_gate2": changed_non_gate2,
        "max_non_thrust_error": max_non_thrust_error,
        "invalid_outputs": invalid_outputs,
        "first_activation": rows[0] if rows else None,
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
    for candidate in (f"{value:03d}" for value in range(37, 52)):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif report["activation_count"] != 0:
            blockers.append(f"candidate{candidate}_unexpected_activation")
    candidate52 = by_candidate.get("052")
    if candidate52 is None:
        blockers.append("candidate052_missing")
    else:
        if candidate52["activation_count"] != 1:
            blockers.append("candidate052_activation_count_not_one")
        if candidate52["active_samples"] != 5:
            blockers.append("candidate052_active_sample_count_not_five")
        if candidate52["changed_samples"] != candidate52["active_samples"]:
            blockers.append("candidate052_full_thrust_not_applied")
        first = candidate52["first_activation"]
        if first is None or not (5.0 <= first["forward_m"] <= 6.0):
            blockers.append("candidate052_activation_range_changed")
    for report in reports:
        if report["changed_non_gate2"]:
            blockers.append(f"candidate{report['candidate']}_changed_non_gate2")
        if report["max_non_thrust_error"] != 0.0:
            blockers.append(f"candidate{report['candidate']}_changed_non_thrust")
        if report["invalid_outputs"]:
            blockers.append(f"candidate{report['candidate']}_invalid_output")
    policy_path = Path(policy.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "authorization": {
            "passive_zero_command_shadow": not blockers,
            "live_flight": False,
            "reason": "offline trace separation authorizes only a pinned passive shadow",
        },
        "contract": {
            "base_policy": "exact N192",
            "scope": "official Gate 2 thrust only",
            "parameters": policy.Gate2RapidVerticalDeficit().snapshot()["parameters"],
            "preserve_pitch_roll_yaw": True,
            "activation_separation": "C052 activates once; C037-C051 remain inactive",
        },
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
        "n192_policy_sha256": _sha256(Path(policy.n192.__file__).resolve()),
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
