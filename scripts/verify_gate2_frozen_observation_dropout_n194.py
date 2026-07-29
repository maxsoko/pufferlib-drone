#!/usr/bin/env python3
"""Verify N195's isolated Gate-2 stale-visibility sanitizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import policy_callable_gate2_frozen_observation_dropout_n194 as policy
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
    sanitizer = policy.Gate2FrozenObservationDropout()
    rows = []
    changed_non_gate2 = 0
    changed_non_visibility = 0
    invalid_outputs = 0
    for sample_index, sample in enumerate(sorted(
        (payload.get("policy_trace") or {}).get("samples") or [],
        key=lambda item: float(item.get("elapsed_s") or 0.0),
    )):
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        if values.shape != (32,):
            continue
        sanitized = sanitizer.sanitize(values)
        changed = np.flatnonzero(np.abs(sanitized - values) > 1.0e-7).tolist()
        gate = tail._gate_index(values)
        if changed and gate != 1:
            changed_non_gate2 += 1
        if any(index != 10 for index in changed):
            changed_non_visibility += 1
        if not np.all(np.isfinite(sanitized)) or np.max(np.abs(sanitized)) > 1.0:
            invalid_outputs += 1
        if sanitizer.active:
            rows.append({
                "sample": sample_index,
                "elapsed_s": float(sample.get("elapsed_s") or 0.0),
                "frozen_samples": sanitizer.frozen_samples,
                "raw_gate_visible": float(values[10]),
                "sanitized_gate_visible": float(sanitized[10]),
            })
    snapshot = sanitizer.snapshot()
    latest = ((payload.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": payload.get("official_active_gate_index"),
        "collision_id": latest.get("collision_id"),
        "activation_count": snapshot["activation_count"],
        "active_samples": snapshot["active_samples"],
        "sanitized_samples": snapshot["sanitized_samples"],
        "changed_non_gate2": changed_non_gate2,
        "changed_non_visibility": changed_non_visibility,
        "invalid_outputs": invalid_outputs,
        "first_activation": rows[0] if rows else None,
        "last_activation": rows[-1] if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers = []
    for candidate in (f"{value:03d}" for value in range(37, 55)):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif candidate not in {"050", "054"} and report["activation_count"] != 0:
            blockers.append(f"candidate{candidate}_unexpected_activation")
    expected_active_samples = {"050": 303, "054": 306}
    for candidate, expected_samples in expected_active_samples.items():
        report = by_candidate.get(candidate)
        if report is None:
            continue
        if report["activation_count"] != 1:
            blockers.append(f"candidate{candidate}_activation_count_not_one")
        if report["active_samples"] != expected_samples:
            blockers.append(f"candidate{candidate}_active_sample_count_changed")
        if report["sanitized_samples"] != report["active_samples"]:
            blockers.append(f"candidate{candidate}_visibility_not_sanitized")
    for report in reports:
        if report["changed_non_gate2"]:
            blockers.append(f"candidate{report['candidate']}_changed_non_gate2")
        if report["changed_non_visibility"]:
            blockers.append(f"candidate{report['candidate']}_changed_non_visibility")
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
            "base_policy": "exact N194",
            "scope": "official Gate 2 stale visibility only",
            "parameters": policy.Gate2FrozenObservationDropout().snapshot()["parameters"],
            "activation_separation": (
                "C050/C054 activate once; C037-C049 and C051-C053 remain inactive"
            ),
        },
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
        "n194_policy_sha256": _sha256(Path(policy.n194.__file__).resolve()),
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
