#!/usr/bin/env python3
"""Verify N190's stateless close Gate-2 vertical-margin floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

import policy_callable_gate2_vertical_margin_n189 as policy
import policy_callable_six_gate_composite as tail


EXPECTED_CHECKPOINT_SHA256 = (
    "ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760"
)
EXPECTED_PARAMETERS = {
    "max_forward_m": 6.0,
    "projected_down_trigger_m": -1.10,
    "thrust_floor_norm": 0.30,
    "intercept_speed_floor_m_s": 3.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    if match is None:
        raise ValueError(f"cannot parse candidate from {path}")
    return match.group(1)


def _observation(sample: dict) -> np.ndarray | None:
    raw = sample.get("observation") or []
    if len(raw) == 32:
        return np.asarray(raw, dtype=np.float32)
    if len(raw) != 23:
        return None
    gate = int(sample.get("official_active_gate_index", -1))
    values = np.zeros(32, dtype=np.float32)
    values[:23] = np.asarray(raw, dtype=np.float32)
    values[23] = np.float32(gate / 6.0)
    if 0 <= gate < 6:
        values[24 + gate] = np.float32(1.0)
    values[30] = np.float32(raw[17])
    values[31] = np.float32(raw[18])
    return values


def analyze(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    controller = policy.Gate2VerticalMarginFloor()
    rows: list[dict] = []
    changed_non_gate2 = 0
    max_non_thrust_error = 0.0
    stronger_thrust_changes = 0
    invalid_outputs = 0
    gate3_samples = 0
    for sample_index, sample in enumerate(
        sorted(
            (payload.get("policy_trace") or {}).get("samples") or [],
            key=lambda item: float(item.get("elapsed_s") or 0.0),
        )
    ):
        values = _observation(sample)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values is None or recorded.shape != (4,):
            continue
        gate = tail._gate_index(values)
        if gate == 2:
            gate3_samples += 1
        governed = np.asarray(controller.apply(values, recorded), dtype=float)
        changed = bool(np.max(np.abs(governed - recorded)) > 1e-7)
        if changed and gate != 1:
            changed_non_gate2 += 1
        if changed:
            max_non_thrust_error = max(
                max_non_thrust_error,
                max(abs(governed[index] - recorded[index]) for index in (0, 1, 3)),
            )
            if recorded[2] >= policy.THRUST_FLOOR_NORM:
                stronger_thrust_changes += 1
        if not np.all(np.isfinite(governed)) or np.max(np.abs(governed)) > 1.0:
            invalid_outputs += 1
        snapshot = controller.snapshot()
        if snapshot["active"]:
            forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
            rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample.get("elapsed_s") or 0.0),
                    "forward_m": forward_m,
                    "projected_down_m": snapshot["last_projected_down_m"],
                    "recorded_action": recorded.tolist(),
                    "governed_action": governed.tolist(),
                    "changed": changed,
                }
            )
    snapshot = controller.snapshot()
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "reported_official_active_gate_index": payload.get("official_active_gate_index"),
        "gate3_samples": gate3_samples,
        "active_samples": snapshot["active_samples"],
        "changed_samples": snapshot["changed_samples"],
        "changed_non_gate2": changed_non_gate2,
        "max_non_thrust_error": max_non_thrust_error,
        "stronger_thrust_changes": stronger_thrust_changes,
        "invalid_outputs": invalid_outputs,
        "rows": rows,
    }


def verify(paths: list[Path], checkpoint: Path, expected_policy_sha256: str) -> dict:
    reports = [analyze(path) for path in paths]
    blockers: list[str] = []
    policy_path = Path(policy.__file__).resolve()
    checkpoint_sha = _sha256(checkpoint)
    policy_sha = _sha256(policy_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        blockers.append("prefix_checkpoint_hash_mismatch")
    if policy_sha != expected_policy_sha256.lower():
        blockers.append("policy_callable_hash_mismatch")
    if policy.Gate2VerticalMarginFloor().snapshot()["parameters"] != EXPECTED_PARAMETERS:
        blockers.append("parameter_vector_mismatch")
    by_candidate = {report["candidate"]: report for report in reports}
    expected = {f"{value:03d}" for value in range(37, 50)}
    for candidate in sorted(expected - by_candidate.keys()):
        blockers.append(f"candidate{candidate}_missing")
    for candidate in ("047", "049"):
        report = by_candidate.get(candidate)
        if report and report["changed_samples"] != 2:
            blockers.append(f"candidate{candidate}_change_count_not_two")
    candidate048 = by_candidate.get("048")
    if candidate048 and candidate048["active_samples"] != 0:
        blockers.append("candidate048_path_not_preserved")
    for candidate in [f"{value:03d}" for value in range(37, 47)] + ["048"]:
        report = by_candidate.get(candidate)
        if report and report["gate3_samples"] == 0:
            blockers.append(f"candidate{candidate}_not_clean_gate2_transition")
    for report in reports:
        name = report["candidate"]
        if report["changed_non_gate2"]:
            blockers.append(f"candidate{name}_changed_non_gate2")
        if report["max_non_thrust_error"] != 0.0:
            blockers.append(f"candidate{name}_changed_non_thrust")
        if report["stronger_thrust_changes"]:
            blockers.append(f"candidate{name}_changed_stronger_thrust")
        if report["invalid_outputs"]:
            blockers.append(f"candidate{name}_invalid_output")

    return {
        "passed": not blockers,
        "blockers": blockers,
        "authorization": {
            "passive_zero_command_shadow": not blockers,
            "live_flight": False,
            "reason": "offline trace invariants authorize only a hash-pinned passive shadow",
        },
        "contract": {
            "mechanism": "stateless close Gate-2 projected vertical margin floor",
            "base_policy": "exact N189",
            "scope": "official Gate 2 thrust only",
            "parameters": EXPECTED_PARAMETERS,
            "preserve_pitch_roll_yaw_and_stronger_thrust": True,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "policy_callable": str(policy_path),
        "policy_callable_sha256": policy_sha,
        "n189_policy_sha256": _sha256(Path(policy.n189.__file__).resolve()),
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.reports, args.checkpoint, args.expected_policy_sha256)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
