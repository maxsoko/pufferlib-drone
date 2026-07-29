#!/usr/bin/env python3
"""Verify the N187 Gate-3 projected-path PD live-shadow contract."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

import policy_callable_gate3_projected_path_pd as policy
import policy_callable_six_gate_composite as tail


EXPECTED_CHECKPOINT_SHA256 = (
    "ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760"
)
EXPECTED_PARAMETERS = {
    "admission_forward_m": 12.0,
    "minimum_closing_m_s": 1.0,
    "path_power": 3.0,
    "terminal_right_m": -0.10,
    "position_gain": 0.20,
    "rate_gain": 1.0,
    "max_roll_norm": 1.0,
    "max_roll_slew_norm_s": 4.0,
    "nominal_control_hz": 60.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    return match.group(1) if match else path.stem


def analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = sorted(
        (report.get("policy_trace") or {}).get("samples") or [],
        key=lambda row: float(row.get("elapsed_s", 0.0)),
    )
    controller = policy.Gate3ProjectedPathPD()
    previous_elapsed: float | None = None
    rows: list[dict] = []
    max_non_roll_error = 0.0
    invalid_roll_outputs = 0
    changed_non_gate3 = 0
    changed_before_admission = 0
    max_controller_slew = 0.0
    activations = 0
    was_active = False
    for sample_index, sample in enumerate(samples):
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            continue
        elapsed = float(sample.get("elapsed_s", 0.0))
        dt_s = 1.0 / policy.PARAMETERS.nominal_control_hz
        if previous_elapsed is not None and elapsed > previous_elapsed:
            dt_s = elapsed - previous_elapsed
        previous_elapsed = elapsed
        governed = np.asarray(
            controller.apply(values, recorded.tolist(), dt_s=dt_s), dtype=float
        )
        snapshot = controller.snapshot()
        gate = tail._gate_index(values)
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        changed = abs(governed[1] - recorded[1]) > 1e-7
        if snapshot["active"] and not was_active:
            activations += 1
        was_active = bool(snapshot["active"])
        if changed and gate != 2:
            changed_non_gate3 += 1
        if changed and forward_m > policy.PARAMETERS.admission_forward_m + 1e-5:
            changed_before_admission += 1
        max_non_roll_error = max(
            max_non_roll_error,
            max(abs(governed[index] - recorded[index]) for index in (0, 2, 3)),
        )
        if not math.isfinite(governed[1]) or not -1.0 <= governed[1] <= 1.0:
            invalid_roll_outputs += 1
        max_controller_slew = max(
            max_controller_slew, float(snapshot["last_slew_norm_s"])
        )
        if gate == 2:
            rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": elapsed,
                    "forward_m": forward_m,
                    "right_m": right_m,
                    "right_rate_m_s": right_rate_m_s,
                    "recorded_roll_norm": float(recorded[1]),
                    "governed_roll_norm": float(governed[1]),
                    "controller_active": bool(snapshot["active"]),
                    "requested_roll_norm": float(
                        snapshot["last_requested_roll_norm"]
                    ),
                    "controller_slew_norm_s": float(
                        snapshot["last_slew_norm_s"]
                    ),
                }
            )

    active_rows = [row for row in rows if row["controller_active"]]
    counter_rows = [row for row in active_rows if row["governed_roll_norm"] < 0.0]
    requested_counter_rows = [
        row for row in active_rows if row["requested_roll_norm"] < 0.0
    ]
    nonpositive_rows = [
        row for row in active_rows if row["governed_roll_norm"] <= 0.0
    ]
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "gate3_samples": len(rows),
        "active_samples": len(active_rows),
        "activations": activations,
        "counter_samples": len(counter_rows),
        "first_counter_elapsed_s": (
            None if not counter_rows else counter_rows[0]["elapsed_s"]
        ),
        "first_counter_forward_m": (
            None if not counter_rows else counter_rows[0]["forward_m"]
        ),
        "first_requested_counter_forward_m": (
            None
            if not requested_counter_rows
            else requested_counter_rows[0]["forward_m"]
        ),
        "first_nonpositive_roll_forward_m": (
            None if not nonpositive_rows else nonpositive_rows[0]["forward_m"]
        ),
        "max_non_roll_error": max_non_roll_error,
        "invalid_roll_outputs": invalid_roll_outputs,
        "changed_non_gate3": changed_non_gate3,
        "changed_before_admission": changed_before_admission,
        "max_controller_slew_norm_s": max_controller_slew,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.trace]
    blockers: list[str] = []
    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        blockers.append("prefix_checkpoint_hash_mismatch")
    policy_path = Path(policy.__file__).resolve()
    policy_sha = _sha256(policy_path)
    if policy_sha != args.expected_policy_sha256.lower():
        blockers.append("policy_callable_hash_mismatch")
    if dataclasses.asdict(policy.PARAMETERS) != EXPECTED_PARAMETERS:
        blockers.append("controller_parameter_vector_mismatch")

    for report in reports:
        candidate = report["candidate"]
        if report["gate3_samples"] and report["active_samples"] == 0:
            blockers.append(f"candidate{candidate}_never_activated")
        if report["activations"] > 1:
            blockers.append(f"candidate{candidate}_multiple_activations")
        if report["max_non_roll_error"] != 0.0:
            blockers.append(f"candidate{candidate}_changed_non_roll")
        if report["invalid_roll_outputs"]:
            blockers.append(f"candidate{candidate}_invalid_roll")
        if report["changed_non_gate3"]:
            blockers.append(f"candidate{candidate}_changed_other_gate")
        if report["changed_before_admission"]:
            blockers.append(f"candidate{candidate}_changed_before_admission")
        if (
            report["max_controller_slew_norm_s"]
            > policy.PARAMETERS.max_roll_slew_norm_s + 1e-5
        ):
            blockers.append(f"candidate{candidate}_slew_limit_exceeded")

    by_candidate = {report["candidate"]: report for report in reports}
    holdout = by_candidate.get("045")
    if holdout is None:
        blockers.append("candidate045_missing")
    elif holdout["counter_samples"] == 0:
        blockers.append("candidate045_no_counter_bank")
    else:
        requested_counter_forward = holdout["first_requested_counter_forward_m"]
        if not 4.0 <= float(requested_counter_forward) <= 6.5:
            blockers.append("candidate045_requested_counter_not_in_4m_to_6p5m")
        nonpositive_forward = holdout["first_nonpositive_roll_forward_m"]
        if not 4.0 <= float(nonpositive_forward) <= 6.5:
            blockers.append("candidate045_nonpositive_roll_not_in_4m_to_6p5m")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "authorization": {
            "passive_zero_command_shadow": not blockers,
            "live_flight": False,
            "reason": (
                "offline invariants can authorize observation-only shadow; "
                "live flight requires a separate passing shadow/parity artifact"
            ),
        },
        "contract": {
            "mechanism": "anchored cubic projected-path PD",
            "scope": "official Gate 3 roll only inside 12 m",
            "preserve_non_roll_channels": True,
            "base_policy": "exact N184 six-gate hybrid",
            "parameters": dataclasses.asdict(policy.PARAMETERS),
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "policy_callable": str(policy_path),
        "policy_callable_sha256": policy_sha,
        "reports": reports,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
