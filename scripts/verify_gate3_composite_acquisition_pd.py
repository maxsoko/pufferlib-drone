#!/usr/bin/env python3
"""Verify N189 branch precedence and bounded behavior on official traces."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

import policy_callable_gate3_composite_acquisition_pd as policy
import policy_callable_gate3_projected_path_pd as projected
import policy_callable_gate3_severe_acquisition as severe
import policy_callable_six_gate_composite as tail


EXPECTED_CHECKPOINT_SHA256 = (
    "ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760"
)
EXPECTED_PROJECTED_PARAMETERS = {
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
EXPECTED_SEVERE_PARAMETERS = {
    "trigger_min_forward_m": 15.0,
    "trigger_max_forward_m": 35.0,
    "trigger_min_abs_right_m": 8.0,
    "release_max_abs_right_m": 4.5,
    "release_max_abs_yaw_error_rad": 0.12,
    "acquisition_pitch_norm": -0.24,
    "acquisition_roll_norm": 0.0,
    "acquisition_thrust_norm": 0.0,
    "max_yaw_step_rad": 0.35,
    "max_dropout_ticks": 30,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _name(path: Path) -> str:
    bounded = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    if bounded:
        return f"candidate{bounded.group(1)}"
    reliability = re.search(r"reliability_129_attempt_(\d{3})", path.name)
    if reliability:
        return f"reliability129_attempt{reliability.group(1)}"
    return path.stem


def _current_observation(sample: dict) -> np.ndarray | None:
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
    samples = sorted(
        (payload.get("policy_trace") or {}).get("samples") or [],
        key=lambda item: float(item.get("elapsed_s") or 0.0),
    )
    controller = policy.Gate3CompositeAcquisitionPD()
    previous_elapsed: float | None = None
    previous_projected_active = False
    projected_activations = 0
    invalid_outputs = 0
    changed_non_gate3 = 0
    projected_non_roll_error = 0.0
    projected_before_admission = 0
    max_projected_slew = 0.0
    gate3_samples = 0
    branch_counts = {"base": 0, "severe": 0, "severe_release": 0, "projected": 0}
    rows: list[dict] = []

    for sample_index, sample in enumerate(samples):
        values = _current_observation(sample)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values is None or recorded.shape != (4,):
            continue
        elapsed = float(sample.get("elapsed_s") or 0.0)
        dt_s = 1.0 / projected.PARAMETERS.nominal_control_hz
        if previous_elapsed is not None and elapsed > previous_elapsed:
            dt_s = elapsed - previous_elapsed
        previous_elapsed = elapsed
        governed = np.asarray(
            controller.apply(values, recorded.tolist(), dt_s=dt_s), dtype=float
        )
        snapshot = controller.snapshot()
        branch = snapshot["branch"]
        branch_counts[branch] += 1
        gate = tail._gate_index(values)
        if gate == 2:
            gate3_samples += 1
        changed = bool(np.max(np.abs(governed - recorded)) > 1e-7)
        if changed and gate != 2:
            changed_non_gate3 += 1
        if not np.all(np.isfinite(governed)) or np.max(np.abs(governed)) > 1.0:
            invalid_outputs += 1

        projected_active = bool(snapshot["projected"]["active"])
        if projected_active and not previous_projected_active:
            projected_activations += 1
        previous_projected_active = projected_active
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        if branch == "projected":
            projected_non_roll_error = max(
                projected_non_roll_error,
                max(abs(governed[index] - recorded[index]) for index in (0, 2, 3)),
            )
            if changed and forward_m > projected.PARAMETERS.admission_forward_m + 1e-5:
                projected_before_admission += 1
            max_projected_slew = max(
                max_projected_slew,
                float(snapshot["projected"]["last_slew_norm_s"]),
            )
        if branch != "base" or changed:
            rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": elapsed,
                    "gate": gate,
                    "branch": branch,
                    "forward_m": forward_m,
                    "right_m": right_m,
                    "recorded_action": recorded.tolist(),
                    "governed_action": governed.tolist(),
                    "requested_roll_norm": float(
                        snapshot["projected"]["last_requested_roll_norm"]
                    ),
                    "projected_slew_norm_s": float(
                        snapshot["projected"]["last_slew_norm_s"]
                    ),
                }
            )

    projected_rows = [row for row in rows if row["branch"] == "projected"]
    counter_rows = [
        row for row in projected_rows if row["governed_action"][1] < 0.0
    ]
    requested_counter_rows = [
        row for row in projected_rows if row["requested_roll_norm"] < 0.0
    ]
    nonpositive_rows = [
        row for row in projected_rows if row["governed_action"][1] <= 0.0
    ]
    final = controller.snapshot()
    return {
        "name": _name(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": payload.get("official_active_gate_index"),
        "passed_gate3": int(payload.get("official_active_gate_index") or 0) >= 3,
        "gate3_samples": gate3_samples,
        "branch_counts": branch_counts,
        "severe_activations": final["severe"]["activations"],
        "severe_releases": final["severe"]["releases"],
        "projected_activations": projected_activations,
        "projected_visibility_resets": final["projected_visibility_resets"],
        "counter_samples": len(counter_rows),
        "first_counter_forward_m": None if not counter_rows else counter_rows[0]["forward_m"],
        "first_requested_counter_forward_m": (
            None if not requested_counter_rows else requested_counter_rows[0]["forward_m"]
        ),
        "first_nonpositive_roll_forward_m": (
            None if not nonpositive_rows else nonpositive_rows[0]["forward_m"]
        ),
        "invalid_outputs": invalid_outputs,
        "changed_non_gate3": changed_non_gate3,
        "projected_non_roll_error": projected_non_roll_error,
        "projected_before_admission": projected_before_admission,
        "max_projected_slew_norm_s": max_projected_slew,
        "rows": rows,
    }


def verify(paths: list[Path], checkpoint: Path, expected_policy_sha256: str) -> dict:
    reports = [analyze(path) for path in paths]
    blockers: list[str] = []
    checkpoint_sha = _sha256(checkpoint)
    policy_path = Path(policy.__file__).resolve()
    policy_sha = _sha256(policy_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        blockers.append("prefix_checkpoint_hash_mismatch")
    if policy_sha != expected_policy_sha256.lower():
        blockers.append("policy_callable_hash_mismatch")
    if dataclasses.asdict(projected.PARAMETERS) != EXPECTED_PROJECTED_PARAMETERS:
        blockers.append("projected_parameter_vector_mismatch")
    if dataclasses.asdict(severe.PARAMETERS) != EXPECTED_SEVERE_PARAMETERS:
        blockers.append("severe_parameter_vector_mismatch")

    for report in reports:
        name = report["name"]
        if report["changed_non_gate3"]:
            blockers.append(f"changed_non_gate3:{name}")
        if report["invalid_outputs"]:
            blockers.append(f"invalid_output:{name}")
        if report["projected_non_roll_error"] != 0.0:
            blockers.append(f"projected_changed_non_roll:{name}")
        if report["projected_before_admission"]:
            blockers.append(f"projected_changed_before_admission:{name}")
        if report["max_projected_slew_norm_s"] > projected.PARAMETERS.max_roll_slew_norm_s + 1e-5:
            blockers.append(f"projected_slew_limit_exceeded:{name}")

    by_name = {report["name"]: report for report in reports}
    candidate046 = by_name.get("candidate046")
    if candidate046 is None:
        blockers.append("candidate046_missing")
    else:
        if candidate046["severe_activations"] != 2:
            blockers.append("candidate046_severe_activation_count")
        if candidate046["branch_counts"]["severe"] == 0:
            blockers.append("candidate046_severe_branch_missing")
        if candidate046["projected_activations"] != 0:
            blockers.append("candidate046_projected_branch_overlap")

    for candidate in ("candidate045", "candidate048"):
        report = by_name.get(candidate)
        if report is None:
            blockers.append(f"{candidate}_missing")
            continue
        if report["severe_activations"] != 0:
            blockers.append(f"{candidate}_unexpected_severe_activation")
        if report["projected_activations"] != 1:
            blockers.append(f"{candidate}_projected_activation_count")
        if report["counter_samples"] == 0:
            blockers.append(f"{candidate}_no_counter_bank")
        for key in (
            "first_requested_counter_forward_m",
            "first_nonpositive_roll_forward_m",
        ):
            value = report[key]
            if value is None or not 4.0 <= float(value) <= 6.5:
                blockers.append(f"{candidate}_{key}_outside_4m_to_6p5m")

    for report in reports:
        if report["name"].startswith("reliability129") and report["severe_activations"]:
            blockers.append(f"protected_severe_activation:{report['name']}")

    return {
        "passed": not blockers,
        "blockers": blockers,
        "authorization": {
            "passive_zero_command_shadow": not blockers,
            "live_flight": False,
            "reason": (
                "offline branch and trace invariants can authorize only a "
                "hash-pinned observation-only shadow"
            ),
        },
        "contract": {
            "mechanism": "severe far-offset acquisition then close projected-path PD",
            "scope": "official Gate 3 only",
            "base_policy": "exact N184 six-gate hybrid",
            "branch_precedence": ["severe", "projected", "base"],
            "visibility_loss": "reset projected controller and delegate to base",
            "projected_parameters": dataclasses.asdict(projected.PARAMETERS),
            "severe_parameters": dataclasses.asdict(severe.PARAMETERS),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "policy_callable": str(policy_path),
        "policy_callable_sha256": policy_sha,
        "projected_policy_sha256": _sha256(Path(projected.__file__).resolve()),
        "severe_policy_sha256": _sha256(Path(severe.__file__).resolve()),
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reports, args.checkpoint, args.expected_policy_sha256)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
