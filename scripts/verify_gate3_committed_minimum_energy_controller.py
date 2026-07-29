#!/usr/bin/env python3
"""Verify N183's committed minimum-energy Gate-3 controller on live traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

try:
    import policy_callable_six_gate_hybrid as hybrid
    import policy_callable_six_gate_composite as tail
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_hybrid as hybrid
    from scripts import policy_callable_six_gate_composite as tail


EXPECTED_CHANGES = {
    "016": 4,
    "018": 5,
    "020": 8,
    "021": 3,
    "022": 3,
    "030": 3,
    "031": 3,
    "034": 3,
    "037": 3,
    "038": 4,
    "039": 7,
    "040": 5,
    "041": 4,
    "042": 2,
    "043": 3,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    if match is None:
        raise ValueError(f"cannot parse candidate number from {path}")
    return match.group(1)


def _decoded_row(sample_index: int, sample: dict, values: np.ndarray) -> dict:
    return {
        "sample": sample_index,
        "elapsed_s": float(sample["elapsed_s"]),
        "current_frame_visible": float(values[10]) >= 0.5,
        "forward_m": tail._inverse_tanh_norm(float(values[11]), 10.0),
        "right_m": tail._inverse_tanh_norm(float(values[12]), 5.0),
        "right_rate_m_s": tail._inverse_tanh_norm(float(values[1]), 3.0),
        "closing_m_s": -tail._inverse_tanh_norm(float(values[0]), 5.0),
        "confidence": float(values[30]),
    }


def analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = (report.get("policy_trace") or {}).get("samples") or []
    hybrid._clear_gate3_state()

    changed_rows: list[dict] = []
    commit: dict | None = None
    level: dict | None = None
    blockers: list[str] = []
    gate3_samples = 0
    changed_samples = 0
    max_non_roll_error = 0.0
    invalid_roll_outputs = 0
    previous_level_latched = False

    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 2:
            continue
        gate3_samples += 1
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"invalid Gate-3 sample {path}:{sample_index}")

        roll_before = hybrid._GATE3_MINIMUM_ENERGY_ROLL_NORM
        level_before = hybrid._GATE3_MINIMUM_ENERGY_LEVEL_LATCHED
        governed = np.asarray(
            hybrid._gate3_minimum_energy_lateral_controller(
                values, recorded.tolist()
            ),
            dtype=float,
        )
        roll_after = hybrid._GATE3_MINIMUM_ENERGY_ROLL_NORM
        level_after = hybrid._GATE3_MINIMUM_ENERGY_LEVEL_LATCHED
        decoded = _decoded_row(sample_index, sample, values)
        decoded["recorded_action"] = recorded.tolist()
        decoded["governed_action"] = governed.tolist()

        non_roll_error = max(
            abs(governed[index] - recorded[index]) for index in (0, 2, 3)
        )
        max_non_roll_error = max(max_non_roll_error, non_roll_error)
        if roll_after is not None and (
            not math.isfinite(governed[1])
            or not -0.5 <= governed[1] <= 0.5
        ):
            invalid_roll_outputs += 1

        changed = governed.tolist() != recorded.tolist()
        if changed:
            changed_samples += 1

        committed_now = roll_before is None and roll_after is not None
        leveled_now = not level_before and level_after
        if committed_now:
            commit = {
                **decoded,
                "committed_roll_norm": roll_after,
            }
            if not decoded["current_frame_visible"]:
                blockers.append("commit_without_current_frame")
            if decoded["confidence"] < hybrid.GATE3_MINIMUM_ENERGY_MIN_CONFIDENCE:
                blockers.append("commit_below_minimum_confidence")
            if not (
                hybrid.GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M
                < decoded["forward_m"]
                <= hybrid.GATE3_MINIMUM_ENERGY_MAX_FORWARD_M
            ):
                blockers.append("commit_outside_range")
            if decoded["closing_m_s"] < 1.0:
                blockers.append("commit_without_closing_speed")
        if leveled_now:
            level = decoded

        if roll_after is not None and not level_after:
            if governed[1] != roll_after:
                blockers.append(f"sample{sample_index}_did_not_hold_commit")
            if roll_before is not None and roll_after != roll_before:
                blockers.append(f"sample{sample_index}_commit_changed")
        if level_after:
            if governed[1] != 0.0:
                blockers.append(f"sample{sample_index}_not_level_after_latch")
            if previous_level_latched and not level_before:
                blockers.append(f"sample{sample_index}_level_latch_regressed")
        previous_level_latched = level_after

        if changed or committed_now or leveled_now:
            decoded["committed_now"] = committed_now
            decoded["leveled_now"] = leveled_now
            changed_rows.append(decoded)

    latest = ((report.get("sitl") or {}).get("latest_telemetry") or {})
    if commit is None:
        blockers.append("missing_commit")
    if level is None:
        blockers.append("missing_level_handoff")
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "reported_official_active_gate_index": report.get(
            "official_active_gate_index"
        ),
        "collision_id": latest.get("collision_id"),
        "collision_threat_level": latest.get("collision_threat_level"),
        "collision_impact": latest.get("collision_impact"),
        "gate3_samples": gate3_samples,
        "changed_samples": changed_samples,
        "max_non_roll_error": max_non_roll_error,
        "invalid_roll_outputs": invalid_roll_outputs,
        "commit": commit,
        "level_handoff": level,
        "blockers": blockers,
        "evidence_rows": changed_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate043-poststop", required=True, type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []
    for candidate, expected in EXPECTED_CHANGES.items():
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
            continue
        if report["changed_samples"] != expected:
            blockers.append(
                f"candidate{candidate}_change_count_not_{expected}"
            )
        blockers.extend(
            f"candidate{candidate}_{blocker}" for blocker in report["blockers"]
        )
        if report["max_non_roll_error"] != 0.0:
            blockers.append(f"candidate{candidate}_changes_non_roll")
        if report["invalid_roll_outputs"]:
            blockers.append(f"candidate{candidate}_invalid_roll_output")

    clean021 = by_candidate.get("021") or {}
    failed043 = by_candidate.get("043") or {}
    clean_commit = clean021.get("commit") or {}
    failed_commit = failed043.get("commit") or {}
    failed_level = failed043.get("level_handoff") or {}
    if clean_commit.get("committed_roll_norm", 0.0) <= 0.0:
        blockers.append("candidate021_commit_not_positive")
    if failed_commit.get("committed_roll_norm", 0.0) >= 0.0:
        blockers.append("candidate043_commit_not_negative")
    if float(failed_level.get("forward_m", math.inf)) > 3.6:
        blockers.append("candidate043_level_handoff_not_early")

    poststop = json.loads(args.candidate043_poststop.read_text(encoding="utf-8"))
    latest = poststop.get("latest_telemetry") or {}
    race = latest.get("race_status") or {}
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate043_poststop_not_command_free")
    if int(race.get("active_gate_index", -1)) != 2:
        blockers.append("candidate043_poststop_gate_index_not_2")
    if int(race.get("race_start_boot_time_ms", -1)) < 0:
        blockers.append("candidate043_poststop_race_not_active")
    if int(race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate043_poststop_finish_unexpected")
    if int(latest.get("base_mode", -1)) != 65:
        blockers.append("candidate043_poststop_not_disarmed")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "mechanism": "one-shot committed minimum-energy intercept",
            "objective": "minimum integral lateral acceleration squared",
            "dynamics": "right_dot=right_rate; right_rate_dot=accel",
            "first_acceleration": "6*(target_right-right)/T^2-(4*right_rate+2*target_rate)/T",
            "commit_once": True,
            "hold_committed_roll_to_handoff": True,
            "level_latched_after_handoff": True,
            "max_forward_m": hybrid.GATE3_MINIMUM_ENERGY_MAX_FORWARD_M,
            "handoff_forward_m": hybrid.GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M,
            "target_right_m": hybrid.GATE3_MINIMUM_ENERGY_TARGET_RIGHT_M,
            "target_right_rate_m_s": hybrid.GATE3_MINIMUM_ENERGY_TARGET_RIGHT_RATE_M_S,
            "min_confidence": hybrid.GATE3_MINIMUM_ENERGY_MIN_CONFIDENCE,
            "min_horizon_s": hybrid.GATE3_MINIMUM_ENERGY_MIN_HORIZON_S,
            "max_roll_rad": hybrid.GATE3_MINIMUM_ENERGY_MAX_ROLL_RAD,
            "policy_max_roll_rad": hybrid.GATE3_POLICY_MAX_ROLL_RAD,
            "preserve_pitch_thrust_yaw": True,
        },
        "candidate043_poststop": {
            "path": str(args.candidate043_poststop),
            "sha256": _sha256(args.candidate043_poststop),
            "reset_sent": poststop.get("reset_sent"),
            "base_mode": latest.get("base_mode"),
            "system_status": latest.get("system_status"),
            "race_status": race,
        },
        "reports": reports,
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
