#!/usr/bin/env python3
"""Verify N167's evidence-separated retirement of the Gate-3 pitch brake."""

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


PASSING_CANDIDATES = {"016", "018", "020", "021", "022"}
FORCED_BRAKE_FAILURES = {"025", "026", "027"}
COUNTER_ROLL_FAILURE = "024"
CLOSE_FORWARD_M = 4.0
RETIRED_PITCH_NORM = -0.2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    if match is None:
        raise ValueError(f"cannot parse candidate number from {path}")
    return match.group(1)


def _forward_m(values: np.ndarray) -> float:
    return tail._inverse_tanh_norm(float(values[11]), 10.0)


def analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = (report.get("policy_trace") or {}).get("samples") or []
    hybrid._clear_gate3_state()
    rows: list[dict] = []
    max_pitch_error = 0.0
    max_yaw_error = 0.0
    channel_changes = [0, 0, 0, 0]
    terminal_level_changes = 0

    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 2:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"invalid Gate-3 sample {path}:{sample_index}")
        level_before = hybrid._GATE3_TERMINAL_LEVEL_LATCHED
        governed = hybrid._gate3_projected_vertical_floor(
            values, recorded.tolist()
        )
        governed = np.asarray(
            hybrid._gate3_terminal_level(values, governed), dtype=float
        )
        level_after = hybrid._GATE3_TERMINAL_LEVEL_LATCHED
        delta = np.abs(governed - recorded)
        max_pitch_error = max(max_pitch_error, float(delta[0]))
        max_yaw_error = max(max_yaw_error, float(delta[3]))
        for index in range(4):
            channel_changes[index] += int(delta[index] > 1e-9)
        if (level_before or level_after) and delta[1] > 1e-9:
            terminal_level_changes += 1

        forward_m = _forward_m(values)
        if (
            float(values[10]) >= 0.5
            and 0.0 < forward_m <= CLOSE_FORWARD_M
        ):
            rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample["elapsed_s"]),
                    "forward_m": forward_m,
                    "recorded_action": recorded.tolist(),
                    "governed_action": governed.tolist(),
                    "terminal_level_latched": level_after,
                }
            )

    latest = ((report.get("sitl") or {}).get("latest_telemetry") or {})
    pitches = [row["recorded_action"][0] for row in rows]
    rolls = [row["recorded_action"][1] for row in rows]
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": int(
            report.get("official_active_gate_index", -1)
        ),
        "collision_id": latest.get("collision_id"),
        "collision_impact": latest.get("collision_impact"),
        "close_samples": len(rows),
        "close_positive_pitch_samples": sum(pitch > 0.0 for pitch in pitches),
        "close_retired_pitch_samples": sum(
            math.isclose(pitch, RETIRED_PITCH_NORM, abs_tol=1e-6)
            for pitch in pitches
        ),
        "close_full_negative_roll_samples": sum(
            math.isclose(roll, -1.0, abs_tol=1e-6) for roll in rolls
        ),
        "last_close_sample": rows[-1] if rows else None,
        "max_pitch_error": max_pitch_error,
        "max_yaw_error": max_yaw_error,
        "governor_channel_changes": {
            "pitch": channel_changes[0],
            "roll": channel_changes[1],
            "thrust": channel_changes[2],
            "yaw": channel_changes[3],
        },
        "terminal_level_roll_changes": terminal_level_changes,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []

    expected = PASSING_CANDIDATES | FORCED_BRAKE_FAILURES | {
        COUNTER_ROLL_FAILURE
    }
    for candidate in sorted(expected - by_candidate.keys()):
        blockers.append(f"candidate{candidate}_missing")

    for candidate in sorted(PASSING_CANDIDATES):
        report = by_candidate.get(candidate)
        if report is None:
            continue
        if report["official_active_gate_index"] < 3:
            blockers.append(f"candidate{candidate}_not_official_gate3_pass")
        if report["close_samples"] < 1:
            blockers.append(f"candidate{candidate}_no_close_samples")
        elif report["close_positive_pitch_samples"] != report["close_samples"]:
            blockers.append(f"candidate{candidate}_close_pitch_not_positive")

    for candidate in sorted(FORCED_BRAKE_FAILURES):
        report = by_candidate.get(candidate)
        if report is None:
            continue
        if report["official_active_gate_index"] != 2:
            blockers.append(f"candidate{candidate}_unexpected_official_progress")
        if report["close_samples"] < 1:
            blockers.append(f"candidate{candidate}_no_close_samples")
        elif report["close_retired_pitch_samples"] != report["close_samples"]:
            blockers.append(f"candidate{candidate}_brake_not_forced_close")

    counter = by_candidate.get(COUNTER_ROLL_FAILURE)
    if counter is not None:
        if counter["official_active_gate_index"] != 2:
            blockers.append("candidate024_unexpected_official_progress")
        if counter["close_full_negative_roll_samples"] < 1:
            blockers.append("candidate024_missing_counter_roll_failure")
        if counter["terminal_level_roll_changes"] < 1:
            blockers.append("candidate024_not_corrected_by_terminal_level")

    for report in reports:
        candidate = report["candidate"]
        if report["max_pitch_error"] != 0.0:
            blockers.append(f"candidate{candidate}_pitch_changed")
        if report["max_yaw_error"] != 0.0:
            blockers.append(f"candidate{candidate}_yaw_changed")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "hypothesis": {
            "retired_pitch_norm": RETIRED_PITCH_NORM,
            "clean_gate3_pass_candidates": sorted(PASSING_CANDIDATES),
            "forced_brake_failure_candidates": sorted(
                FORCED_BRAKE_FAILURES
            ),
            "counter_roll_failure_candidate": COUNTER_ROLL_FAILURE,
            "mechanism": (
                "preserve recurrent pitch; retain projected vertical thrust "
                "floor and terminal roll level"
            ),
            "close_forward_m": CLOSE_FORWARD_M,
        },
        "reports": reports,
        "policy_callable": str(policy_path),
        "policy_callable_sha256": _sha256(policy_path),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
