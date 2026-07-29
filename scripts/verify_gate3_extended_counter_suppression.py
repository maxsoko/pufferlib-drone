#!/usr/bin/env python3
"""Verify N168's Gate-3 counter-bank suppression extension to seven metres."""

from __future__ import annotations

import argparse
import hashlib
import json
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
EXPECTED_EXERCISED = {"019", "024", "026", "028"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    if match is None:
        raise ValueError(f"cannot parse candidate number from {path}")
    return match.group(1)


def analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = (report.get("policy_trace") or {}).get("samples") or []
    rows: list[dict] = []
    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 2:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"invalid Gate-3 sample {path}:{sample_index}")
        if float(values[10]) < 0.5:
            continue
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        mode, _ = tail._promoted_gate3_roll_command(values)
        incremental_window = bool(
            hybrid.GATE3_CLOSE_SEVERE_FORWARD_M
            < forward_m
            <= hybrid.GATE3_COUNTER_SUPPRESSION_FORWARD_M
        )
        if not (
            incremental_window
            and mode == "counter"
            and right_m < 0.0
            and recorded[1] < hybrid.GATE3_CLOSE_COUNTER_ROLL_FLOOR
        ):
            continue
        governed = np.asarray(
            hybrid._gate3_close_severe_boost(values, recorded.tolist()),
            dtype=float,
        )
        rows.append(
            {
                "sample": sample_index,
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": forward_m,
                "right_m": right_m,
                "mode": mode,
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
            }
        )

    latest = ((report.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "reported_official_active_gate_index": report.get(
            "official_active_gate_index"
        ),
        "collision_id": latest.get("collision_id"),
        "collision_impact": latest.get("collision_impact"),
        "incremental_changed_samples": len(rows),
        "max_non_roll_error": max(
            (
                max(
                    abs(row["governed_action"][index] - row["recorded_action"][index])
                    for index in (0, 2, 3)
                )
                for row in rows
            ),
            default=0.0,
        ),
        "roll_floor_violations": sum(
            row["governed_action"][1]
            != hybrid.GATE3_CLOSE_COUNTER_ROLL_FLOOR
            for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate028-poststop", required=True, type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []
    for candidate in sorted(PASSING_CANDIDATES | EXPECTED_EXERCISED):
        if candidate not in by_candidate:
            blockers.append(f"candidate{candidate}_missing")
    for candidate in sorted(PASSING_CANDIDATES):
        report = by_candidate.get(candidate)
        if report and report["incremental_changed_samples"] != 0:
            blockers.append(f"candidate{candidate}_clean_path_changed")
    for candidate in sorted(EXPECTED_EXERCISED):
        report = by_candidate.get(candidate)
        if report and report["incremental_changed_samples"] < 1:
            blockers.append(f"candidate{candidate}_extension_not_exercised")
    candidate28 = by_candidate.get("028")
    if candidate28 is not None:
        if candidate28["incremental_changed_samples"] != 2:
            blockers.append("candidate028_change_count_not_two")
        ranges = [row["forward_m"] for row in candidate28["rows"]]
        if ranges and not (6.4 <= min(ranges) <= max(ranges) <= 7.0):
            blockers.append("candidate028_change_ranges_unexpected")
    for report in reports:
        if report["max_non_roll_error"] != 0.0:
            blockers.append(f"candidate{report['candidate']}_changes_non_roll")
        if report["roll_floor_violations"] != 0:
            blockers.append(f"candidate{report['candidate']}_roll_floor_violation")

    poststop = json.loads(args.candidate028_poststop.read_text(encoding="utf-8"))
    poststop_race = ((poststop.get("latest_telemetry") or {}).get("race_status") or {})
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate028_poststop_not_command_free")
    if int(poststop_race.get("active_gate_index", -1)) != 3:
        blockers.append("candidate028_poststop_does_not_prove_gate3")
    if int(poststop_race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate028_poststop_finish_unexpected")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 2,
            "adaptive_boost_max_forward_m": hybrid.GATE3_CLOSE_SEVERE_FORWARD_M,
            "counter_suppression_max_forward_m": (
                hybrid.GATE3_COUNTER_SUPPRESSION_FORWARD_M
            ),
            "left_side_only": True,
            "counter_roll_floor": hybrid.GATE3_CLOSE_COUNTER_ROLL_FLOOR,
            "preserve_pitch_thrust_yaw": True,
        },
        "candidate028_poststop": {
            "path": str(args.candidate028_poststop),
            "sha256": _sha256(args.candidate028_poststop),
            "reset_sent": poststop.get("reset_sent"),
            "race_status": poststop_race,
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
