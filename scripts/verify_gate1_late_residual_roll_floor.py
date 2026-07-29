#!/usr/bin/env python3
"""Verify N173's late one-sided Gate-1 residual roll floor."""

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


CLEAN_GATE1_CANDIDATES = {
    "011", "013", "014", "015", "016", "017", "018", "019", "020",
    "021", "022", "023", "024", "025", "026", "027", "028", "029",
    "030", "031", "032",
}
EXERCISED_FAILURES = {"008", "033"}


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
        if int(sample.get("official_active_gate_index", -1)) != 0:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"invalid Gate-1 sample {path}:{sample_index}")
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        time_to_plane_s = forward_m / max(
            -forward_rate_m_s,
            hybrid.GATE2_INTERCEPT_SPEED_FLOOR_M_S,
        )
        projected_right_m = right_m + right_rate_m_s * time_to_plane_s
        governed = np.asarray(
            hybrid._gate1_late_residual_roll_floor(values, recorded.tolist()),
            dtype=float,
        )
        if governed.tolist() == recorded.tolist():
            continue
        rows.append(
            {
                "sample": sample_index,
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": forward_m,
                "right_m": right_m,
                "right_rate_m_s": right_rate_m_s,
                "projected_right_at_plane_m": projected_right_m,
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
        "collision_threat_level": latest.get("collision_threat_level"),
        "collision_impact": latest.get("collision_impact"),
        "changed_samples": len(rows),
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
            != hybrid.GATE1_LATE_RESIDUAL_ROLL_FLOOR_NORM
            for row in rows
        ),
        "stronger_roll_changes": sum(
            row["recorded_action"][1]
            <= hybrid.GATE1_LATE_RESIDUAL_ROLL_FLOOR_NORM
            for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate033-poststop", required=True, type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []
    for candidate in sorted(CLEAN_GATE1_CANDIDATES | EXERCISED_FAILURES):
        if candidate not in by_candidate:
            blockers.append(f"candidate{candidate}_missing")
    for candidate in sorted(CLEAN_GATE1_CANDIDATES):
        report = by_candidate.get(candidate)
        if report and report["changed_samples"] != 0:
            blockers.append(f"candidate{candidate}_clean_path_changed")
    expected_failure_changes = {"008": 1, "033": 6}
    for candidate, expected in sorted(expected_failure_changes.items()):
        report = by_candidate.get(candidate)
        if report and report["changed_samples"] != expected:
            blockers.append(f"candidate{candidate}_change_count_not_{expected}")

    for report in reports:
        if report["max_non_roll_error"] != 0.0:
            blockers.append(f"candidate{report['candidate']}_changes_non_roll")
        if report["roll_floor_violations"]:
            blockers.append(f"candidate{report['candidate']}_roll_floor_violation")
        if report["stronger_roll_changes"]:
            blockers.append(f"candidate{report['candidate']}_changes_stronger_roll")

    poststop = json.loads(args.candidate033_poststop.read_text(encoding="utf-8"))
    race = ((poststop.get("latest_telemetry") or {}).get("race_status") or {})
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate033_poststop_not_command_free")
    if int(race.get("active_gate_index", -1)) != 0:
        blockers.append("candidate033_poststop_gate_index_unexpected")
    if int(race.get("last_gate_race_time", -2)) != -1:
        blockers.append("candidate033_poststop_claims_gate_pass")
    if int(race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate033_poststop_finish_unexpected")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 0,
            "max_forward_m": hybrid.GATE1_LATE_RESIDUAL_MAX_FORWARD_M,
            "projected_right_trigger_m": (
                hybrid.GATE1_LATE_RESIDUAL_PROJECTED_RIGHT_TRIGGER_M
            ),
            "normalized_roll_floor": hybrid.GATE1_LATE_RESIDUAL_ROLL_FLOOR_NORM,
            "positive_projected_miss_only": True,
            "preserve_pitch_thrust_yaw_and_stronger_roll": True,
        },
        "candidate033_poststop": {
            "path": str(args.candidate033_poststop),
            "sha256": _sha256(args.candidate033_poststop),
            "reset_sent": poststop.get("reset_sent"),
            "race_status": race,
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
