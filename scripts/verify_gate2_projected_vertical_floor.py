#!/usr/bin/env python3
"""Verify N169's isolated Gate-2 projected-vertical thrust floor."""

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


PASSING_CANDIDATES = {
    "016", "018", "019", "020", "021", "022", "024", "025", "026", "027", "028"
}
EXPECTED_ACTIVATIONS = {"014", "029"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    if match is None:
        raise ValueError(f"cannot parse candidate number from {path}")
    return match.group(1)


def _projected_down(values: np.ndarray) -> tuple[float, float]:
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
    horizon_s = forward_m / max(
        -forward_rate_m_s,
        hybrid.GATE2_INTERCEPT_SPEED_FLOOR_M_S,
    )
    down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
    down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
    return forward_m, down_m + down_rate_m_s * horizon_s


def analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = (report.get("policy_trace") or {}).get("samples") or []
    hybrid._clear_gate2_state()
    rows: list[dict] = []
    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 1:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"invalid Gate-2 sample {path}:{sample_index}")
        floor_before = hybrid._GATE2_VERTICAL_FLOOR_LATCHED
        governed = np.asarray(
            hybrid._gate2_projected_vertical_floor(values, recorded.tolist()),
            dtype=float,
        )
        floor_after = hybrid._GATE2_VERTICAL_FLOOR_LATCHED
        forward_m, projected_down_m = _projected_down(values)
        if floor_before or floor_after:
            rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample["elapsed_s"]),
                    "forward_m": forward_m,
                    "projected_down_m": projected_down_m,
                    "floor_before": floor_before,
                    "floor_after": floor_after,
                    "recorded_action": recorded.tolist(),
                    "governed_action": governed.tolist(),
                }
            )

    first = next((row for row in rows if not row["floor_before"]), None)
    changed = [
        row for row in rows
        if row["governed_action"][2] != row["recorded_action"][2]
    ]
    latest = ((report.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "collision_id": latest.get("collision_id"),
        "collision_impact": latest.get("collision_impact"),
        "first_activation": first,
        "floor_samples": len(rows),
        "changed_thrust_samples": len(changed),
        "max_pitch_roll_yaw_error": max(
            (
                max(
                    abs(row["governed_action"][index] - row["recorded_action"][index])
                    for index in (0, 1, 3)
                )
                for row in rows
            ),
            default=0.0,
        ),
        "stronger_thrust_changes": sum(
            row["recorded_action"][2] >= hybrid.GATE2_VERTICAL_THRUST_FLOOR_NORM
            and row["governed_action"][2] != row["recorded_action"][2]
            for row in rows
        ),
        "floor_violations": sum(
            row["governed_action"][2]
            < hybrid.GATE2_VERTICAL_THRUST_FLOOR_NORM - 1e-9
            for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate029-poststop", required=True, type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []
    for candidate in sorted(PASSING_CANDIDATES | EXPECTED_ACTIVATIONS):
        if candidate not in by_candidate:
            blockers.append(f"candidate{candidate}_missing")
    for candidate in sorted(PASSING_CANDIDATES):
        report = by_candidate.get(candidate)
        if report and report["floor_samples"] != 0:
            blockers.append(f"candidate{candidate}_clean_path_changed")
    for candidate in sorted(EXPECTED_ACTIVATIONS):
        report = by_candidate.get(candidate)
        if report is None:
            continue
        first = report["first_activation"]
        if first is None:
            blockers.append(f"candidate{candidate}_does_not_activate")
        elif first["projected_down_m"] >= hybrid.GATE2_VERTICAL_PROJECTED_DOWN_TRIGGER_M:
            blockers.append(f"candidate{candidate}_unsafe_activation")
        if report["changed_thrust_samples"] < 1:
            blockers.append(f"candidate{candidate}_no_thrust_change")
    candidate29 = by_candidate.get("029")
    if candidate29 is not None:
        first29 = candidate29["first_activation"]
        if first29 is None or not (11.9 <= first29["forward_m"] <= 12.01):
            blockers.append("candidate029_activation_range_changed")
        if candidate29["changed_thrust_samples"] != 5:
            blockers.append("candidate029_change_count_not_five")
    for report in reports:
        if report["max_pitch_roll_yaw_error"] != 0.0:
            blockers.append(f"candidate{report['candidate']}_changes_non_thrust")
        if report["stronger_thrust_changes"] != 0:
            blockers.append(f"candidate{report['candidate']}_changes_stronger_thrust")
        if report["floor_violations"] != 0:
            blockers.append(f"candidate{report['candidate']}_floor_violation")

    poststop = json.loads(args.candidate029_poststop.read_text(encoding="utf-8"))
    poststop_race = ((poststop.get("latest_telemetry") or {}).get("race_status") or {})
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate029_poststop_not_command_free")
    if int(poststop_race.get("active_gate_index", -1)) != 1:
        blockers.append("candidate029_poststop_index_changed")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 1,
            "max_forward_m": hybrid.GATE2_VERTICAL_FLOOR_FORWARD_M,
            "projected_down_trigger_m": hybrid.GATE2_VERTICAL_PROJECTED_DOWN_TRIGGER_M,
            "normalized_thrust_floor": hybrid.GATE2_VERTICAL_THRUST_FLOOR_NORM,
            "preserve_pitch_roll_yaw_and_stronger_thrust": True,
            "latch_until_official_transition": True,
        },
        "candidate029_poststop": {
            "path": str(args.candidate029_poststop),
            "sha256": _sha256(args.candidate029_poststop),
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
