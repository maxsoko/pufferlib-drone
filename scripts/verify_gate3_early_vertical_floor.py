#!/usr/bin/env python3
"""Verify N167's retained one-sided Gate-3 hover-thrust floor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    import policy_callable_six_gate_hybrid as hybrid
    import policy_callable_six_gate_composite as tail
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_hybrid as hybrid
    from scripts import policy_callable_six_gate_composite as tail


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project(values: np.ndarray) -> tuple[float, float, float]:
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
    horizon_s = (
        max(forward_m - 2.0, 0.0) / closing_m_s
        if closing_m_s >= 1.0
        else 0.0
    )
    right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
    down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
    right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
    down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
    return (
        forward_m,
        right_m + right_rate_m_s * horizon_s,
        down_m + down_rate_m_s * horizon_s,
    )


def analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = (report.get("policy_trace") or {}).get("samples") or []
    hybrid._clear_gate3_state()
    rows: list[dict] = []
    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 2:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            continue
        forward_m, projected_right_m, projected_down_m = _project(values)
        floor_before = hybrid._GATE3_VERTICAL_FLOOR_LATCHED
        governed = np.asarray(
            hybrid._gate3_projected_vertical_floor(values, recorded.tolist()),
            dtype=float,
        )
        floor_after = hybrid._GATE3_VERTICAL_FLOOR_LATCHED
        if 0.0 < forward_m <= hybrid.GATE3_VERTICAL_FLOOR_FORWARD_M:
            rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample["elapsed_s"]),
                    "forward_m": forward_m,
                    "projected_right_m": projected_right_m,
                    "projected_down_m": projected_down_m,
                    "floor_before": floor_before,
                    "floor_after": floor_after,
                    "recorded_action": recorded.tolist(),
                    "governed_action": governed.tolist(),
                }
            )

    first_floor = next(
        (row for row in rows if not row["floor_before"] and row["floor_after"]),
        None,
    )
    floored_rows = [row for row in rows if row["floor_after"]]
    changed_rows = [
        row
        for row in floored_rows
        if row["governed_action"][2] != row["recorded_action"][2]
    ]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "first_floor_activation": first_floor,
        "floor_samples": len(floored_rows),
        "changed_thrust_samples": len(changed_rows),
        "max_pitch_roll_yaw_error": max(
            (
                max(
                    abs(
                        row["governed_action"][index]
                        - row["recorded_action"][index]
                    )
                    for index in (0, 1, 3)
                )
                for row in floored_rows
            ),
            default=0.0,
        ),
        "positive_thrust_changes": sum(
            row["recorded_action"][2] >= 0.0
            and row["governed_action"][2] != row["recorded_action"][2]
            for row in floored_rows
        ),
        "thrust_floor_violations": sum(
            row["governed_action"][2]
            < hybrid.GATE3_VERTICAL_FLOOR_THRUST_NORM - 1e-9
            for row in floored_rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    blockers: list[str] = []
    for report in reports:
        name = Path(report["path"]).stem
        first = report["first_floor_activation"]
        if first is None:
            blockers.append(f"{name}:no_vertical_floor_activation")
        elif (
            first["projected_down_m"]
            >= hybrid.GATE3_VERTICAL_FLOOR_PROJECTED_DOWN_TRIGGER_M
        ):
            blockers.append(f"{name}:floor_admitted_without_negative_miss")
        if report["max_pitch_roll_yaw_error"] != 0.0:
            blockers.append(f"{name}:changes_pitch_roll_or_yaw")
        if report["positive_thrust_changes"] != 0:
            blockers.append(f"{name}:changes_positive_thrust")
        if report["thrust_floor_violations"] != 0:
            blockers.append(f"{name}:floor_violation")

    candidate26 = next(
        (
            report
            for report in reports
            if "bounded_026_attempt_001" in report["path"]
        ),
        None,
    )
    if candidate26 is None:
        blockers.append("candidate026_missing")
    else:
        first26 = candidate26["first_floor_activation"]
        if first26 is None or not (9.0 <= first26["forward_m"] <= 10.5):
            blockers.append("candidate026_activation_range_changed")
        if first26 is not None and first26["recorded_action"][2] >= 0.0:
            blockers.append("candidate026_first_thrust_not_below_hover")
        if first26 is not None and first26["governed_action"][2] != 0.0:
            blockers.append("candidate026_first_thrust_not_floored_to_hover")
        if candidate26["changed_thrust_samples"] < 1:
            blockers.append("candidate026_no_changed_thrust_samples")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "mechanism": {
            "official_gate_index": 2,
            "max_forward_m": hybrid.GATE3_VERTICAL_FLOOR_FORWARD_M,
            "negative_projected_down_trigger_m": (
                hybrid.GATE3_VERTICAL_FLOOR_PROJECTED_DOWN_TRIGGER_M
            ),
            "normalized_thrust_floor": (
                hybrid.GATE3_VERTICAL_FLOOR_THRUST_NORM
            ),
            "physical_thrust_floor": "hover=0.27",
            "preserve_pitch_roll_yaw_and_positive_thrust": True,
            "latch_until_official_transition": True,
        },
        "reports": reports,
        "policy_callable": str(Path(hybrid.__file__).resolve()),
        "policy_callable_sha256": _sha256(Path(hybrid.__file__).resolve()),
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
