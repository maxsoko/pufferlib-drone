#!/usr/bin/env python3
"""Verify N167's Gate-3 terminal safe-aperture roll level on live traces."""

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
    right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
    down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
    closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
    right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
    down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
    time_to_terminal_s = (
        max(forward_m - 2.0, 0.0) / closing_m_s
        if closing_m_s >= 1.0
        else 0.0
    )
    return (
        forward_m,
        right_m + right_rate_m_s * time_to_terminal_s,
        down_m + down_rate_m_s * time_to_terminal_s,
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
            raise ValueError(f"invalid Gate-3 sample {path}:{sample_index}")
        forward_m, projected_right_m, projected_down_m = _project(values)
        latched_before = hybrid._GATE3_TERMINAL_LEVEL_LATCHED
        governed = np.asarray(
            hybrid._gate3_terminal_level(values, recorded.tolist()),
            dtype=float,
        )
        latched_after = hybrid._GATE3_TERMINAL_LEVEL_LATCHED
        if latched_before or latched_after:
            rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample["elapsed_s"]),
                    "forward_m": forward_m,
                    "projected_right_m": projected_right_m,
                    "projected_down_m": projected_down_m,
                    "latched_before": latched_before,
                    "latched_after": latched_after,
                    "recorded_action": recorded.tolist(),
                    "governed_action": governed.tolist(),
                }
            )

    first = next((row for row in rows if not row["latched_before"]), None)
    changed = [
        row
        for row in rows
        if row["governed_action"] != row["recorded_action"]
    ]
    latest = ((report.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "collision_id": latest.get("collision_id"),
        "first_activation": first,
        "coast_samples": len(rows),
        "changed_samples": len(changed),
        "max_pitch_thrust_yaw_error": max(
            (
                max(
                    abs(
                        row["governed_action"][index]
                        - row["recorded_action"][index]
                    )
                    for index in (0, 2, 3)
                )
                for row in rows
            ),
            default=0.0,
        ),
        "roll_level_violations": sum(
            row["governed_action"][1] != 0.0 for row in rows
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
    by_candidate = {
        Path(report["path"]).stem.split("_bounded_")[1][:3]: report
        for report in reports
    }
    expected_no_activation = {"016", "018", "020", "022"}
    for candidate in expected_no_activation:
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif report["coast_samples"] != 0:
            blockers.append(f"candidate{candidate}_clean_path_changed")
    for candidate in ("019", "021", "024"):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
            continue
        if report["first_activation"] is None:
            blockers.append(f"candidate{candidate}_does_not_activate")
        if report["max_pitch_thrust_yaw_error"] != 0.0:
            blockers.append(
                f"candidate{candidate}_changes_pitch_thrust_or_yaw"
            )
        if report["roll_level_violations"] != 0:
            blockers.append(f"candidate{candidate}_violates_level_output")

    candidate24 = by_candidate.get("024")
    if candidate24 is not None and candidate24["first_activation"] is not None:
        first24 = candidate24["first_activation"]
        if not (5.0 <= first24["forward_m"] <= 6.0):
            blockers.append("candidate024_activation_range_changed")
        if abs(first24["projected_right_m"]) > 0.5:
            blockers.append("candidate024_lateral_projection_unsafe")
        if abs(first24["projected_down_m"]) > 1.0:
            blockers.append("candidate024_vertical_projection_unsafe")
        if first24["recorded_action"][0] <= 0.0:
            blockers.append("candidate024_does_not_replace_forward_pitch")
        if first24["recorded_action"][1] != -1.0:
            blockers.append("candidate024_does_not_replace_full_counter_roll")
    candidate21 = by_candidate.get("021")
    if candidate21 is not None and candidate21["first_activation"] is not None:
        if candidate21["first_activation"]["recorded_action"][1] != 0.0:
            blockers.append("candidate021_clean_activation_changes_nonlevel_roll")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate_index": 2,
            "max_forward_m": hybrid.GATE3_TERMINAL_LEVEL_FORWARD_M,
            "max_abs_projected_right_m": (
                hybrid.GATE3_TERMINAL_MAX_ABS_PROJECTED_RIGHT_M
            ),
            "max_abs_projected_down_m": (
                hybrid.GATE3_TERMINAL_MAX_ABS_PROJECTED_DOWN_M
            ),
            "level_roll_norm": 0.0,
            "preserve_pitch_thrust_yaw": True,
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
