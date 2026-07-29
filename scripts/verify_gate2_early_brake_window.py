#!/usr/bin/env python3
"""Verify N163's early-brake-only Gate-2 admission on archived live traces."""

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


def _kinematics(values: np.ndarray) -> tuple[float, float, float, float]:
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
    forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
    right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
    time_to_plane_s = forward_m / max(
        -forward_rate_m_s,
        hybrid.GATE2_INTERCEPT_SPEED_FLOOR_M_S,
    )
    projected_right_m = right_m + right_rate_m_s * time_to_plane_s
    return forward_m, right_m, right_rate_m_s, projected_right_m


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
            continue
        visible = float(values[10]) >= 0.5
        if not visible:
            continue
        forward_m, right_m, right_rate_m_s, projected_right_m = _kinematics(
            values
        )
        latched_before = hybrid._GATE2_GOVERNOR_LATCHED
        governed = np.asarray(
            hybrid._gate2_projected_aperture_governor(
                values,
                recorded.tolist(),
            ),
            dtype=float,
        )
        if not 0.0 < forward_m <= hybrid.GATE2_EARLY_BRAKE_FORWARD_M:
            continue
        rows.append(
            {
                "sample": sample_index,
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": forward_m,
                "right_m": right_m,
                "right_rate_m_s": right_rate_m_s,
                "projected_right_at_plane_m": projected_right_m,
                "unsafe_projection": (
                    abs(projected_right_m)
                    > hybrid.GATE2_PROJECTED_MISS_THRESHOLD_M
                ),
                "early_brake_only": forward_m > hybrid.GATE2_GOVERNOR_FORWARD_M,
                "latched_before": latched_before,
                "latched_after": hybrid._GATE2_GOVERNOR_LATCHED,
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
            }
        )

    first_early = next(
        (
            row
            for row in rows
            if row["early_brake_only"]
            and not row["latched_before"]
            and row["latched_after"]
        ),
        None,
    )
    first_close = next(
        (
            row
            for row in rows
            if not row["early_brake_only"] and row["latched_after"]
        ),
        None,
    )
    early_rows = [
        row for row in rows if row["early_brake_only"] and row["latched_after"]
    ]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "first_early_activation": first_early,
        "first_close_governor": first_close,
        "activation_lead_s": (
            None
            if first_early is None or first_close is None
            else first_close["elapsed_s"] - first_early["elapsed_s"]
        ),
        "early_brake_samples": len(early_rows),
        "max_early_roll_error": max(
            (
                abs(row["governed_action"][1] - row["recorded_action"][1])
                for row in early_rows
            ),
            default=0.0,
        ),
        "max_early_thrust_yaw_error": max(
            (
                max(
                    abs(
                        row["governed_action"][index]
                        - row["recorded_action"][index]
                    )
                    for index in (2, 3)
                )
                for row in early_rows
            ),
            default=0.0,
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    blockers: list[str] = []
    for report in reports:
        name = Path(report["path"]).stem
        first = report["first_early_activation"]
        if first is None:
            blockers.append(f"{name}:no_early_activation")
        elif not (
            hybrid.GATE2_GOVERNOR_FORWARD_M
            < first["forward_m"]
            <= hybrid.GATE2_EARLY_BRAKE_FORWARD_M
        ):
            blockers.append(f"{name}:activation_outside_early_window")
        if report["first_close_governor"] is None:
            blockers.append(f"{name}:no_close_governor")
        if report["activation_lead_s"] is None or report["activation_lead_s"] <= 0:
            blockers.append(f"{name}:no_measured_activation_lead")
        if report["max_early_roll_error"] != 0.0:
            blockers.append(f"{name}:early_window_changes_roll")
        if report["max_early_thrust_yaw_error"] != 0.0:
            blockers.append(f"{name}:early_window_changes_thrust_or_yaw")

    candidate23 = next(
        (
            report
            for report in reports
            if "bounded_023_attempt_001" in report["path"]
        ),
        None,
    )
    if candidate23 is None:
        blockers.append("candidate023_trace_missing")
    else:
        first = candidate23["first_early_activation"]
        if first is None or first["forward_m"] < 11.0:
            blockers.append("candidate023_does_not_activate_at_measured_far_sample")
        if (
            candidate23["activation_lead_s"] is None
            or candidate23["activation_lead_s"] < 0.5
        ):
            blockers.append("candidate023_activation_lead_below_half_second")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "mechanism": {
            "gate_index": 1,
            "early_brake_forward_m": hybrid.GATE2_EARLY_BRAKE_FORWARD_M,
            "close_roll_forward_m": hybrid.GATE2_GOVERNOR_FORWARD_M,
            "projected_miss_threshold_m": (
                hybrid.GATE2_PROJECTED_MISS_THRESHOLD_M
            ),
            "brake_pitch_norm": hybrid.GATE2_BRAKE_PITCH_NORM,
            "early_window_preserves_roll_thrust_yaw": True,
            "latches_until_official_transition": True,
        },
        "reports": reports,
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
