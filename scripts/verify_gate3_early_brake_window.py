#!/usr/bin/env python3
"""Verify N165's Gate-3 early-brake-only latch on archived live traces."""

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
    time_to_terminal_s = (
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
            continue
        forward_m, projected_right_m, projected_down_m = _project(values)
        early_before = hybrid._GATE3_EARLY_BRAKE_LATCHED
        governed = np.asarray(
            hybrid._gate3_early_brake_only(values, recorded.tolist()),
            dtype=float,
        )
        early_after = hybrid._GATE3_EARLY_BRAKE_LATCHED
        terminal_before = hybrid._GATE3_TERMINAL_COAST_LATCHED
        hybrid._gate3_terminal_safe_coast(values, governed.tolist())
        terminal_after = hybrid._GATE3_TERMINAL_COAST_LATCHED
        if 0.0 < forward_m <= hybrid.GATE3_EARLY_BRAKE_FORWARD_M:
            rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample["elapsed_s"]),
                    "forward_m": forward_m,
                    "projected_right_m": projected_right_m,
                    "projected_down_m": projected_down_m,
                    "early_before": early_before,
                    "early_after": early_after,
                    "terminal_before": terminal_before,
                    "terminal_after": terminal_after,
                    "recorded_action": recorded.tolist(),
                    "early_governed_action": governed.tolist(),
                }
            )

    first_early = next(
        (row for row in rows if not row["early_before"] and row["early_after"]),
        None,
    )
    first_terminal = next(
        (
            row
            for row in rows
            if not row["terminal_before"] and row["terminal_after"]
        ),
        None,
    )
    early_only = [
        row
        for row in rows
        if row["early_after"] and not row["terminal_after"]
    ]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "official_active_gate_index": report.get("official_active_gate_index"),
        "collision_id": (
            ((report.get("sitl") or {}).get("latest_telemetry") or {}).get(
                "collision_id"
            )
        ),
        "first_early_activation": first_early,
        "first_terminal_activation": first_terminal,
        "early_to_terminal_lead_s": (
            None
            if first_early is None or first_terminal is None
            else first_terminal["elapsed_s"] - first_early["elapsed_s"]
        ),
        "early_only_samples": len(early_only),
        "max_early_roll_thrust_yaw_error": max(
            (
                max(
                    abs(
                        row["early_governed_action"][index]
                        - row["recorded_action"][index]
                    )
                    for index in (1, 2, 3)
                )
                for row in early_only
            ),
            default=0.0,
        ),
        "early_pitch_violations": sum(
            row["early_governed_action"][0]
            > hybrid.GATE3_EARLY_BRAKE_PITCH_NORM + 1e-9
            for row in early_only
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
        first = report["first_early_activation"]
        if first is None:
            blockers.append(f"{name}:no_early_activation")
        elif not (
            hybrid.GATE3_TERMINAL_COAST_FORWARD_M
            < first["forward_m"]
            <= hybrid.GATE3_EARLY_BRAKE_FORWARD_M
        ):
            blockers.append(f"{name}:activation_outside_early_window")
        if report["early_only_samples"] < 1:
            blockers.append(f"{name}:no_early_only_samples")
        if report["max_early_roll_thrust_yaw_error"] != 0.0:
            blockers.append(f"{name}:changes_roll_thrust_or_yaw")
        if report["early_pitch_violations"] != 0:
            blockers.append(f"{name}:brake_pitch_violation")

    for candidate in ("024", "025"):
        report = next(
            (
                item
                for item in reports
                if f"bounded_{candidate}_attempt_001" in item["path"]
            ),
            None,
        )
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif (
            report["early_to_terminal_lead_s"] is None
            or report["early_to_terminal_lead_s"] < 0.5
        ):
            blockers.append(f"candidate{candidate}_lead_below_half_second")

    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "mechanism": {
            "official_gate_index": 2,
            "early_brake_forward_m": hybrid.GATE3_EARLY_BRAKE_FORWARD_M,
            "unsafe_projected_right_m": (
                hybrid.GATE3_EARLY_BRAKE_MAX_ABS_PROJECTED_RIGHT_M
            ),
            "unsafe_projected_down_m": (
                hybrid.GATE3_EARLY_BRAKE_MAX_ABS_PROJECTED_DOWN_M
            ),
            "brake_pitch_norm": hybrid.GATE3_EARLY_BRAKE_PITCH_NORM,
            "preserve_roll_thrust_yaw_until_terminal_coast": True,
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
