#!/usr/bin/env python3
"""Verify N192's post-N191 Gate-3 terminal safe-level rule on live traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

try:
    import policy_callable_gate3_terminal_level_n191 as policy
    import policy_callable_six_gate_composite as tail
except ModuleNotFoundError:
    from scripts import policy_callable_gate3_terminal_level_n191 as policy
    from scripts import policy_callable_six_gate_composite as tail


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt", path.name)
    if match is None:
        raise ValueError(f"cannot identify candidate from {path}")
    return match.group(1)


def _projection(values: np.ndarray) -> tuple[float, float, float]:
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
    horizon_s = (
        max(forward_m - 2.0, 0.0) / closing_m_s
        if closing_m_s >= 1.0 else 0.0
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = (payload.get("policy_trace") or {}).get("samples") or []
    controller = policy.Gate3TerminalSafeLevel()
    rows = []
    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 2:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"invalid Gate-3 sample {path}:{sample_index}")
        before = controller.snapshot()["active"]
        governed = np.asarray(controller.apply(values, recorded), dtype=float)
        after = controller.snapshot()["active"]
        if before or after:
            forward_m, projected_right_m, projected_down_m = _projection(values)
            rows.append({
                "sample": sample_index,
                "elapsed_s": float(sample["elapsed_s"]),
                "forward_m": forward_m,
                "projected_right_m": projected_right_m,
                "projected_down_m": projected_down_m,
                "latched_before": before,
                "latched_after": after,
                "recorded_action": recorded.tolist(),
                "governed_action": governed.tolist(),
            })
    changed = [row for row in rows if row["governed_action"] != row["recorded_action"]]
    latest = ((payload.get("sitl") or {}).get("latest_telemetry") or {})
    return {
        "candidate": _candidate(path),
        "path": str(path),
        "sha256": _sha256(path),
        "reported_official_active_gate_index": payload.get(
            "official_active_gate_index"
        ),
        "collision_id": latest.get("collision_id"),
        "first_activation": rows[0] if rows else None,
        "active_samples": len(rows),
        "changed_samples": len(changed),
        "max_pitch_thrust_yaw_error": max((
            max(abs(row["governed_action"][index] - row["recorded_action"][index])
                for index in (0, 2, 3)) for row in rows
        ), default=0.0),
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
    by_candidate = {report["candidate"]: report for report in reports}
    blockers = []
    for candidate in ("016", "018", "020", "021", "022", "045", "048"):
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
        elif report["changed_samples"] != 0:
            blockers.append(f"candidate{candidate}_protected_path_changed")
    candidate51 = by_candidate.get("051")
    if candidate51 is None:
        blockers.append("candidate051_missing")
    else:
        if candidate51["changed_samples"] != 3:
            blockers.append("candidate051_change_count_not_three")
        first = candidate51["first_activation"]
        if first is None or not (3.5 <= first["forward_m"] <= 3.8):
            blockers.append("candidate051_activation_range_changed")
        if first is not None and abs(first["projected_right_m"]) > 0.5:
            blockers.append("candidate051_lateral_projection_unsafe")
        if first is not None and abs(first["projected_down_m"]) > 1.0:
            blockers.append("candidate051_vertical_projection_unsafe")
    for report in reports:
        if report["max_pitch_thrust_yaw_error"] != 0.0:
            blockers.append(f"candidate{report['candidate']}_non_roll_change")
        if report["roll_level_violations"] != 0:
            blockers.append(f"candidate{report['candidate']}_not_level")
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "base_policy": "exact N191",
            "official_gate_index": 2,
            "max_forward_m": policy.base.GATE3_TERMINAL_LEVEL_FORWARD_M,
            "max_abs_projected_right_m": (
                policy.base.GATE3_TERMINAL_MAX_ABS_PROJECTED_RIGHT_M
            ),
            "max_abs_projected_down_m": (
                policy.base.GATE3_TERMINAL_MAX_ABS_PROJECTED_DOWN_M
            ),
            "level_roll_norm": 0.0,
            "preserve_pitch_thrust_yaw": True,
            "latch_until_official_transition": True,
        },
        "reports": reports,
        "policy_callable": str(Path(policy.__file__).resolve()),
        "policy_callable_sha256": _sha256(Path(policy.__file__).resolve()),
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
