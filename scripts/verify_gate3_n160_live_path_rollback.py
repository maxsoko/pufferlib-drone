#!/usr/bin/env python3
"""Verify N184's exact N160 Gate-3 live-path rollback on official traces."""

from __future__ import annotations

import argparse
import hashlib
import inspect
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


EXPECTED_CHECKPOINT_SHA256 = (
    "ea03965fca1db98a81a5ab43ada3296752a6f254b9b41886be45248797419760"
)
EXPECTED_CHANGED_SAMPLES = {
    "016": 3,
    "018": 0,
    "020": 0,
    "021": 0,
    "022": 0,
    "044": 2,
}
RETIRED_LIVE_HELPERS = (
    "_gate3_terminal_margin_boost",
    "_gate3_projected_vertical_floor",
    "_gate3_terminal_level",
    "_gate3_late_positive_residual_roll_floor",
    "_gate3_minimum_energy_lateral_controller",
)
N160_CANDIDATE021_TERMINAL_ROLL = (
    (8.639, -1.0),
    (8.275, -1.0),
    (7.653, 0.905),
    (6.418, 0.907),
    (6.423, 0.906),
    (4.154, 0.901),
    (3.072, 0.0),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    if match is None:
        raise ValueError(f"cannot parse candidate number from {path}")
    return match.group(1)


def _n160_action(values: np.ndarray, action: list[float]) -> list[float]:
    action = tail._promoted_gate3_intercept(values, action)
    return hybrid._gate3_close_severe_boost(values, action)


def analyze(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = (report.get("policy_trace") or {}).get("samples") or []
    rows: list[dict] = []
    changed_rows: list[dict] = []
    max_non_roll_error = 0.0
    invalid_roll_outputs = 0
    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 2:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"invalid Gate-3 sample {path}:{sample_index}")
        governed = np.asarray(
            _n160_action(values, recorded.tolist()), dtype=float
        )
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        row = {
            "sample": sample_index,
            "elapsed_s": float(sample["elapsed_s"]),
            "forward_m": forward_m,
            "right_m": right_m,
            "right_rate_m_s": right_rate_m_s,
            "promoted_mode": tail._promoted_gate3_roll_command(values)[0],
            "recorded_action": recorded.tolist(),
            "n160_action": governed.tolist(),
        }
        rows.append(row)
        max_non_roll_error = max(
            max_non_roll_error,
            max(abs(governed[index] - recorded[index]) for index in (0, 2, 3)),
        )
        if not math.isfinite(governed[1]) or not -1.0 <= governed[1] <= 1.0:
            invalid_roll_outputs += 1
        if abs(governed[1] - recorded[1]) > 1e-6:
            changed_rows.append(row)

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
        "gate3_samples": len(rows),
        "changed_samples": len(changed_rows),
        "max_non_roll_error": max_non_roll_error,
        "invalid_roll_outputs": invalid_roll_outputs,
        "changed_rows": changed_rows,
        "rows": rows,
    }


def _candidate021_sequence_blockers(report: dict) -> list[str]:
    blockers: list[str] = []
    rows = report.get("rows") or []
    for expected_forward, expected_roll in N160_CANDIDATE021_TERMINAL_ROLL:
        match = min(rows, key=lambda row: abs(row["forward_m"] - expected_forward))
        if abs(match["forward_m"] - expected_forward) > 0.01:
            blockers.append(f"candidate021_missing_forward_{expected_forward}")
        if abs(match["n160_action"][1] - expected_roll) > 0.001:
            blockers.append(
                f"candidate021_roll_at_{expected_forward}_not_{expected_roll}"
            )
    return blockers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidate044-poststop", required=True, type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []

    infer_source = inspect.getsource(hybrid.infer)
    promoted_call = "tail._promoted_gate3_intercept(values, actions)"
    close_call = "_gate3_close_severe_boost(values, actions)"
    if infer_source.count(promoted_call) != 1:
        blockers.append("live_pipeline_promoted_intercept_not_exactly_once")
    if infer_source.count(close_call) != 1:
        blockers.append("live_pipeline_close_severe_not_exactly_once")
    if infer_source.find(promoted_call) >= infer_source.find(close_call):
        blockers.append("live_pipeline_order_not_n160")
    for name in RETIRED_LIVE_HELPERS:
        if name in infer_source:
            blockers.append(f"retired_helper_live:{name}")
        if not hasattr(hybrid, name):
            blockers.append(f"historical_helper_missing:{name}")
    if hybrid.GATE3_COUNTER_SUPPRESSION_FORWARD_M != 4.0:
        blockers.append("counter_suppression_bound_not_4m")

    checkpoint_sha256 = _sha256(args.checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        blockers.append("prefix_checkpoint_hash_mismatch")

    for candidate, expected in EXPECTED_CHANGED_SAMPLES.items():
        report = by_candidate.get(candidate)
        if report is None:
            blockers.append(f"candidate{candidate}_missing")
            continue
        if report["changed_samples"] != expected:
            blockers.append(
                f"candidate{candidate}_change_count_not_{expected}"
            )
        if report["max_non_roll_error"] != 0.0:
            blockers.append(f"candidate{candidate}_changes_non_roll")
        if report["invalid_roll_outputs"]:
            blockers.append(f"candidate{candidate}_invalid_roll_output")
        if candidate != "044" and int(
            report.get("reported_official_active_gate_index") or 0
        ) < 3:
            blockers.append(f"candidate{candidate}_not_historical_gate3_pass")

    clean021 = by_candidate.get("021") or {}
    if int(clean021.get("reported_official_active_gate_index") or 0) < 3:
        blockers.append("candidate021_did_not_pass_gate3")
    blockers.extend(_candidate021_sequence_blockers(clean021))

    failed044 = by_candidate.get("044") or {}
    changed044 = failed044.get("changed_rows") or []
    expected044 = ((5.714, -1.0), (5.220, -1.0))
    if len(changed044) == len(expected044):
        for row, (forward_m, roll) in zip(changed044, expected044):
            if abs(row["forward_m"] - forward_m) > 0.01:
                blockers.append(f"candidate044_change_not_at_{forward_m}")
            if row["n160_action"][1] != roll:
                blockers.append(f"candidate044_change_not_full_counter_{forward_m}")

    poststop = json.loads(args.candidate044_poststop.read_text(encoding="utf-8"))
    latest = poststop.get("latest_telemetry") or {}
    race = latest.get("race_status") or {}
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate044_poststop_not_command_free")
    if int(latest.get("base_mode", -1)) != 65:
        blockers.append("candidate044_poststop_not_disarmed")
    if int(race.get("active_gate_index", -1)) != 2:
        blockers.append("candidate044_poststop_gate_index_not_2")
    if int(race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate044_poststop_finish_unexpected")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "mechanism": "exact N160 Gate-3 live-path rollback",
            "historical_n160_policy_sha256": (
                "587e4b57f7fc4b28fabeaea2e189a81200835ebb645bb3f06465e885f9572953"
            ),
            "live_pipeline": [
                "tail._promoted_gate3_intercept",
                "_gate3_close_severe_boost",
            ],
            "counter_suppression_forward_m": (
                hybrid.GATE3_COUNTER_SUPPRESSION_FORWARD_M
            ),
            "retired_live_helpers": list(RETIRED_LIVE_HELPERS),
            "historical_helpers_retained": True,
            "preserve_non_roll_channels": True,
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "candidate044_poststop": {
            "path": str(args.candidate044_poststop),
            "sha256": _sha256(args.candidate044_poststop),
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
