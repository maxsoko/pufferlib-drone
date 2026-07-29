#!/usr/bin/env python3
"""Verify N175's retirement of the Gate-2 projected-vertical thrust floor."""

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


CLEAN_GATE2_CANDIDATES = {
    "011", "013", "016", "018", "019", "020", "021", "022",
    "024", "025", "026", "027", "028", "030", "031", "034",
}
HISTORICAL_TRIGGER_FAILURES = {"014", "029", "035"}


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
    historical_latched = False
    rows: list[dict] = []
    passthrough_samples = 0
    max_action_error = 0.0
    residual_latch_samples = 0
    hybrid._clear_gate2_state()
    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 1:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"invalid Gate-2 sample {path}:{sample_index}")
        forward_m, projected_down_m = _projected_down(values)
        newly_triggered = (
            not historical_latched
            and float(values[10]) >= 0.5
            and 0.0 < forward_m <= hybrid.GATE2_VERTICAL_FLOOR_FORWARD_M
            and projected_down_m
            < hybrid.GATE2_VERTICAL_PROJECTED_DOWN_TRIGGER_M
        )
        if newly_triggered:
            historical_latched = True
        governed = np.asarray(
            hybrid._gate2_projected_vertical_floor(values, recorded.tolist()),
            dtype=float,
        )
        max_action_error = max(
            max_action_error,
            float(np.max(np.abs(governed - recorded))),
        )
        if hybrid._GATE2_VERTICAL_FLOOR_LATCHED:
            residual_latch_samples += 1
        if historical_latched:
            synthetic_learned = recorded.tolist()
            synthetic_learned[2] = 0.0
            preserved = hybrid._gate2_projected_vertical_floor(
                values, synthetic_learned.copy()
            )
            if preserved == synthetic_learned:
                passthrough_samples += 1
            rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample["elapsed_s"]),
                    "newly_historical_triggered": newly_triggered,
                    "forward_m": forward_m,
                    "projected_down_at_plane_m": projected_down_m,
                    "recorded_action": recorded.tolist(),
                    "retired_action": governed.tolist(),
                    "synthetic_subfloor_action_preserved": (
                        preserved == synthetic_learned
                    ),
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
        "historical_latch_samples": len(rows),
        "synthetic_subfloor_passthrough_samples": passthrough_samples,
        "max_action_error": max_action_error,
        "residual_latch_samples": residual_latch_samples,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate035-poststop", required=True, type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []
    expected = CLEAN_GATE2_CANDIDATES | HISTORICAL_TRIGGER_FAILURES
    for candidate in sorted(expected - by_candidate.keys()):
        blockers.append(f"candidate{candidate}_missing")
    for candidate in sorted(CLEAN_GATE2_CANDIDATES):
        report = by_candidate.get(candidate)
        if report and report["historical_latch_samples"]:
            blockers.append(f"candidate{candidate}_clean_path_historically_latched")
    for candidate in sorted(HISTORICAL_TRIGGER_FAILURES):
        report = by_candidate.get(candidate)
        if report and report["historical_latch_samples"] < 1:
            blockers.append(f"candidate{candidate}_historical_trigger_missing")
    candidate35 = by_candidate.get("035")
    if candidate35:
        if candidate35["historical_latch_samples"] != 10:
            blockers.append("candidate035_historical_latch_count_not_ten")
        if candidate35["synthetic_subfloor_passthrough_samples"] != 10:
            blockers.append("candidate035_subfloor_passthrough_not_ten")
    for report in reports:
        if report["max_action_error"] != 0.0:
            blockers.append(f"candidate{report['candidate']}_action_changed")
        if report["residual_latch_samples"]:
            blockers.append(f"candidate{report['candidate']}_retired_latch_set")

    poststop = json.loads(args.candidate035_poststop.read_text(encoding="utf-8"))
    race = ((poststop.get("latest_telemetry") or {}).get("race_status") or {})
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate035_poststop_not_command_free")
    if int(race.get("active_gate_index", -1)) != 1:
        blockers.append("candidate035_poststop_gate_index_unexpected")
    if int(race.get("last_gate_race_time", -1)) != 5496647834:
        blockers.append("candidate035_poststop_last_gate_time_unexpected")
    if int(race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate035_poststop_finish_unexpected")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "retired": hybrid.GATE2_VERTICAL_FLOOR_RETIRED,
            "historical_max_forward_m": hybrid.GATE2_VERTICAL_FLOOR_FORWARD_M,
            "historical_projected_down_trigger_m": (
                hybrid.GATE2_VERTICAL_PROJECTED_DOWN_TRIGGER_M
            ),
            "historical_normalized_thrust_floor": (
                hybrid.GATE2_VERTICAL_THRUST_FLOOR_NORM
            ),
            "preserve_all_learned_action_channels": True,
            "latch_disabled": True,
        },
        "clean_gate2_candidates": sorted(CLEAN_GATE2_CANDIDATES),
        "historical_trigger_failures": sorted(HISTORICAL_TRIGGER_FAILURES),
        "candidate035_poststop": {
            "path": str(args.candidate035_poststop),
            "sha256": _sha256(args.candidate035_poststop),
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
