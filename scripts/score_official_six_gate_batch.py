#!/usr/bin/env python3
"""Score repeated official six-gate policy attempts lexicographically.

The scorer is deliberately independent of the live launcher.  It consumes a
small manifest that points at the immutable validation and command-free
post-stop JSON files for each attempt.  A valid six-gate finish dominates every
partial result; otherwise robust official gate progress and collision-free
operation dominate elapsed time.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _load(path: Path) -> dict:
    # Windows PowerShell 5 writes UTF-8 manifests with a BOM.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _number(value, default: float = -1.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _attempt_payload(summary: dict) -> dict:
    attempts = summary.get("attempts") or []
    return attempts[-1] if attempts else summary


def score_attempt(record: dict, *, target_gate_count: int) -> dict:
    summary_path = Path(record["summary_path"])
    summary = _load(summary_path)
    attempt = _attempt_payload(summary)
    smoke = attempt.get("smoke") or {}
    poststop_path = Path(record["poststop_path"])
    poststop = _load(poststop_path) if poststop_path.exists() else {}
    post_state = poststop.get("latest_telemetry") or {}
    post_race = post_state.get("race_status") or {}

    in_run_index = max(
        _integer(attempt.get("official_active_gate_index")),
        _integer(smoke.get("official_active_gate_index")),
    )
    poststop_index = _integer(post_race.get("active_gate_index"), -1)
    official_gate_count = max(in_run_index, poststop_index, 0)
    finish_ns = max(
        _integer(attempt.get("official_race_finish_time_ns"), -1),
        _integer(smoke.get("official_race_finish_time_ns"), -1),
        _integer(post_race.get("race_finish_time_ns"), -1),
    )
    crash = smoke.get("crash_detected") is True
    invalid = smoke.get("invalid_run") is True
    accepted = smoke.get("acceptance_passed") is True
    child_exit_code = _integer(attempt.get("exit_code"), 1)
    valid_finish = bool(
        child_exit_code == 0
        and accepted
        and not crash
        and not invalid
        and official_gate_count >= target_gate_count
        and finish_ns >= 0
    )
    elapsed_s = _number(
        smoke.get("completion_time_s"),
        _number((smoke.get("sitl") or {}).get("duration_s"),
                _number(attempt.get("elapsed_s"), math.inf)),
    )
    if valid_finish and finish_ns >= 0:
        elapsed_s = finish_ns / 1.0e9

    return {
        "variant": str(record["variant"]),
        "tag": str(record.get("tag") or ""),
        "summary_path": str(summary_path),
        "poststop_path": str(poststop_path),
        "child_exit_code": child_exit_code,
        "accepted": accepted,
        "crash_detected": crash,
        "invalid_run": invalid,
        "official_gate_count": official_gate_count,
        "in_run_official_gate_count": in_run_index,
        "poststop_official_gate_count": poststop_index,
        "race_finish_time_ns": finish_ns,
        "elapsed_s": elapsed_s,
        "valid_finish": valid_finish,
    }


def _median(values: list[float], default: float) -> float:
    return float(statistics.median(values)) if values else default


def score_variant(variant: str, attempts: list[dict]) -> dict:
    gates = [int(item["official_gate_count"]) for item in attempts]
    valid_times = [
        float(item["elapsed_s"])
        for item in attempts
        if item["valid_finish"] and math.isfinite(float(item["elapsed_s"]))
    ]
    all_times = [
        float(item["elapsed_s"])
        for item in attempts
        if math.isfinite(float(item["elapsed_s"]))
    ]
    valid_finishes = sum(bool(item["valid_finish"]) for item in attempts)
    collision_free = sum(not item["crash_detected"] for item in attempts)
    min_gates = min(gates, default=0)
    median_gates = _median([float(value) for value in gates], 0.0)
    time_tiebreak_s = _median(valid_times, _median(all_times, math.inf))
    # JSON-friendly ordered tuple. Ranking compares every field descending,
    # so elapsed time is negated only at the final tie break.
    ranking_key = [
        valid_finishes,
        min_gates,
        median_gates,
        collision_free,
        -sum(bool(item["crash_detected"]) for item in attempts),
        -time_tiebreak_s,
    ]
    return {
        "variant": variant,
        "attempt_count": len(attempts),
        "valid_finishes": valid_finishes,
        "valid_finish_rate": valid_finishes / len(attempts) if attempts else 0.0,
        "minimum_official_gate_count": min_gates,
        "median_official_gate_count": median_gates,
        "maximum_official_gate_count": max(gates, default=0),
        "collision_free_attempts": collision_free,
        "collision_free_rate": collision_free / len(attempts) if attempts else 0.0,
        "median_valid_lap_time_s": _median(valid_times, -1.0),
        "time_tiebreak_s": time_tiebreak_s,
        "ranking_key": ranking_key,
        "attempts": attempts,
    }


def score_manifest(manifest: dict) -> dict:
    target = _integer(manifest.get("target_gate_count"), 6)
    if target != 6:
        raise ValueError("official competitive batch target_gate_count must be 6")
    records = manifest.get("attempts") or []
    attempts = [score_attempt(record, target_gate_count=target) for record in records]
    grouped: dict[str, list[dict]] = {}
    for attempt in attempts:
        grouped.setdefault(attempt["variant"], []).append(attempt)
    variants = [score_variant(name, values) for name, values in grouped.items()]
    variants.sort(key=lambda item: tuple(item["ranking_key"]), reverse=True)
    for rank, variant in enumerate(variants, 1):
        variant["rank"] = rank
    return {
        "schema_version": 1,
        "target_gate_count": target,
        "objective_order": [
            "valid_finish_count",
            "minimum_official_gate_count",
            "median_official_gate_count",
            "collision_free_count",
            "collision_count",
            "lap_or_attempt_time",
        ],
        "attempt_count": len(attempts),
        "leader": variants[0]["variant"] if variants else None,
        "variants": variants,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = score_manifest(_load(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
