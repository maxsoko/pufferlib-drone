#!/usr/bin/env python3
"""Verify that scheduler starvation, not the corrected prefix ABI, caused N148."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_trace_sample(report: dict) -> dict:
    samples = (report.get("policy_trace") or {}).get("samples") or []
    if not samples:
        raise ValueError("report has no policy trace samples")
    return min(samples, key=lambda sample: float(sample.get("elapsed_s") or 0.0))


def motion_update_rate(report: dict) -> float:
    trace = report.get("policy_trace") or {}
    motion = trace.get("motion_filter") or {}
    updates = int(motion.get("accepted_samples") or 0) + int(
        motion.get("rejected_samples") or 0
    )
    duration_s = float((report.get("sitl") or {}).get("duration_s") or 0.0)
    return updates / duration_s if duration_s > 0.0 else 0.0


def inference_rate(report: dict) -> float:
    trace = report.get("policy_trace") or {}
    ticks = int(trace.get("inference_ticks") or 0)
    duration_s = float((report.get("sitl") or {}).get("duration_s") or 0.0)
    return ticks / duration_s if duration_s > 0.0 else 0.0


def scheduler_contract(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    required = [
        'hot_deadline_s = 0.001 if os.name == "nt" else 0.0',
        "sleep_s = min(max(0.0, wait_s - hot_deadline_s), 0.01)",
        "self._stop.wait(sleep_s)",
    ]
    missing = [fragment for fragment in required if fragment not in source]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "passed": not missing,
        "missing_fragments": missing,
        "windows_hot_deadline_s": 0.001,
        "behavior": "release GIL for most of each publisher period",
    }


def verify(
    historical_reports: list[Path], failed_candidate: Path, runner: Path
) -> dict:
    blockers: list[str] = []
    sources = []
    historical_motion_hz = []
    gate12_passes = 0
    gate3_passes = 0
    for path in historical_reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        official_index = int(report.get("official_active_gate_index") or 0)
        rate = motion_update_rate(report)
        gate12_passes += int(official_index >= 2)
        gate3_passes += int(official_index >= 3)
        historical_motion_hz.append(rate)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "official_active_gate_index": official_index,
                "motion_update_hz": rate,
            }
        )
    if len(historical_reports) != 10:
        blockers.append(f"historical_report_count:{len(historical_reports)}!=10")
    if gate12_passes != len(historical_reports):
        blockers.append(f"historical_gate12:{gate12_passes}!={len(historical_reports)}")
    if gate3_passes < 8:
        blockers.append(f"historical_gate3:{gate3_passes}<8")

    failed = json.loads(failed_candidate.read_text(encoding="utf-8"))
    # Attempt 010 and historical attempt 010 both began with an initially
    # hidden Gate 1, making their fresh recurrent outputs directly comparable.
    historical_first = first_trace_sample(
        json.loads(historical_reports[-1].read_text(encoding="utf-8"))
    )
    failed_first = first_trace_sample(failed)
    first_action_error = max(
        abs(float(a) - float(b))
        for a, b in zip(
            historical_first.get("normalized_action") or [],
            failed_first.get("normalized_action") or [],
        )
    )
    failed_motion_hz = motion_update_rate(failed)
    failed_inference_hz = inference_rate(failed)
    if first_action_error > 0.001:
        blockers.append(f"fresh_action_mismatch:{first_action_error}")
    historical_min_hz = min(historical_motion_hz) if historical_motion_hz else 0.0
    if historical_min_hz < 10.0:
        blockers.append(f"historical_motion_rate:{historical_min_hz}<10")
    if failed_motion_hz >= 10.0:
        blockers.append(f"failed_motion_not_starved:{failed_motion_hz}")
    if failed_inference_hz >= 10.0:
        blockers.append(f"failed_inference_not_starved:{failed_inference_hz}")
    if int(failed.get("official_active_gate_index") or 0) != 0:
        blockers.append("failed_candidate_not_gate1")
    collisions = int(
        (((failed.get("sitl") or {}).get("telemetry") or {}).get("collisions") or 0)
    )
    if collisions != 1:
        blockers.append(f"failed_candidate_collisions:{collisions}!=1")
    contract = scheduler_contract(runner)
    if not contract["passed"]:
        blockers.append("scheduler_contract")

    return {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "required_live_shadow_inference_hz": 10.0,
            "historical_motion_update_floor_hz": 10.0,
            "fresh_action_atol": 0.001,
            "change_scope": "publisher wait scheduling only",
        },
        "historical": {
            "gate12_passes": gate12_passes,
            "gate3_passes": gate3_passes,
            "motion_update_hz_min": historical_min_hz,
            "motion_update_hz_median": (
                statistics.median(historical_motion_hz)
                if historical_motion_hz
                else 0.0
            ),
            "motion_update_hz_max": (
                max(historical_motion_hz) if historical_motion_hz else 0.0
            ),
            "fresh_action": historical_first.get("normalized_action"),
        },
        "failed_candidate": {
            "path": str(failed_candidate),
            "sha256": sha256_file(failed_candidate),
            "official_active_gate_index": int(
                failed.get("official_active_gate_index") or 0
            ),
            "collisions": collisions,
            "motion_update_hz": failed_motion_hz,
            "inference_hz": failed_inference_hz,
            "fresh_action": failed_first.get("normalized_action"),
            "fresh_action_max_error": first_action_error,
        },
        "scheduler": contract,
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-reports", nargs="+", type=Path, required=True)
    parser.add_argument("--failed-candidate", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--json-path", type=Path)
    args = parser.parse_args()
    result = verify(args.historical_reports, args.failed_candidate, args.runner)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
