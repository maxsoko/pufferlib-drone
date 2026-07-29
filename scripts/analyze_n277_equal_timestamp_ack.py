#!/usr/bin/env python3
"""Source-lock N277's equal-millisecond false acknowledgment diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_N277_SHA256 = (
    "9e5e35f5142036c3aef1b58f208c36084650f75d11538e30e8e8d5857d52e690"
)
EXPECTED_N274_SHA256 = (
    "08a16334973d13ca10caa84732d3b60937d4bdf300cf69804fe6c5dfb4d01678"
)
EXPECTED_CONFIG_SHA256 = (
    "9eec139d95b19f2f604db08d44bc65be214347edd540a775dd35b02a8c4dfa08"
)
EXPECTED_REJECTED_RUNNER_SHA256 = (
    "643cdec893c511d59a8a5a9b54bde2bbab1d543a46b7d977ad92867a01e09505"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected}, got {observed}")
    return observed


def interpolate_gate_plane(
    rows: list[dict[str, object]], gate_center_ned_m: np.ndarray
) -> dict[str, object]:
    for previous, current in zip(rows, rows[1:]):
        before = np.asarray(previous["position_ned_m"], dtype=np.float64)
        after = np.asarray(current["position_ned_m"], dtype=np.float64)
        if (before[0] - gate_center_ned_m[0]) * (
            after[0] - gate_center_ned_m[0]
        ) > 0.0:
            continue
        fraction = float(
            (gate_center_ned_m[0] - before[0]) / (after[0] - before[0])
        )
        position = before + fraction * (after - before)
        elapsed_s = float(previous["elapsed_s"]) + fraction * (
            float(current["elapsed_s"]) - float(previous["elapsed_s"])
        )
        return {
            "elapsed_s": elapsed_s,
            "position_ned_m": position.tolist(),
            "absolute_y_error_m": abs(float(position[1] - gate_center_ned_m[1])),
            "absolute_z_error_m": abs(float(position[2] - gate_center_ned_m[2])),
        }
    raise RuntimeError("N277 trace does not cross the Gate-4 NED-x plane")


def build_report(
    n277_path: Path,
    n274_path: Path,
    config_path: Path,
    corrected_runner_path: Path,
) -> dict[str, object]:
    n277_hash = require_sha256(n277_path, EXPECTED_N277_SHA256, "N277")
    n274_hash = require_sha256(n274_path, EXPECTED_N274_SHA256, "N274")
    config_hash = require_sha256(config_path, EXPECTED_CONFIG_SHA256, "config")
    n277 = json.loads(n277_path.read_text())
    n274 = json.loads(n274_path.read_text())
    rejected_runner_hash = n274["sources"]["corrected_runner_sha256"]
    if rejected_runner_hash != EXPECTED_REJECTED_RUNNER_SHA256:
        raise RuntimeError("N274 does not source-lock the N277 runner")
    if n277.get("invalid_reason") != "gate_governor_plane_miss":
        raise RuntimeError("N277 did not fail on the plane-miss guard")
    if int(n277.get("official_active_gate_index", -1)) != 3:
        raise RuntimeError("N277 did not stop in the Gate-4 phase")
    if n277.get("collision") is not None:
        raise RuntimeError("N277 contains a collision")
    abort = n277["plane_miss_abort_diagnostic"]
    crossing_boot_ms = int(abort["crossing_boot_time_ms"])
    status_boot_ms = int(abort["race_status_boot_time_ms"])
    current_boot_ms = int(abort["current_boot_time_ms"])
    if abort["reason"] != "post_crossing_status_still_same_gate":
        raise RuntimeError("N277 did not report a same-gate acknowledgment")
    if status_boot_ms != crossing_boot_ms:
        raise RuntimeError("N277 does not contain the equal-millisecond case")
    grace_ms = round(
        float(n277["contract"]["governor_plane_miss_status_ack_grace_s"]) * 1000.0
    )
    elapsed_ms = current_boot_ms - crossing_boot_ms
    if elapsed_ms >= grace_ms:
        raise RuntimeError("N277 had already exhausted the acknowledgment grace")

    gate_index = 3
    track_gate = n277["latest_telemetry"]["track_gates"][gate_index]
    gate_center = np.asarray(
        [
            track_gate["position_ned_x"],
            track_gate["position_ned_y"],
            track_gate["position_ned_z"],
        ],
        dtype=np.float64,
    )
    rows = [
        row
        for row in n277["trace"]
        if int(row["active_gate_index"]) == gate_index and row.get("governor")
    ]
    crossing = interpolate_gate_plane(rows, gate_center)
    half_width_m = float(track_gate["width_m"]) / 2.0
    half_height_m = float(track_gate["height_m"]) / 2.0
    inside_aperture = (
        crossing["absolute_y_error_m"] < half_width_m
        and crossing["absolute_z_error_m"] < half_height_m
    )
    if not inside_aperture:
        raise RuntimeError("N277 vehicle center did not cross inside the Gate-4 aperture")

    return {
        "experiment": "N278",
        "evidence_kind": "source_hashed_equal_millisecond_ack_diagnosis",
        "sources": {
            "n277": str(n277_path),
            "n277_sha256": n277_hash,
            "n274": str(n274_path),
            "n274_sha256": n274_hash,
            "config": str(config_path),
            "config_sha256": config_hash,
            "rejected_runner_sha256": rejected_runner_hash,
            "corrected_runner": str(corrected_runner_path),
            "corrected_runner_sha256": sha256(corrected_runner_path),
        },
        "n277_result": {
            "official_active_gate_index": int(n277["official_active_gate_index"]),
            "official_gate_times_s": [
                int(item["last_gate_race_time"]) * 1e-9
                for item in n277["gate_transitions"]
            ],
            "collision": n277["collision"],
            "invalid_reason": n277["invalid_reason"],
            "effective_command_hz": float(n277["contract"]["effective_command_hz"]),
        },
        "equal_timestamp_diagnosis": {
            "crossing_boot_time_ms": crossing_boot_ms,
            "race_status_boot_time_ms": status_boot_ms,
            "current_boot_time_ms": current_boot_ms,
            "elapsed_since_crossing_ms": elapsed_ms,
            "ack_grace_ms": grace_ms,
            "strictly_post_crossing_status_observed": status_boot_ms > crossing_boot_ms,
            "interpolated_gate4_plane_crossing": crossing,
            "gate_half_width_m": half_width_m,
            "gate_half_height_m": half_height_m,
            "vehicle_center_inside_aperture": inside_aperture,
            "causal_conclusion": (
                "the rejected runner treated an equal-resolution timestamp as a "
                "post-crossing sample even though the 350 ms grace remained"
            ),
        },
        "correction": {
            "kind": "strictly_newer_status_timestamp_required",
            "comparison_before": "race_status_boot_time_ms >= crossing_boot_time_ms",
            "comparison_after": "race_status_boot_time_ms > crossing_boot_time_ms",
            "distance_and_timeout_guards_unchanged": True,
        },
        "limitations": [
            "N277 remains rejected and is not retried unchanged.",
            "This diagnosis proves a guard-timestamp defect, not an official Gate-4 pass.",
            "A new bounded run must validate the strictly-newer correction before another full lap.",
        ],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n277",
        type=Path,
        default=Path(
            "/mnt/c/Users/anon/code/pufferlib-drone/.n253-shadow/logs/sitl/"
            "n277_v3391_n259_cross9_xgain2_ack350_full_lap_confirm_002.json"
        ),
    )
    parser.add_argument(
        "--n274",
        type=Path,
        default=root / "logs/sitl/n274_n273_plane_miss_ack_diagnosis.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config/drone_race_vq1_v3391_telemetry.ini",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=root / "scripts/run_v3391_telemetry_policy.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "logs/sitl/n278_n277_equal_timestamp_ack_diagnosis.json",
    )
    args = parser.parse_args()
    report = build_report(args.n277, args.n274, args.config, args.runner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"experiment": "N278", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
