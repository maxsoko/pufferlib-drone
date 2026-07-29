#!/usr/bin/env python3
"""Diagnose N273's Gate-4 guard trigger and preregister a bounded successor."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_N273_SHA256 = (
    "6852268b9804d0744658e13640e5fc80ca7d38d8c7e884173158d3d5caa579e1"
)
EXPECTED_N271_SHA256 = (
    "a5d3e09625ae8883f19b111d61c8c691310d361fffba593d6db6ee47ccbe1278"
)
EXPECTED_CONFIG_SHA256 = (
    "9eec139d95b19f2f604db08d44bc65be214347edd540a775dd35b02a8c4dfa08"
)
EXPECTED_N259_SHA256 = (
    "f13d4b8d571d70be16731068c88dd105e81fe7bd00a86eae6f13f81031f00fc0"
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


def load_gate_centers(config_path: Path) -> np.ndarray:
    parser = configparser.ConfigParser()
    if not parser.read(config_path):
        raise FileNotFoundError(config_path)
    env = parser["env"]
    return -np.asarray(
        [
            [
                env.getfloat(f"gate{index}_x"),
                env.getfloat(f"gate{index}_y"),
                env.getfloat(f"gate{index}_z"),
            ]
            for index in range(env.getint("num_gates"))
        ],
        dtype=np.float64,
    )


def interpolate_x_plane_crossing(
    rows: list[dict[str, object]], target_ned_m: np.ndarray
) -> dict[str, object]:
    for previous, current in zip(rows, rows[1:]):
        before = np.asarray(previous["position_ned_m"], dtype=np.float64)
        after = np.asarray(current["position_ned_m"], dtype=np.float64)
        if (before[0] - target_ned_m[0]) * (after[0] - target_ned_m[0]) > 0.0:
            continue
        fraction = float((target_ned_m[0] - before[0]) / (after[0] - before[0]))
        position = before + fraction * (after - before)
        velocity_before = np.asarray(previous["velocity_ned_m_s"], dtype=np.float64)
        velocity_after = np.asarray(current["velocity_ned_m_s"], dtype=np.float64)
        velocity = velocity_before + fraction * (velocity_after - velocity_before)
        elapsed_s = float(previous["elapsed_s"]) + fraction * (
            float(current["elapsed_s"]) - float(previous["elapsed_s"])
        )
        return {
            "elapsed_s": elapsed_s,
            "position_ned_m": position.tolist(),
            "velocity_ned_m_s": velocity.tolist(),
            "aperture_yz_error_m": float(np.linalg.norm(position[1:] - target_ned_m[1:])),
        }
    raise RuntimeError("trace does not cross the configured NED-x gate plane")


def build_report(
    n273_path: Path,
    n271_path: Path,
    config_path: Path,
    runner_path: Path,
) -> dict[str, object]:
    n273_hash = require_sha256(n273_path, EXPECTED_N273_SHA256, "N273")
    n271_hash = require_sha256(n271_path, EXPECTED_N271_SHA256, "N271")
    config_hash = require_sha256(config_path, EXPECTED_CONFIG_SHA256, "config")
    report = json.loads(n273_path.read_text())
    if report.get("accepted"):
        raise RuntimeError("N273 unexpectedly reports accepted")
    if report.get("invalid_reason") != "gate_governor_plane_miss":
        raise RuntimeError("N273 did not stop on the plane-miss guard")
    if int(report.get("official_active_gate_index", -1)) != 3:
        raise RuntimeError("N273 did not reach the Gate-4 phase")
    if report.get("collision") is not None:
        raise RuntimeError("N273 contains a collision")
    if report.get("checkpoint_sha256") != EXPECTED_N259_SHA256:
        raise RuntimeError("N273 checkpoint mismatch")
    telemetry = report["telemetry"]
    if any(
        int(telemetry[field]) != 0
        for field in ("telemetry_dropouts", "malformed_messages", "drain_limit_hits")
    ):
        raise RuntimeError("N273 contains a telemetry fault")

    gates = load_gate_centers(config_path)
    gate_index = 3
    rows = [
        row
        for row in report["trace"]
        if int(row["active_gate_index"]) == gate_index and row.get("governor") is not None
    ]
    crossing = interpolate_x_plane_crossing(rows, gates[gate_index])
    segment_start = gates[gate_index - 1]
    tangent = gates[gate_index] - segment_start
    tangent /= np.linalg.norm(tangent)
    actual_along_speed = float(np.dot(crossing["velocity_ned_m_s"], tangent))

    state = report["latest_telemetry"]
    status = state["race_status"]
    control_start_boot_ms = float(state["attitude_time_boot_ms"]) - 1000.0 * float(
        report["elapsed_wall_s"]
    )
    estimated_crossing_boot_ms = control_start_boot_ms + 1000.0 * float(
        crossing["elapsed_s"]
    )
    status_delta_from_crossing_ms = float(status["sim_boot_time_ms"]) - estimated_crossing_boot_ms
    status_rate_hz = float(telemetry["race_statuses"]) / float(report["elapsed_wall_s"])
    status_period_s = 1.0 / status_rate_hz
    distance_guard_time_s = 1.5 / actual_along_speed
    if crossing["aperture_yz_error_m"] >= 0.45:
        raise RuntimeError("N273 Gate-4 plane crossing was not center-safe")
    if status_delta_from_crossing_ms >= 0.0:
        raise RuntimeError("N273 already observed a post-crossing official status")

    runner_hash = sha256(runner_path)
    return {
        "experiment": "N274",
        "evidence_kind": "source_hashed_gate4_status_ack_diagnosis",
        "sources": {
            "n273": str(n273_path),
            "n273_sha256": n273_hash,
            "n271": str(n271_path),
            "n271_sha256": n271_hash,
            "config": str(config_path),
            "config_sha256": config_hash,
            "corrected_runner": str(runner_path),
            "corrected_runner_sha256": runner_hash,
        },
        "n273_result": {
            "official_active_gate_index": int(report["official_active_gate_index"]),
            "official_gate_times_s": [
                int(item["last_gate_race_time"]) * 1e-9
                for item in report["gate_transitions"]
            ],
            "collision": report["collision"],
            "invalid_reason": report["invalid_reason"],
            "effective_command_hz": float(report["contract"]["effective_command_hz"]),
        },
        "gate4_diagnosis": {
            "configured_aperture_center_ned_m": gates[gate_index].tolist(),
            "interpolated_plane_crossing": crossing,
            "actual_along_speed_at_crossing_m_s": actual_along_speed,
            "legacy_plane_miss_distance_m": 1.5,
            "time_from_plane_to_legacy_distance_at_crossing_speed_s": distance_guard_time_s,
            "observed_race_status_rate_hz": status_rate_hz,
            "observed_race_status_period_s": status_period_s,
            "estimated_control_start_boot_time_ms": control_start_boot_ms,
            "estimated_plane_crossing_boot_time_ms": estimated_crossing_boot_ms,
            "latest_race_status_boot_time_ms": int(status["sim_boot_time_ms"]),
            "latest_status_minus_crossing_ms": status_delta_from_crossing_ms,
            "latest_status_was_post_crossing": False,
            "causal_conclusion": (
                "the distance guard fired before any official status sample could "
                "acknowledge the center-safe Gate-4 plane crossing"
            ),
        },
        "correction": {
            "kind": "fail_closed_post_crossing_status_ack",
            "legacy_distance_m_unchanged": 1.5,
            "status_ack_grace_s": 0.35,
            "abort_conditions": [
                "a post-crossing official status still reports the same gate",
                "350 ms elapse after the crossing without an acknowledging status",
            ],
            "default_behavior_unchanged_when_grace_is_zero": True,
        },
        "preregistered_next_action": {
            "experiment": "N275",
            "tag": "n275_v3391_n259_cross9_xgain2_ack350_gate4_bounded_001",
            "duration_s": 25.0,
            "stop_after_gate_index": 4,
            "governor_parameters": {
                "crossing_speed_m_s": 9.0,
                "maximum_along_speed_m_s": 9.0,
                "slowdown_distance_m": 10.0,
                "cross_track_gain_s_inv": 2.0,
                "maximum_cross_track_correction_m_s": 4.0,
                "plane_miss_abort_m": 1.5,
                "plane_miss_status_ack_grace_s": 0.35,
            },
            "acceptance": (
                "official index 4, finish -1, null collision/invalid reason, one "
                "reset, six gates, clean telemetry/rate, final zero setpoint and disarm"
            ),
            "failure_rule": "reject N275 and forbid an unchanged retry",
        },
        "limitations": [
            "N274 is an offline diagnosis, not proof that Gate 4 scored.",
            "N273 remains rejected and may not be retried unchanged.",
            "Only a bounded official N275 index-4 result can validate the correction.",
        ],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n273",
        type=Path,
        default=Path(
            "/mnt/c/Users/anon/code/pufferlib-drone/.n253-shadow/logs/sitl/"
            "n273_v3391_n259_cross9_xgain2_full_lap_001.json"
        ),
    )
    parser.add_argument(
        "--n271", type=Path, default=root / "logs/sitl/n271_n270_three_trace_governor_sweep.json"
    )
    parser.add_argument(
        "--config", type=Path, default=root / "config/drone_race_vq1_v3391_telemetry.ini"
    )
    parser.add_argument(
        "--runner", type=Path, default=root / "scripts/run_v3391_telemetry_policy.py"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "logs/sitl/n274_n273_plane_miss_ack_diagnosis.json"
    )
    args = parser.parse_args()
    report = build_report(args.n273, args.n271, args.config, args.runner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"experiment": "N274", "output": str(args.output), "next": report["preregistered_next_action"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
