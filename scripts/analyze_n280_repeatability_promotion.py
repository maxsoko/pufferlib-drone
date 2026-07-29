#!/usr/bin/env python3
"""Validate and aggregate N280--N282 as the promoted near-24-second controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


EXPECTED_RUNS = {
    "N280": (
        "n280_v3391_n259_cross9_xgain2_ack350_strict_full_lap_001.json",
        "5b926383063ffe8cf2f881a634925853a73c5fbb81cce4c93418e6c95161abec",
    ),
    "N281": (
        "n281_v3391_n259_cross9_xgain2_ack350_strict_full_lap_confirm_002.json",
        "0fd5a7d4b78ebdcc11562443c11a70b6593769bd15ecc5d9d9b4bc9a29225250",
    ),
    "N282": (
        "n282_v3391_n259_cross9_xgain2_ack350_strict_full_lap_confirm_003.json",
        "27fbf3ba00c62553468f95d515a1144b0aeb0a31555cc5177fcce6c7e9e3fb36",
    ),
}
EXPECTED_N270_SHA256 = (
    "ff466a254fc829dd764484783b20bcf1fdca0c42f96fbba6045f937da2e0a247"
)
EXPECTED_N278_SHA256 = (
    "232c9d190def9ff9216229cd5c448816d34aebd230fe223875e8bd38c39edcce"
)
EXPECTED_N279_SHA256 = (
    "e068a68501ee5fc9a73f9a86b43a222b8c3f3b14e30ae2b910976bb62256ea47"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "f13d4b8d571d70be16731068c88dd105e81fe7bd00a86eae6f13f81031f00fc0"
)
EXPECTED_RUNNER_SHA256 = (
    "de4995a19c5f3c82f0de540b4b07ce810ffae49ad9fd4308befbcea3cfcb34ea"
)
EXPECTED_CONFIG_SHA256 = (
    "9eec139d95b19f2f604db08d44bc65be214347edd540a775dd35b02a8c4dfa08"
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


def validate_run(path: Path, expected_hash: str, experiment: str) -> dict[str, object]:
    artifact_hash = require_sha256(path, expected_hash, experiment)
    report = json.loads(path.read_text())
    if not report.get("accepted") or report.get("acceptance_kind") != "official_six_gate_finish":
        raise RuntimeError(f"{experiment} is not an accepted official finish")
    if int(report.get("official_active_gate_index", -1)) != 6:
        raise RuntimeError(f"{experiment} did not pass all six gates")
    finish_ns = int(report.get("official_race_finish_time_ns", -1))
    if finish_ns < 0 or finish_ns > 24_500_000_000:
        raise RuntimeError(f"{experiment} did not satisfy the 24.5-second gate")
    if report.get("collision") is not None or report.get("invalid_reason") is not None:
        raise RuntimeError(f"{experiment} has a collision or invalid reason")
    if report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"{experiment} checkpoint mismatch")
    transitions = report.get("gate_transitions", [])
    if len(transitions) != 6 or int(transitions[-1]["to_gate_index"]) != 6:
        raise RuntimeError(f"{experiment} gate-transition proof is incomplete")

    contract = report["contract"]
    required_contract = {
        "camera_used": False,
        "command_hz": 60.0,
        "heartbeat_hz": 2.0,
        "stop_after_gate_index": 6,
        "gate_governor_enabled": True,
        "gate_governor_source": "public_track_transfer_and_local_position",
        "governor_plane_miss_abort_m": 1.5,
        "governor_plane_miss_status_ack_grace_s": 0.35,
        "hidden_simulator_state_used": False,
        "exit_disarm_sent": True,
        "exit_stop_setpoints_sent": 1,
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"{experiment} contract mismatch for {key}")
    rate_hz = float(contract["effective_command_hz"])
    if not 50.0 <= rate_hz < 100.0:
        raise RuntimeError(f"{experiment} command rate is out of contract")
    telemetry = report["telemetry"]
    faults = {
        key: int(telemetry[key])
        for key in ("telemetry_dropouts", "malformed_messages", "drain_limit_hits")
    }
    if any(faults.values()):
        raise RuntimeError(f"{experiment} contains a telemetry fault")
    reset = report["reset"]
    if int(reset["reset_commands_sent"]) != 1 or int(reset["track_gate_count"]) != 6:
        raise RuntimeError(f"{experiment} reset/track proof mismatch")
    if int(reset["pre_reset_sim_boot_time_ms"]) - int(reset["post_reset_sim_boot_time_ms"]) <= 1000:
        raise RuntimeError(f"{experiment} reset rollback was not observed")
    if int(reset["race_start_boot_time_ms"]) < 0:
        raise RuntimeError(f"{experiment} race start is invalid")
    transfer = report["gate_transfer_comparison"]
    if int(transfer["count"]) != 6 or float(transfer["max_position_error_m"]) > 1e-6:
        raise RuntimeError(f"{experiment} gate transfer mismatch")

    return {
        "experiment": experiment,
        "artifact": str(path),
        "artifact_sha256": artifact_hash,
        "official_active_gate_index": 6,
        "official_finish_time_s": finish_ns * 1e-9,
        "official_gate_times_s": [
            int(item["last_gate_race_time"]) * 1e-9 for item in transitions
        ],
        "collision": None,
        "invalid_reason": None,
        "effective_command_hz": rate_hz,
        **faults,
        "plane_miss_ack_deferrals": int(contract["plane_miss_ack_deferrals"]),
        "maximum_race_status_lag_ms": int(contract["maximum_race_status_lag_ms"]),
        "exit_stop_setpoints_sent": int(contract["exit_stop_setpoints_sent"]),
        "exit_disarm_sent": bool(contract["exit_disarm_sent"]),
    }


def build_report(
    evidence_dir: Path,
    n270_path: Path,
    n278_path: Path,
    n279_path: Path,
    checkpoint_path: Path,
    runner_path: Path,
    config_path: Path,
) -> dict[str, object]:
    n270_hash = require_sha256(n270_path, EXPECTED_N270_SHA256, "N270")
    n278_hash = require_sha256(n278_path, EXPECTED_N278_SHA256, "N278")
    n279_hash = require_sha256(n279_path, EXPECTED_N279_SHA256, "N279")
    checkpoint_hash = require_sha256(
        checkpoint_path, EXPECTED_CHECKPOINT_SHA256, "checkpoint"
    )
    runner_hash = require_sha256(runner_path, EXPECTED_RUNNER_SHA256, "runner")
    config_hash = require_sha256(config_path, EXPECTED_CONFIG_SHA256, "config")
    laps = [
        validate_run(evidence_dir / filename, expected_hash, experiment)
        for experiment, (filename, expected_hash) in EXPECTED_RUNS.items()
    ]
    finish_times = [float(lap["official_finish_time_s"]) for lap in laps]
    n270 = json.loads(n270_path.read_text())
    old_times = [
        float(lap["official_finish_time_s"]) for lap in n270["official_full_laps"]
    ]
    best = min(finish_times)
    mean = statistics.mean(finish_times)
    worst = max(finish_times)
    old_best = min(old_times)
    old_mean = statistics.mean(old_times)
    if len(laps) != 3 or worst > 24.5:
        raise RuntimeError("repeatability promotion gate did not pass")
    return {
        "experiment": "N283",
        "evidence_kind": "official_near24_three_lap_repeatability_promotion",
        "promotion": {
            "promoted": True,
            "promoted_controller": "N283",
            "supersedes": "N270",
            "official_valid_laps": "3/3",
            "competitive_finish_threshold_s": 24.5,
            "best_official_finish_time_s": best,
            "mean_official_finish_time_s": mean,
            "worst_official_finish_time_s": worst,
            "finish_time_range_s": worst - best,
        },
        "comparison_to_n270": {
            "n270_best_time_s": old_best,
            "n270_mean_time_s": old_mean,
            "best_time_saved_s": old_best - best,
            "mean_time_saved_s": old_mean - mean,
            "every_promoted_lap_faster_than_n270_best": all(
                finish < old_best for finish in finish_times
            ),
        },
        "frozen_artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "runner": str(runner_path),
            "runner_sha256": runner_hash,
            "config": str(config_path),
            "config_sha256": config_hash,
        },
        "governor": {
            "crossing_speed_m_s": 9.0,
            "maximum_along_speed_m_s": 9.0,
            "slowdown_distance_m": 10.0,
            "cross_track_gain_s_inv": 2.0,
            "maximum_cross_track_correction_m_s": 4.0,
            "plane_miss_abort_m": 1.5,
            "plane_miss_status_ack_grace_s": 0.35,
            "post_crossing_status_comparison": "strictly newer timestamp",
        },
        "precursor_evidence": {
            "n270": str(n270_path),
            "n270_sha256": n270_hash,
            "n278_guard_diagnosis": str(n278_path),
            "n278_guard_diagnosis_sha256": n278_hash,
            "n279_bounded_gate4": str(n279_path),
            "n279_bounded_gate4_sha256": n279_hash,
        },
        "official_full_laps": laps,
        "decision": {
            "no_additional_confirmation_required": True,
            "further_optimization_requires_new_offline_evidence_and_unique_tags": True,
            "n259_checkpoint_remains_immutable": True,
        },
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("/mnt/c/Users/anon/code/pufferlib-drone/.n253-shadow/logs/sitl"),
    )
    parser.add_argument(
        "--n270", type=Path, default=root / "logs/sitl/n270_v3391_cross7p5_repeatability_promotion.json"
    )
    parser.add_argument(
        "--n278", type=Path, default=root / "logs/sitl/n278_n277_equal_timestamp_ack_diagnosis.json"
    )
    parser.add_argument(
        "--n279",
        type=Path,
        default=Path(
            "/mnt/c/Users/anon/code/pufferlib-drone/.n253-shadow/logs/sitl/"
            "n279_v3391_n259_cross9_xgain2_ack350_strict_gate4_bounded_001.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "checkpoints/drone_race_vq1_v3391_telemetry/n259_aperture_fullphase_bc_6gate.bin",
    )
    parser.add_argument(
        "--runner", type=Path, default=root / "scripts/run_v3391_telemetry_policy.py"
    )
    parser.add_argument(
        "--config", type=Path, default=root / "config/drone_race_vq1_v3391_telemetry.ini"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "logs/sitl/n283_v3391_cross9_strict_repeatability_promotion.json"
    )
    args = parser.parse_args()
    report = build_report(
        args.evidence_dir,
        args.n270,
        args.n278,
        args.n279,
        args.checkpoint,
        args.runner,
        args.config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"experiment": "N283", "promotion": report["promotion"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
