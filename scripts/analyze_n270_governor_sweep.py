#!/usr/bin/env python3
"""Stress-test the next telemetry-governor rung against N267--N269.

N270 promoted three identical official laps.  This analyzer treats each live
trace as an independent first-order vehicle model, anchors every replay to that
trace's official gate times, and admits a successor only when the same setting
meets the time and crossing-error limits in all three models.  The full replay
residual and the extrapolation beyond observed commands remain explicit; this
is an offline admission test, never an official-time claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from scripts import analyze_n264_governor_sweep as replay_model
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import analyze_n264_governor_sweep as replay_model


EXPECTED_N270_SHA256 = (
    "ff466a254fc829dd764484783b20bcf1fdca0c42f96fbba6045f937da2e0a247"
)
EXPECTED_N259_SHA256 = replay_model.EXPECTED_N259_SHA256
EXPECTED_RUNNER_SHA256 = (
    "27e202604e4e5834b0d726c851cad8cebb19dff826be7ffac50e8a2bee7f4b1e"
)
EXPECTED_CONFIG_SHA256 = (
    "9eec139d95b19f2f604db08d44bc65be214347edd540a775dd35b02a8c4dfa08"
)
EXPECTED_LIVE_SHA256 = {
    "N267": "212c5af1b10909b3a694c60203a3649de84ba116c3911fd7d587eff76a1c8993",
    "N268": "51d36c1154133606b8bdefdf95818a4972c53c17efc8922a87060a68e8b950db",
    "N269": "9be363aa9c96e1623fceea287314f1241ee34269facf9ba8548b3076f319cf73",
}
BASELINE = replay_model.GovernorParameters(
    crossing_speed_m_s=7.5,
    maximum_along_speed_m_s=8.0,
    slowdown_distance_m=10.0,
    cross_track_gain_s_inv=1.5,
    maximum_cross_track_correction_m_s=4.0,
)
TARGET_WORST_CASE_FINISH_S = 24.25
CROSSING_ERROR_LIMIT_M = 0.45


@dataclasses.dataclass(frozen=True)
class LiveModel:
    experiment: str
    path: Path
    artifact_sha256: str
    report: dict[str, object]
    response_axes: tuple[replay_model.ResponseAxis, ...]
    official_gate_times_s: np.ndarray
    baseline_replay: tuple[replay_model.ReplayGate, ...]
    official_minus_raw_residual_s: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_deployment_artifact(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {observed}"
        )
    return observed


def validate_live_report(report: dict[str, object], experiment: str) -> None:
    if not report.get("accepted"):
        raise RuntimeError(f"{experiment} is not accepted")
    if int(report.get("official_active_gate_index", -1)) != 6:
        raise RuntimeError(f"{experiment} is not a six-gate result")
    if int(report.get("official_race_finish_time_ns", -1)) < 0:
        raise RuntimeError(f"{experiment} has no official finish time")
    if report.get("collision") is not None or report.get("invalid_reason") is not None:
        raise RuntimeError(f"{experiment} has collision or invalid evidence")
    if report.get("checkpoint_sha256") != EXPECTED_N259_SHA256:
        raise RuntimeError(f"{experiment} checkpoint identity mismatch")
    contract = report.get("contract", {})
    if not contract.get("gate_governor_enabled"):
        raise RuntimeError(f"{experiment} did not use the public gate governor")
    if not 50.0 <= float(contract.get("effective_command_hz", 0.0)) < 100.0:
        raise RuntimeError(f"{experiment} command rate is invalid")
    if int(contract.get("exit_stop_setpoints_sent", 0)) != 1:
        raise RuntimeError(f"{experiment} final stop-setpoint proof is invalid")
    if not contract.get("exit_disarm_sent"):
        raise RuntimeError(f"{experiment} final disarm proof is absent")
    reset = report.get("reset", {})
    if int(reset.get("reset_commands_sent", 0)) != 1:
        raise RuntimeError(f"{experiment} reset count is not one")
    if int(reset.get("track_gate_count", 0)) != 6:
        raise RuntimeError(f"{experiment} did not transfer six gates")
    telemetry = report.get("telemetry", {})
    for field in ("telemetry_dropouts", "malformed_messages", "drain_limit_hits"):
        if int(telemetry.get(field, -1)) != 0:
            raise RuntimeError(f"{experiment} has nonzero {field}")


def load_live_model(
    experiment: str,
    path: Path,
    gate_centers_ned_m: np.ndarray,
) -> LiveModel:
    expected = EXPECTED_LIVE_SHA256[experiment]
    artifact_hash = validate_deployment_artifact(path, expected, experiment)
    report = json.loads(path.read_text())
    validate_live_report(report, experiment)
    official = replay_model.official_gate_times(report)
    response = replay_model.fit_response_axes(report["trace"])
    baseline_replay = replay_model.replay_governor(
        report["trace"], gate_centers_ned_m, response, (BASELINE,) * 6
    )
    if len(baseline_replay) != 6:
        raise RuntimeError(f"{experiment} baseline replay did not finish")
    raw_times = np.asarray(
        [item.raw_crossing_time_s for item in baseline_replay], dtype=np.float64
    )
    return LiveModel(
        experiment=experiment,
        path=path,
        artifact_sha256=artifact_hash,
        report=report,
        response_axes=response,
        official_gate_times_s=official,
        baseline_replay=baseline_replay,
        official_minus_raw_residual_s=official - raw_times,
    )


def candidate_grid() -> tuple[replay_model.GovernorParameters, ...]:
    candidates = []
    for crossing, maximum, gain in itertools.product(
        (7.5, 8.0, 8.5, 8.75, 9.0),
        (8.0, 8.5, 9.0),
        (1.5, 1.75, 2.0),
    ):
        if maximum < crossing:
            continue
        candidates.append(
            replay_model.GovernorParameters(
                crossing_speed_m_s=crossing,
                maximum_along_speed_m_s=maximum,
                slowdown_distance_m=10.0,
                cross_track_gain_s_inv=gain,
                maximum_cross_track_correction_m_s=4.0,
            )
        )
    return tuple(candidates)


def evaluate_candidate(
    models: Sequence[LiveModel],
    gate_centers_ned_m: np.ndarray,
    candidate: replay_model.GovernorParameters,
) -> dict[str, object]:
    per_trace = []
    for model in models:
        replay = replay_model.replay_governor(
            model.report["trace"],
            gate_centers_ned_m,
            model.response_axes,
            (candidate,) * 6,
        )
        summary = replay_model.corrected_replay_summary(
            replay, model.official_minus_raw_residual_s
        )
        per_trace.append(
            {
                "experiment": model.experiment,
                "completed_gate_count": summary["completed_gate_count"],
                "predicted_finish_time_s": summary["predicted_finish_time_s"],
                "maximum_crossing_error_m": summary["maximum_crossing_error_m"],
                "predicted_gate_crossing_times_s": [
                    item["residual_corrected_crossing_time_s"]
                    for item in summary["gates"]
                ],
                "predicted_gate_crossing_errors_m": [
                    item["crossing_error_m"] for item in summary["gates"]
                ],
            }
        )
    complete = all(item["completed_gate_count"] == 6 for item in per_trace)
    finishes = [item["predicted_finish_time_s"] for item in per_trace]
    errors = [item["maximum_crossing_error_m"] for item in per_trace]
    if complete and all(value is not None for value in finishes + errors):
        worst_finish = max(float(value) for value in finishes)
        mean_finish = float(np.mean(finishes))
        maximum_error = max(float(value) for value in errors)
    else:
        worst_finish = None
        mean_finish = None
        maximum_error = None
    return {
        "parameters": dataclasses.asdict(candidate),
        "parameter_change_count": replay_model.parameter_change_count(
            candidate, BASELINE
        ),
        "completed_all_models": complete,
        "robust_worst_predicted_finish_time_s": worst_finish,
        "mean_predicted_finish_time_s": mean_finish,
        "robust_maximum_crossing_error_m": maximum_error,
        "safe_offline": bool(
            complete
            and maximum_error is not None
            and maximum_error <= CROSSING_ERROR_LIMIT_M
        ),
        "meets_near_24_target": bool(
            complete
            and worst_finish is not None
            and worst_finish <= TARGET_WORST_CASE_FINISH_S
        ),
        "per_trace": per_trace,
    }


def choose_minimal_robust_candidate(
    candidates: Sequence[dict[str, object]],
) -> dict[str, object] | None:
    eligible = [
        item
        for item in candidates
        if item["safe_offline"] and item["meets_near_24_target"]
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item["parameter_change_count"],
            -item["robust_worst_predicted_finish_time_s"],
            item["robust_maximum_crossing_error_m"],
            tuple(item["parameters"].values()),
        ),
    )


def build_report(
    n270_path: Path,
    live_paths: dict[str, Path],
    checkpoint_path: Path,
    runner_path: Path,
    config_path: Path,
) -> dict[str, object]:
    n270_hash = validate_deployment_artifact(
        n270_path, EXPECTED_N270_SHA256, "N270 aggregate"
    )
    checkpoint_hash = validate_deployment_artifact(
        checkpoint_path, EXPECTED_N259_SHA256, "N259 checkpoint"
    )
    runner_hash = validate_deployment_artifact(
        runner_path, EXPECTED_RUNNER_SHA256, "N270 runner"
    )
    config_hash = validate_deployment_artifact(
        config_path, EXPECTED_CONFIG_SHA256, "v3391 config"
    )
    gate_centers, settings = replay_model.load_ned_gate_centers(config_path)
    if len(gate_centers) != 6:
        raise RuntimeError("N271 requires exactly six configured gates")
    if set(live_paths) != set(EXPECTED_LIVE_SHA256):
        raise RuntimeError("N271 requires exactly N267, N268, and N269")
    models = tuple(
        load_live_model(experiment, live_paths[experiment], gate_centers)
        for experiment in ("N267", "N268", "N269")
    )
    sweep = [evaluate_candidate(models, gate_centers, item) for item in candidate_grid()]
    sweep.sort(
        key=lambda item: (
            not item["safe_offline"],
            math.inf
            if item["robust_worst_predicted_finish_time_s"] is None
            else item["robust_worst_predicted_finish_time_s"],
            item["parameter_change_count"],
        )
    )
    recommendation = choose_minimal_robust_candidate(sweep)
    if recommendation is None:
        raise RuntimeError("no robust near-24-second N271 candidate was found")

    baseline_models = []
    for model in models:
        raw_times = np.asarray(
            [item.raw_crossing_time_s for item in model.baseline_replay],
            dtype=np.float64,
        )
        baseline_models.append(
            {
                "experiment": model.experiment,
                "artifact": str(model.path),
                "artifact_sha256": model.artifact_sha256,
                "official_gate_times_s": model.official_gate_times_s.tolist(),
                "official_finish_time_s": float(model.official_gate_times_s[-1]),
                "raw_replay_gate_times_s": raw_times.tolist(),
                "official_minus_raw_replay_gate_residual_s": (
                    model.official_minus_raw_residual_s.tolist()
                ),
                "raw_finish_error_s": float(
                    raw_times[-1] - model.official_gate_times_s[-1]
                ),
                "fitted_velocity_response_axes_ned_xyz": [
                    dataclasses.asdict(item) for item in model.response_axes
                ],
            }
        )

    return {
        "experiment": "N271",
        "evidence_kind": "three_live_trace_robust_governor_sweep",
        "source_artifacts": {
            "n270_aggregate": str(n270_path),
            "n270_aggregate_sha256": n270_hash,
            "n259_checkpoint": str(checkpoint_path),
            "n259_checkpoint_sha256": checkpoint_hash,
            "runner": str(runner_path),
            "runner_sha256": runner_hash,
            "config": str(config_path),
            "config_sha256": config_hash,
        },
        "immutable_n270_baseline": {
            "parameters": dataclasses.asdict(BASELINE),
            "best_official_finish_time_s": 26.591306686,
            "valid_laps": 3,
            "models": baseline_models,
        },
        "sweep_contract": {
            "method": "independent per-live-trace response fits and residual-anchored replay",
            "crossing_speeds_m_s": [7.5, 8.0, 8.5, 8.75, 9.0],
            "maximum_along_speeds_m_s": [8.0, 8.5, 9.0],
            "slowdown_distances_m": [10.0],
            "cross_track_gains_s_inv": [1.5, 1.75, 2.0],
            "maximum_cross_track_corrections_m_s": [4.0],
            "candidate_count": len(sweep),
            "robust_target_finish_time_s": TARGET_WORST_CASE_FINISH_S,
            "crossing_error_limit_m": CROSSING_ERROR_LIMIT_M,
            "native_target_radius_m": settings["gate_radius"],
        },
        "minimal_robust_recommendation": {
            **recommendation,
            "selection_rule": (
                "fewest scalar changes meeting the time and crossing-error limits "
                "in all three independently calibrated live models; among ties "
                "retain the slowest qualifying worst-case prediction"
            ),
        },
        "top_safe_candidates": [item for item in sweep if item["safe_offline"]][
            :20
        ],
        "preregistered_next_action": {
            "experiment": "N272",
            "tag": "n272_v3391_n259_cross9_xgain2_gate3_bounded_001",
            "maximum_duration_s": 20.0,
            "authoritative_stop_gate_index": 3,
            "runner_argv": [
                "--checkpoint",
                "checkpoints\\n259_aperture_fullphase_bc_6gate.bin",
                "--expected-checkpoint-sha256",
                EXPECTED_N259_SHA256,
                "--config",
                "config\\drone_race_vq1_v3391_telemetry.ini",
                "--duration-s",
                "20",
                "--heartbeat-hz",
                "2",
                "--command-hz",
                "60",
                "--trace-hz",
                "10",
                "--stop-after-gate-index",
                "3",
                "--gate-governor",
                "--governor-crossing-speed-m-s",
                "9.0",
                "--governor-maximum-along-speed-m-s",
                "9.0",
                "--governor-slowdown-distance-m",
                "10.0",
                "--governor-cross-track-gain-s-inv",
                "2.0",
                "--governor-maximum-cross-track-correction-m-s",
                "4.0",
                "--governor-plane-miss-abort-m",
                "1.5",
            ],
            "acceptance": {
                "official_active_gate_index": 3,
                "official_race_finish_time_ns": -1,
                "collision": None,
                "invalid_reason": None,
                "effective_command_hz_range": [50.0, 100.0],
                "telemetry_dropouts": 0,
                "malformed_messages": 0,
                "drain_limit_hits": 0,
                "reset_commands_sent": 1,
                "track_gate_count": 6,
                "exit_stop_setpoints_sent": 1,
                "exit_disarm_sent": True,
            },
            "failure_rule": (
                "any failure rejects N272 and forbids an unchanged retry; only a "
                "clean bounded pass may authorize a separately numbered full lap"
            ),
        },
        "limitations": [
            "N271 is offline counterfactual evidence, not an official lap.",
            "Every live model discloses and reapplies its complete baseline gate-time residual.",
            "The 9 m/s command extrapolates beyond N270's observed 8 m/s ceiling.",
            "The recurrent N259 policy trace is replayed by remaining distance rather than counterfactually rerun.",
            "Only the preregistered bounded v3391 N272 run can validate the new command envelope.",
        ],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    windows_logs = Path(
        "/mnt/c/Users/anon/code/pufferlib-drone/.n253-shadow/logs/sitl"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n270",
        type=Path,
        default=root / "logs/sitl/n270_v3391_cross7p5_repeatability_promotion.json",
    )
    for experiment, filename in (
        ("n267", "n267_v3391_n259_cross7p5_full_lap_001.json"),
        ("n268", "n268_v3391_n259_cross7p5_full_lap_confirm_002.json"),
        ("n269", "n269_v3391_n259_cross7p5_full_lap_confirm_003.json"),
    ):
        parser.add_argument(f"--{experiment}", type=Path, default=windows_logs / filename)
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
        "--output",
        type=Path,
        default=root / "logs/sitl/n271_n270_three_trace_governor_sweep.json",
    )
    args = parser.parse_args()
    report = build_report(
        args.n270,
        {"N267": args.n267, "N268": args.n268, "N269": args.n269},
        args.checkpoint,
        args.runner,
        args.config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "experiment": report["experiment"],
                "output": str(args.output),
                "recommendation": report["minimal_robust_recommendation"],
                "next_action": report["preregistered_next_action"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
