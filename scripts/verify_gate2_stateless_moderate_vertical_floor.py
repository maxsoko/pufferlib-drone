#!/usr/bin/env python3
"""Verify N176's stateless moderate Gate-2 projected-vertical floor."""

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(path: Path) -> str:
    match = re.search(r"_bounded_(\d{3})_attempt_", path.name)
    if match is None:
        raise ValueError(f"cannot parse candidate number from {path}")
    return match.group(1)


def _kinematics(values: np.ndarray) -> tuple[float, float]:
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
    activated_once = False
    activation_rows: list[dict] = []
    changed_rows: list[dict] = []
    recovered_rows: list[dict] = []
    synthetic_floor_passes = 0
    synthetic_recovery_passes = 0
    max_non_thrust_error = 0.0
    stronger_thrust_changes = 0
    residual_latch_samples = 0
    hybrid._clear_gate2_state()
    for sample_index, sample in enumerate(samples):
        if int(sample.get("official_active_gate_index", -1)) != 1:
            continue
        values = np.asarray(sample.get("observation") or [], dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action") or [], dtype=float)
        if values.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"invalid Gate-2 sample {path}:{sample_index}")
        forward_m, projected_down_m = _kinematics(values)
        active = (
            float(values[10]) >= 0.5
            and 0.0 < forward_m <= hybrid.GATE2_VERTICAL_FLOOR_FORWARD_M
            and projected_down_m
            < hybrid.GATE2_VERTICAL_PROJECTED_DOWN_TRIGGER_M
        )
        governed = np.asarray(
            hybrid._gate2_projected_vertical_floor(values, recorded.tolist()),
            dtype=float,
        )
        if hybrid._GATE2_VERTICAL_FLOOR_LATCHED:
            residual_latch_samples += 1
        if active:
            activated_once = True
            synthetic = recorded.tolist()
            synthetic[2] = 0.0
            synthetic_out = hybrid._gate2_projected_vertical_floor(
                values, synthetic.copy()
            )
            if synthetic_out[2] == hybrid.GATE2_VERTICAL_THRUST_FLOOR_NORM:
                synthetic_floor_passes += 1
            activation_rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample["elapsed_s"]),
                    "forward_m": forward_m,
                    "projected_down_at_plane_m": projected_down_m,
                    "recorded_action": recorded.tolist(),
                    "governed_action": governed.tolist(),
                }
            )
        elif activated_once:
            synthetic = recorded.tolist()
            synthetic[2] = 0.0
            synthetic_out = hybrid._gate2_projected_vertical_floor(
                values, synthetic.copy()
            )
            if synthetic_out == synthetic:
                synthetic_recovery_passes += 1
            recovered_rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample["elapsed_s"]),
                    "forward_m": forward_m,
                    "projected_down_at_plane_m": projected_down_m,
                }
            )
        if governed.tolist() != recorded.tolist():
            max_non_thrust_error = max(
                max_non_thrust_error,
                max(
                    abs(governed[index] - recorded[index])
                    for index in (0, 1, 3)
                ),
            )
            if recorded[2] >= hybrid.GATE2_VERTICAL_THRUST_FLOOR_NORM:
                stronger_thrust_changes += 1
            changed_rows.append(
                {
                    "sample": sample_index,
                    "elapsed_s": float(sample["elapsed_s"]),
                    "forward_m": forward_m,
                    "projected_down_at_plane_m": projected_down_m,
                    "recorded_action": recorded.tolist(),
                    "governed_action": governed.tolist(),
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
        "activation_samples": len(activation_rows),
        "changed_samples": len(changed_rows),
        "post_activation_recovered_samples": len(recovered_rows),
        "synthetic_floor_passes": synthetic_floor_passes,
        "synthetic_recovery_passes": synthetic_recovery_passes,
        "max_non_thrust_error": max_non_thrust_error,
        "stronger_thrust_changes": stronger_thrust_changes,
        "residual_latch_samples": residual_latch_samples,
        "activation_rows": activation_rows,
        "changed_rows": changed_rows,
        "recovered_rows": recovered_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--candidate036-poststop", required=True, type=Path)
    parser.add_argument("--json-path", required=True, type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    by_candidate = {report["candidate"]: report for report in reports}
    blockers: list[str] = []
    expected = CLEAN_GATE2_CANDIDATES | {"014", "029", "035", "036"}
    for candidate in sorted(expected - by_candidate.keys()):
        blockers.append(f"candidate{candidate}_missing")
    for candidate in sorted(CLEAN_GATE2_CANDIDATES):
        report = by_candidate.get(candidate)
        if report and (report["activation_samples"] or report["changed_samples"]):
            blockers.append(f"candidate{candidate}_clean_path_changed")
    candidate35 = by_candidate.get("035")
    if candidate35:
        if candidate35["activation_samples"] != 5:
            blockers.append("candidate035_activation_count_not_five")
        if candidate35["post_activation_recovered_samples"] != 5:
            blockers.append("candidate035_recovery_count_not_five")
        if candidate35["synthetic_floor_passes"] != 5:
            blockers.append("candidate035_synthetic_floor_not_five")
        if candidate35["synthetic_recovery_passes"] != 5:
            blockers.append("candidate035_synthetic_recovery_not_five")
    candidate36 = by_candidate.get("036")
    if candidate36:
        if candidate36["activation_samples"] != 7:
            blockers.append("candidate036_activation_count_not_seven")
        if candidate36["changed_samples"] != 3:
            blockers.append("candidate036_change_count_not_three")
    for report in reports:
        if report["max_non_thrust_error"] != 0.0:
            blockers.append(f"candidate{report['candidate']}_changes_non_thrust")
        if report["stronger_thrust_changes"]:
            blockers.append(f"candidate{report['candidate']}_changes_stronger_thrust")
        if report["residual_latch_samples"]:
            blockers.append(f"candidate{report['candidate']}_latch_persisted")

    poststop = json.loads(args.candidate036_poststop.read_text(encoding="utf-8"))
    race = ((poststop.get("latest_telemetry") or {}).get("race_status") or {})
    if poststop.get("reset_sent") is not False:
        blockers.append("candidate036_poststop_not_command_free")
    if int(race.get("active_gate_index", -1)) != 1:
        blockers.append("candidate036_poststop_gate_index_unexpected")
    if int(race.get("last_gate_race_time", -1)) != 5081851482:
        blockers.append("candidate036_poststop_last_gate_time_unexpected")
    if int(race.get("race_finish_time_ns", -2)) != -1:
        blockers.append("candidate036_poststop_finish_unexpected")

    policy_path = Path(hybrid.__file__).resolve()
    payload = {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "stateless": hybrid.GATE2_VERTICAL_FLOOR_STATELESS,
            "max_forward_m": hybrid.GATE2_VERTICAL_FLOOR_FORWARD_M,
            "projected_down_trigger_m": (
                hybrid.GATE2_VERTICAL_PROJECTED_DOWN_TRIGGER_M
            ),
            "normalized_thrust_floor": hybrid.GATE2_VERTICAL_THRUST_FLOOR_NORM,
            "preserve_pitch_roll_yaw_and_stronger_thrust": True,
        },
        "clean_gate2_candidates": sorted(CLEAN_GATE2_CANDIDATES),
        "candidate036_poststop": {
            "path": str(args.candidate036_poststop),
            "sha256": _sha256(args.candidate036_poststop),
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
