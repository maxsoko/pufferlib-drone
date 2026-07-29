#!/usr/bin/env python3
"""Collect behavior-cloning (obs, attitude_action) pairs from SITL smoke JSON logs.

Track B phase B4: distill visual-servo expert trajectories into training data
for the 23-float TS-002 contract + 4-float attitude action space.

Usage:
  python scripts/collect_bc_dataset.py --glob "logs/sitl/agent_sprint_*_smoke.json"
  python scripts/collect_bc_dataset.py logs/sitl/agent_sprint_13_smoke.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from drone_policy_contract import (  # noqa: E402
    ATTITUDE_ACTION_FIELDS,
    OBSERVATION_FIELDS,
    build_ts002_observation,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_thrust(thrust: float) -> float:
    hover, minimum, maximum = 0.27, 0.18, 0.42
    span = maximum - hover if thrust >= hover else hover - minimum
    return _clamp((thrust - hover) / span, -1.0, 1.0)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _samples_from_smoke(path: Path, *, min_official_gates: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    official = int(
        payload.get("official_active_gate_index")
        or (payload.get("race_status") or {}).get("active_gate_index")
        or 0
    )
    if official < min_official_gates:
        return []

    control = payload.get("control_inputs") or {}
    mode = str(control.get("control_mode") or payload.get("control_mode") or "")
    if "visual-servo" not in mode and mode != "visual-servo-angles":
        return []

    duration = float(payload.get("duration_s") or control.get("duration_s") or 30.0)
    diagnostics = payload.get("approach_diagnostics") or {}
    samples_raw = diagnostics.get("samples") or []
    out: list[dict] = []

    for sample in samples_raw:
        elapsed = float(sample.get("elapsed_s") or 0.0)
        pose = sample.get("gate_pose") or {}
        actual = sample.get("actual_command") or {}
        if not pose or actual.get("kind") != "visual_servo_angles_target":
            continue

        body = pose.get("body_vector_ned_m") or [0.0, 0.0, 0.0]
        obs = build_ts002_observation(
            gate_visible=True,
            gate_forward_m=float(body[0]),
            gate_right_m=float(body[1]),
            gate_down_m=float(body[2]),
            gate_yaw_error_rad=float(pose.get("yaw_error_rad") or 0.0),
            gate_range_m=float(pose.get("range_camera_m") or 0.0) or None,
            elapsed_fraction=_clamp(elapsed / max(duration, 1e-6), 0.0, 1.0),
            last_cmd=(0.0, 0.0, 0.0, 0.0),
        )

        pitch = float(actual.get("pitch") or actual.get("body_pitch") or 0.0)
        roll = float(actual.get("roll") or actual.get("body_roll") or 0.0)
        yaw = float(actual.get("yaw") or actual.get("body_yaw") or 0.0)
        thrust = float(actual.get("thrust") or 0.17)
        action = (
            _clamp(pitch / 0.5, -1.0, 1.0),
            _clamp(roll / 0.5, -1.0, 1.0),
            _normalize_thrust(thrust),
            _clamp(yaw / math.pi, -1.0, 1.0),
        )
        out.append(
            {
                "source": str(path),
                "elapsed_s": elapsed,
                "observation": list(obs),
                "action": list(action),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke_json", nargs="*")
    parser.add_argument("--glob", dest="glob_pattern", default="")
    parser.add_argument("--min-official-gates", type=int, default=1)
    parser.add_argument(
        "--out",
        default=str(ROOT / "logs" / "track_b" / "bc_dataset.jsonl"),
    )
    args = parser.parse_args()

    paths: list[Path] = [Path(p) for p in args.smoke_json]
    if args.glob_pattern:
        paths.extend(Path(p) for p in glob.glob(args.glob_pattern, recursive=True))
    paths = sorted({p.resolve() for p in paths if p.exists()})

    rows: list[dict] = []
    for path in paths:
        rows.extend(_samples_from_smoke(path, min_official_gates=args.min_official_gates))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "out": str(out_path),
        "sample_count": len(rows),
        "source_count": len(paths),
        "observation_fields": OBSERVATION_FIELDS,
        "action_fields": ATTITUDE_ACTION_FIELDS,
    }
    print(json.dumps(summary, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
