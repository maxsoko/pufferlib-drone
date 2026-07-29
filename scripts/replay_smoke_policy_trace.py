#!/usr/bin/env python3
"""Replay a competition smoke JSON detection trace through a policy callable.

Offline perception transfer check before live official-sim runs. Reconstructs
23-float observations from logged angle_mode_trace / approach samples and
records policy actions (does not drive MAVLink).

Usage:
  set PUFFER_POLICY_CHECKPOINT_PATH=checkpoints/.../step.bin
  python scripts/replay_smoke_policy_trace.py logs/sitl/agent_sprint_13_smoke.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from drone_policy_contract import OBSERVATION_SIZE, validate_observation  # noqa: E402


def load_infer(spec: str):
    path_str, _, function_name = spec.rpartition(":")
    path = Path(path_str).expanduser().resolve()
    module_name = "_replay_policy_callable"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    reset_fn = getattr(module, "reset", None)
    if callable(reset_fn):
        reset_fn()
    infer = getattr(module, function_name)
    if not callable(infer):
        raise ValueError(f"{function_name} not callable in {path}")
    return infer


def observation_from_trace_row(row: dict, *, last_cmd: tuple[float, float, float, float]) -> tuple[float, ...]:
    est_rpy = row.get("est_rpy") or (0.0, 0.0, 0.0)
    est_vel = row.get("est_vel") or (0.0, 0.0, 0.0)
    gate = row.get("gate_obs") or {}
    roll, pitch, yaw = est_rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx, vy, vz = est_vel
    body_forward = c * vx + s * vy
    body_right = -s * vx + c * vy
    body_down = -vz
    obs = [
        max(-1.0, min(1.0, body_forward / 10.0)),
        max(-1.0, min(1.0, body_right / 5.0)),
        max(-1.0, min(1.0, body_down / 5.0)),
        max(-1.0, min(1.0, float(row.get("gyro_roll", 0.0)) / 4.0)),
        max(-1.0, min(1.0, float(row.get("gyro_pitch", 0.0)) / 4.0)),
        max(-1.0, min(1.0, float(row.get("gyro_yaw", 0.0)) / 4.0)),
        max(-1.0, min(1.0, qw)),
        max(-1.0, min(1.0, qx)),
        max(-1.0, min(1.0, qy)),
        max(-1.0, min(1.0, qz)),
        float(gate.get("visible", -1.0)),
        float(gate.get("forward_norm", 0.0)),
        float(gate.get("right_norm", 0.0)),
        float(gate.get("down_norm", 0.0)),
        float(gate.get("yaw_error_norm", 0.0)),
        float(gate.get("pitch_error_norm", 0.0)),
        float(gate.get("size_norm", -1.0)),
        float(gate.get("confidence", -1.0)),
        max(0.0, min(1.0, float(row.get("elapsed_fraction", 0.0)))),
        last_cmd[0],
        last_cmd[1],
        last_cmd[2],
        last_cmd[3],
    ]
    if len(obs) != OBSERVATION_SIZE:
        raise ValueError(f"expected {OBSERVATION_SIZE} obs, got {len(obs)}")
    return validate_observation(obs)


def trace_rows(smoke: dict) -> list[dict]:
    trace = (
        smoke.get("control_inputs", {})
        .get("angle_servo", {})
        .get("trace", [])
    )
    if trace:
        return trace
    approach = smoke.get("approach_diagnostics", {}).get("samples", [])
    rows = []
    duration = float(smoke.get("duration_s", 30.0) or 30.0)
    for sample in approach:
        rows.append(
            {
                "elapsed_s": float(sample.get("elapsed_s", 0.0)),
                "elapsed_fraction": float(sample.get("elapsed_s", 0.0)) / max(duration, 1e-3),
                "est_rpy": sample.get("est_rpy", (0.0, 0.0, 0.0)),
                "est_vel": sample.get("est_vel", (0.0, 0.0, 0.0)),
                "gate_obs": sample.get("gate_obs", {}),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke_json", type=Path)
    parser.add_argument(
        "--policy-callable",
        default="scripts/policy_callable_checkpoint.py:infer",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    smoke = json.loads(args.smoke_json.read_text(encoding="utf-8"))
    rows = trace_rows(smoke)
    if not rows:
        print("no trace rows in smoke JSON", file=sys.stderr)
        return 1

    infer = load_infer(args.policy_callable)
    last_cmd = (0.0, 0.0, 0.0, 0.0)
    replay: list[dict] = []
    for row in rows:
        obs = observation_from_trace_row(row, last_cmd=last_cmd)
        action = infer(obs)
        if len(action) != 4:
            raise ValueError(f"expected 4 action dims, got {len(action)}")
        last_cmd = tuple(max(-1.0, min(1.0, float(v))) for v in action)
        replay.append(
            {
                "elapsed_s": row.get("elapsed_s"),
                "action": last_cmd,
            }
        )

    report = {
        "smoke_json": str(args.smoke_json),
        "policy_callable": args.policy_callable,
        "steps": len(replay),
        "replay": replay,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
