#!/usr/bin/env python3
"""Identify AI-GP Simulator v3379 plant parameters from logs or live probes.

The v3379 attitude interface treats SET_ATTITUDE_TARGET quaternion components as
rate demands (rad/s ≈ gain * cmd_rad), not absolute angles. Translation follows
level attitude thrust with roughly linear vertical response above hover thrust.

Usage:
  # Live open-loop probes (simulator must be running on 14550/5600):
  python scripts/identify_sitl_plant.py --live --output logs/sitl/plant_model.json

  # Offline from a competition smoke JSON (angle_mode_trace + trajectory):
  python scripts/identify_sitl_plant.py --smoke-json logs/sitl/run_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


@dataclass
class RateGainFit:
    axis: str
    cmd_rad: float
    rate_rad_s: float
    gain: float


@dataclass
class SitlPlantModel:
    """Trainable abstraction for ocean/drone_race attitude-setpoint mode."""

    simulator_version: str = "v1.0.3379"
    identified_at_unix_s: float = 0.0
    source: str = ""

    # Open-loop: body rate (rad/s) = gain * commanded quaternion component (rad).
    rate_gain_roll: float = 2.5
    rate_gain_pitch: float = -2.35
    rate_gain_yaw: float = 2.05
    rate_gain_fits: list[dict] = field(default_factory=list)

    # Normalized thrust in [0, 1]; level attitude.
    hover_thrust: float = 0.17
    thrust_climb_onset: float = 0.18
    vertical_accel_per_thrust: float = 35.0  # m/s^2 per unit above hover

    # Translation (optional; fit from forward-flight logs when available).
    linear_drag_coeff: float | None = None
    max_horizontal_accel_m_s2: float | None = None
    rate_lag_tau_s: float = 0.12

    gravity_m_s2: float = 9.80665
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_defaults(path: Path | None = None) -> SitlPlantModel:
    root = Path(__file__).resolve().parents[1]
    defaults_path = path or (root / "config" / "sitl_plant_defaults.json")
    payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    return SitlPlantModel(
        simulator_version=payload.get("simulator_version", "v1.0.3379"),
        identified_at_unix_s=time.time(),
        source=str(defaults_path),
        rate_gain_roll=float(payload["rate_gain_roll"]),
        rate_gain_pitch=float(payload["rate_gain_pitch"]),
        rate_gain_yaw=float(payload["rate_gain_yaw"]),
        hover_thrust=float(payload["hover_thrust"]),
        thrust_climb_onset=float(payload.get("thrust_climb_onset", 0.19)),
        vertical_accel_per_thrust=float(payload["vertical_accel_per_thrust"]),
        linear_drag_coeff=payload.get("linear_drag_coeff"),
        rate_lag_tau_s=float(payload.get("rate_lag_tau_s", 0.12)),
        gravity_m_s2=float(payload.get("gravity_m_s2", 9.80665)),
        notes=list(payload.get("notes", [])),
    )


def fit_rate_lag_from_trace(trace: list[dict]) -> float | None:
    """Estimate first-order rate lag tau from open-loop step segments."""
    taus: list[float] = []
    for i in range(1, len(trace)):
        prev = trace[i - 1]
        curr = trace[i]
        dt = curr["elapsed_s"] - prev["elapsed_s"]
        if dt <= 0.02 or dt > 0.5:
            continue
        cmd = curr.get("cmd_rpy_thrust") or curr.get("cmd_rpy")
        est0 = prev.get("est_rpy")
        est1 = curr.get("est_rpy")
        if not cmd or not est0 or not est1:
            continue
        for axis_idx, gain_sign in enumerate((1.0, -1.0, 1.0)):
            cmd_val = cmd[axis_idx]
            if abs(cmd_val) < 0.08:
                continue
            others_small = all(abs(cmd[j]) < 0.03 for j in range(3) if j != axis_idx)
            if not others_small:
                continue
            delta = (est1[axis_idx] - est0[axis_idx]) / dt
            target_rate = abs(cmd_val) * 2.4 * gain_sign
            if abs(target_rate) < 0.05:
                continue
            frac = abs(delta / target_rate)
            if 0.05 < frac < 0.95:
                taus.append(-dt / math.log(max(frac, 1e-3)))
    if len(taus) < 2:
        return None
    return round(statistics.median(taus), 3)


def _fit_rate_gain(axis: str, cmd_rad: float, rate_rad_s: float) -> RateGainFit | None:
    if abs(cmd_rad) < 1e-6 or abs(rate_rad_s) < 1e-6:
        return None
    return RateGainFit(axis=axis, cmd_rad=cmd_rad, rate_rad_s=rate_rad_s, gain=rate_rad_s / cmd_rad)


def parse_probe_pitch_sign_output(text: str) -> list[RateGainFit]:
    """Parse lines like: pitch +0.30: d_rpy=(...) rates=(+0.000,-0.732,+0.000)"""
    fits: list[RateGainFit] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "rates=" not in line:
            continue
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        header = parts[0].strip()
        tokens = header.split()
        if len(tokens) < 2:
            continue
        axis = tokens[0].lower()
        try:
            cmd = float(tokens[1])
        except ValueError:
            continue
        body = parts[1]
        rates_idx = body.find("rates=")
        if rates_idx < 0:
            continue
        rates_blob = body[rates_idx + len("rates=") :].split(")", 1)[0] + ")"
        rates_blob = rates_blob.strip("()")
        try:
            roll_r, pitch_r, yaw_r = (float(x) for x in rates_blob.split(","))
        except ValueError:
            continue
        rate_by_axis = {"roll": roll_r, "pitch": pitch_r, "yaw": yaw_r}
        if axis not in rate_by_axis:
            continue
        fit = _fit_rate_gain(axis, cmd, rate_by_axis[axis])
        if fit is not None:
            fits.append(fit)
    return fits


def summarize_rate_gains(fits: list[RateGainFit]) -> tuple[float, float, float]:
    by_axis: dict[str, list[float]] = {"roll": [], "pitch": [], "yaw": []}
    for fit in fits:
        by_axis.setdefault(fit.axis, []).append(fit.gain)
    def median_or(default: float, values: list[float]) -> float:
        return statistics.median(values) if values else default

    return (
        median_or(2.5, by_axis["roll"]),
        median_or(-2.35, by_axis["pitch"]),
        median_or(2.05, by_axis["yaw"]),
    )


def parse_probe_thrust_step_output(text: str) -> tuple[float, float]:
    """Return (hover_thrust, climb_onset_thrust) from cy stability lines."""
    stable: list[float] = []
    climb: list[float] = []
    for line in text.splitlines():
        if not line.startswith("thrust "):
            continue
        try:
            thrust = float(line.split()[1])
        except (IndexError, ValueError):
            continue
        if "cy " not in line:
            continue
        cy_part = line.split("cy ", 1)[1]
        start_cy = float(cy_part.split("->", 1)[0].strip())
        end_cy = float(cy_part.split("->", 1)[1].split("(", 1)[0].strip())
        delta = end_cy - start_cy
        if abs(delta) <= 3.0:
            stable.append(thrust)
        elif delta < -20.0:
            climb.append(thrust)
    hover = statistics.median(stable) if stable else 0.17
    climb_onset = min(climb) if climb else hover + 0.02
    return hover, climb_onset


def fit_from_smoke_json(path: Path) -> SitlPlantModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = SitlPlantModel(source=str(path), identified_at_unix_s=time.time())
    trace = (
        payload.get("control_inputs", {})
        .get("angle_servo", {})
        .get("trace", [])
    )
    if not trace:
        model.notes.append("No angle_mode_trace; rate gains unchanged.")
        return model

    # Open-loop segments: desired rpy constant on one axis, others ~0, thrust ~hover.
    segments: list[RateGainFit] = []
    for i in range(1, len(trace)):
        prev = trace[i - 1]
        curr = trace[i]
        dt = curr["elapsed_s"] - prev["elapsed_s"]
        if dt <= 0.05 or dt > 2.0:
            continue
        cmd = curr.get("cmd_rpy_thrust") or curr.get("cmd_rpy")
        est0 = prev.get("est_rpy")
        est1 = curr.get("est_rpy")
        if not cmd or not est0 or not est1:
            continue
        rates = tuple((b - a) / dt for a, b in zip(est0, est1))
        for axis_idx, axis in enumerate(("roll", "pitch", "yaw")):
            cmd_val = cmd[axis_idx]
            if abs(cmd_val) < 0.05:
                continue
            others_small = all(abs(cmd[j]) < 0.03 for j in range(3) if j != axis_idx)
            if not others_small:
                continue
            fit = _fit_rate_gain(axis, cmd_val, rates[axis_idx])
            if fit is not None and 1.0 < abs(fit.gain) < 4.0:
                segments.append(fit)

    if segments:
        roll, pitch, yaw = summarize_rate_gains(segments)
        model.rate_gain_roll = round(roll, 3)
        model.rate_gain_pitch = round(pitch, 3)
        model.rate_gain_yaw = round(yaw, 3)
        model.rate_gain_fits = [asdict(f) for f in segments]
    else:
        model.notes.append(
            "Smoke trace had no open-loop segments; use --live or dedicated probes."
        )

    # Thrust: level segments with stable vertical velocity.
    hover_samples: list[float] = []
    vz_samples: list[tuple[float, float]] = []
    for i in range(1, len(trace)):
        prev = trace[i - 1]
        curr = trace[i]
        dt = curr["elapsed_s"] - prev["elapsed_s"]
        if dt <= 0.05 or dt > 2.0:
            continue
        cmd = curr.get("cmd_rpy_thrust")
        est0 = prev.get("est_vel")
        est1 = curr.get("est_vel")
        if not cmd or not est0 or not est1:
            continue
        rpy = curr.get("est_rpy") or (0.0, 0.0, 0.0)
        if any(abs(v) > 0.08 for v in rpy[:2]):
            continue
        vz = (est1[2] - est0[2]) / dt
        thrust = cmd[3]
        vz_samples.append((thrust, vz))
        if abs(vz) < 0.15:
            hover_samples.append(thrust)

    if hover_samples:
        model.hover_thrust = round(statistics.median(hover_samples), 3)
    if len(vz_samples) >= 3:
        # az ≈ k * (thrust - hover); vz dot is a noisy proxy for az over 0.5s windows.
        hover = model.hover_thrust
        slopes = []
        for thrust, vz in vz_samples:
            delta = thrust - hover
            if abs(delta) > 0.02:
                slopes.append(vz / delta)
        if slopes:
            model.vertical_accel_per_thrust = round(statistics.median(slopes), 2)

    # Drag from horizontal coasting after forward flight.
    coast: list[tuple[float, float]] = []
    for i in range(1, len(trace)):
        prev = trace[i - 1]
        curr = trace[i]
        dt = curr["elapsed_s"] - prev["elapsed_s"]
        if dt <= 0.05 or dt > 2.0:
            continue
        est0 = prev.get("est_vel")
        est1 = curr.get("est_vel")
        if not est0 or not est1:
            continue
        vx0, vy0 = est0[0], est0[1]
        speed = math.hypot(vx0, vy0)
        if speed < 1.0:
            continue
        ax = (est1[0] - vx0) / dt
        ay = (est1[1] - vy0) / dt
        decel = -(vx0 * ax + vy0 * ay) / max(speed, 1e-3)
        if decel > 0.0:
            coast.append((speed, decel))
    if len(coast) >= 3:
        # decel ≈ (b/m) * speed  =>  b/m ≈ decel/speed
        drag_over_mass = [d / s for s, d in coast if s > 0.5]
        model.linear_drag_coeff = round(statistics.median(drag_over_mass), 4)
        model.notes.append("linear_drag_coeff is decel/speed (1/s), not SI b_drag.")

    lag = fit_rate_lag_from_trace(trace)
    if lag is not None:
        model.rate_lag_tau_s = lag
        model.notes.append(f"rate_lag_tau_s fitted from trace: {lag}s")

    return model


def apply_model_to_ini(model: SitlPlantModel, ini_path: Path) -> None:
    """Sync [env] sitl_* keys in a drone_race ini profile."""
    mapping = {
        "sitl_rate_gain_roll": model.rate_gain_roll,
        "sitl_rate_gain_pitch": model.rate_gain_pitch,
        "sitl_rate_gain_yaw": model.rate_gain_yaw,
        "sitl_hover_thrust": model.hover_thrust,
        "sitl_vertical_accel_per_thrust": model.vertical_accel_per_thrust,
        "sitl_gravity_m_s2": model.gravity_m_s2,
        "sitl_rate_lag_tau_s": model.rate_lag_tau_s,
        "sitl_linear_drag": model.linear_drag_coeff or 0.0,
    }
    lines = ini_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_env = False
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if stripped == "[env]":
            in_env = True
            out.append(line)
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_env = False
        if in_env and "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in mapping:
                out.append(f"{key} = {mapping[key]}")
                seen.add(key)
                continue
        out.append(line)
    if in_env or seen:
        pass
    missing = [k for k in mapping if k not in seen]
    if missing:
        try:
            env_idx = next(i for i, line in enumerate(out) if line.strip() == "[env]")
        except StopIteration:
            out.extend(["", "[env]"])
            env_idx = len(out) - 1
        insert_at = env_idx + 1
        for key in missing:
            out.insert(insert_at, f"{key} = {mapping[key]}")
            insert_at += 1
    ini_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def run_live_probes(python: str) -> SitlPlantModel:
    root = Path(__file__).resolve().parents[1]
    pitch_script = root / "scripts" / "probe_pitch_sign.py"
    thrust_script = root / "scripts" / "probe_thrust_step.py"

    pitch = subprocess.run(
        [python, str(pitch_script)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    thrust = subprocess.run(
        [python, str(thrust_script)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    pitch_out = (pitch.stdout or "") + (pitch.stderr or "")
    thrust_out = (thrust.stdout or "") + (thrust.stderr or "")

    fits = parse_probe_pitch_sign_output(pitch_out)
    roll, pitch_g, yaw = summarize_rate_gains(fits)
    hover, climb_onset = parse_probe_thrust_step_output(thrust_out)

    model = SitlPlantModel(
        source="live_probes",
        identified_at_unix_s=time.time(),
        rate_gain_roll=round(roll, 3),
        rate_gain_pitch=round(pitch_g, 3),
        rate_gain_yaw=round(yaw, 3),
        rate_gain_fits=[asdict(f) for f in fits],
        hover_thrust=round(hover, 3),
        thrust_climb_onset=round(climb_onset, 3),
        vertical_accel_per_thrust=round(9.80665 / max(hover, 0.05), 2),
        notes=[
            "Rate gains from probe_pitch_sign.py open-loop holds.",
            "Hover from probe_thrust_step.py gate cy stability (cy delta <= 3 px).",
            "vertical_accel_per_thrust ≈ g/hover_thrust (level hover linearization).",
            "linear_drag_coeff needs forward-flight smoke logs; not measured live.",
        ],
    )
    if pitch.returncode != 0:
        model.notes.append(f"probe_pitch_sign exit {pitch.returncode}")
    if thrust.returncode != 0:
        model.notes.append(f"probe_thrust_step exit {thrust.returncode}")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Run live probes against the simulator")
    parser.add_argument("--smoke-json", type=Path, help="Fit from competition smoke JSON")
    parser.add_argument("--defaults", action="store_true", help="Load config/sitl_plant_defaults.json")
    parser.add_argument(
        "--apply-ini",
        type=Path,
        help="Write sitl_* keys into a drone_race ini profile (e.g. config/drone_race_sitl_plant.ini)",
    )
    parser.add_argument("--output", type=Path, help="Write plant model JSON here")
    parser.add_argument("--python", default=sys.executable, help="Python for live probe subprocesses")
    args = parser.parse_args()

    modes = sum(bool(x) for x in (args.live, args.smoke_json, args.defaults))
    if modes != 1:
        print("Specify exactly one of --live, --smoke-json, or --defaults", file=sys.stderr)
        return 2

    if args.live:
        model = run_live_probes(args.python)
    elif args.defaults:
        model = load_defaults()
    else:
        model = fit_from_smoke_json(args.smoke_json)

    text = json.dumps(model.to_dict(), indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    if args.apply_ini:
        apply_model_to_ini(model, args.apply_ini)
        print(f"updated {args.apply_ini}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
