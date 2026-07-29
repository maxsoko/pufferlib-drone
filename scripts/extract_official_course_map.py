#!/usr/bin/env python3
"""Extract official gate landmarks from SITL smoke JSON into a native course map.

Reads estimator landmarks / trajectory from successful official runs and writes:
  - config/official_course_r1.json
  - config/drone_race_sitl_plant_official.ini (optional)

Usage:
  python scripts/extract_official_course_map.py logs/sitl/agent_sprint_13_smoke.json
  python scripts/extract_official_course_map.py --glob "logs/sitl/*smoke*.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TS002_INNER_HALF_M = 0.75


def _load_smoke(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _landmarks_from_smoke(payload: dict) -> dict[int, tuple[float, float, float]]:
    landmarks: dict[int, tuple[float, float, float]] = {}

    estimator = payload.get("estimator") or {}
    raw = estimator.get("landmarks") or {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, (list, tuple)) or len(value) < 3:
                continue
            landmarks[int(key)] = (float(value[0]), float(value[1]), float(value[2]))

    control = payload.get("control_inputs") or {}
    traj = control.get("trajectory") or payload.get("trajectory") or []
    for sample in traj:
        gate_landmarks = sample.get("landmarks") or {}
        if isinstance(gate_landmarks, dict):
            for key, value in gate_landmarks.items():
                if isinstance(value, (list, tuple)) and len(value) >= 3:
                    landmarks[int(key)] = (float(value[0]), float(value[1]), float(value[2]))

    return landmarks


def _official_gate_count(payload: dict) -> int:
    official = int(
        payload.get("official_active_gate_index")
        or (payload.get("race_status") or {}).get("active_gate_index")
        or 0
    )
    return max(official, 0)


def build_course_map(smoke_paths: list[Path], *, min_official_gates: int = 1) -> dict:
    merged: dict[int, list[tuple[float, float, float]]] = {}
    sources: list[str] = []

    for path in smoke_paths:
        payload = _load_smoke(path)
        if _official_gate_count(payload) < min_official_gates:
            continue
        landmarks = _landmarks_from_smoke(payload)
        if not landmarks:
            continue
        sources.append(str(path))
        for gate_id, pos in landmarks.items():
            merged.setdefault(gate_id, []).append(pos)

    if not merged:
        raise ValueError("no gate landmarks found in the provided smoke artifacts")

    gates = []
    for gate_id in sorted(merged):
        samples = merged[gate_id]
        x = sum(p[0] for p in samples) / len(samples)
        y = sum(p[1] for p in samples) / len(samples)
        z = sum(p[2] for p in samples) / len(samples)
        gates.append(
            {
                "id": gate_id,
                "x_m": round(x, 3),
                "y_m": round(y, 3),
                "z_m": round(z, 3),
                "normal": [1.0, 0.0, 0.0],
                "inner_radius_m": TS002_INNER_HALF_M,
                "sample_count": len(samples),
            }
        )

    return {
        "simulator_version": "v1.0.3379",
        "coordinate_frame": "ned_takeoff",
        "gate_count": len(gates),
        "gates": gates,
        "sources": sources,
    }


def write_ini(course_map: dict, ini_path: Path, *, base_ini: Path) -> None:
    lines = base_ini.read_text(encoding="utf-8").splitlines()
    env_section = False
    out: list[str] = []
    skip_keys = {"num_gates", "use_custom_gate_layout", "gate_radius"}
    gate_keys = {f"gate{i}_x" for i in range(16)} | {f"gate{i}_y" for i in range(16)} | {f"gate{i}_z" for i in range(16)}

    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if line.strip() == "[env]":
            env_section = True
            out.append(line)
            continue
        if env_section and line.startswith("[") and line.endswith("]"):
            env_section = False
        if env_section and (key in skip_keys or key in gate_keys):
            continue
        out.append(line)

    insert_at = next(i for i, line in enumerate(out) if line.strip() == "[env]") + 1
    gate_lines = [
        f"num_gates = {course_map['gate_count']}",
        f"gate_radius = {TS002_INNER_HALF_M}",
        "use_custom_gate_layout = 1",
    ]
    for gate in course_map["gates"]:
        idx = int(gate["id"])
        gate_lines.append(f"gate{idx}_x = {gate['x_m']}")
        gate_lines.append(f"gate{idx}_y = {gate['y_m']}")
        gate_lines.append(f"gate{idx}_z = {gate['z_m']}")
    out[insert_at:insert_at] = gate_lines

    ini_path.parent.mkdir(parents=True, exist_ok=True)
    ini_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke_json", nargs="*", help="Smoke JSON artifacts")
    parser.add_argument("--glob", dest="glob_pattern", default="")
    parser.add_argument("--min-official-gates", type=int, default=1)
    parser.add_argument(
        "--json-out",
        default=str(ROOT / "config" / "official_course_r1.json"),
    )
    parser.add_argument(
        "--ini-out",
        default=str(ROOT / "config" / "drone_race_sitl_plant_official.ini"),
    )
    parser.add_argument(
        "--base-ini",
        default=str(ROOT / "config" / "drone_race_sitl_plant.ini"),
    )
    args = parser.parse_args()

    paths: list[Path] = [Path(p) for p in args.smoke_json]
    if args.glob_pattern:
        paths.extend(Path(p) for p in glob.glob(args.glob_pattern, recursive=True))
    paths = sorted({p.resolve() for p in paths if p.exists()})
    if not paths:
        print("no smoke JSON inputs found", file=sys.stderr)
        return 2

    course_map = build_course_map(paths, min_official_gates=args.min_official_gates)
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(course_map, indent=2) + "\n", encoding="utf-8")
    write_ini(course_map, Path(args.ini_out), base_ini=Path(args.base_ini))

    print(json.dumps({"course_map": str(json_out), "ini": args.ini_out, **course_map}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
