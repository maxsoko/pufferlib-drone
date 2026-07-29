#!/usr/bin/env python3
"""Run the preregistered VQ2-SF007 smooth-spline speed surface."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_native_oracle import run_diagnostic


SPEEDS_M_S = (1.5, 2.0, 2.5, 3.0, 3.5)
AGENTS = 16
EPISODES = 16
SEED = 42007
EPISODE_SECONDS = 60.0


def summarize(report: dict) -> dict:
    metrics = report["metrics"]
    episodes = report["episodes"]
    success_rate = metrics.get("env/success_rate", 0.0)
    return {
        "target_speed_m_s": report["fixed_environment"][
            "teacher_pitch_speed_target_m_s"
        ],
        "episodes": episodes,
        "success_rate": success_rate,
        "successes": int(round(success_rate * episodes)),
        "mean_gates_passed": metrics.get("env/gates_passed", 0.0),
        "crash_rate": metrics.get("env/crash", 0.0),
        "timeout_rate": metrics.get("env/timeout", 0.0),
        "missed_gate_rate": metrics.get("env/missed_gate", 0.0),
        "out_of_order_rate": metrics.get("env/out_of_order", 0.0),
        "terminal_crossing_radial_m": metrics.get(
            "env/terminal_crossing_radial", 0.0
        ),
        "terminal_crossing_right_m": metrics.get(
            "env/terminal_crossing_right", 0.0
        ),
        "terminal_crossing_abs_vertical_m": metrics.get(
            "env/terminal_crossing_abs_vertical", 0.0
        ),
        "completion_time_s": metrics.get("env/completion_time", 0.0),
        "wall_time_seconds": report["wall_time_seconds"],
    }


def selection_key(cell: dict) -> tuple[float, ...]:
    """Fixed solve-first ordering; larger tuple is better."""

    return (
        cell["success_rate"],
        -cell["crash_rate"],
        -cell["out_of_order_rate"],
        cell["mean_gates_passed"],
        -cell["missed_gate_rate"],
        -cell["terminal_crossing_radial_m"],
        -cell["completion_time_s"],
        -cell["target_speed_m_s"],
    )


def main() -> int:
    cells = []
    for speed in SPEEDS_M_S:
        cells.append(summarize(run_diagnostic(
            agents=AGENTS,
            episodes=EPISODES,
            seed=SEED,
            controller="spline",
            target_speed_m_s=speed,
            episode_seconds=EPISODE_SECONDS,
        )))
    selected = max(cells, key=selection_key)
    print(json.dumps({
        "schema": "vq2_native_oracle_spline_speed_surface_v1",
        "tag": "vq2_sf007_native_oracle_spline_speed_surface",
        "common_seed": SEED,
        "episodes_per_cell": EPISODES,
        "cells": cells,
        "selected": selected,
        "selected_for_independent_screen": selected["successes"] > 0,
        "teacher_labels_written": 0,
        "safety": {
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
