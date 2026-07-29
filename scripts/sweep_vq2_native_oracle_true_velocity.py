#!/usr/bin/env python3
"""Run the preregistered VQ2-SF006 true-velocity damping surface."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_native_oracle import run_diagnostic
from scripts.sweep_vq2_native_oracle_pd import selection_key, summarize


ROLL_RATE_GAINS = (0.1, 0.3, 0.5, 0.7)
TARGET_SPEED_M_S = 3.5
ROLL_POSITION_GAIN = 0.35
AGENTS = 16
EPISODES = 16
SEED = 42006
EPISODE_SECONDS = 60.0


def main() -> int:
    cells = []
    for rate_gain in ROLL_RATE_GAINS:
        cells.append(summarize(run_diagnostic(
            agents=AGENTS,
            episodes=EPISODES,
            seed=SEED,
            controller="direct",
            target_speed_m_s=TARGET_SPEED_M_S,
            episode_seconds=EPISODE_SECONDS,
            roll_per_m=ROLL_POSITION_GAIN,
            roll_rate_per_m_s=rate_gain,
            true_velocity_damping=True,
        )))
    selected = max(cells, key=selection_key)
    print(json.dumps({
        "schema": "vq2_native_oracle_true_velocity_surface_v1",
        "tag": "vq2_sf006_native_oracle_true_velocity_surface",
        "common_seed": SEED,
        "episodes_per_cell": EPISODES,
        "true_velocity_damping": True,
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
