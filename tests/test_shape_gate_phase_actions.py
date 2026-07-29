from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from shape_gate_phase_actions import shape_gate_action


def test_shape_gate_action_changes_only_selected_phase_and_action() -> None:
    records = np.zeros((4, 37), dtype=np.float32)
    records[1, 23] = 1.0
    records[2, 24] = 1.0
    records[3, 25] = 1.0
    records[:, 33] = [0.1, 0.2, 0.3, 0.4]

    shaped = shape_gate_action(
        records,
        gate_index=2,
        action_index=1,
        scale=1.0,
        offset=0.25,
    )

    assert shaped == 1
    assert records[:, 33] == pytest.approx([0.1, 0.2, 0.55, 0.4])
    assert np.count_nonzero(records[:, 32]) == 0


def test_shape_gate_action_clips_labels() -> None:
    records = np.zeros((1, 37), dtype=np.float32)
    records[0, 24] = 1.0
    records[0, 33] = 0.9
    shape_gate_action(
        records,
        gate_index=2,
        action_index=1,
        scale=2.0,
        offset=0.0,
    )
    assert records[0, 33] == 1.0


def test_shape_gate_action_ramps_near_gate_without_touching_other_phases() -> None:
    records = np.zeros((4, 37), dtype=np.float32)
    records[:, 24] = [1.0, 1.0, 1.0, 0.0]
    records[3, 25] = 1.0
    records[:, 16] = [0.20, 0.35, 0.50, 0.50]
    records[:, 33] = 0.25

    shaped = shape_gate_action(
        records,
        gate_index=2,
        action_index=1,
        scale=1.0,
        offset=-0.30,
        min_apparent_size=0.20,
        full_apparent_size=0.50,
    )

    assert shaped == 2
    assert records[:, 33] == pytest.approx([0.25, 0.10, -0.05, 0.25])
