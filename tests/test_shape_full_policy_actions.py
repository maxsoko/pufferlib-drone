from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from shape_full_policy_actions import shape_close_roll


def test_shape_close_roll_ramps_only_selected_phase() -> None:
    records = np.zeros((3, 28), dtype=np.float32)
    records[:, 22] = [1 / 3, 1 / 3, 2 / 3]
    records[:, 16] = [0.25, 0.5, 0.5]

    shaped = shape_close_roll(
        records,
        race_phase=1 / 3,
        min_apparent_size=0.2,
        full_apparent_size=0.5,
        roll_offset=0.6,
    )

    assert shaped == 2
    assert records[:, 24] == pytest.approx([0.1, 0.6, 0.0])
