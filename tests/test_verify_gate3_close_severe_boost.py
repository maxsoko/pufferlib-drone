import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.verify_gate3_close_severe_boost import convert_observation


def test_convert_legacy_observation_adds_gate3_onehot():
    values = convert_observation([0.0] * 23, gate=2)

    assert values.shape == (32,)
    assert values[23] == np.float32(2.0 / 6.0)
    assert values[26] == 1.0
    assert sum(values[24:30]) == 1.0


def test_convert_current_observation_preserves_all_values():
    observation = [index / 100.0 for index in range(32)]

    assert convert_observation(observation, gate=2).tolist() == np.asarray(
        observation, dtype=np.float32
    ).tolist()
