import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "crop_teacher_dataset_prefix.py"
SPEC = importlib.util.spec_from_file_location("crop_teacher_dataset_prefix", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_crop_teacher_dataset_prefix_preserves_episode_resets():
    records = np.zeros((9, module.RECORD_WIDTH), dtype=np.float32)
    records[[0, 5], -1] = 1.0

    cropped = module.crop_episode_prefixes(records, 3)

    assert cropped.shape == (6, module.RECORD_WIDTH)
    assert np.flatnonzero(cropped[:, -1] > 0.5).tolist() == [0, 3]


def test_crop_teacher_dataset_prefix_rejects_bad_contract():
    with pytest.raises(ValueError):
        module.crop_episode_prefixes(np.zeros((2, module.RECORD_WIDTH)), 0)
    with pytest.raises(ValueError):
        module.crop_episode_prefixes(np.zeros((2, module.RECORD_WIDTH - 1)), 2)
