import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "collect_vq1_v3391_telemetry_teacher.py"
SPEC = importlib.util.spec_from_file_location(
    "collect_vq1_v3391_telemetry_teacher", MODULE_PATH
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_exact_teacher_rollout_completes_all_six_gates():
    records, report = module.collect_dataset(
        ROOT / "config" / "drone_race_vq1_v3391_telemetry.ini",
        episodes=2,
        seed=3391,
    )

    assert report["success_rate"] == 1.0
    assert report["episodes"] == 2
    assert report["mean_completion_time_s"] < 25.0
    assert report["maximum_crossing_radial_m"] < 0.10
    assert records.shape[1] == 37
    starts = np.flatnonzero(records[:, -1] > 0.5)
    assert len(starts) == 2
    assert records[0, 24] == 1.0
    assert np.allclose(records[:, 35], 0.0)
