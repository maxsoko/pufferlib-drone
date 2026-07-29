import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "verify_six_gate_composite_dataset",
    SCRIPTS / "verify_six_gate_composite_dataset.py",
)
verify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify
SPEC.loader.exec_module(verify)


def test_verify_dataset_replays_resets_phases_and_clamped_actions(tmp_path):
    rows = []
    for episode in range(2):
        for gate in (3, 4, 5):
            row = np.zeros(37, dtype=np.float32)
            row[23] = gate / 6.0
            row[32:36] = [2.0, -2.0, 0.25, -0.25]
            row[36] = float(gate == 3)
            rows.append(row)
    dataset = tmp_path / "dataset.bin"
    np.stack(rows).tofile(dataset)
    resets = []

    report = verify.verify_dataset(
        dataset,
        infer_fn=lambda _obs: [1.0, -1.0, 0.25, -0.25],
        reset_fn=lambda: resets.append(True),
        episodes=2,
        action_atol=1e-7,
    )

    assert report["passed"] is True
    assert report["episodes_replayed"] == 2
    assert report["records_replayed"] == 6
    assert report["phase_counts"] == {"3": 2, "4": 2, "5": 2}
    assert report["action_max_error"] == 0.0
    assert len(resets) == 2
