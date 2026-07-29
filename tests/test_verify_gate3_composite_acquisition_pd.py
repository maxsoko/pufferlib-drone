import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_gate3_composite_acquisition_pd as verifier


def _observation(*, forward, right, yaw=-0.1, closing=8.0, gate=2, size=32):
    values = [0.0] * size
    values[0] = math.tanh(-closing / 5.0)
    values[6] = 1.0
    values[10] = 1.0
    values[11] = math.tanh(forward / 10.0)
    values[12] = math.tanh(right / 5.0)
    values[14] = yaw / (math.pi / 4.0)
    if size == 32:
        values[23] = gate / 6.0
        values[24 + gate] = 1.0
    return values


def _write(path, rows):
    path.write_text(
        json.dumps(
            {
                "official_active_gate_index": 2,
                "policy_trace": {"samples": rows},
            }
        ),
        encoding="utf-8",
    )


def test_analyze_separates_far_severe_and_close_projected(tmp_path):
    path = tmp_path / "policy_bounded_046_attempt_001.json"
    _write(
        path,
        [
            {
                "elapsed_s": 1.0,
                "official_active_gate_index": 2,
                "observation": _observation(forward=28.0, right=-16.0, yaw=-0.5),
                "normalized_action": [0.8, 1.0, -1.0, 0.0],
            },
            {
                "elapsed_s": 1.1,
                "official_active_gate_index": 2,
                "observation": _observation(forward=10.0, right=4.0, yaw=0.1),
                "normalized_action": [0.1, 0.2, 0.3, 0.4],
            },
            {
                "elapsed_s": 1.2,
                "official_active_gate_index": 2,
                "observation": _observation(forward=9.5, right=3.5, yaw=0.08),
                "normalized_action": [0.1, 0.2, 0.3, 0.4],
            },
        ],
    )
    report = verifier.analyze(path)
    assert report["severe_activations"] == 1
    assert report["branch_counts"]["severe"] == 1
    assert report["branch_counts"]["severe_release"] == 1
    assert report["projected_activations"] == 1
    assert report["branch_counts"]["projected"] == 1
    assert report["changed_non_gate3"] == 0


def test_historical_observation_reconstructs_six_gate_phase():
    sample = {
        "official_active_gate_index": 2,
        "observation": _observation(forward=10.0, right=0.0, size=23),
    }
    values = verifier._current_observation(sample)
    assert values is not None
    assert values.shape == (32,)
    assert values[23] == 2 / 6
    assert values[26] == 1.0
