import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_gate3_severe_acquisition as verifier


def _observation(*, forward, right, yaw, gate=2, size=32):
    values = [0.0] * size
    values[6] = 1.0
    values[10] = 1.0
    values[11] = math.tanh(forward / 10.0)
    values[12] = math.tanh(right / 5.0)
    values[14] = yaw / (math.pi / 4.0)
    if size == 32:
        values[23] = gate / 6.0
        values[24 + gate] = 1.0
    return values


def _write(path, rows, official_index=2):
    path.write_text(
        json.dumps(
            {
                "official_active_gate_index": official_index,
                "policy_trace": {"samples": rows},
            }
        ),
        encoding="utf-8",
    )


def test_analyze_changes_only_severe_acquisition_epoch(tmp_path):
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
                "observation": _observation(forward=28.0, right=-4.0, yaw=-0.1),
                "normalized_action": [0.1, 0.2, 0.3, 0.4],
            },
        ],
    )
    report = verifier.analyze(path)
    assert report["name"] == "candidate046"
    assert report["activations"] == 1
    assert report["releases"] == 1
    assert report["changed_samples"] == 1
    assert report["changed_non_gate3"] == 0
    assert report["rows"][0]["governed_action"][:3] == [-0.24, 0.0, 0.0]


def test_historical_23_value_observation_reconstructs_phase():
    sample = {
        "official_active_gate_index": 2,
        "observation": _observation(
            forward=10.0, right=0.0, yaw=0.0, size=23
        ),
    }
    values = verifier._current_observation(sample)
    assert values is not None
    assert values.shape == (32,)
    assert values[23] == 2 / 6
    assert values[26] == 1.0
