import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_gate2_severe_vertical_n190 as verifier


def test_analyze_isolates_severe_floor_to_thrust(tmp_path):
    values = [0.0] * 32
    values[0] = math.tanh(-8.0 / 5.0)
    values[2] = math.tanh(-2.0 / 3.0)
    values[6] = 1.0
    values[10] = 1.0
    values[11] = math.tanh(15.0 / 10.0)
    values[13] = math.tanh(-0.5 / 5.0)
    values[23] = 1.0 / 6.0
    values[25] = 1.0
    path = tmp_path / "policy_bounded_050_attempt_001.json"
    path.write_text(
        json.dumps(
            {
                "official_active_gate_index": 1,
                "policy_trace": {
                    "samples": [
                        {
                            "elapsed_s": 1.0,
                            "official_active_gate_index": 1,
                            "observation": values,
                            "normalized_action": [0.1, 0.2, 0.4, 0.5],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    report = verifier.analyze(path)
    assert report["severe_active_samples"] == 1
    assert report["severe_changed_samples"] == 1
    assert report["max_non_thrust_error"] == 0.0
    assert report["rows"][0]["governed_action"] == [0.1, 0.2, 1.0, 0.5]
