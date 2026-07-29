import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_gate2_vertical_margin_n189 as verifier


def test_analyze_counts_only_weak_thrust_changes(tmp_path):
    values = [0.0] * 32
    values[0] = math.tanh(-8.0 / 5.0)
    values[2] = math.tanh(-2.0 / 3.0)
    values[6] = 1.0
    values[10] = 1.0
    values[11] = math.tanh(5.0 / 10.0)
    values[13] = math.tanh(-0.2 / 5.0)
    values[23] = 1.0 / 6.0
    values[25] = 1.0
    path = tmp_path / "policy_bounded_047_attempt_001.json"
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
                            "normalized_action": [0.1, 0.2, 0.0, 0.4],
                        },
                        {
                            "elapsed_s": 1.1,
                            "official_active_gate_index": 1,
                            "observation": values,
                            "normalized_action": [0.1, 0.2, 0.6, 0.4],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    report = verifier.analyze(path)
    assert report["active_samples"] == 2
    assert report["changed_samples"] == 1
    assert report["max_non_thrust_error"] == 0.0
    assert report["stronger_thrust_changes"] == 0
