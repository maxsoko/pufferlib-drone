import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "verify_gate3_promoted_adapter_traces",
    SCRIPTS / "verify_gate3_promoted_adapter_traces.py",
)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def _sample(mode: str) -> dict:
    observation = [0.0] * 23
    observation[10] = 1.0
    if mode == "adaptive":
        forward, right, forward_rate, right_rate, roll = 20.0, -4.0, -6.0, 0.0, 1.0
    elif mode == "early":
        forward, right, forward_rate, right_rate, roll = 18.0, -1.0, -4.5, 0.0, 0.9
    else:
        forward, right, forward_rate, right_rate, roll = 5.0, -0.5, -5.0, 0.5, -1.0
    observation[11] = float(np.tanh(forward / 10.0))
    observation[12] = float(np.tanh(right / 5.0))
    observation[0] = float(np.tanh(forward_rate / 5.0))
    observation[1] = float(np.tanh(right_rate / 3.0))
    return {
        "official_active_gate_index": 2,
        "observation": observation,
        "normalized_action": [0.0, roll, 0.0, 0.0],
    }


def test_verify_traces_requires_and_accepts_all_promoted_modes(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "acceptance_passed": True,
                "official_active_gate_index": 3,
                "policy_trace": {
                    "samples": [
                        _sample("adaptive"),
                        _sample("early"),
                        _sample("counter"),
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    result = verify.verify_traces([path])

    assert result["passed"] is True
    assert result["gate3_samples"] == 3
    assert result["mode_counts"] == {"adaptive": 1, "counter": 1, "early": 1}
    assert result["projection_max_error"] == 0.0
