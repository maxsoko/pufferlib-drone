import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import verify_gate1_severe_direct_adapter_traces as verify


def _observation(*, forward, right, forward_rate, right_rate):
    values = np.zeros(32, dtype=np.float32)
    values[10] = 1.0
    values[11] = np.tanh(np.float32(forward / 10.0))
    values[12] = np.tanh(np.float32(right / 5.0))
    values[0] = np.tanh(np.float32(forward_rate / 5.0))
    values[1] = np.tanh(np.float32(right_rate / 3.0))
    return values


def test_severe_direct_floor_keeps_candidate005_family():
    observation = _observation(
        forward=12.468,
        right=-1.13,
        forward_rate=-5.192,
        right_rate=-0.084,
    )
    projected, mode, floor, direct_right = verify.severe_direct_positive_intercept(
        observation, [0.0, 0.33, 0.0, 0.0]
    )

    assert mode == "adaptive"
    assert direct_right < -0.8
    assert floor is not None
    assert projected[1] == floor


def test_severe_direct_floor_excludes_candidate006_moderate_family():
    observation = _observation(
        forward=15.709,
        right=-0.466,
        forward_rate=-4.078,
        right_rate=-0.145,
    )
    base = [0.0, 0.22, 0.0, 0.0]
    projected, mode, floor, direct_right = verify.severe_direct_positive_intercept(
        observation, base.copy()
    )

    assert mode == "early"
    assert direct_right > -0.8
    assert floor == 0.9
    assert projected == base


def test_severe_direct_floor_has_margin_from_old_boundary():
    observation = _observation(
        forward=16.0,
        right=-0.300001,
        forward_rate=-4.01,
        right_rate=-0.072,
    )
    base = [0.0, 0.1, 0.0, 0.0]
    projected, _mode, _floor, direct_right = (
        verify.severe_direct_positive_intercept(observation, base.copy())
    )

    assert direct_right > -0.8
    assert projected == base


def test_replay_n145_failure_identifies_first_adapter_change(tmp_path, monkeypatch):
    observation = _observation(
        forward=12.468,
        right=-1.13,
        forward_rate=-5.192,
        right_rate=-0.084,
    )
    base = [0.0, 0.33, 0.0, 0.0]
    recorded, _mode, _floor, _right = verify.severe_direct_positive_intercept(
        observation, base.copy()
    )
    report_path = tmp_path / "candidate007.json"
    report_path.write_text(
        json.dumps(
            {
                "acceptance_passed": False,
                "official_active_gate_index": 0,
                "policy_trace": {
                    "inference_ticks": 1,
                    "samples": [
                        {
                            "elapsed_s": 1.25,
                            "official_active_gate_index": 0,
                            "observation": observation.tolist(),
                            "normalized_action": recorded,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")

    class DummyModel:
        def reset_state(self):
            return None

        def infer(self, _observation):
            return base

    monkeypatch.setattr(
        verify.CheckpointPolicy,
        "load",
        lambda *_args, **_kwargs: DummyModel(),
    )

    result = verify._replay_n145_failure(report_path, checkpoint)

    assert result["trace_samples"] == 1
    assert result["recorded_n145_changes_from_base"] == 1
    assert result["proposal_changes_from_base"] == 1
    assert result["proposal_recorded_mismatches"] == 0
    assert result["first_recorded_change_from_base"]["sample"] == 0
