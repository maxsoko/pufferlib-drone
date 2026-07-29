import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.verify_hybrid_prefix_evidence import (
    exact_hybrid_roundtrip,
    inferred_elapsed_denominator,
    legacy_callable_yaw_contract,
    reconstructed_legacy_confidence,
)


def test_reconstructs_visible_legacy_size_confidence():
    observation = [0.0] * 23
    observation[10] = 1.0
    observation[16] = 0.0625

    assert reconstructed_legacy_confidence(observation) == pytest.approx(0.03125)


def test_hidden_gate_has_zero_legacy_confidence():
    observation = [0.0] * 23
    observation[16] = 0.5

    assert reconstructed_legacy_confidence(observation) == 0.0


def test_reserved_bridge_roundtrips_archived_fields_and_live_previous_yaw():
    observation = [index / 100.0 for index in range(23)]
    observation[17] = 0.03125
    observation[22] = 2.0 / 3.0

    expected = np.asarray(observation, dtype=np.float32)
    expected[22] = np.float32(-0.375)
    assert exact_hybrid_roundtrip(
        observation, gate=2, previous_yaw_action=-0.375
    ) == expected.tolist()


def test_historical_callable_injected_previous_yaw_into_recurrent_field_22():
    contract = legacy_callable_yaw_contract(
        SCRIPTS / "policy_callable_final_gate_handoff.py"
    )

    assert contract["passed"] is True
    assert contract["model_field_22"] == "previous normalized yaw action"


def test_infers_elapsed_denominator_from_trace_samples():
    samples = [
        {"elapsed_s": 2.625, "observation": [0.0] * 18 + [0.25]},
        {"elapsed_s": 5.25, "observation": [0.0] * 18 + [0.5]},
    ]

    assert inferred_elapsed_denominator(samples) == pytest.approx(10.5)
