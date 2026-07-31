from __future__ import annotations

import numpy as np
import pytest

from pufferlib.vq2_public_phase import (
    ENGINE_GATE_CAP,
    LONG_COURSE_GATE_CAP,
    OFFICIAL_PROGRESS_SCALE,
    PUBLIC_STATUS_INTERVAL_STEPS,
    encode_public_gate_index,
    encode_unbounded_gate_progress,
    held_public_phase,
    held_unbounded_gate_progress,
)


def test_public_phase_uses_fixed_engine_cap() -> None:
    assert ENGINE_GATE_CAP == 16
    assert encode_public_gate_index(0) == np.float32(0.0)
    assert encode_public_gate_index(1) == np.float32(1.0 / 16.0)
    assert encode_public_gate_index(11) == np.float32(11.0 / 16.0)
    assert encode_public_gate_index(16) == np.float32(1.0)
    assert encode_public_gate_index(99) == np.float32(1.0)
    assert encode_public_gate_index(-1) == np.float32(0.0)


def test_public_phase_contains_no_total_gate_count() -> None:
    # The same official index is byte-identical on 5- and 12-gate episodes.
    phase_five = encode_public_gate_index(3)
    phase_twelve = encode_public_gate_index(3)
    assert phase_five.tobytes() == phase_twelve.tobytes()


def test_public_phase_is_held_at_four_hz_on_the_64_hz_policy_clock() -> None:
    assert PUBLIC_STATUS_INTERVAL_STEPS == 16
    raw = np.zeros(40, dtype=np.int32)
    raw[5:] = 1
    raw[21:] = 2
    held = held_public_phase(raw)
    assert np.all(held[:16] == np.float32(0.0))
    assert np.all(held[16:32] == np.float32(1.0 / 16.0))
    assert np.all(held[32:] == np.float32(2.0 / 16.0))


def test_public_phase_rejects_non_integer_or_invalid_layout() -> None:
    with pytest.raises(TypeError, match="integer"):
        encode_public_gate_index(1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integers"):
        held_public_phase([0.0, 1.0])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="positive"):
        held_public_phase([0, 1], interval_steps=0)


def test_unbounded_progress_represents_the_observed_long_course() -> None:
    assert OFFICIAL_PROGRESS_SCALE == 6.0
    assert LONG_COURSE_GATE_CAP == 32
    assert encode_unbounded_gate_progress(6) == np.float32(1.0)
    assert encode_unbounded_gate_progress(20) == np.float32(20.0 / 6.0)
    assert encode_unbounded_gate_progress(32) == np.float32(32.0 / 6.0)
    assert encode_unbounded_gate_progress(20) > 1.0


def test_unbounded_progress_holds_without_saturation() -> None:
    raw = np.full(40, 19, dtype=np.int32)
    raw[17:] = 20
    held = held_unbounded_gate_progress(raw)
    assert np.all(held[:32] == np.float32(19.0 / 6.0))
    assert np.all(held[32:] == np.float32(20.0 / 6.0))
    with pytest.raises(ValueError, match="nonnegative"):
        encode_unbounded_gate_progress(-1)
