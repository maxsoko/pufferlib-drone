import pytest

from scripts.collect_vq2_gate_event_windows import (
    EVENT_WINDOW_BOUNDARY_FP32_SCHEMA,
    EVENT_WINDOW_BOUNDARY_SCHEMA,
    EVENT_WINDOW_EXACT_HISTORY_SCHEMA,
    _configure_event_collection,
    _window_indices,
)


def test_configure_event_collection_is_isolated_and_fail_closed() -> None:
    source = {
        "env": {
            "gate_local_start_curriculum": 0,
            "gate_local_start_probability": 0.2,
            "mixed_start_curriculum": 1,
            "segment_start_probability": 0.5,
        }
    }
    configured = _configure_event_collection(source, offset_min=0.75, offset_max=1.5)
    assert source["env"]["gate_local_start_curriculum"] == 0
    assert configured["env"]["gate_local_start_curriculum"] == 1
    assert configured["env"]["gate_local_start_probability"] == 1.0
    assert configured["env"]["gate_local_start_offset_min"] == 0.75
    assert configured["env"]["gate_local_start_offset_max"] == 1.5
    assert configured["env"]["mixed_start_curriculum"] == 0
    assert configured["env"]["segment_start_probability"] == 0.0
    with pytest.raises(ValueError):
        _configure_event_collection(source, offset_min=2.0, offset_max=1.0)


def test_configure_full_start_event_collection_is_uninterrupted() -> None:
    source = {
        "env": {
            "gate_local_start_curriculum": 1,
            "gate_local_start_probability": 1.0,
            "mixed_start_curriculum": 1,
            "segment_start_probability": 0.5,
            "evaluation_episode_limit": 3,
            "evaluation_episode_offset": 0,
            "use_custom_start": 0,
            "start_gate_index": 0,
        }
    }
    configured = _configure_event_collection(
        source,
        offset_min=0.75,
        offset_max=1.5,
        full_start_only=True,
        episode_offset=600,
    )
    assert source["env"]["gate_local_start_curriculum"] == 1
    assert configured["env"]["gate_local_start_curriculum"] == 0
    assert configured["env"]["gate_local_start_probability"] == 0.0
    assert configured["env"]["mixed_start_curriculum"] == 0
    assert configured["env"]["segment_start_probability"] == 0.0
    assert configured["env"]["evaluation_episode_limit"] == 0
    assert configured["env"]["evaluation_episode_offset"] == 600
    with pytest.raises(ValueError):
        _configure_event_collection(
            source,
            offset_min=0.75,
            offset_max=1.5,
            full_start_only=True,
            episode_offset=-1,
        )


def test_configure_full_start_event_collection_refuses_later_gate_custom_start() -> None:
    source = {
        "env": {
            "use_custom_start": 1,
            "start_gate_index": 1,
        }
    }
    with pytest.raises(RuntimeError, match="later-gate start"):
        _configure_event_collection(
            source,
            offset_min=0.75,
            offset_max=1.5,
            full_start_only=True,
        )


def test_window_indices_are_chronological_across_ring_wrap() -> None:
    assert _window_indices(4, 4).tolist() == [0, 1, 2, 3]
    assert _window_indices(6, 4).tolist() == [2, 3, 0, 1]
    with pytest.raises(ValueError):
        _window_indices(3, 4)


def test_boundary_schema_is_versioned_separately() -> None:
    assert EVENT_WINDOW_BOUNDARY_SCHEMA.endswith("_v2")
    assert EVENT_WINDOW_BOUNDARY_FP32_SCHEMA.endswith("_v3")
    assert EVENT_WINDOW_BOUNDARY_FP32_SCHEMA != EVENT_WINDOW_BOUNDARY_SCHEMA
    assert EVENT_WINDOW_EXACT_HISTORY_SCHEMA.endswith("_v4")
    assert EVENT_WINDOW_EXACT_HISTORY_SCHEMA != EVENT_WINDOW_BOUNDARY_FP32_SCHEMA
