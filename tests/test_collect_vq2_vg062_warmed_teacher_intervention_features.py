from __future__ import annotations

import numpy as np
import pytest
import torch

import scripts.collect_vq2_vg062_warmed_teacher_intervention_features as collector
from pufferlib.vq2_recurrent import RecurrentActorOutput

from scripts.collect_vq2_vg062_warmed_teacher_intervention_features import (
    AGENTS,
    EPISODES,
    FEATURE_DTYPE,
    HIDDEN_SIZE,
    INTERVENTION_PHASE_MIN,
    MINIMUM_RECORDS,
    collection_passes,
    select_intervention_plant_actions,
)


def test_feature_dtype_is_compact_and_source_locked() -> None:
    assert FEATURE_DTYPE.itemsize == HIDDEN_SIZE * 2 + 4 * 4 * 2 + 6
    assert FEATURE_DTYPE["hidden"].shape == (HIDDEN_SIZE,)
    assert FEATURE_DTYPE["hidden"].base == np.dtype("<f2")


def test_selector_changes_only_active_public_phase_one_and_later() -> None:
    student = np.arange(20, dtype=np.float32).reshape(5, 4) / 20
    teacher = -student
    active = np.array([True, True, True, False, True])
    phase = np.array([0, 1, 4, 8, 16])
    plant, mask = select_intervention_plant_actions(student, teacher, active, phase)
    np.testing.assert_array_equal(mask, [False, True, True, False, True])
    np.testing.assert_array_equal(plant[~mask], student[~mask])
    np.testing.assert_array_equal(plant[mask], teacher[mask])
    assert INTERVENTION_PHASE_MIN == 1


def test_configured_selector_reads_runtime_phase_boundary(monkeypatch) -> None:
    student = np.arange(20, dtype=np.float32).reshape(5, 4) / 20
    teacher = -student
    active = np.ones(5, dtype=bool)
    phase = np.array([0, 1, 3, 4, 16])
    monkeypatch.setattr(collector, "INTERVENTION_PHASE_MIN", 4)
    plant, mask = collector.select_configured_intervention_plant_actions(
        student, teacher, active, phase
    )
    np.testing.assert_array_equal(mask, [False, False, False, True, True])
    np.testing.assert_array_equal(plant[~mask], student[~mask])
    np.testing.assert_array_equal(plant[mask], teacher[mask])


def test_collection_admission_requires_training_only_successful_intervention() -> None:
    metrics = {
        "env/n": float(EPISODES), "env/success_rate": 0.95,
        "env/crash": 0.0, "env/out_of_order": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
        "env/ordered_gate0_sampled": 1.0,
    }
    for count in range(1, 17):
        metrics[f"env/gate_count{count}_episode"] = 0.125 if 5 <= count <= 12 else 0.0
    report = {
        "metrics": metrics,
        "feature_records": MINIMUM_RECORDS,
        "teacher_plant_actions_executed": MINIMUM_RECORDS,
        "feature_phase_records": [0] + [1] * 11 + [0] * 5,
        "minimum_teacher_phase_index": 1,
        "executed_action_max_error": 0.0,
        "phase_changes_off_tick": 0, "phase_decreases": 0, "phase_skips": 0,
        "raw_phase_encoding_max_error": 0.0, "nonfinite_values": 0,
        "teacher_action_envelope_violations": 0,
        "feature_sha256": "a" * 64,
        "safety": {
            "flight_sim_packets_sent": 0, "submission_authorized": False,
            "runtime_teacher_authorized": False,
        },
    }
    assert collection_passes(report)
    report["safety"]["runtime_teacher_authorized"] = True
    assert not collection_passes(report)
    report["safety"]["runtime_teacher_authorized"] = False
    report["feature_records"] -= 1
    assert not collection_passes(report)


def test_fixed_collection_shape() -> None:
    assert AGENTS == EPISODES == 512


class _DummyActor:
    def forward_step(self, observation, state):
        pre_tanh = observation[:, :4] + state[0, :, :4]
        output = RecurrentActorOutput(
            mean=torch.tanh(pre_tanh),
            pre_tanh_mean=pre_tanh,
            log_std=torch.zeros_like(pre_tanh),
        )
        return output, state + observation[:, :1].unsqueeze(0)


def test_chunked_actor_inference_preserves_row_order_and_state() -> None:
    actor = _DummyActor()
    observation = torch.arange(70, dtype=torch.float32).reshape(10, 7) / 70
    state = torch.arange(60, dtype=torch.float32).reshape(1, 10, 6) / 60
    direct, direct_state = actor.forward_step(observation, state)
    chunked, chunked_state = collector.forward_actor_chunked(
        actor, observation, state, chunk_size=4
    )
    torch.testing.assert_close(chunked.mean, direct.mean, rtol=0, atol=0)
    torch.testing.assert_close(
        chunked.pre_tanh_mean, direct.pre_tanh_mean, rtol=0, atol=0
    )
    torch.testing.assert_close(chunked.log_std, direct.log_std, rtol=0, atol=0)
    torch.testing.assert_close(chunked_state, direct_state, rtol=0, atol=0)


def test_chunked_actor_inference_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        collector.forward_actor_chunked(
            _DummyActor(), torch.zeros(2, 7), torch.zeros(1, 2, 6), chunk_size=0
        )
