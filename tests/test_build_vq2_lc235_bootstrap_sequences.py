import numpy as np

import scripts.build_vq2_lc235_bootstrap_sequences as lc235


def test_source_lock_and_phase_scope() -> None:
    parent = lc235.verify_inputs()
    assert parent["model"]["phase2_sequence_length"] == 1433
    assert (lc235.FIRST_PHASE, lc235.LAST_PHASE) == (3, 23)


def test_sequence_packing_is_deterministic() -> None:
    sequences = {
        3: np.ones((2, 4), dtype=np.float32),
        5: np.full((3, 4), 2.0, dtype=np.float32),
    }
    offsets, actions = lc235.pack_sequences(sequences)
    assert offsets[3].tolist() == [0, 2]
    assert offsets[4].tolist() == [-1, 0]
    assert offsets[5].tolist() == [2, 3]
    assert actions.shape == (5, 4)


def test_actor_observation_is_single_row() -> None:
    legal = np.zeros(lc235.LEGAL_OBS_SIZE, dtype=np.float32)
    observation = lc235.actor_observation(legal, 0.5)
    assert observation.shape == (lc235.OBSERVATION_SIZE,)
    assert observation[-1] == np.float32(0.5)


def test_authority_stays_offline() -> None:
    text = lc235.PREREGISTRATION.read_text()
    assert "sends zero FlightSim packets" in text
    assert "VQ2 Submission remains forbidden" in text
