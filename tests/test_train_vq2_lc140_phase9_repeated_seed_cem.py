from __future__ import annotations

import numpy as np
import torch

import scripts.train_vq2_lc140_phase9_repeated_seed_cem as lc140


def test_lc140_source_lock() -> None:
    payload = lc140.verify_inputs()
    assert payload["numerically_admitted"]
    assert lc140.TOTAL_AGENTS == 512
    assert lc140.ENV_SEED_INDEX == 15
    assert lc140.TARGET_PHASE == 9
    assert lc140.TARGET_RAW_INDEX == 10
    assert np.array_equal(lc140.INITIAL_MEAN, np.zeros(4, dtype=np.float32))


def test_lc140_candidates_are_zero_plus_bounded_antithetic_pairs() -> None:
    mean = np.zeros(4, dtype=np.float32)
    deltas = lc140.candidate_deltas(mean, lc140.INITIAL_STD, generation=1)
    assert deltas.shape == (512, 4)
    assert np.array_equal(deltas[0], np.zeros(4, dtype=np.float32))
    assert np.array_equal(deltas[1], mean)
    assert np.all(np.abs(deltas) <= lc140.MAXIMUM_ABS_DELTA + 1e-7)
    assert np.allclose(deltas[2:257], -deltas[257:])


def test_lc140_checkpoint_changes_only_phase9_output_bias() -> None:
    parent = lc140.verify_inputs()
    delta = np.array((0.01, -0.02, 0.03, -0.004), dtype=np.float32)
    checkpoint, frozen = lc140.checkpoint_with_delta(parent, delta)
    assert frozen
    assert checkpoint["schema"] == lc140.CHECKPOINT_SCHEMA
    assert checkpoint[lc140.FROZEN_STATE_FIELD]
    assert checkpoint[lc140.DELTA_FIELD] == delta.tolist()
    before = parent["model_state"]
    after = checkpoint["model_state"]
    for name in before:
        if name == lc140.PHASE_OUTPUT_BIAS:
            keep = torch.arange(before[name].shape[0]) != lc140.TARGET_PHASE
            assert torch.equal(before[name][keep], after[name][keep])
            assert torch.equal(
                after[name][lc140.TARGET_PHASE],
                before[name][lc140.TARGET_PHASE] + torch.from_numpy(delta),
            )
        else:
            assert torch.equal(before[name], after[name])


def test_native_seed_offset_hook_is_default_preserving() -> None:
    source = (lc140.ROOT / "src/vecenv.h").read_text()
    assert 'dict_get_unsafe(\n        vec_kwargs, "env_seed_index_offset")' in source
    assert "env_seed_index += env_seed_index_offset;" in source


def test_selected_phase_return_offset_tracks_target_index() -> None:
    source = (lc140.ROOT / "scripts/train_vq2_lc140_phase9_repeated_seed_cem.py").read_text()
    assert "1_000.0 * TARGET_RAW_INDEX" in source
