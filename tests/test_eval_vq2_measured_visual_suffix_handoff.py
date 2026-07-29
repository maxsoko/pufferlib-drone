from __future__ import annotations

import numpy as np
import pytest
import torch

from pufferlib.vq2_informed import MASK_SIZE
from scripts import eval_vq2_measured_visual_suffix_handoff as handoff


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    values = mask.reshape(64, 64)
    yy, xx = np.indices((64, 64), dtype=np.float32)
    mass = values.sum()
    return float((values * xx).sum() / mass), float((values * yy).sum() / mass)


def test_mask_zoom_preserves_centroid_and_increases_mass():
    observation = np.zeros((1, 4119), dtype=np.float32)
    mask = observation[0, :MASK_SIZE].reshape(64, 64)
    mask[20:23, 39:42] = 1.0
    before = _centroid(mask)
    zoomed = handoff.centered_mask_zoom(observation, 1.75)
    after_mask = zoomed[0, :MASK_SIZE]
    after = _centroid(after_mask)
    assert after == pytest.approx(before, abs=0.05)
    assert after_mask.sum() > mask.sum()
    assert np.all(zoomed[:, MASK_SIZE:] == observation[:, MASK_SIZE:])


def test_mask_zoom_identity_is_exact_and_does_not_alias():
    rng = np.random.default_rng(7)
    observation = rng.normal(size=(2, 4119)).astype(np.float32)
    result = handoff.centered_mask_zoom(observation, 1.0)
    assert np.array_equal(result, observation)
    assert not np.shares_memory(result, observation)


@pytest.mark.parametrize("zoom", [0.0, -1.0, float("nan")])
def test_mask_zoom_rejects_invalid_scale(zoom: float):
    with pytest.raises(ValueError, match="finite and positive"):
        handoff.centered_mask_zoom(np.zeros((1, 4119), dtype=np.float32), zoom)


def test_prefix_and_measured_action_source_locks():
    prefix = handoff.load_warm_prefix()
    assert prefix["visual_observation"].shape == (208, 4119)
    assert prefix["legacy_observation"].shape == (208, 32)
    assert not prefix["suffix_selected"].any()
    action = handoff.load_measured_previous_action()
    assert action == pytest.approx([-0.044653, 0.231316, 0.1124859, 0.000208])


def test_measured_config_is_teacher_free_and_exact():
    try:
        from pufferlib import _C, pufferl
    except ImportError:
        pytest.skip("native Puffer binding unavailable")
    if getattr(_C, "env_name", None) != handoff.BACKEND_ENV_NAME:
        pytest.skip("drone_race_vision binding is not loaded")
    config, _ = handoff.measured_suffix_config(
        pufferl, agents=2, seed=43003, camera_pitch_rad=0.0
    )
    env = config["env"]
    assert env["num_gates"] == 6
    assert env["use_custom_start"] == 1
    assert env["start_gate_index"] == 1
    assert env["start_elapsed_time"] == pytest.approx(3.281)
    assert env["start_vx"] == pytest.approx(4.677)
    assert env["gate1_y"] == pytest.approx(8.70)
    assert env["gate2_x"] == pytest.approx(60.0)
    assert env["gate5_x"] == pytest.approx(150.0)
    assert env["teacher_action_blend"] == 0.0
    assert env["observable_gate_index_denominator"] == 6.0

    from pufferlib.torch_pufferl import _cpu_tensor

    vector = _C.create_vec(config, gpu=0)
    observations = _cpu_tensor(
        vector.obs_ptr, (2, handoff.ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (2,), torch.float32)
    try:
        vector.reset()
        phase = observations.numpy()[:, handoff.PHASE_PRIVILEGED_INDEX]
        assert phase == pytest.approx([1.0 / 6.0, 1.0 / 6.0])
        zero_actions = torch.zeros((2, handoff.ACTION_SIZE), dtype=torch.float32)
        vector.cpu_step(zero_actions.data_ptr())
        assert np.array_equal(terminals.numpy(), [0.0, 0.0])
    finally:
        vector.close()


def test_stepwise_suffix_warm_replays_source_trace():
    if not torch.cuda.is_available():
        pytest.skip("source trace was generated with CUDA")
    from scripts.eval_vq2_n294_visual_suffix_composite import load_suffix

    prefix = handoff.load_warm_prefix()
    device = torch.device("cuda")
    suffix, _ = load_suffix(device)
    replayed, state = handoff.warm_suffix_stepwise(
        suffix, prefix["visual_observation"], device=device
    )
    assert replayed == pytest.approx(prefix["suffix_action"], abs=5e-5)
    assert state.shape[1] == 1


def test_c007_candidate_is_source_locked_and_admitted():
    actor, payload, sources = handoff.load_candidate_suffix(
        torch.device("cpu"), "c007"
    )
    assert actor.training is False
    assert payload["combined_numerical_admission"] is True
    assert sources == (handoff.C007_CHECKPOINT, handoff.C007_REPORT)


def test_c009_candidate_is_source_locked_and_admitted():
    actor, payload, sources = handoff.load_candidate_suffix(
        torch.device("cpu"), "c009"
    )
    assert actor.training is False
    assert payload["combined_numerical_admission"] is True
    assert sources == (handoff.C009_CHECKPOINT, handoff.C009_REPORT)


def test_c012_candidate_is_source_locked_and_admitted():
    actor, payload, sources = handoff.load_candidate_suffix(
        torch.device("cpu"), "c012"
    )
    assert actor.training is False
    assert payload["combined_numerical_admission"] is True
    assert sources == (handoff.C012_CHECKPOINT, handoff.C012_REPORT)


def test_unknown_candidate_is_rejected():
    with pytest.raises(ValueError, match="unsupported suffix candidate"):
        handoff.load_candidate_suffix(torch.device("cpu"), "unknown")


def test_candidate_label_is_not_reused_for_recurrent_state():
    source = handoff.Path(handoff.__file__).read_text()
    assert "suffix_output, suffix_candidate =" not in source
    assert '"suffix_candidate": suffix_candidate' in source
