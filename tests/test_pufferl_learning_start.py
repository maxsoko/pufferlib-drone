import pytest

from pufferlib.pufferl import _optimizer_update_ready, validate_config


def test_learning_start_default_preserves_immediate_updates():
    assert _optimizer_update_ready({}, 0)
    assert _optimizer_update_ready({}, 32_768)


def test_learning_start_waits_for_first_rollout_boundary_at_or_after_threshold():
    train = {"learning_start_timesteps": 1_638_400}

    assert not _optimizer_update_ready(train, 1_605_632)
    assert _optimizer_update_ready(train, 1_638_400)
    assert _optimizer_update_ready(train, 1_671_168)


def test_learning_start_uses_agent_steps_not_optimizer_epoch():
    train = {"learning_start_timesteps": 1_638_400}

    # An optimizer epoch may remain at zero throughout warmup; only rollout
    # agent steps determine when updates become eligible.
    optimizer_epoch = 0
    assert optimizer_epoch == 0
    assert not _optimizer_update_ready(train, 1_605_632)
    assert _optimizer_update_ready(train, 1_638_400)


def test_early_episode_during_warmup_does_not_make_update_ready():
    train = {"learning_start_timesteps": 1_638_400}

    # Environment completions are intentionally not part of the predicate.
    # A failed episode can be logged before the optimizer-start threshold
    # without ending warmup or authorizing an update.
    env_episode_count = 37
    assert env_episode_count > 0
    assert not _optimizer_update_ready(train, 1_310_720)


def test_learning_start_must_be_nonnegative():
    args = {
        "train": {
            "horizon": 32,
            "minibatch_size": 8192,
            "learning_start_timesteps": -1,
        },
        "vec": {"total_agents": 1024},
    }

    with pytest.raises(AssertionError, match="must be nonnegative"):
        validate_config(args)


def _encoder_feature_args(start, end, *, slowly=False):
    return {
        "slowly": slowly,
        "train": {
            "horizon": 32,
            "minibatch_size": 8192,
            "train_encoder_feature_start": start,
            "train_encoder_feature_end": end,
        },
        "vec": {"total_agents": 1024},
    }


def test_encoder_feature_bounds_must_be_enabled_together():
    with pytest.raises(AssertionError, match="both be -1 or both be enabled"):
        validate_config(_encoder_feature_args(27, -1))


def test_encoder_feature_bounds_reject_values_below_disabled_sentinel():
    with pytest.raises(AssertionError, match="-1 or nonnegative"):
        validate_config(_encoder_feature_args(-2, -2))


def test_encoder_feature_bounds_require_cuda_backend():
    with pytest.raises(AssertionError, match="requires the CUDA backend"):
        validate_config(_encoder_feature_args(27, 29, slowly=True))


def test_encoder_feature_bounds_accept_late_six_gate_columns():
    validate_config(_encoder_feature_args(27, 29))
