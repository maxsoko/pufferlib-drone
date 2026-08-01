from __future__ import annotations

import scripts.train_vq2_lc132_phase8_progress_return_ppo as lc132


def test_lc132_source_lock_and_reward_contract() -> None:
    payload = lc132.verify_inputs()
    assert payload["numerically_admitted"]
    assert lc132.REWARD_OVERRIDES["w_progress"] == 5.0
    assert lc132.REWARD_OVERRIDES["w_ordered_gate"] == 30.0
    assert lc132.REWARD_OVERRIDES["w_body_rate"] == 0.0
    assert lc132.REWARD_OVERRIDES["w_action_teacher"] == 0.0


def test_lc132_nested_configuration_restores_modules() -> None:
    dense_before = (
        lc132.dense.TAG,
        lc132.dense.ROLLOUTS,
        lc132.dense.PPO_EPOCHS,
        lc132.dense.LEARNING_RATE,
        lc132.dense.PHASE_RETURN_ENV_OVERRIDES,
    )
    outer = lc132.configure()
    inner = lc132.dense.configure()
    try:
        assert lc132.dense.base.TAG == lc132.TAG
        assert lc132.dense.base.ROLLOUTS == 6
        assert lc132.dense.base.PPO_EPOCHS == 2
        assert lc132.dense.base.LEARNING_RATE == 1e-5
        config = {"vec": {}, "env": {}}
        assert lc132.dense.PHASE_RETURN_ENV_OVERRIDES["w_body_rate"] == 0.0
        config["env"].update(lc132.dense.PHASE_RETURN_ENV_OVERRIDES)
        assert config["env"]["w_progress"] == 5.0
    finally:
        lc132.dense.restore(inner)
        lc132.restore(outer)
    dense_after = (
        lc132.dense.TAG,
        lc132.dense.ROLLOUTS,
        lc132.dense.PPO_EPOCHS,
        lc132.dense.LEARNING_RATE,
        lc132.dense.PHASE_RETURN_ENV_OVERRIDES,
    )
    assert dense_after == dense_before
