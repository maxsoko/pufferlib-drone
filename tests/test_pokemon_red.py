import pytest

pytest.importorskip("pokegym")

from pufferlib.environments.pokemon_red import env_creator


def test_pokemon_red_step():
    env = env_creator()()
    ob, info = env.reset()
    assert ob is not None
    for _ in range(10):
        ob, reward, terminal, truncated, info = env.step(env.action_space.sample())
        assert ob is not None
        if terminal or truncated:
            break
    env.close()
