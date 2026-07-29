from __future__ import annotations

import numpy as np
import torch

from scripts.train_vq2_exact_two_gate_ppo import (
    CONFIG,
    compute_gae,
    evaluation_rank,
    exact_two_gate_config,
    official_phase_from_training,
)


class FakePufferl:
    @staticmethod
    def load_config(_name: str) -> dict:
        return {
            "backend_env_name": "drone_race_vision",
            "seed": 1,
            "vec": {"total_agents": 64},
            "env": {"dt": 0.015625},
        }


def test_training_phase_maps_back_to_official_six_gate_denominator() -> None:
    raw = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    assert np.allclose(official_phase_from_training(raw), [0.0, 1 / 6, 2 / 6])


def test_exact_training_config_is_two_gate_and_teacher_free() -> None:
    config = exact_two_gate_config(FakePufferl, agents=CONFIG.agents, seed=42068, evaluation=False)
    env = config["env"]
    assert env["num_gates"] == 2
    assert env["teacher_action_blend"] == 0.0
    assert env["w_action_teacher"] == 0.0
    assert env["reset_position_noise_xy"] == 0.0
    assert env["gate_position_domain_randomize"] == 0
    assert env["w_gate_camera_alignment"] == 20.0


def test_gae_cuts_value_bootstrap_at_terminal() -> None:
    reward = torch.tensor([[1.0], [2.0]])
    value = torch.zeros_like(reward)
    done = torch.tensor([[0.0], [1.0]])
    advantage, returns = compute_gae(
        reward, value, done, torch.tensor([99.0]), gamma=1.0, gae_lambda=1.0
    )
    assert torch.equal(advantage, torch.tensor([[3.0], [2.0]]))
    assert torch.equal(returns, advantage)


def test_evaluation_rank_does_not_treat_missing_crossing_as_zero_error() -> None:
    absent = {
        "metrics": {
            "env/success_rate": 0.0,
            "env/crash": 0.0,
            "env/gates_passed": 1.0,
            "env/terminal_crossing_sampled": 0.0,
            "env/terminal_crossing_radial": 0.0,
            "env/closest_gate_range": 8.5,
        }
    }
    observed = {
        "metrics": {
            "env/success_rate": 0.0,
            "env/crash": 0.0,
            "env/gates_passed": 1.0,
            "env/terminal_crossing_sampled": 1.0,
            "env/terminal_crossing_radial": 0.4,
            "env/closest_gate_range": 8.5,
        }
    }
    assert evaluation_rank(absent)[-1] == 8.5
    assert evaluation_rank(observed)[-1] == 0.4
