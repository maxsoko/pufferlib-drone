from __future__ import annotations

import torch

from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE, VQ2PhaseRecurrentActor
from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseMLPResidualActor
from scripts.eval_vq2_vg070_nonlinear_residual_count5_scale_bracket import (
    ALPHAS,
    MAX_STEPS,
    NUM_GATES,
    PARENT_CHECKPOINT,
    load_endpoints,
    qualifies,
)


def test_endpoint_mapping_scales_only_nonlinear_outputs() -> None:
    parent, update = load_endpoints()
    p = parent["model_state"]
    u = update["model_state"]
    torch.testing.assert_close(
        p["indexed_phase_residual_input"],
        u["indexed_phase_residual_input"], rtol=0, atol=0,
    )
    torch.testing.assert_close(
        p["indexed_phase_residual_input_bias"],
        u["indexed_phase_residual_input_bias"], rtol=0, atol=0,
    )
    assert torch.count_nonzero(p["indexed_phase_residual_output"]) == 0
    assert torch.count_nonzero(p["indexed_phase_residual_output_bias"]) == 0
    assert torch.count_nonzero(u["indexed_phase_residual_output"]) > 0


def test_alpha_zero_actor_is_parent_behavior() -> None:
    parent, _ = load_endpoints()
    actor = VQ2IndexedPhaseMLPResidualActor(
        hidden_size=256, residual_size=64, initial_std=0.15
    )
    actor.load_state_dict(parent["model_state"])
    assert torch.count_nonzero(actor.indexed_phase_residual_output) == 0
    assert torch.count_nonzero(actor.indexed_phase_residual_output_bias) == 0
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    base = VQ2PhaseRecurrentActor(hidden_size=256, initial_std=0.15)
    base.load_state_dict(payload["model_state"])
    observation = torch.zeros(3, PHASE_LEGAL_OBS_SIZE)
    observation[:, -1] = torch.tensor([0.0, 0.25, 0.75])
    with torch.no_grad():
        base_output, base_state = base.forward_step(observation)
        selected_output, selected_state = actor.forward_step(observation)
    torch.testing.assert_close(selected_output.mean, base_output.mean, rtol=0, atol=0)
    torch.testing.assert_close(
        selected_output.pre_tanh_mean, base_output.pre_tanh_mean, rtol=0, atol=0
    )
    torch.testing.assert_close(selected_state, base_state, rtol=0, atol=0)


def test_qualification_requires_more_finishes_without_more_crashes() -> None:
    parent = {
        "all_hard_transport_pass": True, "successes": 10, "crashes": 4,
        "gate_reach": {str(i): 20 for i in range(1, 6)},
    }
    candidate = {
        "all_hard_transport_pass": True, "successes": 11, "crashes": 4,
        "gate_reach": {str(i): 20 for i in range(1, 6)},
    }
    assert qualifies(parent, candidate)
    candidate["successes"] = 10
    assert not qualifies(parent, candidate)
    candidate["successes"] = 11; candidate["crashes"] = 5
    assert not qualifies(parent, candidate)


def test_fixed_long_horizon_scale_contract() -> None:
    assert NUM_GATES == 5
    assert MAX_STEPS == 23040
    assert ALPHAS == (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
