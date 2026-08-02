from __future__ import annotations

import numpy as np
import torch

from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseActionSequenceActor
from scripts.policy_callable_vq2_all24_sequence import NumpyLC216Policy
import scripts.policy_callable_vq2_all24_sequence as callable_module


CHECKPOINT = "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc216_all24_action_sequence_001/policy_selected.pt"
NUMPY_CHECKPOINT = "checkpoints/vq2_lc216_all24_action_sequence_numpy.npz"


def test_numpy_callable_matches_torch_recurrent_actor() -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload["model"]
    actor = VQ2PhaseActionSequenceActor(
        target_phase=contract["adapter_target_phase"],
        sequence_phase_min=contract["sequence_phase_min"],
        sequence_phase_max_exclusive=contract["sequence_phase_max_exclusive"],
        sequence_length=contract["sequence_length"],
        hidden_size=contract["hidden_size"],
        residual_size=contract["residual_size"],
        adapter_size=contract["adapter_size"],
        initial_std=contract["initial_std"],
    )
    actor.load_state_dict(payload["model_state"])
    actor.eval()
    state = actor.initial_state(1, device="cpu")
    numpy_policy = NumpyLC216Policy(NUMPY_CHECKPOINT)
    rng = np.random.default_rng(219)

    maximum_error = 0.0
    for phase in (0, 0, 15, 15, 16, 16, 23):
        observation = rng.normal(0.0, 0.1, 4119).astype(np.float32)
        observation[:4096] = rng.random(4096, dtype=np.float32)
        observation[4117:] = phase / 6.0
        with torch.no_grad():
            output, state = actor.forward_step(
                torch.from_numpy(observation[None]), state
            )
        actual = np.asarray(numpy_policy(observation), dtype=np.float32)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(actual - output.mean[0].numpy()))),
        )

    assert maximum_error <= 3e-7


def test_runtime_reset_eagerly_loads_then_reuses_archive(monkeypatch) -> None:
    monkeypatch.setenv("PUFFER_POLICY_CHECKPOINT_PATH", NUMPY_CHECKPOINT)
    monkeypatch.setattr(callable_module, "_POLICY", None)
    monkeypatch.setattr(callable_module, "_CHECKPOINT", None)

    callable_module.reset()
    loaded = callable_module._POLICY
    assert loaded is not None
    loaded.sequence_counter = 17

    callable_module.reset()
    assert callable_module._POLICY is loaded
    assert loaded.sequence_counter == 0
