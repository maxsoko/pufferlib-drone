import json

import numpy as np
import pytest

from scripts.capture_native_policy_trace import (
    assemble_trace_blocks,
    build_capture_overrides,
    parse_config_overrides,
    sha256_file,
    verify_trace_replay,
)
from scripts.convert_policy_checkpoint_layout import pack_layout, tensor_counts


def _block(terminals):
    terminals = np.asarray(terminals, dtype=np.float32)
    time, agents = terminals.shape
    return {
        "observations": np.zeros((time, agents, 3), dtype=np.float32),
        "actions": np.zeros((time, agents, 2), dtype=np.float32),
        "rewards": np.zeros((time, agents), dtype=np.float32),
        "terminals": terminals,
        "initial_states": np.zeros((1, agents, 4), dtype=np.float32),
    }


def test_assemble_trace_blocks_marks_first_complete_episode_only():
    trace = assemble_trace_blocks([
        _block([[0, 0], [0, 1]]),
        _block([[1, 0], [0, 0]]),
    ])
    assert trace["lengths"].tolist() == [3, 2]
    assert trace["valid"][:, 0].tolist() == [True, True, True, False]
    assert trace["valid"][:, 1].tolist() == [True, True, False, False]
    assert trace["rollout_initial_states"].shape == (2, 1, 2, 4)


def test_assemble_trace_blocks_rejects_inconsistent_shapes():
    block = _block([[0], [1]])
    block["actions"] = np.zeros((3, 1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="dimensions do not agree"):
        assemble_trace_blocks([block])


def test_build_capture_overrides_allows_six_gate_contract_to_win_last():
    overrides = build_capture_overrides(
        total_agents=8,
        horizon=64,
        floor=4.1875,
        config_overrides=(
            "--env.num-gates", "6",
            "--env.observable-gate-phase-onehot", "1",
        ),
    )
    assert overrides[-4:] == [
        "--env.num-gates", "6",
        "--env.observable-gate-phase-onehot", "1",
    ]


def test_build_capture_overrides_rejects_unpaired_values():
    with pytest.raises(ValueError, match="key/value pairs"):
        build_capture_overrides(
            total_agents=1,
            horizon=1,
            floor=4.1875,
            config_overrides=("--env.num-gates",),
        )


def test_parse_config_overrides_accepts_dash_prefixed_keys():
    assert parse_config_overrides([
        "--env.num-gates=6",
        "--env.observable-gate-phase-onehot=1",
    ]) == (
        "--env.num-gates", "6",
        "--env.observable-gate-phase-onehot", "1",
    )


def test_parse_config_overrides_rejects_missing_value():
    with pytest.raises(ValueError, match="section.key=value"):
        parse_config_overrides(["--env.num-gates"])


def test_verify_trace_replay_checks_chunk_states_actions_and_terminal_resets(tmp_path):
    input_dim, hidden_dim, num_layers, num_actions = 3, 4, 1, 2
    counts = tensor_counts(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )
    rng = np.random.default_rng(9)
    tensors = [rng.normal(0, 0.1, count).astype(np.float32) for count in counts]
    checkpoint = tmp_path / "policy.bin"
    pack_layout(tensors, counts, precision_bytes=4).tofile(checkpoint)

    from scripts.policy_callable_checkpoint import CheckpointPolicy

    observations = rng.normal(size=(4, 1, input_dim)).astype(np.float32)
    terminals = np.asarray([[1], [0], [0], [1]], dtype=np.float32)
    policy = CheckpointPolicy.load(
        str(checkpoint),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
        layout_precision_bytes=4,
    )
    states = []
    actions = []
    for step in range(4):
        if step % 2 == 0:
            states.append(policy.state.copy())
        if terminals[step, 0] > 0.5:
            policy.reset_state()
        actions.append(policy.infer(observations[step, 0]))
    trace_path = tmp_path / "trace.npz"
    metadata = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "horizon": 2,
    }
    np.savez_compressed(
        trace_path,
        observations=observations,
        actions=np.asarray(actions, dtype=np.float32)[:, None, :],
        terminals=terminals,
        rollout_initial_states=np.asarray(states)[:, :, None, :],
        metadata=np.asarray(json.dumps(metadata)),
    )
    report = verify_trace_replay(
        checkpoint,
        trace_path,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
        layout_precision_bytes=4,
        action_atol=0.0,
        state_atol=0.0,
    )
    assert report["passed"]
    assert report["terminal_resets_replayed"] == 2
    assert report["action_max_error"] == 0.0
    assert report["state_max_error"] == 0.0
