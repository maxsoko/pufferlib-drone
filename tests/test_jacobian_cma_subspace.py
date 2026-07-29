import json
from pathlib import Path

import numpy as np
import pytest

from scripts.convert_policy_checkpoint_layout import pack_layout, tensor_counts, unpack_layout
from scripts.jacobian_cma_subspace import (
    extract_trainable_vector,
    focus_mask,
    load_checkpoint_tensors,
    materialize_candidate,
    project_anchor_null_and_orthonormalize,
    replace_trainable_vector,
    trace_action_drift,
    torch_recurrent_actions,
)


def _checkpoint(path: Path):
    input_dim, hidden_dim, num_layers, num_actions = 3, 4, 1, 2
    counts = tensor_counts(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )
    rng = np.random.default_rng(7)
    tensors = [rng.normal(0, 0.1, count).astype(np.float32) for count in counts]
    pack_layout(tensors, counts, precision_bytes=4).tofile(path)
    return counts, tensors


def test_trainable_vector_excludes_logstd_and_value_row(tmp_path):
    _, tensors = _checkpoint(tmp_path / "parent.bin")
    original_logstd = tensors[2].copy()
    original_value = tensors[1].reshape(3, 4)[2].copy()
    vector = extract_trainable_vector(tensors, hidden_dim=4, num_actions=2)
    replaced = replace_trainable_vector(
        tensors, vector + np.float32(0.25), hidden_dim=4, num_actions=2
    )
    assert np.array_equal(replaced[2], original_logstd)
    assert np.array_equal(replaced[1].reshape(3, 4)[2], original_value)
    assert not np.array_equal(replaced[0], tensors[0])


def test_materialize_round_trip_keeps_excluded_tensors(tmp_path):
    parent = tmp_path / "parent.bin"
    counts, parent_tensors = _checkpoint(parent)
    trainable = extract_trainable_vector(parent_tensors, hidden_dim=4, num_actions=2)
    basis = np.zeros((2, trainable.size), dtype=np.float32)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    metadata = {
        "coefficient_scale": 0.01,
        "layout_precision_bytes": 4,
        "policy": {"input_dim": 3, "hidden_dim": 4, "num_layers": 1, "num_actions": 2},
    }
    basis_path = tmp_path / "basis.npz"
    np.savez_compressed(basis_path, basis=basis, metadata=np.asarray(json.dumps(metadata)))
    output = tmp_path / "candidate.bin"
    report = materialize_candidate(parent, basis_path, np.array([1.0, -1.0]), output)
    candidate = unpack_layout(np.fromfile(output, np.float32), counts, precision_bytes=4)
    assert np.array_equal(candidate[2], parent_tensors[2])
    assert np.array_equal(candidate[1].reshape(3, 4)[2], parent_tensors[1].reshape(3, 4)[2])
    assert report["changed_trainable_values"] == 2


def test_parallel_recurrent_replay_has_full_history_gradient():
    torch = pytest.importorskip("torch")
    input_dim, hidden_dim, num_layers, num_actions = 2, 3, 1, 2
    size = hidden_dim * input_dim + num_actions * hidden_dim + 3 * hidden_dim * hidden_dim
    theta = torch.linspace(-0.2, 0.3, size, requires_grad=True)
    observations = torch.tensor(
        [[[0.1, -0.2], [0.2, 0.3], [-0.1, 0.4]]], requires_grad=True
    )
    actions = torch_recurrent_actions(
        observations,
        theta,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )
    early_observation_gradient = torch.autograd.grad(actions[0, -1].sum(), observations)[0]
    assert actions.shape == (1, 3, 2)
    assert torch.count_nonzero(early_observation_gradient[0, 0]).item() > 0


def test_parallel_recurrent_replay_matches_sequential_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    parent = tmp_path / "parent.bin"
    _checkpoint(parent)
    _, tensors = load_checkpoint_tensors(
        parent,
        input_dim=3,
        hidden_dim=4,
        num_layers=1,
        num_actions=2,
        layout_precision_bytes=4,
    )
    theta = torch.tensor(extract_trainable_vector(tensors, hidden_dim=4, num_actions=2))
    observations = np.random.default_rng(13).normal(size=(2, 7, 3)).astype(np.float32)
    parallel = torch_recurrent_actions(
        torch.tensor(observations),
        theta,
        input_dim=3,
        hidden_dim=4,
        num_layers=1,
        num_actions=2,
    ).detach().numpy()

    from scripts.policy_callable_checkpoint import CheckpointPolicy

    sequential = np.empty_like(parallel)
    for batch in range(observations.shape[0]):
        policy = CheckpointPolicy.load(
            str(parent),
            input_dim=3,
            hidden_dim=4,
            num_layers=1,
            num_actions=2,
            layout_precision_bytes=4,
        )
        for step in range(observations.shape[1]):
            hidden = policy.hidden_features(observations[batch, step])
            sequential[batch, step] = policy.decoder[:2] @ hidden
    assert parallel == pytest.approx(sequential, abs=2e-6)


def test_anchor_null_basis_is_deterministic_and_orthonormal():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(3385)
    directions = torch.randn(3, 12, generator=generator)
    anchor_rows = torch.randn(2, 12, generator=generator)
    first, first_null, first_gram = project_anchor_null_and_orthonormalize(
        directions, anchor_rows
    )
    second, second_null, second_gram = project_anchor_null_and_orthonormalize(
        directions, anchor_rows
    )
    assert torch.equal(first, second)
    assert first @ first.T == pytest.approx(torch.eye(3), abs=5e-6)
    assert first_null < 5e-6
    assert first_gram < 5e-6
    assert second_null == first_null
    assert second_gram == first_gram


def test_focus_mask_includes_gate2_and_late_gate1():
    observations = np.zeros((1, 10, 32), dtype=np.float32)
    observations[0, :7, 23] = 1.0
    observations[0, 7:, 24] = 1.0
    selected = focus_mask(observations, np.ones((1, 10), dtype=bool), prefix_steps=2)
    assert np.flatnonzero(selected[0]).tolist() == [5, 6, 7, 8, 9]


def test_trace_action_drift_is_zero_for_same_checkpoint(tmp_path):
    parent = tmp_path / "parent.bin"
    _checkpoint(parent)
    observations = np.zeros((2, 1, 3), dtype=np.float32)
    trace = tmp_path / "trace.npz"
    np.savez_compressed(
        trace,
        observations=observations,
        lengths=np.asarray([2], dtype=np.int32),
    )
    drift = trace_action_drift(
        parent,
        parent,
        trace,
        agents=1,
        input_dim=3,
        hidden_dim=4,
        num_layers=1,
        num_actions=2,
        layout_precision_bytes=4,
    )
    assert drift["full_rms"] == 0.0
    assert drift["focus_rms"] == 0.0
