#!/usr/bin/env python3
"""Build and apply a recurrent action-Jacobian policy-parameter subspace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from convert_policy_checkpoint_layout import pack_layout, tensor_counts, unpack_layout
except ModuleNotFoundError:  # Imported as scripts.jacobian_cma_subspace.
    from scripts.convert_policy_checkpoint_layout import (
        pack_layout,
        tensor_counts,
        unpack_layout,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_checkpoint_tensors(
    path: Path,
    *,
    input_dim: int = 32,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_actions: int = 4,
    layout_precision_bytes: int = 4,
) -> tuple[list[int], list[np.ndarray]]:
    counts = tensor_counts(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )
    serialized = np.fromfile(path, dtype=np.float32)
    return counts, unpack_layout(
        serialized, counts, precision_bytes=layout_precision_bytes
    )


def extract_trainable_vector(
    tensors: list[np.ndarray],
    *,
    hidden_dim: int = 128,
    num_actions: int = 4,
) -> np.ndarray:
    if len(tensors) < 4:
        raise ValueError("checkpoint must contain encoder, decoder, log_std, and recurrent tensors")
    decoder = tensors[1].reshape(num_actions + 1, hidden_dim)
    parts = [tensors[0].reshape(-1), decoder[:num_actions].reshape(-1)]
    parts.extend(tensor.reshape(-1) for tensor in tensors[3:])
    return np.concatenate(parts).astype(np.float32, copy=True)


def replace_trainable_vector(
    tensors: list[np.ndarray],
    vector: np.ndarray,
    *,
    hidden_dim: int = 128,
    num_actions: int = 4,
) -> list[np.ndarray]:
    result = [np.asarray(tensor, dtype=np.float32).copy() for tensor in tensors]
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    cursor = 0

    encoder_count = result[0].size
    result[0] = flat[cursor:cursor + encoder_count].copy()
    cursor += encoder_count

    decoder = result[1].reshape(num_actions + 1, hidden_dim)
    decoder_count = num_actions * hidden_dim
    decoder[:num_actions] = flat[cursor:cursor + decoder_count].reshape(
        num_actions, hidden_dim
    )
    cursor += decoder_count
    result[1] = decoder.reshape(-1)

    for tensor_index in range(3, len(result)):
        count = result[tensor_index].size
        result[tensor_index] = flat[cursor:cursor + count].copy()
        cursor += count
    if cursor != flat.size:
        raise ValueError(
            f"trainable vector size mismatch: consumed {cursor}, got {flat.size}"
        )
    return result


def materialize_candidate(
    parent: Path,
    basis_path: Path,
    coefficients: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    basis_data = np.load(basis_path, allow_pickle=False)
    metadata = json.loads(str(basis_data["metadata"]))
    basis = np.asarray(basis_data["basis"], dtype=np.float32)
    coefficient_scale = float(metadata["coefficient_scale"])
    z = np.asarray(coefficients, dtype=np.float32).reshape(-1)
    if z.size != basis.shape[0]:
        raise ValueError(f"expected {basis.shape[0]} coefficients, got {z.size}")

    dims = metadata["policy"]
    counts, tensors = load_checkpoint_tensors(
        parent,
        input_dim=int(dims["input_dim"]),
        hidden_dim=int(dims["hidden_dim"]),
        num_layers=int(dims["num_layers"]),
        num_actions=int(dims["num_actions"]),
        layout_precision_bytes=int(metadata["layout_precision_bytes"]),
    )
    parent_vector = extract_trainable_vector(
        tensors,
        hidden_dim=int(dims["hidden_dim"]),
        num_actions=int(dims["num_actions"]),
    )
    delta = coefficient_scale * (z @ basis)
    candidate_vector = np.asarray(parent_vector + delta, dtype=np.float32)
    candidate_tensors = replace_trainable_vector(
        tensors,
        candidate_vector,
        hidden_dim=int(dims["hidden_dim"]),
        num_actions=int(dims["num_actions"]),
    )
    serialized = pack_layout(
        candidate_tensors,
        counts,
        precision_bytes=int(metadata["layout_precision_bytes"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized.tofile(output)
    actual_delta = extract_trainable_vector(
        load_checkpoint_tensors(
            output,
            input_dim=int(dims["input_dim"]),
            hidden_dim=int(dims["hidden_dim"]),
            num_layers=int(dims["num_layers"]),
            num_actions=int(dims["num_actions"]),
            layout_precision_bytes=int(metadata["layout_precision_bytes"]),
        )[1],
        hidden_dim=int(dims["hidden_dim"]),
        num_actions=int(dims["num_actions"]),
    ) - parent_vector
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "coefficients": z.astype(float).tolist(),
        "coefficient_scale": coefficient_scale,
        "requested_delta_l2": float(np.linalg.norm(delta)),
        "actual_delta_l2": float(np.linalg.norm(actual_delta)),
        "changed_trainable_values": int(np.count_nonzero(actual_delta)),
    }


def trace_action_drift(
    parent: Path,
    candidate: Path,
    trace_path: Path,
    *,
    agents: int = 2,
    input_dim: int = 32,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_actions: int = 4,
    layout_precision_bytes: int = 4,
) -> dict[str, float]:
    try:
        from policy_callable_checkpoint import CheckpointPolicy
    except ModuleNotFoundError:
        from scripts.policy_callable_checkpoint import CheckpointPolicy

    trace = np.load(trace_path, allow_pickle=False)
    observations = np.asarray(trace["observations"], dtype=np.float32)
    lengths = np.asarray(trace["lengths"], dtype=np.int32)
    count = min(int(agents), lengths.size)
    all_delta = []
    focus_delta = []
    for agent in range(count):
        length = int(lengths[agent])
        parent_policy = CheckpointPolicy.load(
            str(parent),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_actions=num_actions,
            layout_precision_bytes=layout_precision_bytes,
        )
        candidate_policy = CheckpointPolicy.load(
            str(candidate),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_actions=num_actions,
            layout_precision_bytes=layout_precision_bytes,
        )
        parent_actions = np.asarray(
            [parent_policy.infer(obs) for obs in observations[:length, agent]],
            dtype=np.float32,
        )
        candidate_actions = np.asarray(
            [candidate_policy.infer(obs) for obs in observations[:length, agent]],
            dtype=np.float32,
        )
        delta = candidate_actions - parent_actions
        if observations.shape[-1] > 24:
            valid = np.ones((1, length), dtype=np.bool_)
            selected = focus_mask(
                observations[:length, agent][None, ...], valid
            )[0]
        else:
            selected = np.ones(length, dtype=np.bool_)
        all_delta.append(delta.reshape(-1))
        focus_delta.append(delta[selected].reshape(-1))
    if not all_delta:
        raise ValueError("trace contains no selected agents")
    flat = np.concatenate(all_delta)
    focus = np.concatenate(focus_delta)
    return {
        "full_rms": float(np.sqrt(np.mean(np.square(flat, dtype=np.float64)))),
        "full_max": float(np.max(np.abs(flat))),
        "focus_rms": float(np.sqrt(np.mean(np.square(focus, dtype=np.float64)))),
        "focus_max": float(np.max(np.abs(focus))),
        "finite": bool(np.isfinite(flat).all()),
    }


def _split_torch_parameters(theta, *, input_dim, hidden_dim, num_layers, num_actions):
    cursor = 0
    encoder_count = hidden_dim * input_dim
    encoder = theta[cursor:cursor + encoder_count].reshape(hidden_dim, input_dim)
    cursor += encoder_count
    decoder_count = num_actions * hidden_dim
    decoder = theta[cursor:cursor + decoder_count].reshape(num_actions, hidden_dim)
    cursor += decoder_count
    recurrent = []
    recurrent_count = 3 * hidden_dim * hidden_dim
    for _ in range(num_layers):
        recurrent.append(
            theta[cursor:cursor + recurrent_count].reshape(3 * hidden_dim, hidden_dim)
        )
        cursor += recurrent_count
    if cursor != theta.numel():
        raise ValueError(f"unexpected trainable vector size: consumed {cursor}, got {theta.numel()}")
    return encoder, decoder, recurrent


def torch_recurrent_actions(
    observations,
    theta,
    *,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_actions: int,
):
    """Parallel full-history MinGRU replay, differentiable in ``theta``."""
    import torch
    import torch.nn.functional as F

    if observations.ndim == 2:
        observations = observations.unsqueeze(0)
    if observations.ndim != 3 or observations.shape[-1] != input_dim:
        raise ValueError("observations must have shape [batch, time, input_dim]")
    encoder, decoder, recurrent = _split_torch_parameters(
        theta,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )
    h = F.linear(observations, encoder)
    for weights in recurrent:
        hidden, gate, highway = F.linear(h, weights).chunk(3, dim=-1)
        log_coefficients = -F.softplus(gate)
        log_g = torch.where(
            hidden >= 0,
            torch.log(F.relu(hidden) + 0.5),
            -F.softplus(-hidden),
        )
        log_values = -F.softplus(-gate) + log_g
        cumulative = log_coefficients.cumsum(dim=1)
        recurrent_out = torch.exp(
            cumulative + torch.logcumsumexp(log_values - cumulative, dim=1)
        )
        highway_gate = torch.sigmoid(highway)
        h = highway_gate * recurrent_out + (1.0 - highway_gate) * h
    return F.linear(h, decoder)


def _load_trace_agent_batch(path: Path, agents: int) -> tuple[np.ndarray, np.ndarray]:
    trace = np.load(path, allow_pickle=False)
    lengths = np.asarray(trace["lengths"], dtype=np.int32)
    count = min(int(agents), lengths.size)
    if count <= 0:
        raise ValueError("trace contains no agents")
    length = int(lengths[:count].min())
    observations = np.asarray(trace["observations"][:length, :count], dtype=np.float32)
    observations = np.transpose(observations, (1, 0, 2)).copy()
    mask = np.asarray(trace["valid"][:length, :count], dtype=np.bool_).T.copy()
    return observations, mask


def focus_mask(observations: np.ndarray, valid: np.ndarray, prefix_steps: int = 64) -> np.ndarray:
    if observations.shape[:2] != valid.shape:
        raise ValueError("observation and valid masks disagree")
    selected = (observations[..., 24] > 0.5) & valid
    phase_one = (observations[..., 23] > 0.5) & valid
    for batch in range(observations.shape[0]):
        indices = np.flatnonzero(phase_one[batch])
        if indices.size:
            selected[batch, indices[-prefix_steps:]] = True
    return selected


def _rademacher_projection(actions, mask, generator):
    import torch

    signs = torch.randint(
        0,
        2,
        actions.shape,
        device=actions.device,
        generator=generator,
        dtype=torch.int64,
    ).to(actions.dtype)
    signs = signs.mul_(2.0).sub_(1.0)
    selected = mask.unsqueeze(-1).expand_as(actions)
    projection = torch.where(selected, signs, torch.zeros_like(signs))
    norm = torch.linalg.vector_norm(projection)
    if float(norm) == 0.0:
        raise ValueError("focus mask selects no actions")
    return projection / norm


def project_anchor_null_and_orthonormalize(generalized_directions, anchor_matrix):
    """Project direction rows from sampled anchor rows, then FP32-orthonormalize."""
    import torch

    dtype = generalized_directions.dtype
    device = generalized_directions.device
    anchor_row_norms = torch.linalg.vector_norm(anchor_matrix, dim=1, keepdim=True)
    normalized_anchor = anchor_matrix / torch.clamp(
        anchor_row_norms, min=torch.finfo(dtype).tiny
    )
    anchor_probes = anchor_matrix.shape[0]
    anchor_gram = normalized_anchor @ normalized_anchor.T
    null_ridge = torch.finfo(dtype).eps * float(anchor_probes)
    projection_coefficients = torch.linalg.solve(
        anchor_gram + null_ridge * torch.eye(anchor_probes, device=device),
        normalized_anchor @ generalized_directions.T,
    )
    projected = generalized_directions - projection_coefficients.T @ normalized_anchor
    basis_columns, _ = torch.linalg.qr(projected.T, mode="reduced")
    basis = basis_columns.T
    anchor_null_residual = float(torch.max(torch.abs(normalized_anchor @ basis.T)))
    gram = basis @ basis.T
    orthonormal_error = float(
        torch.max(
            torch.abs(
                gram
                - torch.eye(
                    generalized_directions.shape[0], device=device, dtype=dtype
                )
            )
        )
    )
    return basis, anchor_null_residual, orthonormal_error


def build_subspace(
    parent: Path,
    failure_trace: Path,
    anchor_trace: Path,
    output: Path,
    *,
    dimension: int = 12,
    failure_probes: int = 24,
    anchor_probes: int = 8,
    damping_fraction: float = 1e-3,
    seed: int = 3385,
    trace_agents: int = 2,
    target_failure_action_rms: float = 0.005,
    maximum_anchor_action_rms: float = 0.001,
    input_dim: int = 32,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_actions: int = 4,
    layout_precision_bytes: int = 4,
    device: str = "auto",
) -> dict[str, Any]:
    import torch

    if not 0 < dimension <= failure_probes:
        raise ValueError("dimension must be positive and no larger than failure probes")
    counts, tensors = load_checkpoint_tensors(
        parent,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
        layout_precision_bytes=layout_precision_bytes,
    )
    del counts
    parent_vector = extract_trainable_vector(
        tensors, hidden_dim=hidden_dim, num_actions=num_actions
    )
    failure_obs_np, failure_valid_np = _load_trace_agent_batch(
        failure_trace, trace_agents
    )
    anchor_obs_np, anchor_valid_np = _load_trace_agent_batch(anchor_trace, trace_agents)
    failure_mask_np = focus_mask(failure_obs_np, failure_valid_np)
    anchor_mask_np = focus_mask(anchor_obs_np, anchor_valid_np)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    dtype = torch.float32
    theta = torch.tensor(parent_vector, device=torch_device, dtype=dtype, requires_grad=True)
    failure_obs = torch.tensor(failure_obs_np, device=torch_device, dtype=dtype)
    anchor_obs = torch.tensor(anchor_obs_np, device=torch_device, dtype=dtype)
    failure_mask = torch.tensor(failure_mask_np, device=torch_device)
    anchor_mask = torch.tensor(anchor_mask_np, device=torch_device)

    def replay(obs, value):
        return torch_recurrent_actions(
            obs,
            value,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_actions=num_actions,
        )

    generator = torch.Generator(device=torch_device)
    generator.manual_seed(seed)
    anchor_actions = replay(anchor_obs, theta)
    anchor_gradient_square = torch.zeros_like(theta)
    anchor_gradients = []
    for probe in range(anchor_probes):
        projection = _rademacher_projection(anchor_actions, anchor_mask, generator)
        gradient = torch.autograd.grad(
            (anchor_actions * projection).sum(),
            theta,
            retain_graph=probe + 1 < anchor_probes,
        )[0]
        anchor_gradients.append(gradient.detach())
        anchor_gradient_square.add_(gradient.square())
    anchor_diagonal = anchor_gradient_square / float(anchor_probes)
    positive = anchor_diagonal[anchor_diagonal > 0]
    diagonal_reference = positive.median() if positive.numel() else torch.tensor(
        1.0, device=torch_device
    )
    damping = torch.clamp(
        diagonal_reference * float(damping_fraction), min=torch.finfo(dtype).tiny
    )

    failure_actions = replay(failure_obs, theta)
    whitened_gradients = []
    failure_gradients = []
    denominator = torch.sqrt(anchor_diagonal + damping)
    for probe in range(failure_probes):
        projection = _rademacher_projection(failure_actions, failure_mask, generator)
        gradient = torch.autograd.grad(
            (failure_actions * projection).sum(),
            theta,
            retain_graph=probe + 1 < failure_probes,
        )[0]
        failure_gradients.append(gradient.detach())
        whitened_gradients.append(gradient / denominator)
    gradient_matrix = torch.stack(whitened_gradients)
    _, singular_values, right_vectors = torch.linalg.svd(
        gradient_matrix, full_matrices=False
    )
    # The SVD lives in anchor-whitened coordinates. Map its right vectors back
    # through D_anchor^-1/2, then orthonormalize in actual checkpoint space.
    # Treating the whitened vectors themselves as raw parameter directions
    # would discard the generalized-eigenproblem's anchor constraint.
    generalized_directions = right_vectors[:dimension] / denominator.unsqueeze(0)
    anchor_matrix = torch.stack(anchor_gradients)
    basis_torch, anchor_null_residual, orthonormal_error = (
        project_anchor_null_and_orthonormalize(
            generalized_directions, anchor_matrix
        )
    )

    # Every Rademacher projection is normalized by sqrt(selected action count),
    # so E[(r^T Jb)^2] is the action-space mean square of direction b. Reuse the
    # stable reverse-mode products here: long-sequence forward-mode JVP through
    # logcumsumexp is numerically less stable despite finite forward actions.
    failure_projection = torch.stack(failure_gradients) @ basis_torch.T
    anchor_projection = torch.stack(anchor_gradients) @ basis_torch.T
    failure_expected_rms_per_unit = torch.sqrt(
        failure_projection.square().sum(dim=1).mean()
    )
    anchor_expected_rms_per_unit = torch.sqrt(
        anchor_projection.square().sum(dim=1).mean()
    )
    failure_sensitivity = float(failure_expected_rms_per_unit)
    anchor_sensitivity = float(anchor_expected_rms_per_unit)
    if not np.isfinite(failure_sensitivity) or failure_sensitivity <= 0.0:
        raise RuntimeError("subspace has no finite failure-trace action sensitivity")
    coefficient_scale = float(target_failure_action_rms / failure_sensitivity)
    predicted_anchor_rms = coefficient_scale * anchor_sensitivity

    basis = basis_torch.detach().cpu().numpy().astype(np.float32)
    singular = singular_values.detach().cpu().numpy().astype(np.float64)
    metadata: dict[str, Any] = {
        "method": "matrix_free_failure_jacobian_with_diagonal_anchor_whitening",
        "parent": str(parent),
        "parent_sha256": sha256_file(parent),
        "failure_trace": str(failure_trace),
        "failure_trace_sha256": sha256_file(failure_trace),
        "anchor_trace": str(anchor_trace),
        "anchor_trace_sha256": sha256_file(anchor_trace),
        "basis_sha256": sha256_array(basis),
        "dimension": dimension,
        "trainable_parameter_count": int(parent_vector.size),
        "failure_probes": failure_probes,
        "anchor_probes": anchor_probes,
        "trace_agents": trace_agents,
        "seed": seed,
        "damping_fraction": damping_fraction,
        "damping": float(damping.detach().cpu()),
        "orthonormal_max_error": orthonormal_error,
        "anchor_null_projection_max_residual": anchor_null_residual,
        "failure_focus_steps": int(failure_mask_np.sum()),
        "anchor_focus_steps": int(anchor_mask_np.sum()),
        "failure_action_rms_per_unit_coefficient": failure_sensitivity,
        "anchor_action_rms_per_unit_coefficient": anchor_sensitivity,
        "action_sensitivity_calibration": "normalized_hutchinson_vjp",
        "target_failure_action_rms": target_failure_action_rms,
        "maximum_anchor_action_rms": maximum_anchor_action_rms,
        "predicted_anchor_action_rms": predicted_anchor_rms,
        "anchor_calibration_feasible": predicted_anchor_rms <= maximum_anchor_action_rms,
        "coefficient_scale": coefficient_scale,
        "layout_precision_bytes": layout_precision_bytes,
        "device": str(torch_device),
        "policy": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_actions": num_actions,
            "excluded": ["log_std", "decoder_value_row"],
        },
        "singular_values": singular.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        basis=basis,
        anchor_diagonal=anchor_diagonal.detach().cpu().numpy().astype(np.float32),
        singular_values=singular,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("parent", type=Path)
    build.add_argument("failure_trace", type=Path)
    build.add_argument("anchor_trace", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--dimension", type=int, default=12)
    build.add_argument("--failure-probes", type=int, default=24)
    build.add_argument("--anchor-probes", type=int, default=8)
    build.add_argument("--damping-fraction", type=float, default=1e-3)
    build.add_argument("--seed", type=int, default=3385)
    build.add_argument("--trace-agents", type=int, default=2)
    build.add_argument("--target-failure-action-rms", type=float, default=0.005)
    build.add_argument("--maximum-anchor-action-rms", type=float, default=0.001)
    build.add_argument("--device", default="auto")

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("parent", type=Path)
    materialize.add_argument("basis", type=Path)
    materialize.add_argument("output", type=Path)
    materialize.add_argument("coefficients", nargs="+", type=float)
    args = parser.parse_args()

    if args.command == "build":
        report = build_subspace(
            args.parent,
            args.failure_trace,
            args.anchor_trace,
            args.output,
            dimension=args.dimension,
            failure_probes=args.failure_probes,
            anchor_probes=args.anchor_probes,
            damping_fraction=args.damping_fraction,
            seed=args.seed,
            trace_agents=args.trace_agents,
            target_failure_action_rms=args.target_failure_action_rms,
            maximum_anchor_action_rms=args.maximum_anchor_action_rms,
            device=args.device,
        )
    else:
        report = materialize_candidate(
            args.parent,
            args.basis,
            np.asarray(args.coefficients, dtype=np.float32),
            args.output,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
