from pathlib import Path

import numpy as np
import torch

from scripts.audit_vq2_actor_low_sensitivity_progress_channel import (
    _canonical_actor_tangent_null_direction as _audited_tangent_direction,
)
from scripts.audit_vq2_rotation_invariant_progress_geometry import (
    _progress_geometry_loss as _audited_invariant_progress_loss,
)
from scripts.continue_vq2_success_prefix_representation_joint_offline import (
    _EpochBatchSampler,
    _canonical_actor_tangent_null_direction,
    _candidate_key,
    _candidate_should_replace,
    _fixed_reference_event_chunks,
    _frozen_readout_contract_exact,
    _invariant_progress_loss,
    _invariant_source_contract_exact,
    _load_partitioned_arrays,
    _preservation,
    _tangent_residual_losses,
    _tangent_source_contract_exact,
    _weighted_delta_loss,
    _weighted_plane_loss,
)


class Bounds:
    maximum_action_rmse = 0.001
    maximum_action_max_abs = 0.01
    maximum_dense_feature_rmse = 0.02


class DonorBounds:
    minimum_validation_correlation = 0.8
    maximum_validation_mae_m = 0.5
    maximum_validation_near_plane_mae_m = 0.6
    minimum_validation_direction_accuracy = 0.75


def _candidate(*, preserved: bool, correlation: float, step: int) -> dict:
    return {
        "step": step,
        "validation_progress": {
            "correlation": correlation,
            "near_plane_mae_m": 0.3,
            "direction_accuracy": 0.8,
            "mae_m": 0.4,
        },
        "preservation_gates": {"bounded": preserved},
    }


def test_candidate_selection_prioritizes_preservation_then_progress() -> None:
    preserved = _candidate(preserved=True, correlation=0.7, step=100)
    drifted = _candidate(preserved=False, correlation=0.99, step=200)
    better = _candidate(preserved=True, correlation=0.8, step=300)
    assert _candidate_key(preserved) > _candidate_key(drifted)
    assert _candidate_key(better) > _candidate_key(preserved)


def test_epoch_batch_sampler_exposes_each_item_before_reuse() -> None:
    sampler = _EpochBatchSampler(
        np.arange(12), 3, np.random.default_rng(17)
    )
    first_epoch = np.concatenate([sampler.draw() for _ in range(4)])
    second_epoch = np.concatenate([sampler.draw() for _ in range(4)])
    assert sorted(first_epoch.tolist()) == list(range(12))
    assert sorted(second_epoch.tolist()) == list(range(12))
    assert all(len(set(sampler.draw().tolist())) == 3 for _ in range(4))


def test_donor_selection_prioritizes_complete_progress_pass() -> None:
    admitted = _candidate(preserved=True, correlation=0.90, step=350)
    regressed = _candidate(preserved=True, correlation=0.92, step=400)
    regressed["validation_progress"]["mae_m"] = 0.51
    assert _candidate_key(admitted, DonorBounds()) > _candidate_key(
        regressed, DonorBounds()
    )


def test_fixed_final_selection_ignores_validation_rank() -> None:
    early = _candidate(preserved=True, correlation=0.99, step=100)
    final = _candidate(preserved=False, correlation=0.01, step=357)
    args = DonorBounds()
    args.fixed_final_selection = True
    assert _candidate_should_replace(early, final, args)
    args.fixed_final_selection = False
    assert not _candidate_should_replace(early, final, args)


def test_preservation_checks_hard_actions_and_latents() -> None:
    validation = {"rmse": 0.0009, "max_abs": 0.009}
    dense = {
        "deterministic": {"rmse": 0.019, "max_abs": 1.0},
        "logits": {"rmse": 0.019, "max_abs": 1.0},
        "hard_action": {"rmse": 0.0009, "max_abs": 0.009},
    }
    assert all(_preservation(validation, dense, Bounds()).values())
    dense["hard_action"]["max_abs"] = 0.011
    gates = _preservation(validation, dense, Bounds())
    assert not gates["dense_action_max_bounded"]


def test_partition_loader_keeps_frozen_train_then_validation_order(tmp_path) -> None:
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    np.savez(train, vector_step=np.asarray([1, 2]), value=np.asarray([[10], [20]]))
    np.savez(validation, vector_step=np.asarray([3]), value=np.asarray([[30]]))
    arrays, split = _load_partitioned_arrays(train, validation)
    assert arrays["vector_step"].tolist() == [1, 2, 3]
    assert arrays["value"].ravel().tolist() == [10, 20, 30]
    assert split.tolist() == [0, 0, 1]


def test_weighted_plane_loss_corrects_crop_exposure_and_near_support() -> None:
    prediction = torch.zeros((1, 2), dtype=torch.float32)
    normalized_target = torch.asarray([[1.0, 2.0]])
    target_m = torch.asarray([[0.2, 2.0]])
    loss, weight = _weighted_plane_loss(
        prediction,
        normalized_target,
        target_m,
        np.asarray([0]),
        np.asarray([1, 2]),
        burn_in=0,
        near_plane_m=1.0,
        near_plane_multiplier=2.0,
    )
    assert torch.allclose(weight, torch.asarray([[2.0, 0.5]]))
    assert torch.isclose(loss, torch.asarray(1.6))


def test_fixed_reference_chunks_reproduce_the_audited_batch_shape() -> None:
    chunks = _fixed_reference_event_chunks(np.arange(336), 416)
    assert len(chunks) == 1
    batch, valid = chunks[0]
    assert valid == 336
    assert len(batch) == 416
    assert batch[:336].tolist() == list(range(336))
    assert batch[336:].tolist() == list(range(80))


def test_weighted_delta_loss_uses_pair_exposure_and_either_near_endpoint() -> None:
    prediction = torch.zeros((1, 4), dtype=torch.float32)
    normalized_target = torch.asarray([[0.0, 1.0, 3.0, 4.0]])
    target_m = torch.asarray([[3.0, 2.0, 0.5, 0.0]])
    loss, weight = _weighted_delta_loss(
        prediction,
        normalized_target,
        target_m,
        np.asarray([0]),
        np.asarray([0, 1, 2, 1]),
        burn_in=0,
        near_plane_m=1.0,
        near_plane_multiplier=2.0,
    )
    assert torch.allclose(weight, torch.asarray([[1.0, 1.0, 2.0]]))
    assert torch.isclose(loss, torch.asarray(1.75))


def test_tangent_direction_matches_source_locked_audit_geometry() -> None:
    generator = torch.Generator().manual_seed(700)
    weight = torch.randn((3, 8), generator=generator)
    actual, geometry = _canonical_actor_tangent_null_direction(weight, 4, 2, 2)
    expected, expected_geometry = _audited_tangent_direction(weight, 4, 2, 2)
    assert torch.equal(actual, expected)
    assert geometry == expected_geometry
    assert geometry["actor_first_layer_residual_max_abs"] <= 1e-8
    assert geometry["simplex_tangent_residual_max_abs"] <= 1e-8
    assert geometry["stochastic_direction_norm"] > 0.0


def test_tangent_residual_loss_backpropagates_direct_feature_target() -> None:
    child = torch.zeros((1, 4, 8), requires_grad=True)
    parent_det = torch.zeros((1, 4, 4))
    parent_logits = torch.zeros((1, 4, 2, 2))
    direction = torch.asarray(
        [0.5, -0.5, 0.0, 0.0, 0.25, -0.25, 0.25, -0.25]
    )
    normalized = torch.asarray([[0.0, 0.5, 1.0]])
    target_m = torch.asarray([[2.0, 0.5, 0.0]])
    loss, plane, delta = _tangent_residual_losses(
        child,
        parent_det,
        parent_logits,
        direction,
        normalized,
        target_m,
        np.asarray([0]),
        np.ones(8, dtype=np.int64),
        np.ones(8, dtype=np.int64),
        scale=0.05,
        burn_in=1,
        near_plane_m=1.0,
        near_plane_multiplier=2.0,
        delta_weight=1.0,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(plane)
    assert torch.isfinite(delta)
    loss.backward()
    assert child.grad is not None
    assert child.grad.abs().max() > 0.0


def test_n700_source_contract_is_exact() -> None:
    report = (
        Path(__file__).resolve().parents[1]
        / "logs/drone_race_vq2_informed_dreamer/n700_n698_n699_tangent_source_contract/report.json"
    )
    assert _tangent_source_contract_exact(report)


def test_invariant_progress_loss_matches_source_locked_audit() -> None:
    generator = torch.Generator().manual_seed(705)
    feature = torch.randn((2, 7, 16), generator=generator)
    target = torch.randn((2, 7), generator=generator)
    level_weight = torch.rand((2, 7), generator=generator) + 0.1
    pair_weight = torch.rand((2, 6), generator=generator) + 0.1
    actual, actual_metrics = _invariant_progress_loss(
        feature,
        target,
        level_weight,
        pair_weight,
        level_coefficient=1.0,
        delta_coefficient=1.0,
    )
    expected, expected_metrics = _audited_invariant_progress_loss(
        feature,
        target,
        level_weight,
        pair_weight,
        level_weight_coefficient=1.0,
        delta_weight_coefficient=1.0,
    )
    assert torch.equal(actual, expected)
    for name in ("level", "delta"):
        for metric in (
            "score",
            "feature_trace_variance",
            "target_variance",
            "cross_covariance_norm",
        ):
            assert actual_metrics[name][metric] == expected_metrics[name][metric]


def test_n704_source_contract_is_exact() -> None:
    report = (
        Path(__file__).resolve().parents[1]
        / "logs/drone_race_vq2_informed_dreamer/n704_n587_rotation_invariant_progress_geometry/report.json"
    )
    assert _invariant_source_contract_exact(report)


def test_n706_frozen_readout_contract_is_exact() -> None:
    report = (
        Path(__file__).resolve().parents[1]
        / "logs/drone_race_vq2_informed_dreamer/n706_n705_frozen_ridge_readout_smoke/report.json"
    )
    assert _frozen_readout_contract_exact(report)
