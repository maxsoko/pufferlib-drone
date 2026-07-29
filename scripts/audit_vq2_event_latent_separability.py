#!/usr/bin/env python3
"""Read-only held-out gate-event separability audit for frozen VQ2 latents.

The probe consumes only deterministic N604 states built from the legal
observation slice and Puffer actions.  Privileged phase/reward values supply
offline labels and grouping metadata only.  No model parameter is optimized or
saved; a closed-form linear ridge probe is fit on group-disjoint event windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import VQ2InformedDreamer
from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from scripts.continue_vq2_event_balanced_world_offline import (
    _event_batch,
    _load_event_dataset,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.train_vq2_informed_dreamer import _autocast_context, _burn_in_state


AUDIT_SCHEMA = "vq2_frozen_legal_latent_event_separability_v1"


def _hash_order(seed: int, split: str, phase: int, group: int) -> bytes:
    return hashlib.sha256(f"{seed}:{split}:{phase}:{group}".encode()).digest()


def _stratified_group_split(
    groups: np.ndarray,
    phases: np.ndarray,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> np.ndarray:
    """Assign whole groups to train/validation/test with phase coverage."""

    groups = np.asarray(groups, dtype=np.int64)
    phases = np.asarray(phases, dtype=np.int64)
    if groups.ndim != 1 or phases.shape != groups.shape or groups.size == 0:
        raise ValueError("groups and phases must be aligned nonempty vectors")
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("validation/test fractions must lie in (0,1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation plus test fraction must be below one")

    unique_groups = set(int(value) for value in np.unique(groups))
    group_phases = {
        group: set(int(value) for value in phases[groups == group])
        for group in unique_groups
    }
    assignment = {group: "train" for group in unique_groups}
    remaining = set(unique_groups)
    phase_values = sorted(int(value) for value in np.unique(phases))

    for split, fraction in (("test", test_fraction), ("validation", validation_fraction)):
        selected: set[int] = set()
        targets = {
            phase: max(1, int(math.ceil((phases == phase).sum() * fraction)))
            for phase in phase_values
        }
        # Protect the rarest phase first; a multi-phase simultaneous group can
        # satisfy more than one target but is never split across partitions.
        ordered_phases = sorted(phase_values, key=lambda p: ((phases == p).sum(), p))
        for phase in ordered_phases:
            while sum(phase in group_phases[group] for group in selected) < targets[phase]:
                candidates = [
                    group
                    for group in remaining
                    if group not in selected and phase in group_phases[group]
                ]
                if not candidates:
                    raise RuntimeError(f"cannot provide {split} coverage for phase {phase}")
                candidates.sort(key=lambda group: _hash_order(seed, split, phase, group))
                selected.add(candidates[0])
        for group in selected:
            assignment[group] = split
        remaining.difference_update(selected)

    encoded = np.asarray(
        [{"train": 0, "validation": 1, "test": 2}[assignment[int(group)]] for group in groups],
        dtype=np.int8,
    )
    for phase in phase_values:
        for split_value in (0, 1, 2):
            if not np.any((phases == phase) & (encoded == split_value)):
                raise RuntimeError(
                    f"phase {phase} missing from split {split_value}; reduce holdout fractions"
                )
    return encoded


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels]
    negative = scores[~labels]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("ROC AUC needs positive and negative samples")
    comparison = positive[:, None] - negative[None, :]
    return float(((comparison > 0.0) + 0.5 * (comparison == 0.0)).mean())


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if not labels.any():
        raise ValueError("average precision needs a positive sample")
    # Stable ordering makes exact ties deterministic. This is the standard
    # non-interpolated average of precision at each positive rank.
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order].astype(np.float64)
    precision = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
    return float((precision * ordered).sum() / ordered.sum())


def _balanced_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores) >= threshold
    tpr = float(predicted[labels].mean())
    tnr = float((~predicted[~labels]).mean())
    return 0.5 * (tpr + tnr)


def _validation_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(scores)
    candidates = np.concatenate(
        ([np.nextafter(candidates.max(), np.inf)], candidates, [np.nextafter(candidates.min(), -np.inf)])
    )
    ranked = [(_balanced_accuracy(labels, scores, value), value) for value in candidates]
    # Prefer the larger threshold on an exact balanced-accuracy tie.
    return float(max(ranked, key=lambda item: (item[0], item[1]))[1])


def _metrics(labels: np.ndarray, scores: np.ndarray, *, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores) >= threshold
    return {
        "samples": int(labels.size),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
        "roc_auc": _roc_auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "balanced_accuracy": _balanced_accuracy(labels, scores, threshold),
        "true_positive_rate": float(predicted[labels].mean()),
        "false_positive_rate": float(predicted[~labels].mean()),
        "score_mean_positive": float(np.asarray(scores)[labels].mean()),
        "score_mean_negative": float(np.asarray(scores)[~labels].mean()),
    }


def _fit_ridge_probe(
    features: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    *,
    ridge_values: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit validation-selected class-balanced linear ridge regression."""

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    split = np.asarray(split)
    train = split == 0
    validation = split == 1
    if features.ndim != 2 or features.shape[0] != labels.size:
        raise ValueError("probe features and labels are misaligned")
    if not train.any() or not validation.any():
        raise ValueError("probe needs train and validation samples")
    if not ridge_values or any(value <= 0.0 for value in ridge_values):
        raise ValueError("ridge values must be positive")

    mean = features[train].mean(0)
    std = features[train].std(0)
    active = std > 1e-6
    if not active.any():
        raise RuntimeError("all probe features are constant")
    standardized = (features[:, active] - mean[active]) / std[active]
    design = np.concatenate((standardized, np.ones((features.shape[0], 1))), 1)
    target = labels.astype(np.float64) * 2.0 - 1.0
    positive_weight = 0.5 / labels[train].sum()
    negative_weight = 0.5 / (~labels[train]).sum()
    sample_weight = np.where(labels[train], positive_weight, negative_weight)
    weighted_design = design[train] * np.sqrt(sample_weight[:, None])
    weighted_target = target[train] * np.sqrt(sample_weight)
    gram = weighted_design.T @ weighted_design
    rhs = weighted_design.T @ weighted_target

    candidates: list[dict[str, float | np.ndarray]] = []
    for ridge in ridge_values:
        regularizer = np.eye(gram.shape[0]) * ridge
        regularizer[-1, -1] = 0.0
        coefficient = np.linalg.solve(gram + regularizer, rhs)
        scores = design @ coefficient
        candidates.append(
            {
                "ridge": ridge,
                "validation_roc_auc": _roc_auc(labels[validation], scores[validation]),
                "validation_average_precision": _average_precision(
                    labels[validation], scores[validation]
                ),
                "coefficient": coefficient,
                "scores": scores,
            }
        )
    # Prefer stronger regularization only after validation AUC and AP ties.
    selected = max(
        candidates,
        key=lambda row: (
            float(row["validation_roc_auc"]),
            float(row["validation_average_precision"]),
            float(row["ridge"]),
        ),
    )
    scores = np.asarray(selected["scores"])
    threshold = _validation_threshold(labels[validation], scores[validation])
    summary = {
        "input_features": int(features.shape[1]),
        "active_standardized_features": int(active.sum()),
        "selected_ridge": float(selected["ridge"]),
        "validation_threshold": threshold,
        "ridge_grid": [
            {
                "ridge": float(row["ridge"]),
                "validation_roc_auc": float(row["validation_roc_auc"]),
                "validation_average_precision": float(row["validation_average_precision"]),
            }
            for row in candidates
        ],
    }
    return scores, summary


@torch.no_grad()
def _extract_event_samples(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    *,
    feature_context: int,
    negative_offsets: tuple[int, ...],
    event_threshold: float,
    amp_dtype: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_index = observation.shape[1] - 1
    target_indices = tuple(event_index - value for value in negative_offsets) + (event_index,)
    if min(target_indices) < feature_context:
        raise ValueError("feature context exceeds the earliest target history")
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    selected_actions: list[torch.Tensor] = []
    for target in target_indices:
        with _autocast_context(observation.device, amp_dtype):
            state = _burn_in_state(
                model,
                observation[:, target - feature_context : target],
                action[:, target - feature_context : target],
            )
            prior = model.rssm.imagine_step(
                state,
                action[:, target],
                deterministic_latent=True,
            )
        current_action = action[:, target].float()
        features.append(torch.cat((prior.features.float(), current_action), -1).cpu())
        labels.append((reward[:, target] > event_threshold).cpu())
        selected_actions.append(current_action.cpu())
    # Event-major layout keeps every positive and its hard negatives in one
    # group for leakage-free splitting and within-window ranking.
    feature = torch.stack(features, 1).numpy()
    label = torch.stack(labels, 1).numpy().astype(bool)
    selected_action = torch.stack(selected_actions, 1).numpy()
    if not label[:, -1].all() or label[:, :-1].any():
        raise RuntimeError("event-window targets are not one positive after hard negatives")
    return feature, label, selected_action


def _phase_split_counts(
    split: np.ndarray, phases: np.ndarray, groups: np.ndarray
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in (("train", 0), ("validation", 1), ("test", 2)):
        chosen = split == value
        result[name] = {
            "events": int(chosen.sum()),
            "groups": int(np.unique(groups[chosen]).size),
            "phase_counts": {
                str(phase): int(((phases == phase) & chosen).sum())
                for phase in sorted(np.unique(phases))
            },
        }
    return result


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_sha_before = _sha256(args.checkpoint)
    event_sha_before = _sha256(args.event_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    if not isinstance(training_args, dict):
        raise RuntimeError("checkpoint lacks training arguments")
    model = VQ2InformedDreamer(
        deterministic_size=int(training_args["deterministic_size"]),
        stochastic_groups=int(training_args["stochastic_groups"]),
        stochastic_classes=int(training_args["stochastic_classes"]),
        actor_initial_std=float(training_args.get("actor_initial_std", 0.20)),
        actor_distribution_mode=str(
            training_args.get("actor_distribution_mode", "legacy_tanh_normal")
        ),
        distributional_reward=bool(training_args.get("distributional_reward", False)),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    source_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = _load_event_dataset(
        args.event_dataset,
        context_length=int(dataset_context := 16),
        event_threshold=args.event_threshold,
    )
    observation, action, reward, _continuation = _event_batch(
        dataset,
        np.arange(dataset["mask"].shape[0]),
        device=device,
    )
    negative_offsets = tuple(int(value) for value in args.negative_offsets.split(","))
    if not negative_offsets or any(value <= 0 for value in negative_offsets):
        raise ValueError("negative offsets must be comma-separated positive integers")
    if len(set(negative_offsets)) != len(negative_offsets):
        raise ValueError("negative offsets must be unique")
    negative_offsets = tuple(sorted(negative_offsets, reverse=True))
    feature, label, selected_action = _extract_event_samples(
        model,
        observation,
        action,
        reward,
        feature_context=args.feature_context,
        negative_offsets=negative_offsets,
        event_threshold=args.event_threshold,
        amp_dtype=args.amp_dtype,
    )

    counterfactual = observation.clone()
    counterfactual[..., LEGAL_OBS_SIZE:] = 123.456
    counterfactual_feature, counterfactual_label, _ = _extract_event_samples(
        model,
        counterfactual,
        action,
        reward,
        feature_context=args.feature_context,
        negative_offsets=negative_offsets,
        event_threshold=args.event_threshold,
        amp_dtype=args.amp_dtype,
    )
    privilege_feature_max_abs_error = float(
        np.max(np.abs(counterfactual_feature - feature))
    )
    if not np.array_equal(counterfactual_label, label):
        raise RuntimeError("privilege counterfactual changed event labels")

    event_groups = dataset["vector_step"].astype(np.int64)
    phases = np.rint(dataset["phase_after_event"].astype(np.float64) * 6.0).astype(
        np.int64
    )
    event_split = _stratified_group_split(
        event_groups,
        phases,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    samples_per_event = label.shape[1]
    flat_feature = feature.reshape(-1, feature.shape[-1])
    flat_action = selected_action.reshape(-1, selected_action.shape[-1])
    flat_label = label.reshape(-1)
    flat_split = np.repeat(event_split, samples_per_event)
    flat_phase = np.repeat(phases, samples_per_event)
    flat_group = np.repeat(event_groups, samples_per_event)

    ridge_values = tuple(float(value) for value in args.ridge_values.split(","))
    latent_scores, latent_fit = _fit_ridge_probe(
        flat_feature,
        flat_label,
        flat_split,
        ridge_values=ridge_values,
    )
    action_scores, action_fit = _fit_ridge_probe(
        flat_action,
        flat_label,
        flat_split,
        ridge_values=ridge_values,
    )
    validation = flat_split == 1
    test = flat_split == 2
    latent_threshold = float(latent_fit["validation_threshold"])
    action_threshold = float(action_fit["validation_threshold"])
    latent_test = _metrics(flat_label[test], latent_scores[test], threshold=latent_threshold)
    action_test = _metrics(flat_label[test], action_scores[test], threshold=action_threshold)

    phase_auc = {
        str(phase): _roc_auc(
            flat_label[test & (flat_phase == phase)],
            latent_scores[test & (flat_phase == phase)],
        )
        for phase in sorted(np.unique(phases))
    }
    score_by_event = latent_scores.reshape(label.shape)
    heldout_events = event_split == 2
    within_window_all_negative_rank = float(
        (
            score_by_event[heldout_events, -1]
            > score_by_event[heldout_events, :-1].max(1)
        ).mean()
    )
    model_changes = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, source_state[name])
    ]
    checkpoint_sha_after = _sha256(args.checkpoint)
    event_sha_after = _sha256(args.event_dataset)
    gates = {
        "test_roc_auc_at_least_0p90": float(latent_test["roc_auc"]) >= 0.90,
        "test_average_precision_at_least_0p65": float(latent_test["average_precision"]) >= 0.65,
        "within_window_rank_at_least_0p75": within_window_all_negative_rank >= 0.75,
        "minimum_phase_auc_at_least_0p75": min(phase_auc.values()) >= 0.75,
        "auc_margin_over_action_at_least_0p10": (
            float(latent_test["roc_auc"]) - float(action_test["roc_auc"])
        ) >= 0.10,
        "privilege_counterfactual_exact": privilege_feature_max_abs_error == 0.0,
        "model_exact": not model_changes,
        "checkpoint_exact": checkpoint_sha_after == checkpoint_sha_before,
        "event_dataset_exact": event_sha_after == event_sha_before,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha_after,
        "event_dataset": str(args.event_dataset),
        "event_dataset_sha256": event_sha_after,
        "dataset_context": dataset_context,
        "feature_context": args.feature_context,
        "negative_offsets": list(negative_offsets),
        "target_indices": [dataset_context - value for value in negative_offsets]
        + [dataset_context],
        "event_threshold": args.event_threshold,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "event_split": _phase_split_counts(event_split, phases, event_groups),
        "train_validation_group_overlap": int(
            len(set(flat_group[flat_split == 0]) & set(flat_group[flat_split == 1]))
        ),
        "train_test_group_overlap": int(
            len(set(flat_group[flat_split == 0]) & set(flat_group[flat_split == 2]))
        ),
        "validation_test_group_overlap": int(
            len(set(flat_group[flat_split == 1]) & set(flat_group[flat_split == 2]))
        ),
        "latent_action_fit": latent_fit,
        "action_only_fit": action_fit,
        "latent_action_validation": _metrics(
            flat_label[validation], latent_scores[validation], threshold=latent_threshold
        ),
        "latent_action_test": latent_test,
        "action_only_test": action_test,
        "test_auc_margin_over_action": float(latent_test["roc_auc"])
        - float(action_test["roc_auc"]),
        "test_phase_roc_auc": phase_auc,
        "test_minimum_phase_roc_auc": min(phase_auc.values()),
        "test_within_window_all_negative_rank_fraction": within_window_all_negative_rank,
        "privilege_counterfactual_feature_max_abs_error": privilege_feature_max_abs_error,
        "model_tensor_changes": model_changes,
        "gates": gates,
        "passed": all(gates.values()),
        "probe_model": "class-balanced closed-form linear ridge",
        "probe_fits": len(ridge_values) * 2,
        "world_updates": 0,
        "reward_updates": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--event-dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--feature-context", type=int, default=12)
    parser.add_argument("--negative-offsets", default="4,3,2,1")
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=628)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--ridge-values", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
