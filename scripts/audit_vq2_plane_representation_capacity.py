#!/usr/bin/env python3
"""Read-only continuous plane-distance capacity probe for frozen N604 features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import VQ2InformedDreamer
from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE, VISUAL_HEIGHT, VISUAL_WIDTH
from scripts.audit_vq2_event_latent_separability import _stratified_group_split
from scripts.audit_vq2_plane_progress_support import _correlation
from scripts.continue_vq2_event_balanced_world_offline import (
    _event_batch,
    _load_event_dataset,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.train_vq2_informed_dreamer import _autocast_context, _burn_in_state


AUDIT_SCHEMA = "vq2_frozen_continuous_plane_representation_capacity_v1"


@torch.no_grad()
def _extract_features(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    *,
    targets: tuple[int, ...],
    feature_context: int,
    amp_dtype: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    collected: dict[str, list[torch.Tensor]] = {
        "image_encoder": [],
        "sensor_encoder": [],
        "fused_encoder": [],
        "current_deterministic": [],
        "current_posterior": [],
        "decoder_hidden": [],
    }
    parent_prediction: list[torch.Tensor] = []
    target_forward: list[torch.Tensor] = []
    for target in targets:
        with _autocast_context(observation.device, amp_dtype):
            state = _burn_in_state(
                model,
                observation[:, target - feature_context : target],
                action[:, target - feature_context : target],
            )
            legal = observation[:, target - 1, :LEGAL_OBS_SIZE]
            image = model.rssm.encoder.image(
                legal[:, :MASK_SIZE].reshape(-1, 1, VISUAL_HEIGHT, VISUAL_WIDTH)
            )
            sensor = model.rssm.encoder.sensors(legal[:, MASK_SIZE:])
            fused = model.rssm.encoder.fusion(torch.cat((image, sensor), -1))
            hidden = model.privileged_decoder[:-1](state.features)
            decoded = model.privileged_decoder[-1](hidden)[..., 3]
        values = {
            "image_encoder": image,
            "sensor_encoder": sensor,
            "fused_encoder": fused,
            "current_deterministic": state.deterministic,
            "current_posterior": state.features,
            "decoder_hidden": hidden,
        }
        for name, value in values.items():
            collected[name].append(value.float().cpu())
        parent_prediction.append(decoded.float().cpu())
        target_forward.append(
            observation[:, target - 1, LEGAL_OBS_SIZE + 3].float().cpu()
        )
    return (
        {name: torch.stack(value, 1).numpy() for name, value in collected.items()},
        torch.stack(parent_prediction, 1).numpy(),
        torch.stack(target_forward, 1).numpy(),
    )


def _fit_continuous_ridge(
    features: np.ndarray,
    target: np.ndarray,
    event_split: np.ndarray,
    *,
    ridge_values: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    samples = target.shape[1]
    flat_features = np.asarray(features, dtype=np.float64).reshape(
        -1, features.shape[-1]
    )
    flat_target = np.asarray(target, dtype=np.float64).reshape(-1)
    flat_split = np.repeat(event_split, samples)
    train = flat_split == 0
    validation = flat_split == 1
    mean = flat_features[train].mean(0)
    std = flat_features[train].std(0)
    active = std > 1e-6
    standardized = (flat_features[:, active] - mean[active]) / std[active]
    design = np.concatenate((standardized, np.ones((flat_features.shape[0], 1))), 1)
    target_mean = flat_target[train].mean()
    target_std = flat_target[train].std()
    if target_std <= 0.0:
        raise RuntimeError("training plane-distance target is constant")
    normalized_target = (flat_target - target_mean) / target_std
    gram = design[train].T @ design[train] / train.sum()
    rhs = design[train].T @ normalized_target[train] / train.sum()
    candidates: list[dict[str, object]] = []
    for ridge in ridge_values:
        regularizer = np.eye(gram.shape[0]) * ridge
        regularizer[-1, -1] = 0.0
        coefficient = np.linalg.solve(gram + regularizer, rhs)
        prediction = (design @ coefficient) * target_std + target_mean
        candidates.append(
            {
                "ridge": ridge,
                "prediction": prediction,
                "validation_correlation": _correlation(
                    flat_target[validation], prediction[validation]
                ),
                "validation_mae": float(
                    np.abs(flat_target[validation] - prediction[validation]).mean()
                ),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["validation_correlation"]),
            -float(row["validation_mae"]),
            float(row["ridge"]),
        ),
    )
    return np.asarray(selected["prediction"]).reshape(target.shape), {
        "input_features": int(features.shape[-1]),
        "active_standardized_features": int(active.sum()),
        "selected_ridge": float(selected["ridge"]),
        "ridge_grid": [
            {
                "ridge": float(row["ridge"]),
                "validation_correlation": float(row["validation_correlation"]),
                "validation_mae": float(row["validation_mae"]),
            }
            for row in candidates
        ],
    }


def _regression_result(
    target: np.ndarray,
    prediction: np.ndarray,
    event_split: np.ndarray,
) -> dict[str, float]:
    test = event_split == 2
    error = prediction[test] - target[test]
    step = float(np.abs(np.diff(target[test], axis=1)).mean())
    return {
        "correlation": _correlation(target[test], prediction[test]),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_true_step": step,
        "mae_in_true_step_units": float(np.abs(error).mean()) / step,
        "strictly_decreasing_window_fraction": float(
            (np.diff(prediction[test], axis=1) < 0.0).all(1).mean()
        ),
        "final_is_minimum_window_fraction": float(
            (prediction[test, -1] < prediction[test, :-1].min(1)).mean()
        ),
    }


def _passes_capacity(result: dict[str, float]) -> bool:
    return bool(
        result["correlation"] >= 0.80
        and result["mae_in_true_step_units"] <= 2.0
        and result["strictly_decreasing_window_fraction"] >= 0.75
        and result["final_is_minimum_window_fraction"] >= 0.75
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_hashes_before = {
        "checkpoint": _sha256(args.checkpoint),
        "event_dataset": _sha256(args.event_dataset),
        "n631_report": _sha256(args.n631_report),
    }
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
    dataset_context = 16
    dataset = _load_event_dataset(
        args.event_dataset,
        context_length=dataset_context,
        event_threshold=args.event_threshold,
    )
    observation, action, _reward, _continuation = _event_batch(
        dataset,
        np.arange(dataset["mask"].shape[0]),
        device=device,
    )
    negative_offsets = tuple(
        sorted((int(value) for value in args.negative_offsets.split(",")), reverse=True)
    )
    targets = tuple(dataset_context - value for value in negative_offsets) + (
        dataset_context,
    )
    features, parent_prediction, target_forward = _extract_features(
        model,
        observation,
        action,
        targets=targets,
        feature_context=args.feature_context,
        amp_dtype=args.amp_dtype,
    )
    groups = dataset["vector_step"].astype(np.int64)
    phases = np.rint(dataset["phase_after_event"].astype(np.float64) * 6.0).astype(
        np.int64
    )
    event_split = _stratified_group_split(
        groups,
        phases,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    ridge_values = tuple(float(value) for value in args.ridge_values.split(","))
    results: dict[str, object] = {}
    passes: dict[str, bool] = {}
    for name, feature in features.items():
        prediction, fit = _fit_continuous_ridge(
            feature,
            target_forward,
            event_split,
            ridge_values=ridge_values,
        )
        regression = _regression_result(target_forward, prediction, event_split)
        results[name] = {"fit": fit, "test": regression}
        passes[name] = _passes_capacity(regression)
    parent_result = _regression_result(
        target_forward, parent_prediction, event_split
    )
    source_hashes_after = {
        "checkpoint": _sha256(args.checkpoint),
        "event_dataset": _sha256(args.event_dataset),
        "n631_report": _sha256(args.n631_report),
    }
    model_changes = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, source_state[name])
    ]
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "targets": list(targets),
        "feature_context": args.feature_context,
        "negative_offsets": list(negative_offsets),
        "parent_decoder_test": parent_result,
        "variant_results": results,
        "variant_passes_capacity_gate": passes,
        "best_test_correlation": max(
            float(result["test"]["correlation"]) for result in results.values()
        ),
        "model_tensor_changes": model_changes,
        "model_exact": not model_changes,
        "probe_fits": len(results) * len(ridge_values),
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
    parser.add_argument("--n631-report", type=Path, required=True)
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
