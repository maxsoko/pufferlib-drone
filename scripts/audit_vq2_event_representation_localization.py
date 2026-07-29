#!/usr/bin/env python3
"""Localize where frozen N604 loses hard gate-event separability.

This read-only audit repeats N628's exact group split and targets while probing
intermediate legal-only encoder, posterior, and prior representations.  Linear
ridge probes are diagnostics only; no learned head or checkpoint is saved.
"""

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
from scripts.audit_vq2_event_latent_separability import (
    _fit_ridge_probe,
    _metrics,
    _roc_auc,
    _stratified_group_split,
)
from scripts.continue_vq2_event_balanced_world_offline import (
    _event_batch,
    _load_event_dataset,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.train_vq2_informed_dreamer import _autocast_context, _burn_in_state


AUDIT_SCHEMA = "vq2_frozen_event_representation_localization_v1"


@torch.no_grad()
def _extract_variants(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    *,
    feature_context: int,
    negative_offsets: tuple[int, ...],
    event_threshold: float,
    amp_dtype: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    event_index = observation.shape[1] - 1
    targets = tuple(event_index - value for value in negative_offsets) + (event_index,)
    if min(targets) < feature_context:
        raise ValueError("feature context exceeds earliest hard-negative history")
    collected: dict[str, list[torch.Tensor]] = {
        "action_only": [],
        "image_encoder_action": [],
        "sensor_encoder_action": [],
        "fused_encoder_action": [],
        "current_deterministic_action": [],
        "current_stochastic_action": [],
        "current_posterior_action": [],
        "next_deterministic_action": [],
        "next_stochastic_action": [],
        "next_prior_action": [],
    }
    labels: list[torch.Tensor] = []
    for target in targets:
        with _autocast_context(observation.device, amp_dtype):
            state = _burn_in_state(
                model,
                observation[:, target - feature_context : target],
                action[:, target - feature_context : target],
            )
            current_legal = observation[:, target - 1, :LEGAL_OBS_SIZE]
            flat_mask = current_legal[:, :MASK_SIZE].reshape(
                -1, 1, VISUAL_HEIGHT, VISUAL_WIDTH
            )
            image_feature = model.rssm.encoder.image(flat_mask)
            sensor_feature = model.rssm.encoder.sensors(current_legal[:, MASK_SIZE:])
            fused_feature = model.rssm.encoder.fusion(
                torch.cat((image_feature, sensor_feature), -1)
            )
            prior = model.rssm.imagine_step(
                state,
                action[:, target],
                deterministic_latent=True,
            )
        current_action = action[:, target].float()
        current_stochastic = state.stochastic.flatten(-2).float()
        next_stochastic = prior.stochastic.flatten(-2).float()
        values = {
            "action_only": current_action,
            "image_encoder_action": torch.cat((image_feature.float(), current_action), -1),
            "sensor_encoder_action": torch.cat((sensor_feature.float(), current_action), -1),
            "fused_encoder_action": torch.cat((fused_feature.float(), current_action), -1),
            "current_deterministic_action": torch.cat(
                (state.deterministic.float(), current_action), -1
            ),
            "current_stochastic_action": torch.cat(
                (current_stochastic, current_action), -1
            ),
            "current_posterior_action": torch.cat(
                (state.features.float(), current_action), -1
            ),
            "next_deterministic_action": torch.cat(
                (prior.deterministic.float(), current_action), -1
            ),
            "next_stochastic_action": torch.cat((next_stochastic, current_action), -1),
            "next_prior_action": torch.cat((prior.features.float(), current_action), -1),
        }
        for name, value in values.items():
            collected[name].append(value.cpu())
        labels.append((reward[:, target] > event_threshold).cpu())
    label = torch.stack(labels, 1).numpy().astype(bool)
    if not label[:, -1].all() or label[:, :-1].any():
        raise RuntimeError("event-window labels are not one positive after hard negatives")
    variants = {
        name: torch.stack(values, 1).numpy()
        for name, values in collected.items()
    }
    return variants, label


def _variant_result(
    features: np.ndarray,
    labels: np.ndarray,
    event_split: np.ndarray,
    phases: np.ndarray,
    *,
    ridge_values: tuple[float, ...],
) -> dict[str, object]:
    samples_per_event = labels.shape[1]
    flat_features = features.reshape(-1, features.shape[-1])
    flat_labels = labels.reshape(-1)
    flat_split = np.repeat(event_split, samples_per_event)
    flat_phases = np.repeat(phases, samples_per_event)
    scores, fit = _fit_ridge_probe(
        flat_features,
        flat_labels,
        flat_split,
        ridge_values=ridge_values,
    )
    validation = flat_split == 1
    test = flat_split == 2
    threshold = float(fit["validation_threshold"])
    test_metrics = _metrics(flat_labels[test], scores[test], threshold=threshold)
    phase_auc = {
        str(phase): _roc_auc(
            flat_labels[test & (flat_phases == phase)],
            scores[test & (flat_phases == phase)],
        )
        for phase in sorted(np.unique(phases))
    }
    score_by_event = scores.reshape(labels.shape)
    heldout_event = event_split == 2
    within_window_rank = float(
        (
            score_by_event[heldout_event, -1]
            > score_by_event[heldout_event, :-1].max(1)
        ).mean()
    )
    return {
        "fit": fit,
        "validation": _metrics(
            flat_labels[validation], scores[validation], threshold=threshold
        ),
        "test": test_metrics,
        "test_phase_roc_auc": phase_auc,
        "test_minimum_phase_roc_auc": min(phase_auc.values()),
        "test_within_window_all_negative_rank_fraction": within_window_rank,
    }


def _passes_event_gate(result: dict[str, object], *, action_auc: float) -> bool:
    test = result["test"]
    assert isinstance(test, dict)
    return bool(
        float(test["roc_auc"]) >= 0.90
        and float(test["average_precision"]) >= 0.65
        and float(result["test_within_window_all_negative_rank_fraction"]) >= 0.75
        and float(result["test_minimum_phase_roc_auc"]) >= 0.75
        and float(test["roc_auc"]) - action_auc >= 0.10
    )


def _classify_bottleneck(passes: dict[str, bool]) -> str:
    if passes["current_posterior_action"] and not passes["next_prior_action"]:
        return "learned_prior_transition"
    if passes["fused_encoder_action"] and not passes["current_posterior_action"]:
        return "posterior_recurrent_compression"
    if passes["image_encoder_action"] and not passes["fused_encoder_action"]:
        return "encoder_fusion"
    if passes["sensor_encoder_action"] and not passes["fused_encoder_action"]:
        return "encoder_fusion_sensor_path"
    if passes["next_prior_action"]:
        return "n628_linear_fit_inconsistency"
    if any(passes.values()):
        return "partial_intermediate_signal"
    return "no_frozen_linear_representation_separates_events"


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_hashes_before = {
        "checkpoint": _sha256(args.checkpoint),
        "event_dataset": _sha256(args.event_dataset),
        "n628_report": _sha256(args.n628_report),
    }
    n628 = json.loads(args.n628_report.read_text(encoding="utf-8"))
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
    observation, action, reward, _continuation = _event_batch(
        dataset,
        np.arange(dataset["mask"].shape[0]),
        device=device,
    )
    negative_offsets = tuple(
        sorted((int(value) for value in args.negative_offsets.split(",")), reverse=True)
    )
    variants, labels = _extract_variants(
        model,
        observation,
        action,
        reward,
        feature_context=args.feature_context,
        negative_offsets=negative_offsets,
        event_threshold=args.event_threshold,
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
    results = {
        name: _variant_result(
            feature,
            labels,
            event_split,
            phases,
            ridge_values=ridge_values,
        )
        for name, feature in variants.items()
    }
    action_auc = float(results["action_only"]["test"]["roc_auc"])
    passes = {
        name: _passes_event_gate(result, action_auc=action_auc)
        for name, result in results.items()
        if name != "action_only"
    }
    n628_prior_auc = float(n628["latent_action_test"]["roc_auc"])
    n628_prior_ap = float(n628["latent_action_test"]["average_precision"])
    reproduced_prior_auc = float(results["next_prior_action"]["test"]["roc_auc"])
    reproduced_prior_ap = float(
        results["next_prior_action"]["test"]["average_precision"]
    )
    reproduction_error = max(
        abs(reproduced_prior_auc - n628_prior_auc),
        abs(reproduced_prior_ap - n628_prior_ap),
    )
    model_changes = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, source_state[name])
    ]
    source_hashes_after = {
        "checkpoint": _sha256(args.checkpoint),
        "event_dataset": _sha256(args.event_dataset),
        "n628_report": _sha256(args.n628_report),
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "feature_context": args.feature_context,
        "negative_offsets": list(negative_offsets),
        "seed": args.seed,
        "variant_results": results,
        "variant_passes_n628_gate": passes,
        "bottleneck_classification": _classify_bottleneck(passes),
        "n628_prior_metric_reproduction_max_abs_error": reproduction_error,
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
    parser.add_argument("--n628-report", type=Path, required=True)
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
