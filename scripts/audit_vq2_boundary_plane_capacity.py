#!/usr/bin/env python3
"""Read-only plane-capacity audit using source-locked legal RSSM boundaries.

N643 stores the N604 posterior immediately before each short event window. The
boundary was produced only from legal observations, previous Puffer actions,
and recurrent carry. This audit measures whether restoring that carry exposes
continuous gate-forward progress that was hidden by N632's zero-state rolling
reconstruction. Privileged pose is used only as an offline regression label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer
from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_event_latent_separability import (
    _phase_split_counts,
    _stratified_group_split,
)
from scripts.audit_vq2_plane_representation_capacity import (
    _fit_continuous_ridge,
    _passes_capacity,
    _regression_result,
)
from scripts.continue_vq2_event_balanced_world_offline import _event_batch
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.train_vq2_informed_dreamer import _burn_in_state


AUDIT_SCHEMA = "vq2_frozen_boundary_continuous_plane_capacity_v1"
BOUNDARY_DATASET_FIELDS = {
    "mask",
    "tail",
    "action",
    "reward",
    "continuation",
    "agent",
    "vector_step",
    "phase_after_event",
    "initial_deterministic",
    "initial_stochastic_index",
    "initial_logits",
    "preevent_deterministic",
    "preevent_stochastic_index",
    "preevent_logits",
}


def _load_boundary_event_dataset(
    path: Path,
    *,
    context_length: int,
    event_threshold: float,
    deterministic_size: int,
    stochastic_groups: int,
    stochastic_classes: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != BOUNDARY_DATASET_FIELDS:
            raise RuntimeError("boundary event corpus schema mismatch")
        dataset = {name: archive[name].copy() for name in archive.files}
    events = int(dataset["mask"].shape[0])
    length = context_length + 1
    expected = {
        "mask": (events, length, MASK_SIZE),
        "tail": (events, length, ENV_OBS_SIZE - MASK_SIZE),
        "action": (events, length, 4),
        "reward": (events, length),
        "continuation": (events, length),
        "agent": (events,),
        "vector_step": (events,),
        "phase_after_event": (events,),
        "initial_deterministic": (events, deterministic_size),
        "initial_stochastic_index": (events, stochastic_groups),
        "initial_logits": (events, stochastic_groups, stochastic_classes),
        "preevent_deterministic": (events, deterministic_size),
        "preevent_stochastic_index": (events, stochastic_groups),
        "preevent_logits": (events, stochastic_groups, stochastic_classes),
    }
    if events <= 0:
        raise RuntimeError("boundary event corpus is empty")
    for name, shape in expected.items():
        if dataset[name].shape != shape:
            raise RuntimeError(
                f"boundary event corpus {name} shape mismatch: "
                f"{dataset[name].shape} != {shape}"
            )
        if not np.isfinite(dataset[name]).all():
            raise RuntimeError(f"boundary event corpus {name} is non-finite")
    if not (dataset["reward"][:, -1] > event_threshold).all():
        raise RuntimeError("boundary event corpus final reward misses threshold")
    if not (dataset["reward"][:, :-1] <= event_threshold).all():
        raise RuntimeError("boundary event corpus contains an early event")
    if not (dataset["continuation"][:, :-1] == 1).all():
        raise RuntimeError("boundary event corpus crosses a pre-event terminal")
    if np.abs(dataset["action"]).max() > 1.0:
        raise RuntimeError("boundary event corpus action exceeds normalized bounds")
    indices = dataset["initial_stochastic_index"]
    preevent_indices = dataset["preevent_stochastic_index"]
    if indices.min() < 0 or indices.max() >= stochastic_classes:
        raise RuntimeError("initial stochastic index is outside the class range")
    if preevent_indices.min() < 0 or preevent_indices.max() >= stochastic_classes:
        raise RuntimeError("preevent stochastic index is outside the class range")
    return dataset


def _boundary_state(
    dataset: dict[str, np.ndarray],
    *,
    device: torch.device,
    stochastic_classes: int,
) -> RSSMState:
    index = torch.from_numpy(
        dataset["initial_stochastic_index"].astype(np.int64)
    ).to(device)
    return RSSMState(
        torch.from_numpy(dataset["initial_deterministic"].astype(np.float32)).to(
            device
        ),
        F.one_hot(index, stochastic_classes).to(torch.float32),
        torch.from_numpy(dataset["initial_logits"].astype(np.float32)).to(device),
    )


def _state_at(output: object, index: int) -> RSSMState:
    posterior = output.posterior
    return RSSMState(
        posterior.deterministic[:, index],
        posterior.stochastic[:, index],
        posterior.logits[:, index],
    )


@torch.no_grad()
def _extract_history_features(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    initial_state: RSSMState,
    *,
    targets: tuple[int, ...],
    rolling_context: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract aligned states before each target action under three histories."""

    maximum_target = max(targets)
    exact_output = model.observe_sequence(
        observation[:, :maximum_target],
        action[:, :maximum_target],
        initial_state=initial_state,
        deterministic_latent=True,
    )
    zero_full_output = model.observe_sequence(
        observation[:, :maximum_target],
        action[:, :maximum_target],
        deterministic_latent=True,
    )
    states: dict[str, list[RSSMState]] = {
        "exact_boundary": [],
        "zero_full_prefix": [],
        "zero_rolling": [],
    }
    for target in targets:
        states["exact_boundary"].append(_state_at(exact_output, target - 1))
        states["zero_full_prefix"].append(_state_at(zero_full_output, target - 1))
        if target < rolling_context:
            raise ValueError("rolling context exceeds an audited target")
        states["zero_rolling"].append(
            _burn_in_state(
                model,
                observation[:, target - rolling_context : target],
                action[:, target - rolling_context : target],
            )
        )

    features: dict[str, np.ndarray] = {}
    decoded: dict[str, np.ndarray] = {}
    for history_name, history_states in states.items():
        deterministic = torch.stack(
            [state.deterministic.float() for state in history_states], 1
        )
        posterior = torch.stack(
            [state.features.float() for state in history_states], 1
        )
        decoder_hidden = model.privileged_decoder[:-1](posterior)
        decoded_forward = model.privileged_decoder[-1](decoder_hidden)[..., 3]
        features[f"{history_name}_deterministic"] = deterministic.cpu().numpy()
        features[f"{history_name}_posterior"] = posterior.cpu().numpy()
        if history_name == "exact_boundary":
            features[f"{history_name}_decoder_hidden"] = (
                decoder_hidden.float().cpu().numpy()
            )
        decoded[history_name] = decoded_forward.float().cpu().numpy()

    return features, decoded


def _boundary_replay_result(
    final_state: RSSMState,
    dataset: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, float | int]:
    expected_deterministic = torch.from_numpy(
        dataset["preevent_deterministic"].astype(np.float32)
    ).to(device)
    expected_logits = torch.from_numpy(
        dataset["preevent_logits"].astype(np.float32)
    ).to(device)
    expected_index = torch.from_numpy(
        dataset["preevent_stochastic_index"].astype(np.int64)
    ).to(device)
    return {
        "deterministic_max_abs_error": float(
            (final_state.deterministic.float() - expected_deterministic)
            .abs()
            .max()
            .cpu()
        ),
        "logits_max_abs_error": float(
            (final_state.logits.float() - expected_logits).abs().max().cpu()
        ),
        "stochastic_index_mismatch_count": int(
            (final_state.stochastic.argmax(-1) != expected_index).sum().cpu()
        ),
    }


@torch.no_grad()
def _replay_final_boundary(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    initial_state: RSSMState,
    *,
    event_index: int,
) -> RSSMState:
    output = model.observe_sequence(
        observation[:, :event_index],
        action[:, :event_index],
        initial_state=initial_state,
        deterministic_latent=True,
    )
    return _state_at(output, event_index - 1)


def _decoded_result(
    target: np.ndarray,
    prediction: np.ndarray,
    event_split: np.ndarray,
) -> dict[str, float]:
    return _regression_result(target, prediction, event_split)


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_paths = {
        "checkpoint": args.checkpoint,
        "event_dataset": args.event_dataset,
        "n632_report": args.n632_report,
        "n643_report": args.n643_report,
    }
    source_hashes_before = {
        name: _sha256(path) for name, path in source_paths.items()
    }
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    if not isinstance(training_args, dict):
        raise RuntimeError("checkpoint lacks training arguments")
    deterministic_size = int(training_args["deterministic_size"])
    stochastic_groups = int(training_args["stochastic_groups"])
    stochastic_classes = int(training_args["stochastic_classes"])
    model = VQ2InformedDreamer(
        deterministic_size=deterministic_size,
        stochastic_groups=stochastic_groups,
        stochastic_classes=stochastic_classes,
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

    dataset = _load_boundary_event_dataset(
        args.event_dataset,
        context_length=args.context_length,
        event_threshold=args.event_threshold,
        deterministic_size=deterministic_size,
        stochastic_groups=stochastic_groups,
        stochastic_classes=stochastic_classes,
    )
    observation, action, reward, _continuation = _event_batch(
        dataset,
        np.arange(dataset["mask"].shape[0]),
        device=device,
    )
    initial_state = _boundary_state(
        dataset, device=device, stochastic_classes=stochastic_classes
    )
    negative_offsets = tuple(
        sorted(
            (int(value) for value in args.negative_offsets.split(",")),
            reverse=True,
        )
    )
    if not negative_offsets or any(value <= 0 for value in negative_offsets):
        raise ValueError("negative offsets must be unique positive integers")
    if len(set(negative_offsets)) != len(negative_offsets):
        raise ValueError("negative offsets must be unique positive integers")
    targets = tuple(args.context_length - value for value in negative_offsets) + (
        args.context_length,
    )
    if min(targets) < args.rolling_context:
        raise ValueError("rolling context exceeds the earliest target")

    features, decoded = _extract_history_features(
        model,
        observation,
        action,
        initial_state,
        targets=targets,
        rolling_context=args.rolling_context,
    )
    counterfactual = observation.clone()
    counterfactual[..., LEGAL_OBS_SIZE:] = 123.456
    counterfactual_features, counterfactual_decoded = _extract_history_features(
        model,
        counterfactual,
        action,
        initial_state,
        targets=targets,
        rolling_context=args.rolling_context,
    )
    privilege_feature_max_abs_error = max(
        float(np.max(np.abs(features[name] - counterfactual_features[name])))
        for name in features
    )
    privilege_decoded_max_abs_error = max(
        float(np.max(np.abs(decoded[name] - counterfactual_decoded[name])))
        for name in decoded
    )

    target_forward = torch.stack(
        [observation[:, target - 1, LEGAL_OBS_SIZE + 3] for target in targets],
        1,
    ).float().cpu().numpy()
    labels = torch.stack(
        [reward[:, target] > args.event_threshold for target in targets], 1
    ).cpu().numpy()
    if not labels[:, -1].all() or labels[:, :-1].any():
        raise RuntimeError("audited targets are not four hard negatives and one event")

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
    variant_results: dict[str, object] = {}
    variant_passes: dict[str, bool] = {}
    for name, feature in features.items():
        prediction, fit = _fit_continuous_ridge(
            feature,
            target_forward,
            event_split,
            ridge_values=ridge_values,
        )
        result = _regression_result(target_forward, prediction, event_split)
        variant_results[name] = {"fit": fit, "test": result}
        variant_passes[name] = _passes_capacity(result)
    decoded_results = {
        name: _decoded_result(target_forward, prediction, event_split)
        for name, prediction in decoded.items()
    }

    final_boundary = _replay_final_boundary(
        model,
        observation,
        action,
        initial_state,
        event_index=args.context_length,
    )
    boundary_replay = _boundary_replay_result(
        final_boundary, dataset, device=device
    )
    source_hashes_after = {
        name: _sha256(path) for name, path in source_paths.items()
    }
    model_changes = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, source_state[name])
    ]
    exact_names = [name for name in variant_results if name.startswith("exact_boundary")]
    zero_names = [name for name in variant_results if name.startswith("zero_")]
    best_exact_name = max(
        exact_names,
        key=lambda name: float(variant_results[name]["test"]["correlation"]),
    )
    best_zero_name = max(
        zero_names,
        key=lambda name: float(variant_results[name]["test"]["correlation"]),
    )
    best_exact_correlation = float(
        variant_results[best_exact_name]["test"]["correlation"]
    )
    best_zero_correlation = float(
        variant_results[best_zero_name]["test"]["correlation"]
    )
    exact_capacity_pass = any(variant_passes[name] for name in exact_names)
    process_gates = {
        "sources_exact": source_hashes_after == source_hashes_before,
        "model_exact": not model_changes,
        "boundary_deterministic_error_at_most_0p001": float(
            boundary_replay["deterministic_max_abs_error"]
        )
        <= 0.001,
        "boundary_logits_error_at_most_0p001": float(
            boundary_replay["logits_max_abs_error"]
        )
        <= 0.001,
        "boundary_stochastic_index_exact": int(
            boundary_replay["stochastic_index_mismatch_count"]
        )
        == 0,
        "privileged_counterfactual_exact": privilege_feature_max_abs_error == 0.0
        and privilege_decoded_max_abs_error == 0.0,
        "all_splits_cover_all_phases": all(
            all(int(count) > 0 for count in split["phase_counts"].values())
            for split in _phase_split_counts(event_split, phases, groups).values()
        ),
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "dataset_context": args.context_length,
        "rolling_context": args.rolling_context,
        "negative_offsets": list(negative_offsets),
        "targets": list(targets),
        "event_split": _phase_split_counts(event_split, phases, groups),
        "boundary_replay": boundary_replay,
        "variant_results": variant_results,
        "variant_passes_capacity_gate": variant_passes,
        "decoded_forward_results": decoded_results,
        "history_diagnosis": {
            "best_exact_variant": best_exact_name,
            "best_exact_test_correlation": best_exact_correlation,
            "best_zero_variant": best_zero_name,
            "best_zero_test_correlation": best_zero_correlation,
            "exact_minus_zero_test_correlation": best_exact_correlation
            - best_zero_correlation,
            "exact_boundary_capacity_pass": exact_capacity_pass,
        },
        "privilege_counterfactual_feature_max_abs_error": privilege_feature_max_abs_error,
        "privilege_counterfactual_decoded_max_abs_error": privilege_decoded_max_abs_error,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "model_tensor_changes": model_changes,
        "model_exact": not model_changes,
        "probe_fits": len(variant_results) * len(ridge_values),
        "test_labels_read_only_after_fixed_split": 1,
        "optimizer_steps": 0,
        "world_updates": 0,
        "reward_updates": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "device": str(device),
        "semantic_precision": "float32",
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
    parser.add_argument("--n632-report", type=Path, required=True)
    parser.add_argument("--n643-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--rolling-context", type=int, default=12)
    parser.add_argument("--negative-offsets", default="4,3,2,1")
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=644)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--ridge-values", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.context_length <= 0 or args.rolling_context <= 0:
        parser.error("context lengths must be positive")
    if args.report.exists():
        parser.error("audit refuses to overwrite an existing report")
    return args


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
