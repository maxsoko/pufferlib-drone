#!/usr/bin/env python3
"""Audit continuous near-plane support in unchanged N604.

The native active-gate-relative pose is an offline label only.  This read-only
audit compares its true within-window approach signal with N604's existing
nonlinear privileged decoder and a privileged timing oracle on N628's exact
group split.  It performs no model update and saves no fitted head.
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
from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from scripts.audit_vq2_event_latent_separability import (
    _fit_ridge_probe,
    _metrics,
    _stratified_group_split,
    _validation_threshold,
)
from scripts.continue_vq2_event_balanced_world_offline import (
    _event_batch,
    _load_event_dataset,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256
from scripts.train_vq2_informed_dreamer import _autocast_context, _burn_in_state


AUDIT_SCHEMA = "vq2_frozen_continuous_plane_progress_support_v1"
GATE_RELATIVE_POSITION = slice(3, 6)


def _correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.std() == 0.0 or prediction.std() == 0.0:
        return 0.0
    return float(np.corrcoef(target, prediction)[0, 1])


def _direct_score_result(
    labels: np.ndarray,
    scores: np.ndarray,
    event_split: np.ndarray,
) -> dict[str, object]:
    samples = labels.shape[1]
    flat_labels = labels.reshape(-1)
    flat_scores = scores.reshape(-1)
    flat_split = np.repeat(event_split, samples)
    validation = flat_split == 1
    test = flat_split == 2
    threshold = _validation_threshold(flat_labels[validation], flat_scores[validation])
    return {
        "validation_threshold": threshold,
        "validation": _metrics(
            flat_labels[validation], flat_scores[validation], threshold=threshold
        ),
        "test": _metrics(flat_labels[test], flat_scores[test], threshold=threshold),
        "test_within_window_all_negative_rank_fraction": float(
            (
                scores[event_split == 2, -1]
                > scores[event_split == 2, :-1].max(1)
            ).mean()
        ),
    }


@torch.no_grad()
def _decode_current_pose(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    *,
    targets: tuple[int, ...],
    feature_context: int,
    amp_dtype: str,
) -> np.ndarray:
    decoded: list[torch.Tensor] = []
    for target in targets:
        with _autocast_context(observation.device, amp_dtype):
            state = _burn_in_state(
                model,
                observation[:, target - feature_context : target],
                action[:, target - feature_context : target],
            )
            current = model.privileged_decoder(state.features)[
                ..., GATE_RELATIVE_POSITION
            ]
        decoded.append(current.float().cpu())
    return torch.stack(decoded, 1).numpy()


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_hashes_before = {
        "checkpoint": _sha256(args.checkpoint),
        "event_dataset": _sha256(args.event_dataset),
        "n629_report": _sha256(args.n629_report),
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
    observation, action, reward, _continuation = _event_batch(
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
    decoded_pose = _decode_current_pose(
        model,
        observation,
        action,
        targets=targets,
        feature_context=args.feature_context,
        amp_dtype=args.amp_dtype,
    )
    true_pose = torch.stack(
        [
            observation[:, target - 1, LEGAL_OBS_SIZE + GATE_RELATIVE_POSITION.start : LEGAL_OBS_SIZE + GATE_RELATIVE_POSITION.stop]
            for target in targets
        ],
        1,
    ).float().cpu().numpy()
    previous_pose = torch.stack(
        [
            observation[:, target - 2, LEGAL_OBS_SIZE + GATE_RELATIVE_POSITION.start : LEGAL_OBS_SIZE + GATE_RELATIVE_POSITION.stop]
            for target in targets
        ],
        1,
    ).float().cpu().numpy()
    pose_delta = true_pose - previous_pose
    selected_action = torch.stack(
        [action[:, target] for target in targets], 1
    ).float().cpu().numpy()
    labels = torch.stack(
        [reward[:, target] > args.event_threshold for target in targets], 1
    ).cpu().numpy()
    if not labels[:, -1].all() or labels[:, :-1].any():
        raise RuntimeError("event labels do not match hard-negative contract")

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
    flat_oracle = np.concatenate((true_pose, pose_delta, selected_action), -1).reshape(
        -1, 10
    )
    flat_labels = labels.reshape(-1)
    flat_split = np.repeat(event_split, labels.shape[1])
    oracle_scores, oracle_fit = _fit_ridge_probe(
        flat_oracle,
        flat_labels,
        flat_split,
        ridge_values=ridge_values,
    )
    oracle_threshold = float(oracle_fit["validation_threshold"])
    oracle_test = flat_split == 2
    oracle_score_by_event = oracle_scores.reshape(labels.shape)

    test_events = event_split == 2
    regression: dict[str, object] = {}
    axes = ("forward", "right", "down")
    for axis, name in enumerate(axes):
        error = decoded_pose[test_events, :, axis] - true_pose[test_events, :, axis]
        regression[name] = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "correlation": _correlation(
                true_pose[test_events, :, axis], decoded_pose[test_events, :, axis]
            ),
            "target_std": float(true_pose[test_events, :, axis].std()),
            "prediction_std": float(decoded_pose[test_events, :, axis].std()),
        }
    true_step = float(np.abs(pose_delta[test_events, :, 0]).mean())
    truth_forward = _direct_score_result(labels, -true_pose[..., 0], event_split)
    decoded_forward = _direct_score_result(labels, -decoded_pose[..., 0], event_split)
    oracle_result = {
        "fit": oracle_fit,
        "test": _metrics(
            flat_labels[oracle_test],
            oracle_scores[oracle_test],
            threshold=oracle_threshold,
        ),
        "test_within_window_all_negative_rank_fraction": float(
            (
                oracle_score_by_event[test_events, -1]
                > oracle_score_by_event[test_events, :-1].max(1)
            ).mean()
        ),
    }
    source_hashes_after = {
        "checkpoint": _sha256(args.checkpoint),
        "event_dataset": _sha256(args.event_dataset),
        "n629_report": _sha256(args.n629_report),
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
        "true_forward_event_support": truth_forward,
        "decoded_forward_event_support": decoded_forward,
        "privileged_pose_rate_action_oracle": oracle_result,
        "decoded_current_pose_test_regression": regression,
        "test_true_forward_mean_abs_step": true_step,
        "test_decoded_forward_mae_in_true_step_units": (
            float(regression["forward"]["mae"]) / true_step
        ),
        "test_true_forward_strictly_decreasing_window_fraction": float(
            (np.diff(true_pose[test_events, :, 0], axis=1) < 0.0).all(1).mean()
        ),
        "test_decoded_forward_strictly_decreasing_window_fraction": float(
            (np.diff(decoded_pose[test_events, :, 0], axis=1) < 0.0).all(1).mean()
        ),
        "model_tensor_changes": model_changes,
        "model_exact": not model_changes,
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
    parser.add_argument("--n629-report", type=Path, required=True)
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
