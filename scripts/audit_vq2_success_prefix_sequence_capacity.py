#!/usr/bin/env python3
"""Disposable recurrent capacity audit on replay-exact successful prefixes."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE, VISUAL_HEIGHT, VISUAL_WIDTH
from scripts.audit_vq2_event_latent_separability import _stratified_group_split
from scripts.audit_vq2_gate1_sequence_capacity import PROBE_SPECS, RecurrentVisualProbe
from scripts.audit_vq2_plane_progress_support import _correlation
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256


AUDIT_SCHEMA = "vq2_training_only_success_prefix_recurrent_capacity_v1"


def _forward_metres(normalized: np.ndarray) -> np.ndarray:
    return 10.0 * np.arctanh(np.clip(normalized, -0.999999, 0.999999))


def _crop_index(event_count: int, crop_starts: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    if event_count <= 0 or not crop_starts or any(start < 0 for start in crop_starts):
        raise ValueError("event count and crop starts must be valid")
    return (
        np.repeat(np.arange(event_count, dtype=np.int64), len(crop_starts)),
        np.tile(np.asarray(crop_starts, dtype=np.int64), event_count),
    )


def _crop_batch(
    mask: np.ndarray,
    tail: np.ndarray,
    event: np.ndarray,
    start: np.ndarray,
    *,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event = np.asarray(event, dtype=np.int64)
    start = np.asarray(start, dtype=np.int64)
    if event.ndim != 1 or start.shape != event.shape:
        raise ValueError("crop event/start vectors must align")
    offsets = np.arange(sequence_length, dtype=np.int64)
    time_index = start[:, None] + offsets[None, :]
    if np.any(event < 0) or np.any(event >= mask.shape[0]) or np.any(time_index >= mask.shape[1]):
        raise ValueError("crop falls outside source histories")
    selected_mask = np.asarray(mask[event[:, None], time_index], dtype=np.uint8)
    full_tail = np.asarray(tail[event[:, None], time_index], dtype=np.float32)
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    selected_tail = full_tail[..., :legal_tail_size]
    target = _forward_metres(full_tail[..., legal_tail_size + 3]).astype(np.float32)
    return selected_mask, selected_tail, target


def _crop_targets(
    tail: np.ndarray,
    event: np.ndarray,
    start: np.ndarray,
    *,
    sequence_length: int,
) -> np.ndarray:
    event = np.asarray(event, dtype=np.int64)
    start = np.asarray(start, dtype=np.int64)
    offsets = np.arange(sequence_length, dtype=np.int64)
    time_index = start[:, None] + offsets[None, :]
    full_tail = np.asarray(tail[event[:, None], time_index], dtype=np.float32)
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    return _forward_metres(full_tail[..., legal_tail_size + 3]).astype(np.float32)


def _metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    evaluation_indices: np.ndarray,
    near_plane_m: float,
    minimum_pair_delta_m: float,
) -> dict[str, float | int]:
    selected_target = target[:, evaluation_indices]
    selected_prediction = prediction[:, evaluation_indices]
    error = selected_prediction - selected_target
    near = np.abs(selected_target) <= near_plane_m
    true_delta = np.diff(selected_target, axis=1)
    predicted_delta = np.diff(selected_prediction, axis=1)
    directional = np.abs(true_delta) >= minimum_pair_delta_m
    direction_accuracy = float(
        (np.sign(true_delta[directional]) == np.sign(predicted_delta[directional])).mean()
    ) if directional.any() else 0.0
    return {
        "samples": int(selected_target.size),
        "correlation": _correlation(selected_target, selected_prediction),
        "mae_m": float(np.abs(error).mean()),
        "rmse_m": float(np.sqrt(np.square(error).mean())),
        "near_plane_samples": int(near.sum()),
        "near_plane_correlation": _correlation(
            selected_target[near], selected_prediction[near]
        ),
        "near_plane_mae_m": float(np.abs(error[near]).mean()),
        "direction_pairs": int(directional.sum()),
        "direction_accuracy": direction_accuracy,
        "strictly_decreasing_crop_fraction": float(
            (np.diff(selected_prediction, axis=1) < 0.0).all(1).mean()
        ),
        "final_is_minimum_crop_fraction": float(
            (selected_prediction[:, -1] < selected_prediction[:, :-1].min(1)).mean()
        ),
    }


def _selection_key(result: dict[str, float | int], seed: int, name: str) -> tuple:
    return (
        float(result["correlation"]),
        -float(result["near_plane_mae_m"]),
        float(result["direction_accuracy"]),
        -float(result["mae_m"]),
        -seed,
        name,
    )


def _predict(
    probe: RecurrentVisualProbe,
    mask: np.ndarray,
    tail: np.ndarray,
    crop_event: np.ndarray,
    crop_start: np.ndarray,
    indices: np.ndarray,
    *,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probe.eval()
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            chosen = indices[offset : offset + batch_size]
            batch_mask, batch_tail, batch_target = _crop_batch(
                mask,
                tail,
                crop_event[chosen],
                crop_start[chosen],
                sequence_length=sequence_length,
            )
            mask_tensor = torch.from_numpy(batch_mask).to(device=device, dtype=torch.float32)
            mask_tensor = mask_tensor.reshape(
                len(chosen), sequence_length, VISUAL_HEIGHT, VISUAL_WIDTH
            ) / 255.0
            tail_tensor = torch.from_numpy(batch_tail).to(device)
            prediction = probe(mask_tensor, tail_tensor)
            targets.append(batch_target)
            predictions.append(prediction.float().cpu().numpy() * target_std + target_mean)
    return np.concatenate(targets), np.concatenate(predictions)


def _train_candidate(
    *,
    name: str,
    seed: int,
    mask: np.ndarray,
    tail: np.ndarray,
    crop_event: np.ndarray,
    crop_start: np.ndarray,
    crop_split: np.ndarray,
    evaluation_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor], float, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    probe = RecurrentVisualProbe(
        legal_tail_size=LEGAL_OBS_SIZE - MASK_SIZE,
        hidden_size=args.hidden_size,
        spec=PROBE_SPECS[name],
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train = np.flatnonzero(crop_split == 0)
    validation = np.flatnonzero(crop_split == 1)
    train_target = _crop_targets(
        tail,
        crop_event[train],
        crop_start[train],
        sequence_length=args.sequence_length,
    )[:, args.burn_in :]
    target_mean = float(train_target.mean())
    target_std = float(train_target.std())
    if target_std <= 0.0:
        raise RuntimeError("training target is constant")
    del train_target
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, float | int] | None = None
    best_step = 0
    final_loss = float("nan")
    maximum_gradient_norm = 0.0
    for update in range(1, args.steps + 1):
        chosen = rng.choice(train, args.batch_size, replace=True)
        batch_mask, batch_tail, batch_target = _crop_batch(
            mask,
            tail,
            crop_event[chosen],
            crop_start[chosen],
            sequence_length=args.sequence_length,
        )
        mask_tensor = torch.from_numpy(batch_mask).to(device=device, dtype=torch.float32)
        mask_tensor = mask_tensor.reshape(
            len(chosen), args.sequence_length, VISUAL_HEIGHT, VISUAL_WIDTH
        ) / 255.0
        tail_tensor = torch.from_numpy(batch_tail).to(device)
        target_tensor = torch.from_numpy((batch_target - target_mean) / target_std).to(device)
        probe.train()
        prediction = probe(mask_tensor, tail_tensor)[:, args.burn_in :]
        target_supervised = target_tensor[:, args.burn_in :]
        regression_loss = F.mse_loss(prediction, target_supervised)
        delta_loss = F.mse_loss(
            torch.diff(prediction, dim=1), torch.diff(target_supervised, dim=1)
        )
        loss = regression_loss + args.delta_weight * delta_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(probe.parameters(), args.gradient_clip).detach().cpu()
        )
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if update % args.validation_interval and update != args.steps:
            continue
        validation_target, validation_prediction = _predict(
            probe,
            mask,
            tail,
            crop_event,
            crop_start,
            validation,
            sequence_length=args.sequence_length,
            batch_size=args.evaluation_batch_size,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        result = _metrics(
            validation_target,
            validation_prediction,
            evaluation_indices=evaluation_indices,
            near_plane_m=args.near_plane_m,
            minimum_pair_delta_m=args.minimum_pair_delta_m,
        )
        if best_validation is None or _selection_key(result, seed, name) > _selection_key(
            best_validation, seed, name
        ):
            best_validation = result
            best_step = update
            best_state = {
                key: value.detach().cpu().clone() for key, value in probe.state_dict().items()
            }
    assert best_state is not None and best_validation is not None
    return (
        {
            "name": name,
            "seed": seed,
            "parameters": sum(parameter.numel() for parameter in probe.parameters()),
            "best_step": best_step,
            "best_validation": best_validation,
            "final_training_loss": final_loss,
            "maximum_preclip_gradient_norm": maximum_gradient_norm,
            "optimizer_steps": args.steps,
        },
        best_state,
        target_mean,
        target_std,
    )


def _passes_capacity(result: dict[str, float | int], args: argparse.Namespace) -> bool:
    return bool(
        float(result["correlation"]) >= args.minimum_correlation
        and float(result["mae_m"]) <= args.maximum_mae_m
        and float(result["near_plane_mae_m"]) <= args.maximum_near_plane_mae_m
        and float(result["direction_accuracy"]) >= args.minimum_direction_accuracy
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    start_time = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before = {
        "dataset": _sha256(args.dataset),
        "source_report": _sha256(args.source_report),
    }
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source_report.get("output_dataset_sha256") != source_before["dataset"]:
        raise RuntimeError("source report dataset hash mismatch")
    with np.load(args.dataset) as loaded:
        mask = loaded["mask"][:, :-1].copy()
        tail = loaded["tail"][:, :-1].astype(np.float32)
        vector_step = loaded["vector_step"].astype(np.int64)
        phase_after = np.rint(loaded["phase_after_event"].astype(np.float64) * 6.0).astype(np.int64)
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    phase_before = np.rint(tail[..., legal_tail_size + 32] * 6.0).astype(np.int64)
    if not np.all(phase_before == 0) or not np.all(phase_after == 1):
        raise RuntimeError("successful-prefix phase contract mismatch")
    crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    if not crop_starts or len(set(crop_starts)) != len(crop_starts):
        raise ValueError("crop starts must be nonempty and unique")
    if min(crop_starts) < 0 or max(crop_starts) + args.sequence_length > mask.shape[1]:
        raise ValueError("crop starts exceed pre-event history")
    event_split = _stratified_group_split(
        vector_step,
        phase_after,
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    crop_event, crop_start = _crop_index(len(vector_step), crop_starts)
    crop_split = event_split[crop_event]
    evaluation_indices = np.arange(
        args.burn_in - 1, args.sequence_length, args.evaluation_stride, dtype=np.int64
    )
    if len(evaluation_indices) < 2:
        raise ValueError("evaluation grid must contain at least two points")
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("probe seeds must be nonempty and unique")
    candidates: list[dict[str, object]] = []
    states: dict[tuple[str, int], tuple[dict[str, torch.Tensor], float, float]] = {}
    for name in PROBE_SPECS:
        for seed in seeds:
            summary, state, target_mean, target_std = _train_candidate(
                name=name,
                seed=seed,
                mask=mask,
                tail=tail,
                crop_event=crop_event,
                crop_start=crop_start,
                crop_split=crop_split,
                evaluation_indices=evaluation_indices,
                args=args,
                device=device,
            )
            candidates.append(summary)
            states[(name, seed)] = (state, target_mean, target_std)
    selected = max(
        candidates,
        key=lambda row: _selection_key(
            row["best_validation"], int(row["seed"]), str(row["name"])
        ),
    )
    selected_name = str(selected["name"])
    selected_seed = int(selected["seed"])
    selected_state, target_mean, target_std = states[(selected_name, selected_seed)]
    probe = RecurrentVisualProbe(
        legal_tail_size=legal_tail_size,
        hidden_size=args.hidden_size,
        spec=PROBE_SPECS[selected_name],
    ).to(device)
    probe.load_state_dict(selected_state)
    test = np.flatnonzero(crop_split == 2)
    test_target, test_prediction = _predict(
        probe,
        mask,
        tail,
        crop_event,
        crop_start,
        test,
        sequence_length=args.sequence_length,
        batch_size=args.evaluation_batch_size,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_result = _metrics(
        test_target,
        test_prediction,
        evaluation_indices=evaluation_indices,
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )
    source_after = {
        "dataset": _sha256(args.dataset),
        "source_report": _sha256(args.source_report),
    }
    split_groups = {
        name: int(np.unique(vector_step[event_split == value]).size)
        for name, value in (("train", 0), ("validation", 1), ("test", 2))
    }
    process_gates = {
        "sources_exact": source_after == source_before,
        "source_report_matches_dataset": source_report.get("output_dataset_sha256") == source_after["dataset"],
        "whole_event_group_split": sum(split_groups.values()) == int(np.unique(vector_step).size),
        "all_preevent_phase_zero": bool(np.all(phase_before == 0)),
        "all_events_enter_phase_one": bool(np.all(phase_after == 1)),
        "multiple_crop_offsets": len(crop_starts) > 1,
        "selected_on_validation_before_test": True,
        "legal_inputs_only": True,
        "probe_not_saved": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "sources_unchanged": source_after == source_before,
        "events": int(len(vector_step)),
        "event_groups": int(np.unique(vector_step).size),
        "crop_starts": list(crop_starts),
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "evaluation_indices": evaluation_indices.tolist(),
        "crops": int(len(crop_event)),
        "split_event_counts": {
            name: int((event_split == value).sum())
            for name, value in (("train", 0), ("validation", 1), ("test", 2))
        },
        "split_group_counts": split_groups,
        "probe_candidates": candidates,
        "selected_probe": selected_name,
        "selected_seed": selected_seed,
        "selected_best_step": int(selected["best_step"]),
        "selected_validation": selected["best_validation"],
        "selected_test": test_result,
        "selected_passes_capacity_gate": _passes_capacity(test_result, args),
        "capacity_gates": {
            "minimum_correlation": args.minimum_correlation,
            "maximum_mae_m": args.maximum_mae_m,
            "maximum_near_plane_mae_m": args.maximum_near_plane_mae_m,
            "minimum_direction_accuracy": args.minimum_direction_accuracy,
        },
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "probe_optimizer_steps": len(candidates) * args.steps,
        "probe_weights_saved": 0,
        "model_loaded": 0,
        "model_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "device": str(device),
        "wall_seconds": time.perf_counter() - start_time,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--evaluation-stride", type=int, default=16)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=668)
    parser.add_argument("--seeds", default="669,670")
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--minimum-correlation", type=float, default=0.80)
    parser.add_argument("--maximum-mae-m", type=float, default=0.50)
    parser.add_argument("--maximum-near-plane-mae-m", type=float, default=0.25)
    parser.add_argument("--minimum-direction-accuracy", type=float, default=0.75)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for name in (
        "sequence_length", "burn_in", "evaluation_stride", "hidden_size",
        "steps", "validation_interval", "batch_size", "evaluation_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.burn_in >= args.sequence_length:
        parser.error("burn-in must be shorter than sequence length")
    if args.report.exists():
        parser.error("audit refuses to overwrite an existing report")
    return args


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))
