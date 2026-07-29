#!/usr/bin/env python3
"""Disposable recurrent legal-visual capacity probe on dense Gate-1 replay."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE, VISUAL_HEIGHT, VISUAL_WIDTH
from scripts.audit_vq2_broad_visual_association import (
    _load_replay_arrays,
    _metrics,
    _pair_indices,
    _replay_hashes,
    _selection_key,
)
from scripts.audit_vq2_dense_replay_coverage import (
    _episode_groups,
    _parse_phase_values,
    _stratified_episode_group_split,
    _temporal_candidates,
)


AUDIT_SCHEMA = "vq2_training_only_gate1_recurrent_sequence_capacity_v1"


@dataclass(frozen=True)
class ProbeSpec:
    use_mask: bool
    use_tail: bool


PROBE_SPECS = {
    "tail_sequence": ProbeSpec(False, True),
    "mask_sequence": ProbeSpec(True, False),
    "mask_tail_sequence": ProbeSpec(True, True),
}


class RecurrentVisualProbe(nn.Module):
    def __init__(self, *, legal_tail_size: int, hidden_size: int, spec: ProbeSpec) -> None:
        super().__init__()
        self.spec = spec
        if spec.use_mask:
            self.visual = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, 64),
                nn.SiLU(),
            )
        else:
            self.visual = None
        if spec.use_tail:
            self.tail = nn.Sequential(nn.Linear(legal_tail_size, 32), nn.SiLU())
        else:
            self.tail = None
        input_size = (64 if spec.use_mask else 0) + (32 if spec.use_tail else 0)
        if input_size <= 0:
            raise ValueError("probe specification selects no legal input")
        self.recurrent = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def forward(self, mask: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 4 or tail.ndim != 3 or mask.shape[:2] != tail.shape[:2]:
            raise ValueError("mask/tail must align as [batch,time,...]")
        batch, time = mask.shape[:2]
        parts: list[torch.Tensor] = []
        if self.spec.use_mask:
            assert self.visual is not None
            visual = self.visual(mask.reshape(batch * time, 1, VISUAL_HEIGHT, VISUAL_WIDTH))
            parts.append(visual.reshape(batch, time, -1))
        if self.spec.use_tail:
            assert self.tail is not None
            parts.append(self.tail(tail))
        encoded = torch.cat(parts, -1)
        recurrent, _state = self.recurrent(encoded)
        return self.head(recurrent).squeeze(-1)


def _forward_metres(normalized: np.ndarray) -> np.ndarray:
    return 10.0 * np.arctanh(np.clip(normalized, -0.999999, 0.999999))


def _sequence_candidates(
    arrays: dict[str, np.memmap],
    order: np.ndarray,
    *,
    sequence_length: int,
    sample_stride: int,
    phase_values: tuple[int, ...],
    target_min_m: float,
    target_max_m: float,
) -> dict[str, np.ndarray]:
    valid = np.asarray(arrays["valid"][order], dtype=bool)
    continuation = np.asarray(arrays["continuation"][order], dtype=bool)
    groups, _complete = _episode_groups(
        valid,
        continuation,
        starts_at_known_boundary=(
            len(order) < arrays["valid"].shape[0] and int(order[0]) == 0
        ),
    )
    candidates = _temporal_candidates(
        groups, depth=sequence_length, stride=sample_stride
    )
    step = candidates[:, 0]
    agent = candidates[:, 1]
    physical = order[step]
    tail = np.asarray(arrays["tail"][physical, agent], dtype=np.float32)
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    target_m = _forward_metres(tail[:, legal_tail_size + 3])
    phase = np.rint(tail[:, legal_tail_size + 32] * 6.0).astype(np.int64)
    keep = (
        np.isin(phase, np.asarray(phase_values, dtype=np.int64))
        & (target_m >= target_min_m)
        & (target_m <= target_max_m)
    )
    step = step[keep]
    agent = agent[keep]
    return {
        "step": step,
        "agent": agent,
        "group": groups[step, agent],
        "phase": phase[keep],
        "target_m": target_m[keep].astype(np.float32),
    }


def _gather_sequences(
    arrays: dict[str, np.memmap],
    order: np.ndarray,
    step: np.ndarray,
    agent: np.ndarray,
    *,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = np.asarray(step, dtype=np.int64)
    agent = np.asarray(agent, dtype=np.int64)
    if step.ndim != 1 or agent.shape != step.shape or np.any(step < sequence_length - 1):
        raise ValueError("sequence endpoints are invalid")
    offsets = np.arange(sequence_length - 1, -1, -1, dtype=np.int64)
    physical = order[step[:, None] - offsets[None, :]]
    agent_grid = agent[:, None]
    mask = np.asarray(arrays["mask"][physical, agent_grid], dtype=np.uint8)
    full_tail = np.asarray(arrays["tail"][physical, agent_grid], dtype=np.float32)
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    tail = full_tail[..., :legal_tail_size]
    target = _forward_metres(full_tail[..., legal_tail_size + 3]).astype(np.float32)
    return mask, tail, target


def _gather_target_sequences(
    arrays: dict[str, np.memmap],
    order: np.ndarray,
    step: np.ndarray,
    agent: np.ndarray,
    *,
    sequence_length: int,
) -> np.ndarray:
    step = np.asarray(step, dtype=np.int64)
    agent = np.asarray(agent, dtype=np.int64)
    if step.ndim != 1 or agent.shape != step.shape or np.any(step < sequence_length - 1):
        raise ValueError("sequence endpoints are invalid")
    offsets = np.arange(sequence_length - 1, -1, -1, dtype=np.int64)
    physical = order[step[:, None] - offsets[None, :]]
    full_tail = np.asarray(arrays["tail"][physical, agent[:, None]], dtype=np.float32)
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    return _forward_metres(full_tail[..., legal_tail_size + 3]).astype(np.float32)


def _trajectory_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "correlation": float(np.corrcoef(target.reshape(-1), prediction.reshape(-1))[0, 1]),
        "mae_m": float(np.abs(error).mean()),
        "rmse_m": float(np.sqrt(np.square(error).mean())),
        "delta_mae_m": float(np.abs(np.diff(error, axis=1)).mean()),
    }


def _predict_subset(
    probe: RecurrentVisualProbe,
    arrays: dict[str, np.memmap],
    order: np.ndarray,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    probe.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            mask, tail, target = _gather_sequences(
                arrays,
                order,
                data["step"][chosen],
                data["agent"][chosen],
                sequence_length=sequence_length,
            )
            mask_tensor = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
            mask_tensor = mask_tensor.reshape(
                len(chosen), sequence_length, VISUAL_HEIGHT, VISUAL_WIDTH
            ) / 255.0
            tail_tensor = torch.from_numpy(tail).to(device)
            prediction = probe(mask_tensor, tail_tensor)
            predictions.append((prediction.float().cpu().numpy() * target_std + target_mean))
            targets.append(target)
    return np.concatenate(targets), np.concatenate(predictions)


def _endpoint_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    phase: np.ndarray,
    group: np.ndarray,
    step: np.ndarray,
    *,
    sample_stride: int,
    near_plane_m: float,
    minimum_pair_delta_m: float,
) -> dict[str, object]:
    pair_left, pair_right = _pair_indices(group, step, sample_stride=sample_stride)
    chosen = np.ones(len(target), dtype=bool)
    return _metrics(
        target,
        prediction,
        phase,
        chosen,
        pair_left,
        pair_right,
        near_plane_m=near_plane_m,
        minimum_pair_delta_m=minimum_pair_delta_m,
    )


def _train_candidate(
    *,
    name: str,
    spec: ProbeSpec,
    seed: int,
    arrays: dict[str, np.memmap],
    order: np.ndarray,
    data: dict[str, np.ndarray],
    split: np.ndarray,
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
        spec=spec,
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train = np.flatnonzero(split == 0)
    near_train = np.flatnonzero((split == 0) & (np.abs(data["target_m"]) <= args.near_plane_m))
    validation = np.flatnonzero(split == 1)
    train_target = _gather_target_sequences(
        arrays,
        order,
        data["step"][train],
        data["agent"][train],
        sequence_length=args.sequence_length,
    )
    target_mean = float(train_target.mean())
    target_std = float(train_target.std())
    del train_target
    if target_std <= 0.0:
        raise RuntimeError("training target is constant")
    rng = np.random.default_rng(seed)
    near_count = int(round(args.batch_size * args.near_sample_fraction))
    broad_count = args.batch_size - near_count
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, object] | None = None
    best_step = 0
    final_loss = float("nan")
    maximum_gradient_norm = 0.0
    for update in range(1, args.steps + 1):
        chosen = np.concatenate(
            (
                rng.choice(train, broad_count, replace=True),
                rng.choice(near_train, near_count, replace=True),
            )
        )
        rng.shuffle(chosen)
        mask, tail, target = _gather_sequences(
            arrays,
            order,
            data["step"][chosen],
            data["agent"][chosen],
            sequence_length=args.sequence_length,
        )
        mask_tensor = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
        mask_tensor = mask_tensor.reshape(
            len(chosen), args.sequence_length, VISUAL_HEIGHT, VISUAL_WIDTH
        ) / 255.0
        tail_tensor = torch.from_numpy(tail).to(device)
        target_tensor = torch.from_numpy((target - target_mean) / target_std).to(device)
        probe.train()
        prediction = probe(mask_tensor, tail_tensor)
        regression_loss = F.mse_loss(prediction, target_tensor)
        delta_loss = F.mse_loss(torch.diff(prediction, dim=1), torch.diff(target_tensor, dim=1))
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
        validation_target, validation_prediction = _predict_subset(
            probe,
            arrays,
            order,
            data,
            validation,
            sequence_length=args.sequence_length,
            batch_size=args.evaluation_batch_size,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        validation_result = _endpoint_metrics(
            validation_target[:, -1],
            validation_prediction[:, -1],
            data["phase"][validation],
            data["group"][validation],
            data["step"][validation],
            sample_stride=args.sample_stride,
            near_plane_m=args.near_plane_m,
            minimum_pair_delta_m=args.minimum_pair_delta_m,
        )
        if best_validation is None or _selection_key(
            validation_result, seed, name
        ) > _selection_key(best_validation, seed, name):
            best_validation = validation_result
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


def _passes_capacity(result: dict[str, object], args: argparse.Namespace) -> bool:
    return bool(
        float(result["correlation"]) >= args.minimum_correlation
        and float(result["mae_m"]) <= args.maximum_mae_m
        and float(result["minimum_phase_correlation"]) >= args.minimum_phase_correlation
        and float(result["near_plane_mae_m"]) <= args.maximum_near_plane_mae_m
        and float(result["direction_accuracy"]) >= args.minimum_direction_accuracy
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    audit_start = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before = _replay_hashes(args.replay_dir, args.n665_report)
    source_before["n665_report"] = source_before.pop("n648_report")
    metadata, arrays, order = _load_replay_arrays(args.replay_dir)
    phase_values = _parse_phase_values(args.phase_values)
    data = _sequence_candidates(
        arrays,
        order,
        sequence_length=args.sequence_length,
        sample_stride=args.sample_stride,
        phase_values=phase_values,
        target_min_m=args.target_min_m,
        target_max_m=args.target_max_m,
    )
    split = _stratified_episode_group_split(
        data["group"],
        data["phase"],
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("probe seeds must be nonempty and unique")
    candidates: list[dict[str, object]] = []
    states: dict[tuple[str, int], tuple[dict[str, torch.Tensor], float, float]] = {}
    for name, spec in PROBE_SPECS.items():
        for seed in seeds:
            summary, state, target_mean, target_std = _train_candidate(
                name=name,
                spec=spec,
                seed=seed,
                arrays=arrays,
                order=order,
                data=data,
                split=split,
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
    selected_probe = RecurrentVisualProbe(
        legal_tail_size=LEGAL_OBS_SIZE - MASK_SIZE,
        hidden_size=args.hidden_size,
        spec=PROBE_SPECS[selected_name],
    ).to(device)
    selected_probe.load_state_dict(selected_state)
    test = np.flatnonzero(split == 2)
    test_target, test_prediction = _predict_subset(
        selected_probe,
        arrays,
        order,
        data,
        test,
        sequence_length=args.sequence_length,
        batch_size=args.evaluation_batch_size,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_endpoint = _endpoint_metrics(
        test_target[:, -1],
        test_prediction[:, -1],
        data["phase"][test],
        data["group"][test],
        data["step"][test],
        sample_stride=args.sample_stride,
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )
    source_after = _replay_hashes(args.replay_dir, args.n665_report)
    source_after["n665_report"] = source_after.pop("n648_report")
    split_groups = {
        name: int(np.unique(data["group"][split == value]).size)
        for name, value in (("train", 0), ("validation", 1), ("test", 2))
    }
    process_gates = {
        "sources_exact": source_after == source_before,
        "whole_episode_split": sum(split_groups.values()) == int(np.unique(data["group"]).size),
        "all_required_phases_present": set(phase_values).issubset(
            set(int(value) for value in data["phase"])
        ),
        "selected_on_validation_before_test": True,
        "legal_inputs_only": True,
        "probe_not_saved": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_after,
        "sources_unchanged": source_after == source_before,
        "replay_metadata": metadata,
        "sequence_length": args.sequence_length,
        "sample_stride": args.sample_stride,
        "samples": int(len(data["step"])),
        "groups": int(np.unique(data["group"]).size),
        "phase_values": list(phase_values),
        "split_sample_counts": {
            name: int((split == value).sum())
            for name, value in (("train", 0), ("validation", 1), ("test", 2))
        },
        "split_group_counts": split_groups,
        "probe_candidates": candidates,
        "selection_rule": "validation endpoint correlation, near-plane MAE, direction, MAE, lower seed, name",
        "selected_probe": selected_name,
        "selected_seed": selected_seed,
        "selected_best_step": int(selected["best_step"]),
        "selected_validation": selected["best_validation"],
        "selected_test_endpoint": test_endpoint,
        "selected_test_trajectory": _trajectory_metrics(test_target, test_prediction),
        "selected_passes_capacity_gate": _passes_capacity(test_endpoint, args),
        "capacity_gates": {
            "minimum_correlation": args.minimum_correlation,
            "maximum_mae_m": args.maximum_mae_m,
            "minimum_phase_correlation": args.minimum_phase_correlation,
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
        "wall_seconds": time.perf_counter() - audit_start,
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
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n665-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--phase-values", default="0")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--target-min-m", type=float, default=-1.0)
    parser.add_argument("--target-max-m", type=float, default=7.0)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=664)
    parser.add_argument("--seeds", default="666,667")
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--near-sample-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--minimum-correlation", type=float, default=0.80)
    parser.add_argument("--maximum-mae-m", type=float, default=0.50)
    parser.add_argument("--minimum-phase-correlation", type=float, default=0.65)
    parser.add_argument("--maximum-near-plane-mae-m", type=float, default=0.25)
    parser.add_argument("--minimum-direction-accuracy", type=float, default=0.75)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for name in (
        "sequence_length", "sample_stride", "hidden_size", "steps",
        "validation_interval", "batch_size", "evaluation_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.near_sample_fraction <= 1.0:
        parser.error("--near-sample-fraction must lie in [0,1]")
    try:
        _parse_phase_values(args.phase_values)
    except ValueError as error:
        parser.error(str(error))
    if args.report.exists():
        parser.error("audit refuses to overwrite an existing report")
    return args


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
