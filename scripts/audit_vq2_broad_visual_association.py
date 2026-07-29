#!/usr/bin/env python3
"""Disposable nonlinear visual-association probe on broad VQ2 replay."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE, MASK_SIZE, VISUAL_HEIGHT, VISUAL_WIDTH
from scripts.audit_vq2_dense_replay_coverage import (
    _chronological_indices,
    _episode_groups,
    _parse_phase_values,
    _stratified_episode_group_split,
    _temporal_candidates,
)
from scripts.audit_vq2_nonlinear_visual_association import (
    ProbeSpec,
    VisualAssociationProbe,
)
from scripts.audit_vq2_plane_progress_support import _correlation
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256


AUDIT_SCHEMA = "vq2_training_only_broad_visual_association_capacity_v2"
PROBE_SPECS = {
    "tail_only": ProbeSpec(False, True, False),
    "mask_only": ProbeSpec(True, False, False),
    "mask_legal_tail": ProbeSpec(True, True, False),
}


def _replay_hashes(replay_dir: Path, n648_report: Path) -> dict[str, str]:
    result = {"n648_report": _sha256(n648_report)}
    for path in sorted(replay_dir.iterdir()):
        if path.is_file():
            result[f"replay/{path.name}"] = _sha256(path)
    return result


def _load_replay_arrays(
    replay_dir: Path,
) -> tuple[dict[str, object], dict[str, np.memmap], np.ndarray]:
    metadata = json.loads((replay_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "vq2_quantized_sequence_replay_v1":
        raise RuntimeError("replay schema mismatch")
    capacity = int(metadata["capacity"])
    agents = int(metadata["agents"])
    size = int(metadata["size"])
    position = int(metadata["position"])
    shapes = {
        "mask": (capacity, agents, MASK_SIZE),
        "tail": (capacity, agents, ENV_OBS_SIZE - MASK_SIZE),
        "continuation": (capacity, agents),
        "valid": (capacity, agents),
    }
    dtypes = {
        "mask": np.uint8,
        "tail": np.float16,
        "continuation": np.uint8,
        "valid": np.uint8,
    }
    arrays = {
        name: np.memmap(
            replay_dir / f"{name}.dat",
            dtype=dtypes[name],
            mode="r",
            shape=shape,
        )
        for name, shape in shapes.items()
    }
    order = _chronological_indices(capacity, size, position)
    return metadata, arrays, order


def _candidate_data(
    arrays: dict[str, np.memmap],
    order: np.ndarray,
    *,
    temporal_depth: int,
    sample_stride: int,
    target_min_m: float,
    target_max_m: float,
    phase_values: tuple[int, ...] = tuple(range(6)),
) -> dict[str, np.ndarray]:
    valid = np.asarray(arrays["valid"][order], dtype=bool)
    continuation = np.asarray(arrays["continuation"][order], dtype=bool)
    groups, _complete = _episode_groups(
        valid,
        continuation,
        starts_at_known_boundary=len(order) < arrays["valid"].shape[0] and int(order[0]) == 0,
    )
    candidates = _temporal_candidates(
        groups, depth=temporal_depth, stride=sample_stride
    )
    step = candidates[:, 0]
    agent = candidates[:, 1]
    physical = order[step]
    tail = np.asarray(arrays["tail"][physical, agent], dtype=np.float32)
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    normalized_forward = tail[:, legal_tail_size + 3]
    forward_m = 10.0 * np.arctanh(np.clip(normalized_forward, -0.999999, 0.999999))
    phase = np.rint(tail[:, legal_tail_size + 32] * 6.0).astype(np.int64)
    keep = (
        np.isin(phase, np.asarray(phase_values, dtype=np.int64))
        & (forward_m >= target_min_m)
        & (forward_m <= target_max_m)
    )
    step = step[keep]
    agent = agent[keep]
    tail = tail[keep, :legal_tail_size]
    forward_m = forward_m[keep]
    phase = phase[keep]
    group = groups[step, agent]
    masks = np.stack(
        [
            np.asarray(arrays["mask"][order[step - lag], agent], dtype=np.uint8)
            for lag in range(temporal_depth)
        ],
        1,
    ).reshape(-1, temporal_depth, VISUAL_HEIGHT, VISUAL_WIDTH)
    return {
        "mask": masks,
        "tail": tail.astype(np.float32),
        "target_m": forward_m.astype(np.float32),
        "phase": phase,
        "group": group,
        "step": step,
    }


def _pair_indices(
    group: np.ndarray,
    step: np.ndarray,
    *,
    sample_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    adjacent = (group[1:] == group[:-1]) & ((step[1:] - step[:-1]) == sample_stride)
    left = np.nonzero(adjacent)[0]
    return left, left + 1


def _metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    phase: np.ndarray,
    chosen: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    *,
    near_plane_m: float,
    minimum_pair_delta_m: float,
) -> dict[str, object]:
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    chosen = np.asarray(chosen, dtype=bool)
    error = prediction[chosen] - target[chosen]
    near = chosen & (np.abs(target) <= near_plane_m)
    near_error = prediction[near] - target[near]
    per_phase: dict[str, object] = {}
    for value in sorted(np.unique(phase)):
        selected = chosen & (phase == value)
        phase_error = prediction[selected] - target[selected]
        per_phase[str(int(value))] = {
            "samples": int(selected.sum()),
            "correlation": _correlation(target[selected], prediction[selected]),
            "mae_m": float(np.abs(phase_error).mean()),
        }
    pair_chosen = chosen[pair_left] & chosen[pair_right]
    true_delta = target[pair_right] - target[pair_left]
    predicted_delta = prediction[pair_right] - prediction[pair_left]
    pair_chosen &= np.abs(true_delta) >= minimum_pair_delta_m
    direction_accuracy = float(
        (np.sign(true_delta[pair_chosen]) == np.sign(predicted_delta[pair_chosen])).mean()
    ) if pair_chosen.any() else 0.0
    return {
        "samples": int(chosen.sum()),
        "correlation": _correlation(target[chosen], prediction[chosen]),
        "mae_m": float(np.abs(error).mean()),
        "rmse_m": float(np.sqrt(np.square(error).mean())),
        "near_plane_samples": int(near.sum()),
        "near_plane_correlation": _correlation(target[near], prediction[near]),
        "near_plane_mae_m": float(np.abs(near_error).mean()),
        "direction_pairs": int(pair_chosen.sum()),
        "direction_accuracy": direction_accuracy,
        "per_phase": per_phase,
        "minimum_phase_correlation": min(
            float(result["correlation"]) for result in per_phase.values()
        ),
    }


def _selection_key(result: dict[str, object], seed: int, name: str) -> tuple:
    return (
        float(result["correlation"]),
        -float(result["near_plane_mae_m"]),
        float(result["direction_accuracy"]),
        -float(result["mae_m"]),
        -seed,
        name,
    )


def _predict(
    probe: VisualAssociationProbe,
    mask: torch.Tensor,
    tail: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    probe.eval()
    output: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, mask.shape[0], batch_size):
            stop = min(start + batch_size, mask.shape[0])
            batch_mask = mask[start:stop].to(device=device, dtype=torch.float32) / 255.0
            batch_tail = tail[start:stop].to(device)
            boundary = torch.empty((stop - start, 0), device=device)
            output.append(probe(batch_mask, batch_tail, boundary).float().cpu())
    normalized = torch.cat(output).numpy()
    return normalized * target_std + target_mean


def _train_candidate(
    *,
    name: str,
    spec: ProbeSpec,
    seed: int,
    mask: torch.Tensor,
    tail: torch.Tensor,
    target: np.ndarray,
    phase: np.ndarray,
    split: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor], float, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    probe = VisualAssociationProbe(
        temporal_depth=args.temporal_depth,
        legal_tail_size=tail.shape[-1],
        boundary_size=0,
        spec=spec,
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_indices = np.nonzero(split == 0)[0]
    near_train_indices = np.nonzero((split == 0) & (np.abs(target) <= args.near_plane_m))[0]
    validation = split == 1
    target_mean = float(target[train_indices].mean())
    target_std = float(target[train_indices].std())
    normalized_target = (target - target_mean) / target_std
    rng = np.random.default_rng(seed)
    near_batch = int(round(args.batch_size * args.near_sample_fraction))
    broad_batch = args.batch_size - near_batch
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, object] | None = None
    best_step = 0
    final_loss = float("nan")
    maximum_gradient_norm = 0.0
    for update in range(1, args.steps + 1):
        indices = np.concatenate(
            (
                rng.choice(train_indices, broad_batch, replace=True),
                rng.choice(near_train_indices, near_batch, replace=True),
            )
        )
        rng.shuffle(indices)
        batch_mask = mask[indices].to(device=device, dtype=torch.float32) / 255.0
        batch_tail = tail[indices].to(device)
        batch_target = torch.from_numpy(normalized_target[indices].astype(np.float32)).to(device)
        boundary = torch.empty((len(indices), 0), device=device)
        probe.train()
        prediction = probe(batch_mask, batch_tail, boundary)
        loss = F.mse_loss(prediction, batch_target)
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
        prediction_all = _predict(
            probe,
            mask,
            tail,
            batch_size=args.evaluation_batch_size,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        validation_result = _metrics(
            target,
            prediction_all,
            phase,
            validation,
            pair_left,
            pair_right,
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
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_hashes_before = _replay_hashes(args.replay_dir, args.n648_report)
    metadata, arrays, order = _load_replay_arrays(args.replay_dir)
    phase_values = _parse_phase_values(args.phase_values)
    data = _candidate_data(
        arrays,
        order,
        temporal_depth=args.temporal_depth,
        sample_stride=args.sample_stride,
        target_min_m=args.target_min_m,
        target_max_m=args.target_max_m,
        phase_values=phase_values,
    )
    split = _stratified_episode_group_split(
        data["group"],
        data["phase"],
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    pair_left, pair_right = _pair_indices(
        data["group"], data["step"], sample_stride=args.sample_stride
    )
    mask = torch.from_numpy(data["mask"])
    tail = torch.from_numpy(data["tail"])
    seeds = tuple(int(value) for value in args.seeds.split(","))
    candidates: list[dict[str, object]] = []
    states: dict[tuple[str, int], tuple[dict[str, torch.Tensor], float, float]] = {}
    for name, spec in PROBE_SPECS.items():
        for seed in seeds:
            summary, state, target_mean, target_std = _train_candidate(
                name=name,
                spec=spec,
                seed=seed,
                mask=mask,
                tail=tail,
                target=data["target_m"],
                phase=data["phase"],
                split=split,
                pair_left=pair_left,
                pair_right=pair_right,
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
    selected_probe = VisualAssociationProbe(
        temporal_depth=args.temporal_depth,
        legal_tail_size=tail.shape[-1],
        boundary_size=0,
        spec=PROBE_SPECS[selected_name],
    ).to(device)
    selected_probe.load_state_dict(selected_state)
    selected_prediction = _predict(
        selected_probe,
        mask,
        tail,
        batch_size=args.evaluation_batch_size,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_result = _metrics(
        data["target_m"],
        selected_prediction,
        data["phase"],
        split == 2,
        pair_left,
        pair_right,
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )
    source_hashes_after = _replay_hashes(args.replay_dir, args.n648_report)
    split_groups = {
        name: int(np.unique(data["group"][split == value]).size)
        for name, value in (("train", 0), ("validation", 1), ("test", 2))
    }
    process_gates = {
        "sources_exact": source_hashes_after == source_hashes_before,
        "whole_episode_split": sum(split_groups.values()) == int(np.unique(data["group"]).size),
        "selected_on_validation_before_test": True,
        "probe_not_saved": True,
        "legal_input_only": True,
        "all_required_phases_present": set(phase_values).issubset(
            set(int(value) for value in data["phase"])
        ),
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "replay_metadata": metadata,
        "samples": int(len(data["target_m"])),
        "groups": int(np.unique(data["group"]).size),
        "phase_counts": {
            str(value): int((data["phase"] == value).sum()) for value in range(6)
        },
        "split_sample_counts": {
            name: int((split == value).sum())
            for name, value in (("train", 0), ("validation", 1), ("test", 2))
        },
        "split_group_counts": split_groups,
        "target_range_m": [args.target_min_m, args.target_max_m],
        "phase_values": list(phase_values),
        "near_plane_m": args.near_plane_m,
        "temporal_depth": args.temporal_depth,
        "sample_stride": args.sample_stride,
        "probe_candidates": candidates,
        "selection_rule": "validation correlation, negative near-plane MAE, direction, negative MAE, lower seed, name",
        "selected_probe": selected_name,
        "selected_seed": selected_seed,
        "selected_best_step": int(selected["best_step"]),
        "selected_validation": selected["best_validation"],
        "selected_test": test_result,
        "selected_passes_capacity_gate": _passes_capacity(test_result, args),
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
    parser.add_argument("--n648-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--temporal-depth", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--target-min-m", type=float, default=-1.0)
    parser.add_argument("--target-max-m", type=float, default=12.0)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--phase-values", default="0,1,2,3,4,5")
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=647)
    parser.add_argument("--seeds", default="649,650")
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--near-sample-fraction", type=float, default=0.50)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--minimum-correlation", type=float, default=0.80)
    parser.add_argument("--maximum-mae-m", type=float, default=0.50)
    parser.add_argument("--minimum-phase-correlation", type=float, default=0.65)
    parser.add_argument("--maximum-near-plane-mae-m", type=float, default=0.25)
    parser.add_argument("--minimum-direction-accuracy", type=float, default=0.75)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for name in ("temporal_depth", "sample_stride", "steps", "validation_interval", "batch_size", "evaluation_batch_size"):
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
