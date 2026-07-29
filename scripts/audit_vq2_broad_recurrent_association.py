#!/usr/bin/env python3
"""Broad visual capacity probe conditioned on exact legal N604 episode state."""

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

from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer
from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_broad_visual_association import (
    _candidate_data,
    _load_replay_arrays,
    _metrics,
    _pair_indices,
    _replay_hashes,
    _selection_key,
)
from scripts.audit_vq2_dense_replay_coverage import (
    _episode_groups,
    _stratified_episode_group_split,
)
from scripts.audit_vq2_nonlinear_visual_association import (
    ProbeSpec,
    VisualAssociationProbe,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256


AUDIT_SCHEMA = "vq2_training_only_broad_recurrent_visual_capacity_v1"
PROBE_SPECS = {
    "recurrent_tail_only": ProbeSpec(False, True, True),
    "mask_tail_recurrent": ProbeSpec(True, True, True),
}


def _source_hashes(
    replay_dir: Path, n649_report: Path, checkpoint: Path
) -> dict[str, str]:
    replay = _replay_hashes(replay_dir, n649_report)
    replay["n649_report"] = replay.pop("n648_report")
    replay["checkpoint"] = _sha256(checkpoint)
    return replay


def _masked_state(
    candidate: RSSMState,
    zero: RSSMState,
    active: torch.Tensor,
) -> RSSMState:
    if active.ndim != 1 or active.shape[0] != candidate.deterministic.shape[0]:
        raise ValueError("active mask does not align with recurrent state")
    values: list[torch.Tensor] = []
    for candidate_value, zero_value in zip(candidate, zero, strict=True):
        shape = (active.shape[0],) + (1,) * (candidate_value.ndim - 1)
        values.append(torch.where(active.reshape(shape), candidate_value, zero_value))
    return RSSMState(*values)


@torch.no_grad()
def _extract_candidate_states(
    model: VQ2InformedDreamer,
    arrays: dict[str, np.memmap],
    order: np.ndarray,
    action: np.memmap,
    candidate_step: np.ndarray,
    candidate_agent: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float | int]]:
    valid = np.asarray(arrays["valid"][order], dtype=bool)
    continuation = np.asarray(arrays["continuation"][order], dtype=bool)
    groups, _complete = _episode_groups(
        valid,
        continuation,
        starts_at_known_boundary=len(order) < arrays["valid"].shape[0] and int(order[0]) == 0,
    )
    capture: dict[int, list[int]] = {}
    for index, step in enumerate(candidate_step):
        capture.setdefault(int(step), []).append(index)
    states = np.empty((len(candidate_step), model.rssm.deterministic_size), dtype=np.float16)
    zero = model.rssm.initial(valid.shape[1], device=device)
    state = zero
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    start = time.perf_counter()
    captured = 0
    for chronological_step, physical in enumerate(order):
        active = torch.from_numpy(groups[chronological_step] >= 0).to(device)
        mask = torch.from_numpy(
            np.asarray(arrays["mask"][physical], dtype=np.float32)
        ).to(device)
        mask /= 255.0
        tail = torch.from_numpy(
            np.asarray(arrays["tail"][physical, :, :legal_tail_size], dtype=np.float32)
        ).to(device)
        legal = torch.cat((mask, tail), -1)
        previous_action = torch.from_numpy(
            np.asarray(action[physical], dtype=np.float32)
        ).to(device)
        candidate, _prior = model.rssm.observe_step(
            state,
            previous_action,
            legal,
            deterministic_latent=True,
        )
        state = _masked_state(candidate, zero, active)
        for index in capture.get(chronological_step, ()):
            agent = int(candidate_agent[index])
            if groups[chronological_step, agent] < 0:
                raise RuntimeError("candidate state falls outside a clean episode")
            states[index] = state.deterministic[agent].float().cpu().numpy().astype(np.float16)
            captured += 1
    if captured != len(candidate_step):
        raise RuntimeError(f"captured {captured} recurrent states, expected {len(candidate_step)}")
    return states.astype(np.float32), {
        "chronological_steps_replayed": int(len(order)),
        "agent_transitions_replayed": int(len(order) * valid.shape[1]),
        "candidate_states_captured": captured,
        "wall_seconds": time.perf_counter() - start,
    }


def _predict(
    probe: VisualAssociationProbe,
    mask: torch.Tensor,
    tail: torch.Tensor,
    recurrent: torch.Tensor,
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
            output.append(
                probe(
                    mask[start:stop].to(device=device, dtype=torch.float32) / 255.0,
                    tail[start:stop].to(device),
                    recurrent[start:stop].to(device),
                ).float().cpu()
            )
    return torch.cat(output).numpy() * target_std + target_mean


def _train_candidate(
    *,
    name: str,
    spec: ProbeSpec,
    seed: int,
    mask: torch.Tensor,
    tail: torch.Tensor,
    recurrent: torch.Tensor,
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
        boundary_size=recurrent.shape[-1],
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
        prediction = probe(
            mask[indices].to(device=device, dtype=torch.float32) / 255.0,
            tail[indices].to(device),
            recurrent[indices].to(device),
        )
        batch_target = torch.from_numpy(normalized_target[indices].astype(np.float32)).to(device)
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
            recurrent,
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
    source_hashes_before = _source_hashes(
        args.replay_dir, args.n649_report, args.checkpoint
    )
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    if not isinstance(training_args, dict):
        raise RuntimeError("checkpoint lacks training arguments")
    model = VQ2InformedDreamer(
        deterministic_size=int(training_args["deterministic_size"]),
        stochastic_groups=int(training_args["stochastic_groups"]),
        stochastic_classes=int(training_args["stochastic_classes"]),
        actor_initial_std=float(training_args.get("actor_initial_std", 0.20)),
        actor_distribution_mode=str(training_args.get("actor_distribution_mode", "legacy_tanh_normal")),
        distributional_reward=bool(training_args.get("distributional_reward", False)),
        action_conditioned_reward=bool(training_args.get("action_conditioned_reward", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    source_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata, arrays, order = _load_replay_arrays(args.replay_dir)
    action = np.memmap(
        args.replay_dir / "action.dat",
        dtype=np.float16,
        mode="r",
        shape=(int(metadata["capacity"]), int(metadata["agents"]), 4),
    )
    data = _candidate_data(
        arrays,
        order,
        temporal_depth=args.temporal_depth,
        sample_stride=args.sample_stride,
        target_min_m=args.target_min_m,
        target_max_m=args.target_max_m,
    )
    valid = np.asarray(arrays["valid"][order], dtype=bool)
    continuation = np.asarray(arrays["continuation"][order], dtype=bool)
    group_grid, _complete = _episode_groups(
        valid,
        continuation,
        starts_at_known_boundary=len(order) < arrays["valid"].shape[0] and int(order[0]) == 0,
    )
    group_to_agent = {}
    for agent_index in range(group_grid.shape[1]):
        for group_value in np.unique(group_grid[:, agent_index]):
            if group_value >= 0:
                group_to_agent[int(group_value)] = agent_index
    candidate_agent = np.asarray([group_to_agent[int(group)] for group in data["group"]], dtype=np.int64)
    recurrent_np, recurrent_replay = _extract_candidate_states(
        model,
        arrays,
        order,
        action,
        data["step"],
        candidate_agent,
        device=device,
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
    recurrent = torch.from_numpy(recurrent_np)
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
                recurrent=recurrent,
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
        boundary_size=recurrent.shape[-1],
        spec=PROBE_SPECS[selected_name],
    ).to(device)
    selected_probe.load_state_dict(selected_state)
    selected_prediction = _predict(
        selected_probe,
        mask,
        tail,
        recurrent,
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
    source_hashes_after = _source_hashes(
        args.replay_dir, args.n649_report, args.checkpoint
    )
    model_changes = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, source_model_state[name])
    ]
    process_gates = {
        "sources_exact": source_hashes_after == source_hashes_before,
        "n604_model_exact": not model_changes,
        "all_candidate_states_captured": int(recurrent_replay["candidate_states_captured"]) == len(data["step"]),
        "selected_on_validation_before_test": True,
        "probe_not_saved": True,
        "legal_recurrent_input_only": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "samples": int(len(data["target_m"])),
        "groups": int(np.unique(data["group"]).size),
        "recurrent_replay": recurrent_replay,
        "recurrent_state_size": int(recurrent.shape[-1]),
        "probe_candidates": candidates,
        "selected_probe": selected_name,
        "selected_seed": selected_seed,
        "selected_best_step": int(selected["best_step"]),
        "selected_validation": selected["best_validation"],
        "selected_test": test_result,
        "selected_passes_capacity_gate": _passes_capacity(test_result, args),
        "model_tensor_changes": model_changes,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "probe_optimizer_steps": len(candidates) * args.steps,
        "probe_weights_saved": 0,
        "n604_optimizer_steps": 0,
        "n604_updates": 0,
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--n649-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--temporal-depth", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--target-min-m", type=float, default=-1.0)
    parser.add_argument("--target-max-m", type=float, default=12.0)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=647)
    parser.add_argument("--seeds", default="651,652")
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
    if args.report.exists():
        parser.error("audit refuses to overwrite an existing report")
    return args


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
