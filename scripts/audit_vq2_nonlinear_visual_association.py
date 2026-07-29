#!/usr/bin/env python3
"""Train-only nonlinear capacity probe for legal visual gate association.

The probe is diagnostic: it is selected using group-disjoint validation data,
tested once, never saved, and never attached to N604. Inputs are a fixed-depth
legal mask sequence, optional legal sensor/action timing values, and optionally
the exact legal-derived N604 deterministic recurrent state. Privileged gate
position supplies only the offline regression target.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer
from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE, VISUAL_HEIGHT, VISUAL_WIDTH
from scripts.audit_vq2_boundary_plane_capacity import (
    _boundary_state,
    _load_boundary_event_dataset,
)
from scripts.audit_vq2_event_latent_separability import (
    _phase_split_counts,
    _stratified_group_split,
)
from scripts.audit_vq2_plane_progress_support import _correlation
from scripts.audit_vq2_plane_representation_capacity import _passes_capacity
from scripts.continue_vq2_event_balanced_world_offline import _event_batch
from scripts.continue_vq2_informed_dreamer_actor_offline import _sha256


AUDIT_SCHEMA = "vq2_training_only_nonlinear_visual_association_capacity_v1"


@dataclass(frozen=True)
class ProbeSpec:
    use_mask: bool
    use_tail: bool
    use_boundary: bool


PROBE_SPECS = {
    "boundary_tail_only": ProbeSpec(False, True, True),
    "mask_only": ProbeSpec(True, False, False),
    "mask_legal_tail": ProbeSpec(True, True, False),
    "mask_legal_tail_boundary": ProbeSpec(True, True, True),
}


class VisualAssociationProbe(nn.Module):
    def __init__(
        self,
        *,
        temporal_depth: int,
        legal_tail_size: int,
        boundary_size: int,
        spec: ProbeSpec,
    ) -> None:
        super().__init__()
        self.spec = spec
        if spec.use_mask:
            self.visual = nn.Sequential(
                nn.Conv2d(temporal_depth, 8, kernel_size=5, stride=2, padding=2),
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
        input_size = 64 if spec.use_mask else 0
        input_size += legal_tail_size if spec.use_tail else 0
        input_size += boundary_size if spec.use_boundary else 0
        if input_size <= 0:
            raise ValueError("probe specification selects no legal input")
        self.head = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        mask: torch.Tensor,
        tail: torch.Tensor,
        boundary: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.spec.use_mask:
            assert self.visual is not None
            parts.append(self.visual(mask))
        if self.spec.use_tail:
            parts.append(tail)
        if self.spec.use_boundary:
            parts.append(boundary)
        return self.head(torch.cat(parts, -1)).squeeze(-1)


def _temporal_masks(
    mask: np.ndarray,
    targets: tuple[int, ...],
    *,
    depth: int,
) -> np.ndarray:
    if depth <= 0 or min(targets) < depth:
        raise ValueError("temporal depth exceeds available target history")
    return np.stack(
        [
            np.stack(
                [mask[:, target - 1 - lag] for lag in range(depth)],
                1,
            )
            for target in targets
        ],
        1,
    )


def _state_at(posterior: RSSMState, index: int) -> RSSMState:
    return RSSMState(
        posterior.deterministic[:, index],
        posterior.stochastic[:, index],
        posterior.logits[:, index],
    )


@torch.no_grad()
def _exact_boundary_deterministic(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
    initial_state: RSSMState,
    *,
    targets: tuple[int, ...],
) -> torch.Tensor:
    output = model.observe_sequence(
        observation[:, : max(targets)],
        action[:, : max(targets)],
        initial_state=initial_state,
        deterministic_latent=True,
    )
    return torch.stack(
        [_state_at(output.posterior, target - 1).deterministic.float() for target in targets],
        1,
    )


def _partition_result(
    target: np.ndarray,
    prediction: np.ndarray,
    chosen: np.ndarray,
) -> dict[str, float | int]:
    selected_target = np.asarray(target)[chosen]
    selected_prediction = np.asarray(prediction)[chosen]
    error = selected_prediction - selected_target
    step = float(np.abs(np.diff(selected_target, axis=1)).mean())
    return {
        "events": int(chosen.sum()),
        "correlation": _correlation(selected_target, selected_prediction),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_true_step": step,
        "mae_in_true_step_units": float(np.abs(error).mean()) / step,
        "strictly_decreasing_window_fraction": float(
            (np.diff(selected_prediction, axis=1) < 0.0).all(1).mean()
        ),
        "final_is_minimum_window_fraction": float(
            (
                selected_prediction[:, -1]
                < selected_prediction[:, :-1].min(1)
            ).mean()
        ),
    }


def _selection_key(result: dict[str, float | int], *, seed: int, name: str) -> tuple:
    return (
        float(result["correlation"]),
        float(result["final_is_minimum_window_fraction"]),
        float(result["strictly_decreasing_window_fraction"]),
        -float(result["mae_in_true_step_units"]),
        -seed,
        name,
    )


def _flatten_selected(value: torch.Tensor, chosen: np.ndarray) -> torch.Tensor:
    return value[torch.from_numpy(chosen).to(value.device)].flatten(0, 1)


def _predict_events(
    probe: VisualAssociationProbe,
    mask: torch.Tensor,
    tail: torch.Tensor,
    boundary: torch.Tensor,
    *,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    probe.eval()
    with torch.no_grad():
        prediction = probe(
            mask.flatten(0, 1),
            tail.flatten(0, 1),
            boundary.flatten(0, 1),
        ).reshape(mask.shape[:2])
    return (prediction.float().cpu().numpy() * target_std + target_mean)


def _train_candidate(
    *,
    name: str,
    spec: ProbeSpec,
    seed: int,
    mask: torch.Tensor,
    tail: torch.Tensor,
    boundary: torch.Tensor,
    target: np.ndarray,
    event_split: np.ndarray,
    steps: int,
    validation_interval: int,
    learning_rate: float,
    weight_decay: float,
    delta_weight: float,
    rank_weight: float,
    rank_margin: float,
    gradient_clip: float,
) -> tuple[dict[str, object], dict[str, torch.Tensor], float, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if mask.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    probe = VisualAssociationProbe(
        temporal_depth=mask.shape[2],
        legal_tail_size=tail.shape[-1],
        boundary_size=boundary.shape[-1],
        spec=spec,
    ).to(mask.device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train = event_split == 0
    validation = event_split == 1
    target_mean = float(target[train].mean())
    target_std = float(target[train].std())
    if target_std <= 0.0:
        raise RuntimeError("training target is constant")
    normalized = torch.from_numpy(
        ((target - target_mean) / target_std).astype(np.float32)
    ).to(mask.device)
    train_mask = _flatten_selected(mask, train)
    train_tail = _flatten_selected(tail, train)
    train_boundary = _flatten_selected(boundary, train)
    train_target = normalized[torch.from_numpy(train).to(mask.device)]
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, float | int] | None = None
    best_step = 0
    final_loss = float("nan")
    maximum_gradient_norm = 0.0
    for step in range(1, steps + 1):
        probe.train()
        prediction = probe(train_mask, train_tail, train_boundary).reshape(
            train_target.shape
        )
        regression_loss = F.mse_loss(prediction, train_target)
        delta_loss = F.mse_loss(
            torch.diff(prediction, dim=1), torch.diff(train_target, dim=1)
        )
        rank_loss = F.relu(
            rank_margin + prediction[:, 1:] - prediction[:, :-1]
        ).mean()
        loss = regression_loss + delta_weight * delta_loss + rank_weight * rank_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(probe.parameters(), gradient_clip).detach().cpu()
        )
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % validation_interval and step != steps:
            continue
        prediction_all = _predict_events(
            probe,
            mask,
            tail,
            boundary,
            target_mean=target_mean,
            target_std=target_std,
        )
        validation_result = _partition_result(target, prediction_all, validation)
        if best_validation is None or _selection_key(
            validation_result, seed=seed, name=name
        ) > _selection_key(best_validation, seed=seed, name=name):
            best_validation = validation_result
            best_step = step
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
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
            "optimizer_steps": steps,
        },
        best_state,
        target_mean,
        target_std,
    )


def audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_paths = {
        "checkpoint": args.checkpoint,
        "event_dataset": args.event_dataset,
        "n645_report": args.n645_report,
    }
    source_hashes_before = {name: _sha256(path) for name, path in source_paths.items()}
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
    dataset = _load_boundary_event_dataset(
        args.event_dataset,
        context_length=args.context_length,
        event_threshold=args.event_threshold,
        deterministic_size=deterministic_size,
        stochastic_groups=stochastic_groups,
        stochastic_classes=stochastic_classes,
    )
    observation, action, reward, _continuation = _event_batch(
        dataset, np.arange(dataset["mask"].shape[0]), device=device
    )
    negative_offsets = tuple(
        sorted((int(value) for value in args.negative_offsets.split(",")), reverse=True)
    )
    if not negative_offsets or any(value <= 0 for value in negative_offsets):
        raise ValueError("negative offsets must be unique positive integers")
    if len(set(negative_offsets)) != len(negative_offsets):
        raise ValueError("negative offsets must be unique positive integers")
    targets = tuple(args.context_length - value for value in negative_offsets) + (args.context_length,)
    initial_state = _boundary_state(
        dataset, device=device, stochastic_classes=stochastic_classes
    )
    exact_boundary = _exact_boundary_deterministic(
        model, observation, action, initial_state, targets=targets
    )
    counterfactual = observation.clone()
    counterfactual[..., LEGAL_OBS_SIZE:] = 123.456
    counterfactual_boundary = _exact_boundary_deterministic(
        model, counterfactual, action, initial_state, targets=targets
    )
    privilege_counterfactual_max_abs_error = float(
        (exact_boundary - counterfactual_boundary).abs().max().cpu()
    )
    mask_np = dataset["mask"].astype(np.float32).reshape(
        dataset["mask"].shape[0], args.context_length + 1, VISUAL_HEIGHT, VISUAL_WIDTH
    ) / 255.0
    temporal_mask = torch.from_numpy(
        _temporal_masks(mask_np, targets, depth=args.temporal_depth)
    ).to(device)
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    legal_tail = torch.from_numpy(
        np.stack(
            [dataset["tail"][:, target - 1, :legal_tail_size] for target in targets],
            1,
        ).astype(np.float32)
    ).to(device)
    target = np.stack(
        [
            dataset["tail"][:, target_index - 1, legal_tail_size + 3]
            for target_index in targets
        ],
        1,
    ).astype(np.float32)
    labels = np.stack(
        [dataset["reward"][:, target_index] > args.event_threshold for target_index in targets],
        1,
    )
    if not labels[:, -1].all() or labels[:, :-1].any():
        raise RuntimeError("audited targets are not four hard negatives and one event")
    groups = dataset["vector_step"].astype(np.int64)
    phases = np.rint(dataset["phase_after_event"].astype(np.float64) * 6.0).astype(np.int64)
    event_split = _stratified_group_split(
        groups,
        phases,
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("probe seeds must be nonempty and unique")
    candidates: list[dict[str, object]] = []
    saved_states: dict[tuple[str, int], tuple[dict[str, torch.Tensor], float, float]] = {}
    for name, spec in PROBE_SPECS.items():
        for seed in seeds:
            summary, state, target_mean, target_std = _train_candidate(
                name=name,
                spec=spec,
                seed=seed,
                mask=temporal_mask,
                tail=legal_tail,
                boundary=exact_boundary,
                target=target,
                event_split=event_split,
                steps=args.steps,
                validation_interval=args.validation_interval,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                delta_weight=args.delta_weight,
                rank_weight=args.rank_weight,
                rank_margin=args.rank_margin,
                gradient_clip=args.gradient_clip,
            )
            candidates.append(summary)
            saved_states[(name, seed)] = (state, target_mean, target_std)
    selected = max(
        candidates,
        key=lambda row: _selection_key(
            row["best_validation"], seed=int(row["seed"]), name=str(row["name"])
        ),
    )
    selected_name = str(selected["name"])
    selected_seed = int(selected["seed"])
    selected_state, target_mean, target_std = saved_states[(selected_name, selected_seed)]
    selected_probe = VisualAssociationProbe(
        temporal_depth=args.temporal_depth,
        legal_tail_size=legal_tail.shape[-1],
        boundary_size=exact_boundary.shape[-1],
        spec=PROBE_SPECS[selected_name],
    ).to(device)
    selected_probe.load_state_dict(selected_state)
    selected_prediction = _predict_events(
        selected_probe,
        temporal_mask,
        legal_tail,
        exact_boundary,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_result = _partition_result(target, selected_prediction, event_split == 2)
    source_hashes_after = {name: _sha256(path) for name, path in source_paths.items()}
    model_changes = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, source_model_state[name])
    ]
    split_summary = _phase_split_counts(event_split, phases, groups)
    process_gates = {
        "sources_exact": source_hashes_after == source_hashes_before,
        "n604_model_exact": not model_changes,
        "privileged_counterfactual_exact": privilege_counterfactual_max_abs_error == 0.0,
        "all_splits_cover_all_phases": all(
            all(int(count) > 0 for count in split["phase_counts"].values())
            for split in split_summary.values()
        ),
        "selected_on_validation_before_test": True,
        "probe_not_saved": True,
    }
    report: dict[str, object] = {
        "contract": AUDIT_SCHEMA,
        "source_hashes": source_hashes_after,
        "sources_unchanged": source_hashes_after == source_hashes_before,
        "targets": list(targets),
        "negative_offsets": list(negative_offsets),
        "temporal_depth": args.temporal_depth,
        "event_split": split_summary,
        "probe_candidates": candidates,
        "selection_rule": "validation correlation, final-minimum, monotonic, negative step-MAE, lower seed, name",
        "selected_probe": selected_name,
        "selected_seed": selected_seed,
        "selected_best_step": int(selected["best_step"]),
        "selected_validation": selected["best_validation"],
        "selected_test": test_result,
        "selected_passes_capacity_gate": _passes_capacity(test_result),
        "privilege_counterfactual_max_abs_error": privilege_counterfactual_max_abs_error,
        "model_tensor_changes": model_changes,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "probe_optimizer_steps": len(candidates) * args.steps,
        "probe_weights_saved": 0,
        "n604_optimizer_steps": 0,
        "n604_updates": 0,
        "world_updates": 0,
        "reward_updates": 0,
        "actor_updates": 0,
        "critic_updates": 0,
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
    parser.add_argument("--event-dataset", type=Path, required=True)
    parser.add_argument("--n645-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--temporal-depth", type=int, default=4)
    parser.add_argument("--negative-offsets", default="4,3,2,1")
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=644)
    parser.add_argument("--seeds", default="646,647")
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--validation-interval", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for name in ("context_length", "temporal_depth", "steps", "validation_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.report.exists():
        parser.error("audit refuses to overwrite an existing report")
    return args


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
