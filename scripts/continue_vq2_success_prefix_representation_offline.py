#!/usr/bin/env python3
"""One exact-boundary, actor-frozen N587 representation integration step."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer
from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_event_latent_separability import _stratified_group_split
from scripts.audit_vq2_success_prefix_sequence_capacity import (
    _crop_index,
    _forward_metres,
    _metrics,
    _selection_key,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import (
    _replay_hashes,
    _sha256,
)
from scripts.train_vq2_informed_dreamer import (
    QuantizedSequenceReplay,
    _finite_clip_grad_norm,
    _save_checkpoint,
)


CONTINUATION_SCHEMA = "vq2_success_prefix_soft_posterior_representation_v1"
ALLOWED_MODEL_PREFIXES = (
    "rssm.encoder.",
    "rssm.sequence.",
    "rssm.posterior.",
)


def _replay_geometry(storage_dir: Path) -> tuple[int, int]:
    metadata_path = storage_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != QuantizedSequenceReplay.SCHEMA:
        raise RuntimeError("dense replay schema mismatch")
    capacity = int(metadata.get("capacity", 0))
    agents = int(metadata.get("agents", 0))
    size = int(metadata.get("size", -1))
    position = int(metadata.get("position", -1))
    if capacity <= 0 or agents <= 0:
        raise RuntimeError("dense replay geometry must be positive")
    if not 0 <= size <= capacity or not 0 <= position < capacity:
        raise RuntimeError("dense replay metadata bounds are invalid")
    return capacity, agents


class SoftPlaneProbe(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature).squeeze(-1)


def _model_from_payload(payload: dict, device: torch.device) -> VQ2InformedDreamer:
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
    return model


def _soft_posterior_features(state: RSSMState) -> torch.Tensor:
    probability = state.logits.float().softmax(-1)
    return torch.cat((state.deterministic.float(), probability.flatten(-2)), -1)


def _observe_legal_sequence(
    model: VQ2InformedDreamer,
    legal: torch.Tensor,
    action: torch.Tensor,
    initial_state: RSSMState,
) -> RSSMState:
    if legal.ndim != 3 or legal.shape[-1] != LEGAL_OBS_SIZE:
        raise ValueError("legal sequence must have shape [batch,time,4118]")
    if action.shape[:2] != legal.shape[:2]:
        raise ValueError("action sequence does not align with legal observations")
    state = initial_state
    rows: list[RSSMState] = []
    for step in range(legal.shape[1]):
        state, _prior = model.rssm.observe_step(
            state,
            action[:, step],
            legal[:, step],
            deterministic_latent=True,
        )
        rows.append(state)
    return RSSMState(
        *(torch.stack([getattr(row, field) for row in rows], 1) for field in RSSMState._fields)
    )


def _initial_state(
    arrays: dict[str, np.ndarray], event: np.ndarray, *, device: torch.device
) -> RSSMState:
    event = np.asarray(event, dtype=np.int64)
    logits = torch.from_numpy(arrays["initial_logits"][event].astype(np.float32)).to(device)
    index = torch.from_numpy(
        arrays["initial_stochastic_index"][event].astype(np.int64)
    ).to(device)
    return RSSMState(
        torch.from_numpy(
            arrays["initial_deterministic"][event].astype(np.float32)
        ).to(device),
        F.one_hot(index, logits.shape[-1]).to(torch.float32),
        logits,
    )


def _crop_initial_state(
    arrays: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    reference_event: np.ndarray,
    original_event: np.ndarray,
    crop_start: np.ndarray,
    *,
    device: torch.device,
) -> RSSMState:
    reference_event = np.asarray(reference_event, dtype=np.int64)
    original_event = np.asarray(original_event, dtype=np.int64)
    crop_start = np.asarray(crop_start, dtype=np.int64)
    if not (
        reference_event.shape == original_event.shape == crop_start.shape
    ) or reference_event.ndim != 1:
        raise ValueError("crop boundary vectors must align")
    deterministic: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    stochastic_index: list[np.ndarray] = []
    for ref, original, start in zip(
        reference_event, original_event, crop_start, strict=True
    ):
        if start == 0:
            deterministic.append(arrays["initial_deterministic"][original])
            logits.append(arrays["initial_logits"][original])
            stochastic_index.append(arrays["initial_stochastic_index"][original])
        elif start > 0:
            deterministic.append(reference["deterministic"][ref, start - 1])
            logits.append(reference["logits"][ref, start - 1])
            stochastic_index.append(
                reference["logits"][ref, start - 1].argmax(-1)
            )
        else:
            raise ValueError("crop start cannot be negative")
    logits_tensor = torch.from_numpy(np.stack(logits).astype(np.float32)).to(device)
    index_tensor = torch.from_numpy(np.stack(stochastic_index).astype(np.int64)).to(device)
    return RSSMState(
        torch.from_numpy(np.stack(deterministic).astype(np.float32)).to(device),
        F.one_hot(index_tensor, logits_tensor.shape[-1]).to(torch.float32),
        logits_tensor,
    )


def _legal_action_batch(
    arrays: dict[str, np.ndarray],
    event: np.ndarray,
    start: np.ndarray,
    *,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    event = np.asarray(event, dtype=np.int64)
    start = np.asarray(start, dtype=np.int64)
    offsets = np.arange(sequence_length, dtype=np.int64)
    time_index = start[:, None] + offsets[None, :]
    if np.any(time_index >= arrays["mask"].shape[1] - 1):
        raise ValueError("crop includes the event observation")
    mask = arrays["mask"][event[:, None], time_index].astype(np.float32) / 255.0
    tail = arrays["tail"][event[:, None], time_index, : LEGAL_OBS_SIZE - MASK_SIZE].astype(
        np.float32
    )
    legal = torch.from_numpy(np.concatenate((mask, tail), -1)).to(device)
    action = torch.from_numpy(
        arrays["action"][event[:, None], time_index].astype(np.float32)
    ).to(device)
    return legal, action


def _target_batch(
    arrays: dict[str, np.ndarray],
    event: np.ndarray,
    start: np.ndarray,
    *,
    sequence_length: int,
) -> np.ndarray:
    offsets = np.arange(sequence_length, dtype=np.int64)
    time_index = np.asarray(start, dtype=np.int64)[:, None] + offsets[None, :]
    full_tail = arrays["tail"][np.asarray(event, dtype=np.int64)[:, None], time_index]
    normalized = full_tail[..., (LEGAL_OBS_SIZE - MASK_SIZE) + 3]
    return _forward_metres(normalized.astype(np.float32)).astype(np.float32)


@torch.no_grad()
def _parent_reference(
    model: VQ2InformedDreamer,
    arrays: dict[str, np.ndarray],
    event_ids: np.ndarray,
    *,
    microbatch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {
        "deterministic": [],
        "logits": [],
        "action": [],
    }
    full_length = arrays["mask"].shape[1] - 1
    for offset in range(0, len(event_ids), microbatch_size):
        chosen = event_ids[offset : offset + microbatch_size]
        legal, action = _legal_action_batch(
            arrays,
            chosen,
            np.zeros(len(chosen), dtype=np.int64),
            sequence_length=full_length,
            device=device,
        )
        state = _observe_legal_sequence(
            model, legal, action, _initial_state(arrays, chosen, device=device)
        )
        actor_action = model.deterministic_actor_action(
            model.actor_distribution(state.features)
        )
        values["deterministic"].append(state.deterministic.float().cpu().numpy())
        values["logits"].append(state.logits.float().cpu().numpy())
        values["action"].append(actor_action.float().cpu().numpy())
    return {name: np.concatenate(rows) for name, rows in values.items()}


def _soft_reference(reference: dict[str, np.ndarray]) -> np.ndarray:
    logits = torch.from_numpy(reference["logits"])
    probability = logits.softmax(-1).numpy()
    return np.concatenate(
        (reference["deterministic"], probability.reshape(*probability.shape[:2], -1)),
        -1,
    ).astype(np.float32)


def _crop_full(
    value: np.ndarray,
    reference_event: np.ndarray,
    crop_start: np.ndarray,
    *,
    sequence_length: int,
) -> np.ndarray:
    offsets = np.arange(sequence_length, dtype=np.int64)
    time_index = np.asarray(crop_start, dtype=np.int64)[:, None] + offsets[None, :]
    return value[np.asarray(reference_event, dtype=np.int64)[:, None], time_index]


def _probe_prediction(
    probe: SoftPlaneProbe,
    feature: np.ndarray,
    reference_event: np.ndarray,
    crop_start: np.ndarray,
    indices: np.ndarray,
    *,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    probe.eval()
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            chosen = indices[offset : offset + batch_size]
            batch = _crop_full(
                feature,
                reference_event[chosen],
                crop_start[chosen],
                sequence_length=sequence_length,
            )
            prediction = probe(torch.from_numpy(batch).to(device)).float().cpu().numpy()
            rows.append(prediction * target_std + target_mean)
    return np.concatenate(rows)


def _fit_probe(
    feature: np.ndarray,
    arrays: dict[str, np.ndarray],
    original_event: np.ndarray,
    reference_event: np.ndarray,
    crop_start: np.ndarray,
    crop_split: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SoftPlaneProbe, dict[str, object], float, float]:
    torch.manual_seed(args.probe_seed)
    np.random.seed(args.probe_seed)
    random.seed(args.probe_seed)
    probe = SoftPlaneProbe(feature.shape[-1], args.probe_hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.probe_learning_rate, weight_decay=args.probe_weight_decay
    )
    train = np.flatnonzero(crop_split == 0)
    validation = np.flatnonzero(crop_split == 1)
    train_target = _target_batch(
        arrays,
        original_event[train],
        crop_start[train],
        sequence_length=args.sequence_length,
    )[:, args.burn_in :]
    target_mean = float(train_target.mean())
    target_std = float(train_target.std())
    if target_std <= 0.0:
        raise RuntimeError("plane target is constant")
    rng = np.random.default_rng(args.probe_seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, float | int] | None = None
    best_step = 0
    final_loss = float("nan")
    for step in range(1, args.probe_steps + 1):
        chosen = rng.choice(train, args.probe_batch_size, replace=True)
        batch_feature = _crop_full(
            feature,
            reference_event[chosen],
            crop_start[chosen],
            sequence_length=args.sequence_length,
        )
        batch_target = _target_batch(
            arrays,
            original_event[chosen],
            crop_start[chosen],
            sequence_length=args.sequence_length,
        )
        feature_tensor = torch.from_numpy(batch_feature).to(device)
        target_tensor = torch.from_numpy((batch_target - target_mean) / target_std).to(device)
        prediction = probe(feature_tensor)[:, args.burn_in :]
        target_supervised = target_tensor[:, args.burn_in :]
        regression = F.mse_loss(prediction, target_supervised)
        delta = F.mse_loss(
            torch.diff(prediction, dim=1), torch.diff(target_supervised, dim=1)
        )
        loss = regression + args.delta_weight * delta
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), args.probe_gradient_clip)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step % args.probe_validation_interval and step != args.probe_steps:
            continue
        prediction_np = _probe_prediction(
            probe,
            feature,
            reference_event,
            crop_start,
            validation,
            sequence_length=args.sequence_length,
            batch_size=args.probe_evaluation_batch_size,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        target_np = _target_batch(
            arrays,
            original_event[validation],
            crop_start[validation],
            sequence_length=args.sequence_length,
        )
        result = _metrics(
            target_np,
            prediction_np,
            evaluation_indices=evaluation_indices,
            near_plane_m=args.near_plane_m,
            minimum_pair_delta_m=args.minimum_pair_delta_m,
        )
        if best_validation is None or _selection_key(
            result, args.probe_seed, "soft_posterior"
        ) > _selection_key(best_validation, args.probe_seed, "soft_posterior"):
            best_validation = result
            best_step = step
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in probe.state_dict().items()
            }
    assert best_state is not None and best_validation is not None
    probe.load_state_dict(best_state)
    return probe, {
        "optimizer_steps": args.probe_steps,
        "best_step": best_step,
        "best_validation": best_validation,
        "final_training_loss": final_loss,
        "parameters": sum(parameter.numel() for parameter in probe.parameters()),
    }, target_mean, target_std


def _changed_model_keys(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> tuple[list[str], list[str]]:
    changed = [name for name, value in after.items() if not torch.equal(value, before[name])]
    forbidden = [name for name in changed if not name.startswith(ALLOWED_MODEL_PREFIXES)]
    return changed, forbidden


def _drift(after: torch.Tensor, before: torch.Tensor) -> dict[str, float]:
    difference = after.float() - before.float()
    return {
        "rmse": float(difference.square().mean().sqrt().cpu()),
        "max_abs": float(difference.abs().max().cpu()),
    }


@torch.no_grad()
def _validation_outputs(
    model: VQ2InformedDreamer,
    probe: SoftPlaneProbe,
    arrays: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    original_event: np.ndarray,
    reference_event: np.ndarray,
    crop_start: np.ndarray,
    indices: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    target_rows: list[np.ndarray] = []
    prediction_rows: list[np.ndarray] = []
    action_rows: list[torch.Tensor] = []
    parent_action_rows: list[torch.Tensor] = []
    model.eval()
    probe.eval()
    for offset in range(0, len(indices), args.microbatch_size):
        chosen = indices[offset : offset + args.microbatch_size]
        original = original_event[chosen]
        ref = reference_event[chosen]
        starts = crop_start[chosen]
        legal, action = _legal_action_batch(
            arrays,
            original,
            starts,
            sequence_length=args.sequence_length,
            device=device,
        )
        state = _observe_legal_sequence(
            model,
            legal,
            action,
            _crop_initial_state(
                arrays, reference, ref, original, starts, device=device
            ),
        )
        prediction = probe(_soft_posterior_features(state))
        actor_action = model.deterministic_actor_action(
            model.actor_distribution(state.features)
        )
        parent_action = _crop_full(
            reference["action"], ref, starts, sequence_length=args.sequence_length
        )
        target_rows.append(
            _target_batch(
                arrays, original, starts, sequence_length=args.sequence_length
            )
        )
        prediction_rows.append(
            prediction.float().cpu().numpy() * target_std + target_mean
        )
        action_rows.append(actor_action.float().cpu())
        parent_action_rows.append(torch.from_numpy(parent_action))
    return (
        np.concatenate(target_rows),
        np.concatenate(prediction_rows),
        torch.cat(action_rows),
        torch.cat(parent_action_rows),
    )


def continue_representation(args: argparse.Namespace) -> dict[str, object]:
    if args.output_checkpoint.resolve() == args.checkpoint.resolve():
        raise ValueError("integration cannot overwrite its parent")
    if args.output_checkpoint.exists() or args.report.exists():
        raise ValueError("integration refuses to overwrite outputs")
    start_time = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_before = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "dataset": _sha256(args.dataset),
        "source_report": _sha256(args.source_report),
        "n669_report": _sha256(args.n669_report),
        "dense_replay": _replay_hashes(args.dense_replay_dir),
    }
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    n669_report = json.loads(args.n669_report.read_text(encoding="utf-8"))
    if source_report.get("contract") != (
        "vq2_actor_frozen_gate_event_windows_replay_filtered_v1"
    ) or not source_report.get("process_passed"):
        raise RuntimeError("N668 source report contract did not pass")
    if source_report.get("output_dataset_sha256") != source_before["dataset"]:
        raise RuntimeError("N668 report does not match its dataset")
    if n669_report.get("contract") != (
        "vq2_training_only_success_prefix_recurrent_capacity_v1"
    ) or not n669_report.get("process_passed"):
        raise RuntimeError("N669 source report contract did not pass")
    if not n669_report.get("selected_passes_capacity_gate"):
        raise RuntimeError("N669 capacity gate did not pass")
    if n669_report.get("source_hashes", {}).get("dataset") != source_before["dataset"]:
        raise RuntimeError("N669 report does not lock N668 dataset")
    if (
        n669_report.get("source_hashes", {}).get("source_report")
        != source_before["source_report"]
    ):
        raise RuntimeError("N669 report does not lock the N668 report")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _model_from_payload(payload, device)
    model.eval()
    parent_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    with np.load(args.dataset) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    phase_after = np.rint(arrays["phase_after_event"].astype(np.float64) * 6.0).astype(
        np.int64
    )
    legal_tail_size = LEGAL_OBS_SIZE - MASK_SIZE
    phase_before = np.rint(
        arrays["tail"][:, :-1, legal_tail_size + 32].astype(np.float64) * 6.0
    ).astype(np.int64)
    if not np.all(phase_before == 0) or not np.all(phase_after == 1):
        raise RuntimeError("N668 phase contract mismatch")
    event_split = _stratified_group_split(
        arrays["vector_step"].astype(np.int64),
        phase_after,
        seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    crop_starts = tuple(int(value) for value in args.crop_starts.split(","))
    if tuple(n669_report.get("crop_starts", ())) != crop_starts:
        raise RuntimeError("crop starts differ from N669")
    if n669_report.get("sequence_length") != args.sequence_length:
        raise RuntimeError("sequence length differs from N669")
    if n669_report.get("burn_in") != args.burn_in:
        raise RuntimeError("burn-in differs from N669")
    original_event, crop_start = _crop_index(len(phase_after), crop_starts)
    crop_split = event_split[original_event]
    allowed_events = np.flatnonzero(event_split != 2)
    event_to_reference = np.full(len(event_split), -1, dtype=np.int64)
    event_to_reference[allowed_events] = np.arange(len(allowed_events), dtype=np.int64)
    reference_event = event_to_reference[original_event]
    if np.any(reference_event[crop_split != 2] < 0):
        raise RuntimeError("train/validation crop lacks a legal parent reference")
    evaluation_indices = np.arange(
        args.burn_in - 1, args.sequence_length, args.evaluation_stride, dtype=np.int64
    )
    reference = _parent_reference(
        model,
        arrays,
        allowed_events,
        microbatch_size=args.reference_microbatch_size,
        device=device,
    )
    expected_det = arrays["preevent_deterministic"][allowed_events].astype(np.float32)
    expected_logits = arrays["preevent_logits"][allowed_events].astype(np.float32)
    boundary_replay = {
        "deterministic_max_abs_error": float(
            np.abs(reference["deterministic"][:, -1] - expected_det).max()
        ),
        "logits_max_abs_error": float(
            np.abs(reference["logits"][:, -1] - expected_logits).max()
        ),
        "stochastic_index_mismatch_count": int(
            (
                reference["logits"][:, -1].argmax(-1)
                != arrays["preevent_stochastic_index"][allowed_events]
            ).sum()
        ),
    }
    soft_reference = _soft_reference(reference)
    probe, probe_fit, target_mean, target_std = _fit_probe(
        soft_reference,
        arrays,
        original_event,
        reference_event,
        crop_start,
        crop_split,
        evaluation_indices,
        args=args,
        device=device,
    )
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    train = np.flatnonzero(crop_split == 0)
    validation = np.flatnonzero(crop_split == 1)
    pre_validation_prediction = _probe_prediction(
        probe,
        soft_reference,
        reference_event,
        crop_start,
        validation,
        sequence_length=args.sequence_length,
        batch_size=args.probe_evaluation_batch_size,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    validation_target = _target_batch(
        arrays,
        original_event[validation],
        crop_start[validation],
        sequence_length=args.sequence_length,
    )
    pre_validation_metrics = _metrics(
        validation_target,
        pre_validation_prediction,
        evaluation_indices=evaluation_indices,
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )

    replay_capacity, replay_agents = _replay_geometry(args.dense_replay_dir)
    replay = QuantizedSequenceReplay(
        replay_capacity,
        replay_agents,
        storage_dir=args.dense_replay_dir,
        resume=True,
    )
    dense_observation, dense_action, _reward, _continuation = replay.sample(
        args.dense_batch_size,
        args.dense_sequence_length,
        device=device,
        rng=random.Random(args.dense_seed),
    )
    dense_legal = dense_observation[..., :LEGAL_OBS_SIZE]
    zero_state = model.rssm.initial(args.dense_batch_size, device=device)
    with torch.no_grad():
        dense_parent_state = _observe_legal_sequence(
            model, dense_legal, dense_action, zero_state
        )
        dense_parent_action = model.deterministic_actor_action(
            model.actor_distribution(dense_parent_state.features)
        ).detach()
        dense_parent_det = dense_parent_state.deterministic.detach()
        dense_parent_logits = dense_parent_state.logits.detach()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_parameters = [
        *model.rssm.encoder.parameters(),
        *model.rssm.sequence.parameters(),
        *model.rssm.posterior.parameters(),
    ]
    for parameter in train_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(train_parameters, lr=args.learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    aggregate = {
        "plane": 0.0,
        "delta": 0.0,
        "success_action_anchor": 0.0,
        "dense_feature_anchor": 0.0,
        "dense_action_anchor": 0.0,
    }
    model.train()
    for offset in range(0, len(train), args.microbatch_size):
        chosen = train[offset : offset + args.microbatch_size]
        original = original_event[chosen]
        ref = reference_event[chosen]
        starts = crop_start[chosen]
        legal, action = _legal_action_batch(
            arrays,
            original,
            starts,
            sequence_length=args.sequence_length,
            device=device,
        )
        state = _observe_legal_sequence(
            model,
            legal,
            action,
            _crop_initial_state(
                arrays, reference, ref, original, starts, device=device
            ),
        )
        prediction = probe(_soft_posterior_features(state))[:, args.burn_in :]
        target_np = _target_batch(
            arrays, original, starts, sequence_length=args.sequence_length
        )
        target_tensor = torch.from_numpy((target_np - target_mean) / target_std).to(device)
        target_supervised = target_tensor[:, args.burn_in :]
        plane_loss = F.mse_loss(prediction, target_supervised)
        delta_loss = F.mse_loss(
            torch.diff(prediction, dim=1), torch.diff(target_supervised, dim=1)
        )
        actor_action = model.deterministic_actor_action(
            model.actor_distribution(state.features)
        )
        parent_action = torch.from_numpy(
            _crop_full(
                reference["action"], ref, starts, sequence_length=args.sequence_length
            )
        ).to(device)
        action_anchor = F.mse_loss(actor_action.float(), parent_action.float())
        weight = len(chosen) / len(train)
        loss = (
            plane_loss
            + args.delta_weight * delta_loss
            + args.success_action_anchor_weight * action_anchor
        )
        (loss * weight).backward()
        aggregate["plane"] += float(plane_loss.detach().cpu()) * weight
        aggregate["delta"] += float(delta_loss.detach().cpu()) * weight
        aggregate["success_action_anchor"] += float(action_anchor.detach().cpu()) * weight

    dense_child_state = _observe_legal_sequence(
        model,
        dense_legal,
        dense_action,
        model.rssm.initial(args.dense_batch_size, device=device),
    )
    dense_child_action = model.deterministic_actor_action(
        model.actor_distribution(dense_child_state.features)
    )
    dense_feature_anchor = F.mse_loss(
        dense_child_state.deterministic.float(), dense_parent_det.float()
    ) + F.mse_loss(dense_child_state.logits.float(), dense_parent_logits.float())
    dense_action_anchor = F.mse_loss(
        dense_child_action.float(), dense_parent_action.float()
    )
    dense_loss = (
        args.dense_feature_anchor_weight * dense_feature_anchor
        + args.dense_action_anchor_weight * dense_action_anchor
    )
    dense_loss.backward()
    aggregate["dense_feature_anchor"] = float(dense_feature_anchor.detach().cpu())
    aggregate["dense_action_anchor"] = float(dense_action_anchor.detach().cpu())
    gradient_by_component = {}
    for name, module in (
        ("encoder", model.rssm.encoder),
        ("sequence", model.rssm.sequence),
        ("posterior", model.rssm.posterior),
    ):
        gradients = [
            parameter.grad.detach().abs().max()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        gradient_by_component[name] = float(torch.stack(gradients).max().cpu()) if gradients else 0.0
    gradient_norm = _finite_clip_grad_norm(
        train_parameters, args.max_gradient_norm, label="success-prefix-representation"
    )
    optimizer.step()
    preprojection_max_parameter_delta = 0.0
    projected_parameter_values = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            delta = parameter - parent_model_state[name]
            preprojection_max_parameter_delta = max(
                preprojection_max_parameter_delta, float(delta.abs().max().cpu())
            )
            clipped = delta.clamp(-args.max_parameter_delta, args.max_parameter_delta)
            projected_parameter_values += int((clipped != delta).sum().cpu())
            parameter.copy_(parent_model_state[name] + clipped)
    model.eval()

    post_target, post_prediction, post_action, validation_parent_action = _validation_outputs(
        model,
        probe,
        arrays,
        reference,
        original_event,
        reference_event,
        crop_start,
        validation,
        args=args,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    validation_metrics = _metrics(
        post_target,
        post_prediction,
        evaluation_indices=evaluation_indices,
        near_plane_m=args.near_plane_m,
        minimum_pair_delta_m=args.minimum_pair_delta_m,
    )
    validation_action_drift = _drift(post_action, validation_parent_action)
    with torch.no_grad():
        dense_post_state = _observe_legal_sequence(
            model,
            dense_legal,
            dense_action,
            model.rssm.initial(args.dense_batch_size, device=device),
        )
        dense_post_action = model.deterministic_actor_action(
            model.actor_distribution(dense_post_state.features)
        )
    dense_drift = {
        "deterministic": _drift(dense_post_state.deterministic, dense_parent_det),
        "logits": _drift(dense_post_state.logits, dense_parent_logits),
        "action": _drift(dense_post_action, dense_parent_action),
    }
    changed, forbidden = _changed_model_keys(parent_model_state, model.state_dict())
    postprojection_max_parameter_delta = max(
        (
            float((value - parent_model_state[name]).abs().max().cpu())
            for name, value in model.state_dict().items()
            if name in changed
        ),
        default=0.0,
    )
    output_payload = dict(payload)
    output_payload["model"] = model.state_dict()
    output_payload["success_prefix_representation_aux"] = {
        "schema": CONTINUATION_SCHEMA,
        "probe_state": {
            name: value.detach().cpu() for name, value in probe.state_dict().items()
        },
        "target_mean": target_mean,
        "target_std": target_std,
        "crop_starts": crop_starts,
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "split_seed": args.split_seed,
        "representation_updates": 1,
    }
    _save_checkpoint(args.output_checkpoint, output_payload)
    source_after = {
        "executable": _sha256(Path(__file__)),
        "checkpoint": _sha256(args.checkpoint),
        "dataset": _sha256(args.dataset),
        "source_report": _sha256(args.source_report),
        "n669_report": _sha256(args.n669_report),
        "dense_replay": _replay_hashes(args.dense_replay_dir),
    }
    process_gates = {
        "sources_exact": source_after == source_before,
        "n669_capacity_passed": bool(n669_report.get("selected_passes_capacity_gate")),
        "boundary_replay_deterministic_at_most_0p002": boundary_replay["deterministic_max_abs_error"] <= 0.002,
        "boundary_replay_logits_at_most_0p002": boundary_replay["logits_max_abs_error"] <= 0.002,
        "boundary_replay_stochastic_exact": boundary_replay["stochastic_index_mismatch_count"] == 0,
        "soft_gradient_reaches_encoder": gradient_by_component["encoder"] > 0.0,
        "soft_gradient_reaches_sequence": gradient_by_component["sequence"] > 0.0,
        "soft_gradient_reaches_posterior": gradient_by_component["posterior"] > 0.0,
        "model_changed": bool(changed),
        "only_allowed_model_tensors_changed": not forbidden,
        "test_labels_not_evaluated": True,
        "actor_weights_exact": not any(name.startswith("actor.") for name in changed),
    }
    validation_admission_gates = {
        "correlation_improves_by_minimum": (
            validation_metrics["correlation"]
            >= pre_validation_metrics["correlation"]
            + args.minimum_validation_correlation_gain
        ),
        "near_plane_mae_improves": (
            validation_metrics["near_plane_mae_m"]
            <= pre_validation_metrics["near_plane_mae_m"]
            * args.maximum_validation_near_mae_ratio
        ),
        "direction_does_not_regress": (
            validation_metrics["direction_accuracy"]
            >= pre_validation_metrics["direction_accuracy"]
            + args.minimum_validation_direction_gain
        ),
        "successful_action_rmse_bounded": (
            validation_action_drift["rmse"] <= args.maximum_action_rmse
        ),
        "successful_action_max_bounded": (
            validation_action_drift["max_abs"] <= args.maximum_action_max_abs
        ),
        "dense_deterministic_rmse_bounded": (
            dense_drift["deterministic"]["rmse"]
            <= args.maximum_dense_feature_rmse
        ),
        "dense_logits_rmse_bounded": (
            dense_drift["logits"]["rmse"] <= args.maximum_dense_feature_rmse
        ),
        "dense_action_rmse_bounded": (
            dense_drift["action"]["rmse"] <= args.maximum_action_rmse
        ),
        "dense_action_max_bounded": (
            dense_drift["action"]["max_abs"] <= args.maximum_action_max_abs
        ),
        "parameter_delta_bounded": (
            postprojection_max_parameter_delta <= args.max_parameter_delta
        ),
    }
    cuda_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    validation_admission_gates["cuda_peak_bounded"] = (
        device.type != "cuda" or cuda_peak < args.maximum_cuda_bytes
    )
    report: dict[str, object] = {
        "contract": CONTINUATION_SCHEMA,
        "source_hashes": source_after,
        "sources_unchanged": source_after == source_before,
        "output_checkpoint": str(args.output_checkpoint),
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "events": int(len(event_split)),
        "train_events": int((event_split == 0).sum()),
        "validation_events": int((event_split == 1).sum()),
        "test_events": int((event_split == 2).sum()),
        "train_crops": int(len(train)),
        "validation_crops": int(len(validation)),
        "test_evaluations": 0,
        "test_observations_evaluated": 0,
        "test_privileged_labels_evaluated": 0,
        "crop_starts": list(crop_starts),
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "evaluation_indices": evaluation_indices.tolist(),
        "boundary_replay": boundary_replay,
        "soft_plane_probe": probe_fit,
        "target_mean": target_mean,
        "target_std": target_std,
        "preupdate_validation_progress": pre_validation_metrics,
        "validation_progress": validation_metrics,
        "validation_action_drift": validation_action_drift,
        "dense_drift": dense_drift,
        "losses": aggregate,
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "gradient_abs_max_by_component": gradient_by_component,
        "learning_rate": args.learning_rate,
        "max_parameter_delta": args.max_parameter_delta,
        "preprojection_max_parameter_delta": preprojection_max_parameter_delta,
        "postprojection_max_parameter_delta": postprojection_max_parameter_delta,
        "projected_parameter_values": projected_parameter_values,
        "changed_model_keys": changed,
        "forbidden_model_key_changes": forbidden,
        "process_gates": process_gates,
        "process_passed": all(process_gates.values()),
        "validation_admission_gates": validation_admission_gates,
        "validation_admitted": (
            all(process_gates.values()) and all(validation_admission_gates.values())
        ),
        "probe_optimizer_steps": args.probe_steps,
        "posterior_representation_updates": 1,
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
        "wall_seconds": time.perf_counter() - start_time,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = cuda_peak
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--n669-report", type=Path, required=True)
    parser.add_argument("--dense-replay-dir", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-starts", default="0,32,64,96,128,160,192")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--evaluation-stride", type=int, default=16)
    parser.add_argument("--near-plane-m", type=float, default=1.0)
    parser.add_argument("--minimum-pair-delta-m", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=668)
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--probe-seed", type=int, default=670)
    parser.add_argument("--probe-hidden-size", type=int, default=128)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--probe-validation-interval", type=int, default=50)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--probe-evaluation-batch-size", type=int, default=32)
    parser.add_argument("--probe-learning-rate", type=float, default=0.0003)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0001)
    parser.add_argument("--probe-gradient-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=670)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-parameter-delta", type=float, default=0.000001)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--success-action-anchor-weight", type=float, default=10.0)
    parser.add_argument("--dense-feature-anchor-weight", type=float, default=1.0)
    parser.add_argument("--dense-action-anchor-weight", type=float, default=10.0)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--dense-sequence-length", type=int, default=12)
    parser.add_argument("--dense-seed", type=int, default=670001)
    parser.add_argument("--reference-microbatch-size", type=int, default=4)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--max-gradient-norm", type=float, default=100.0)
    parser.add_argument("--minimum-validation-correlation-gain", type=float, default=0.01)
    parser.add_argument("--maximum-validation-near-mae-ratio", type=float, default=0.99)
    parser.add_argument("--minimum-validation-direction-gain", type=float, default=0.0)
    parser.add_argument("--maximum-dense-feature-rmse", type=float, default=0.001)
    parser.add_argument("--maximum-action-rmse", type=float, default=0.0001)
    parser.add_argument("--maximum-action-max-abs", type=float, default=0.001)
    parser.add_argument("--maximum-cuda-bytes", type=int, default=7_730_941_132)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for name in (
        "sequence_length", "burn_in", "evaluation_stride", "probe_hidden_size",
        "probe_steps", "probe_validation_interval", "probe_batch_size",
        "probe_evaluation_batch_size", "dense_batch_size", "dense_sequence_length",
        "reference_microbatch_size", "microbatch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.burn_in >= args.sequence_length:
        parser.error("burn-in must be shorter than sequence length")
    if args.max_parameter_delta <= 0.0:
        parser.error("maximum parameter delta must be positive")
    if not 0.0 < args.maximum_validation_near_mae_ratio <= 1.0:
        parser.error("maximum validation near-MAE ratio must lie in (0,1]")
    if args.maximum_dense_feature_rmse <= 0.0:
        parser.error("maximum dense feature RMSE must be positive")
    if args.maximum_action_rmse <= 0.0 or args.maximum_action_max_abs <= 0.0:
        parser.error("maximum action drift bounds must be positive")
    if args.maximum_cuda_bytes <= 0:
        parser.error("maximum CUDA bytes must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_representation(parse_args()), indent=2, sort_keys=True))
