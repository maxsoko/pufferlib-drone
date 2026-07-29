#!/usr/bin/env python3
"""Train only the VQ2 one-step ordered-phase transition offline.

The event corpus supplies true current-policy phase transitions. Reward,
continuation, visual encoder/posterior, actor, critic, and every non-phase
privileged output are frozen. Privileged phase remains a training-only target.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import VQ2InformedDreamer
from pufferlib.vq2_informed import PRIVILEGED_SIZE
from scripts.continue_vq2_event_balanced_world_offline import (
    _event_batch,
    _load_event_dataset,
    _mix_batch,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import (
    _replay_hashes,
    _restore_rng,
    _sha256,
)
from scripts.train_vq2_informed_dreamer import (
    QuantizedSequenceReplay,
    _autocast_context,
    _burn_in_state,
    _finite_clip_grad_norm,
    _save_checkpoint,
)


PHASE_INDEX = 32


def _row_mask(shape: torch.Size, row: int) -> torch.Tensor:
    if len(shape) < 1 or not 0 <= row < shape[0]:
        raise ValueError("phase row is out of range")
    mask = torch.zeros(shape)
    mask[row] = 1.0
    return mask


def _configure_phase_parameters(
    model: VQ2InformedDreamer,
) -> tuple[list[torch.nn.Parameter], list[torch.utils.hooks.RemovableHandle]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = [*model.rssm.sequence.parameters(), *model.rssm.prior.parameters()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    head = model.privileged_decoder[-1]
    if not isinstance(head, torch.nn.Linear):
        raise TypeError("privileged decoder output head must be linear")
    head.weight.requires_grad_(True)
    head.bias.requires_grad_(True)
    weight_mask = _row_mask(head.weight.shape, PHASE_INDEX).to(head.weight.device)
    bias_mask = _row_mask(head.bias.shape, PHASE_INDEX).to(head.bias.device)
    handles = [
        head.weight.register_hook(lambda gradient: gradient * weight_mask),
        head.bias.register_hook(lambda gradient: gradient * bias_mask),
    ]
    parameters.extend((head.weight, head.bias))
    return parameters, handles


def _phase_targets(
    observation: torch.Tensor,
    *,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = observation[:, context_length - 1, -PRIVILEGED_SIZE + PHASE_INDEX]
    next_phase = observation[:, context_length, -PRIVILEGED_SIZE + PHASE_INDEX]
    return current, next_phase


def _crossing_loss(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    logit_scale: float,
) -> torch.Tensor:
    if logit_scale <= 0.0:
        raise ValueError("crossing logit scale must be positive")
    threshold = 0.5 / 6.0
    target = (target_delta >= threshold).to(predicted_delta.dtype)
    logits = (predicted_delta - threshold) * logit_scale
    return F.binary_cross_entropy_with_logits(logits, target)


@torch.no_grad()
def _phase_metrics(
    model: VQ2InformedDreamer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    context_length: int,
    microbatch_size: int,
    amp_dtype: str,
) -> dict[str, float]:
    observation, action, _reward, _continuation = batch
    predicted_current: list[torch.Tensor] = []
    predicted_next: list[torch.Tensor] = []
    target_current: list[torch.Tensor] = []
    target_next: list[torch.Tensor] = []
    for start in range(0, observation.shape[0], microbatch_size):
        stop = min(start + microbatch_size, observation.shape[0])
        with _autocast_context(observation.device, amp_dtype):
            state = _burn_in_state(
                model,
                observation[start:stop, :context_length],
                action[start:stop, :context_length],
            )
            current_phase = model.privileged_decoder(state.features)[..., PHASE_INDEX]
            prior = model.rssm.imagine_step(
                state,
                action[start:stop, context_length],
                deterministic_latent=True,
            )
            next_phase = model.privileged_decoder(prior.features)[..., PHASE_INDEX]
        current_target, next_target = _phase_targets(
            observation[start:stop], context_length=context_length
        )
        predicted_current.append(current_phase.float().cpu())
        predicted_next.append(next_phase.float().cpu())
        target_current.append(current_target.float().cpu())
        target_next.append(next_target.float().cpu())
    predicted_current_t = torch.cat(predicted_current)
    predicted_next_t = torch.cat(predicted_next)
    target_current_t = torch.cat(target_current)
    target_next_t = torch.cat(target_next)
    predicted_delta = predicted_next_t - predicted_current_t
    target_delta = target_next_t - target_current_t
    threshold = 0.5 / 6.0
    return {
        "next_phase_mae": float((predicted_next_t - target_next_t).abs().mean()),
        "phase_delta_mae": float((predicted_delta - target_delta).abs().mean()),
        "predicted_crossing_fraction": float(
            (predicted_delta >= threshold).float().mean()
        ),
        "target_crossing_fraction": float((target_delta >= threshold).float().mean()),
        "predicted_next_phase_mean": float(predicted_next_t.mean()),
        "target_next_phase_mean": float(target_next_t.mean()),
        "predicted_delta_mean": float(predicted_delta.mean()),
        "target_delta_mean": float(target_delta.mean()),
    }


def _phase_update(
    model: VQ2InformedDreamer,
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    context_length: int,
    microbatch_size: int,
    amp_dtype: str,
    crossing_bce_weight: float,
    crossing_logit_scale: float,
) -> tuple[float, float, int]:
    observation, action, _reward, _continuation = batch
    optimizer.zero_grad(set_to_none=True)
    logical_loss = torch.zeros((), device=observation.device)
    microbatches = 0
    for start in range(0, observation.shape[0], microbatch_size):
        stop = min(start + microbatch_size, observation.shape[0])
        weight = (stop - start) / observation.shape[0]
        with torch.no_grad(), _autocast_context(observation.device, amp_dtype):
            state = _burn_in_state(
                model,
                observation[start:stop, :context_length],
                action[start:stop, :context_length],
            )
            current_phase = model.privileged_decoder(state.features)[..., PHASE_INDEX]
        with _autocast_context(observation.device, amp_dtype):
            prior = model.rssm.imagine_step(
                state.detached(),
                action[start:stop, context_length],
                deterministic_latent=False,
            )
            next_phase = model.privileged_decoder(prior.features)[..., PHASE_INDEX]
            target_current, target_next = _phase_targets(
                observation[start:stop], context_length=context_length
            )
            target_delta = target_next - target_current
            predicted_delta = next_phase - current_phase.detach()
            loss = F.smooth_l1_loss(next_phase, target_next) + F.smooth_l1_loss(
                predicted_delta, target_delta
            )
            if crossing_bce_weight > 0.0:
                loss = loss + crossing_bce_weight * _crossing_loss(
                    predicted_delta,
                    target_delta,
                    logit_scale=crossing_logit_scale,
                )
            weighted_loss = loss * weight
        weighted_loss.backward()
        logical_loss += loss.detach().float() * weight
        microbatches += 1
    gradient_norm = _finite_clip_grad_norm(parameters, 100.0, label="phase-transition")
    optimizer.step()
    return float(logical_loss.cpu()), float(gradient_norm.cpu()), microbatches


def continue_phase(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("phase continuation cannot overwrite its parent")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parent_sha = _sha256(args.checkpoint)
    replay_hashes_before = _replay_hashes(args.replay_dir)
    event_sha = _sha256(args.event_dataset)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    replay_state = payload.get("replay")
    if not isinstance(training_args, dict) or not isinstance(replay_state, dict):
        raise RuntimeError("parent checkpoint lacks training or replay state")
    if int(training_args["replay_context"]) != args.context_length:
        raise RuntimeError("phase context mismatch")
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
    parent_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    parameters, handles = _configure_phase_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)

    replay = QuantizedSequenceReplay(
        int(replay_state["capacity"]),
        int(replay_state["agents"]),
        storage_dir=args.replay_dir,
        resume=True,
    )
    observed = replay.state_dict()
    for key in ("schema", "capacity", "agents", "position", "size"):
        if observed[key] != replay_state[key]:
            raise RuntimeError(f"source replay {key} mismatch")
    dataset = _load_event_dataset(
        args.event_dataset,
        context_length=args.context_length,
        event_threshold=args.event_threshold,
    )
    dense_count = args.batch_size - args.event_batch_size
    if dense_count <= 0:
        raise ValueError("phase update requires dense samples")
    rng = _restore_rng(payload, device)
    validation_rng = random.Random()
    validation_rng.setstate(rng.getstate())
    dense_validation = replay.sample(
        args.batch_size,
        args.context_length + 1,
        device=device,
        rng=validation_rng,
    )
    event_validation = _event_batch(
        dataset, np.arange(dataset["mask"].shape[0]), device=device
    )
    pre_event = _phase_metrics(
        model,
        event_validation,
        context_length=args.context_length,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
    )
    pre_dense = _phase_metrics(
        model,
        dense_validation,
        context_length=args.context_length,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
    )
    started = time.perf_counter()
    last_loss = 0.0
    last_gradient = 0.0
    microbatches = 0
    try:
        for update in range(args.updates):
            indices = rng.sample(range(dataset["mask"].shape[0]), args.event_batch_size)
            event_batch = _event_batch(dataset, indices, device=device)
            dense_batch = replay.sample(
                dense_count,
                args.context_length + 1,
                device=device,
                rng=rng,
            )
            batch = _mix_batch(
                dense_batch,
                event_batch,
                permutation=torch.randperm(args.batch_size, device=device),
            )
            last_loss, last_gradient, count = _phase_update(
                model,
                optimizer,
                parameters,
                batch,
                context_length=args.context_length,
                microbatch_size=args.microbatch_size,
                amp_dtype=args.amp_dtype,
                crossing_bce_weight=args.crossing_bce_weight,
                crossing_logit_scale=args.crossing_logit_scale,
            )
            microbatches += count
    finally:
        for handle in handles:
            handle.remove()
    post_event = _phase_metrics(
        model,
        event_validation,
        context_length=args.context_length,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
    )
    post_dense = _phase_metrics(
        model,
        dense_validation,
        context_length=args.context_length,
        microbatch_size=args.microbatch_size,
        amp_dtype=args.amp_dtype,
    )
    child_state = model.state_dict()
    changed = sorted(
        name for name, value in child_state.items() if not torch.equal(value, parent_state[name])
    )
    allowed = (
        "rssm.sequence.",
        "rssm.prior.",
        "privileged_decoder.5.weight",
        "privileged_decoder.5.bias",
    )
    forbidden = [name for name in changed if not name.startswith(allowed)]
    head = model.privileged_decoder[-1]
    parent_weight = parent_state["privileged_decoder.5.weight"]
    parent_bias = parent_state["privileged_decoder.5.bias"]
    nonphase_decoder_rows_changed = int(
        not torch.equal(head.weight[:PHASE_INDEX], parent_weight[:PHASE_INDEX])
        or not torch.equal(head.weight[PHASE_INDEX + 1 :], parent_weight[PHASE_INDEX + 1 :])
        or not torch.equal(head.bias[:PHASE_INDEX], parent_bias[:PHASE_INDEX])
        or not torch.equal(head.bias[PHASE_INDEX + 1 :], parent_bias[PHASE_INDEX + 1 :])
    )
    if not changed or forbidden or nonphase_decoder_rows_changed:
        raise RuntimeError(
            f"phase continuation tensor isolation failed: {forbidden}, rows={nonphase_decoder_rows_changed}"
        )
    if _sha256(args.checkpoint) != parent_sha:
        raise RuntimeError("phase continuation changed parent checkpoint")
    if _replay_hashes(args.replay_dir) != replay_hashes_before:
        raise RuntimeError("phase continuation changed dense replay")
    if _sha256(args.event_dataset) != event_sha:
        raise RuntimeError("phase continuation changed event corpus")

    previous = int(payload.get("offline_phase_transition_updates", 0))
    report: dict[str, object] = {
        "contract": "vq2_phase_transition_only_v1",
        "parent_checkpoint": str(args.checkpoint),
        "parent_checkpoint_sha256": parent_sha,
        "dense_replay_hashes": replay_hashes_before,
        "dense_replay_unchanged": 1,
        "event_dataset_sha256": event_sha,
        "event_dataset_unchanged": 1,
        "updates": args.updates,
        "cumulative_offline_phase_transition_updates": previous + args.updates,
        "batch_size": args.batch_size,
        "event_batch_size": args.event_batch_size,
        "dense_batch_size": dense_count,
        "learning_rate": args.learning_rate,
        "crossing_bce_weight": args.crossing_bce_weight,
        "crossing_logit_scale": args.crossing_logit_scale,
        "context_length": args.context_length,
        "microbatch_size": args.microbatch_size,
        "microbatches_processed": microbatches,
        "amp_dtype": args.amp_dtype,
        "pre_event_phase": pre_event,
        "post_event_phase": post_event,
        "pre_dense_phase": pre_dense,
        "post_dense_phase": post_dense,
        "last_phase_loss": last_loss,
        "last_gradient_norm": last_gradient,
        "changed_model_keys": changed,
        "forbidden_model_key_changes": forbidden,
        "nonphase_decoder_rows_changed": nonphase_decoder_rows_changed,
        "reward_tensor_changes": sum(name.startswith("reward_predictor.") for name in changed),
        "continuation_tensor_changes": sum(name.startswith("continue_predictor.") for name in changed),
        "encoder_tensor_changes": sum(name.startswith("rssm.encoder.") for name in changed),
        "posterior_tensor_changes": sum(name.startswith("rssm.posterior.") for name in changed),
        "actor_tensor_changes": sum(name.startswith("actor.") for name in changed),
        "critic_tensor_changes": sum(name.startswith(("critic.", "target_critic.")) for name in changed),
        "actor_updates": 0,
        "critic_updates": 0,
        "reward_updates": 0,
        "continuation_updates": 0,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "flightsim_packets": 0,
        "privileged_actor_input": 0,
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    output = dict(payload)
    output["model"] = child_state
    output["offline_phase_transition_optimizer"] = optimizer.state_dict()
    output["offline_phase_transition_updates"] = previous + args.updates
    output["offline_phase_transition_parent_sha256"] = parent_sha
    output["offline_phase_transition_event_dataset_sha256"] = event_sha
    output["rng_state"] = {
        "python": rng.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    output["metrics"] = report
    _save_checkpoint(args.output_checkpoint, output)
    report["output_checkpoint"] = str(args.output_checkpoint)
    report["output_checkpoint_sha256"] = _sha256(args.output_checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--event-dataset", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--event-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--crossing-bce-weight", type=float, default=0.0)
    parser.add_argument("--crossing-logit-scale", type=float, default=20.0)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for name in ("updates", "batch_size", "event_batch_size", "context_length", "microbatch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.event_batch_size >= args.batch_size:
        parser.error("--event-batch-size must leave dense samples")
    if args.microbatch_size > args.batch_size:
        parser.error("--microbatch-size cannot exceed batch size")
    if args.learning_rate <= 0.0 or args.event_threshold <= 0.0:
        parser.error("learning rate and event threshold must be positive")
    if args.crossing_bce_weight < 0.0 or args.crossing_logit_scale <= 0.0:
        parser.error("crossing BCE weight/scale are invalid")
    if args.output_checkpoint.exists() or args.report.exists():
        parser.error("phase continuation refuses to overwrite outputs")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_phase(parse_args()), indent=2, sort_keys=True))
