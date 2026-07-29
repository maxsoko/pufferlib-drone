#!/usr/bin/env python3
"""Finetune only the VQ2 world model on source-locked paired causal replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import VQ2InformedDreamer  # noqa: E402
from pufferlib.vq2_informed import ENV_OBS_SIZE, MASK_SIZE  # noqa: E402
from scripts.train_vq2_informed_dreamer import (  # noqa: E402
    QuantizedSequenceReplay,
    _save_checkpoint,
    _world_model_update,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _restore_rng(payload: dict, device: torch.device) -> random.Random:
    state = payload.get("rng_state")
    if not isinstance(state, dict):
        raise RuntimeError("parent checkpoint does not contain RNG state")
    rng = random.Random()
    rng.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        cuda_state = state.get("torch_cuda")
        if not cuda_state:
            raise RuntimeError("parent checkpoint does not contain CUDA RNG state")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])
    return rng


def _anchored_channel_batch(
    replay: QuantizedSequenceReplay,
    *,
    batch_size: int,
    sequence_length: int,
    steps_per_channel: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balance every CTBR channel/value from the exact initial record."""

    channels = 4
    values = 8
    base_batch = channels * values
    if batch_size % base_batch:
        raise ValueError("anchored causal batch size must be a multiple of 32")
    records_per_channel = steps_per_channel + 1
    if sequence_length > records_per_channel:
        raise ValueError("anchored causal sequence exceeds one channel segment")
    expected_capacity = channels * records_per_channel + channels - 1
    if replay.capacity != expected_capacity or replay.size != expected_capacity:
        raise RuntimeError("anchored causal replay layout mismatch")

    observations = []
    actions = []
    rewards = []
    continuations = []
    for _repeat in range(batch_size // base_batch):
        for channel in range(channels):
            logical_start = channel * (records_per_channel + 1)
            physical = replay._physical(
                np.arange(logical_start, logical_start + sequence_length)
            )
            for value in range(values):
                agent = value * (replay.agents // values) + rng.randrange(
                    replay.agents // values
                )
                mask = replay.mask[physical, agent].astype(np.float32) / 255.0
                tail = replay.tail[physical, agent].astype(np.float32)
                observation = np.concatenate((mask, tail), -1)
                if observation.shape != (sequence_length, ENV_OBS_SIZE):
                    raise RuntimeError("anchored causal observation shape mismatch")
                observations.append(observation)
                actions.append(replay.action[physical, agent].astype(np.float32))
                rewards.append(replay.reward[physical, agent].astype(np.float32))
                continuations.append(
                    replay.continuation[physical, agent].astype(np.float32)
                )
    return tuple(
        torch.from_numpy(np.stack(value)).to(device)
        for value in (observations, actions, rewards, continuations)
    )


def finetune(args: argparse.Namespace) -> dict:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("causal finetune must not overwrite its parent")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    replay_hashes_before = _replay_hashes(args.replay_dir)
    parent_sha256 = _sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    if not isinstance(training_args, dict):
        raise RuntimeError("parent checkpoint does not contain training arguments")
    if int(payload.get("actor_updates", 0)) != 0:
        raise RuntimeError("causal world finetune refuses a parent with actor updates")

    model = VQ2InformedDreamer(
        deterministic_size=int(training_args["deterministic_size"]),
        stochastic_groups=int(training_args["stochastic_groups"]),
        stochastic_classes=int(training_args["stochastic_classes"]),
        actor_initial_std=float(training_args.get("actor_initial_std", 0.20)),
        distributional_reward=bool(training_args.get("distributional_reward", False)),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(payload["model"])
    parent_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    world_parameters = [
        *model.rssm.parameters(),
        *model.privileged_decoder.parameters(),
        *model.reward_predictor.parameters(),
        *model.continue_predictor.parameters(),
    ]
    optimizer = torch.optim.AdamW(world_parameters, lr=args.learning_rate)

    metadata = json.loads((args.replay_dir / "metadata.json").read_text())
    replay = QuantizedSequenceReplay(
        int(metadata["capacity"]),
        int(metadata["agents"]),
        storage_dir=args.replay_dir,
        resume=True,
    )
    if replay.size != int(metadata["size"]):
        raise RuntimeError("causal replay size mismatch")
    sample_length = args.replay_context + args.world_sequence_length
    if sample_length > replay.capacity:
        raise ValueError("causal sequence exceeds replay capacity")
    rng = _restore_rng(payload, device)
    last_loss = None
    last_gradient = None
    microbatches = 0
    started = time.perf_counter()
    for update in range(args.updates):
        if args.anchored_channel_steps:
            observation, action, reward, continuation = _anchored_channel_batch(
                replay,
                batch_size=args.batch_size,
                sequence_length=sample_length,
                steps_per_channel=args.anchored_channel_steps,
                device=device,
                rng=rng,
            )
        else:
            observation, action, reward, continuation = replay.sample(
                args.batch_size, sample_length, device=device, rng=rng
            )
        last_loss, _start, last_gradient, count = _world_model_update(
            model=model,
            optimizer=optimizer,
            observation=observation,
            action=action,
            reward=reward,
            continuation=continuation,
            replay_context=args.replay_context,
            microbatch_size=args.microbatch_size,
            amp_dtype=args.amp_dtype,
            free_nats=args.free_nats,
            world_parameters=world_parameters,
            deterministic_latent=True,
        )
        microbatches += count
        if args.progress_every and (update + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "causal_world_updates": update + 1,
                        "world_total": float(last_loss.total.detach().cpu()),
                        "raw_kl": float(last_loss.dynamics_kl_raw.detach().cpu()),
                        "wall_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    assert last_loss is not None and last_gradient is not None
    child_model_state = model.state_dict()
    changed_model_keys = sorted(
        name
        for name, value in child_model_state.items()
        if not torch.equal(value, parent_model_state[name])
    )
    allowed_prefixes = (
        "rssm.",
        "privileged_decoder.",
        "reward_predictor.",
        "continue_predictor.",
    )
    unexpected = [
        name
        for name in changed_model_keys
        if not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise RuntimeError(f"causal finetune changed policy/value tensors: {unexpected}")
    if not changed_model_keys:
        raise RuntimeError("causal finetune changed no world-model tensors")
    replay_hashes_after = _replay_hashes(args.replay_dir)
    if replay_hashes_after != replay_hashes_before:
        raise RuntimeError("paired causal replay changed during finetune")

    elapsed = time.perf_counter() - started
    previous_causal_updates = int(payload.get("causal_world_updates", 0))
    starting_world_updates = int(payload.get("updates", 0))
    report: dict[str, object] = {
        "parent_checkpoint": str(args.checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "causal_replay_dir": str(args.replay_dir),
        "causal_replay_hashes": replay_hashes_after,
        "causal_replay_unchanged": 1,
        "general_replay_reference_preserved": 1,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "causal_world_updates": args.updates,
        "cumulative_causal_world_updates": previous_causal_updates + args.updates,
        "starting_world_updates": starting_world_updates,
        "ending_world_updates": starting_world_updates + args.updates,
        "changed_model_keys": changed_model_keys,
        "changed_model_key_count": len(changed_model_keys),
        "forbidden_policy_value_key_changes": len(unexpected),
        "optimizer_reset_for_causal_replay": 1,
        "replay_context": args.replay_context,
        "world_sequence_length": args.world_sequence_length,
        "logical_batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "microbatches_processed": microbatches,
        "free_nats": args.free_nats,
        "learning_rate": args.learning_rate,
        "last_world_gradient_norm": float(last_gradient.detach().cpu()),
        "wall_seconds": elapsed,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "flightsim_packets": 0,
        "anchored_channel_steps": args.anchored_channel_steps,
    }
    for field in fields(last_loss):
        report[f"world_{field.name}"] = float(
            getattr(last_loss, field.name).detach().cpu()
        )
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )

    output = dict(payload)
    output["args"] = dict(training_args)
    output["args"]["free_nats"] = args.free_nats
    output["model"] = child_model_state
    output["world_optimizer"] = optimizer.state_dict()
    output["updates"] = starting_world_updates + args.updates
    output["causal_world_updates"] = previous_causal_updates + args.updates
    output["causal_world_parent_checkpoint"] = str(args.checkpoint)
    output["causal_world_parent_sha256"] = parent_sha256
    output["causal_replay_dir"] = str(args.replay_dir)
    output["rng_state"] = {
        "python": rng.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    output["metrics"] = report
    _save_checkpoint(args.output_checkpoint, output)
    report["output_checkpoint"] = str(args.output_checkpoint)
    report["output_checkpoint_sha256"] = _sha256(args.output_checkpoint)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--replay-context", type=int, default=16)
    parser.add_argument("--world-sequence-length", type=int, default=32)
    parser.add_argument(
        "--anchored-channel-steps",
        type=int,
        default=0,
        help="nonzero uses balanced reset-anchored channel/value sequences",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--free-nats", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")
    if args.replay_context < 0 or args.world_sequence_length <= 0:
        parser.error("invalid causal replay lengths")
    if args.anchored_channel_steps < 0:
        parser.error("--anchored-channel-steps cannot be negative")
    if args.anchored_channel_steps and args.replay_context != 0:
        parser.error("anchored causal replay requires --replay-context 0")
    if not 1 <= args.microbatch_size <= args.batch_size:
        parser.error("--microbatch-size must lie in [1,batch-size]")
    if args.free_nats < 0.0 or args.learning_rate <= 0.0:
        parser.error("invalid free nats or learning rate")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    return args


if __name__ == "__main__":
    print(json.dumps(finetune(parse_args()), indent=2, sort_keys=True))
