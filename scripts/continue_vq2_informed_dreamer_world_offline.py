#!/usr/bin/env python3
"""Continue only a VQ2 Informed-Dreamer world model from frozen replay.

This executable creates no native vector environment and has no policy
collection path. It reuses a source-locked memory-mapped replay segment,
preserves actor/critic optimizer state byte-for-byte, and advances only the
world model optimizer before writing a new checkpoint.
"""

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


def continue_world(args: argparse.Namespace) -> dict:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("offline continuation must not overwrite its parent")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parent_sha256 = _sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    if not isinstance(training_args, dict):
        raise RuntimeError("parent checkpoint does not contain training arguments")
    if payload.get("actor_updates", 0) != 0:
        raise RuntimeError("world-only admission refuses a parent with actor updates")

    model = VQ2InformedDreamer(
        deterministic_size=int(training_args["deterministic_size"]),
        stochastic_groups=int(training_args["stochastic_groups"]),
        stochastic_classes=int(training_args["stochastic_classes"]),
        actor_initial_std=float(training_args.get("actor_initial_std", 0.20)),
        distributional_reward=bool(
            training_args.get("distributional_reward", False)
        ),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(payload["model"])
    world_parameters = [
        *model.rssm.parameters(),
        *model.privileged_decoder.parameters(),
        *model.reward_predictor.parameters(),
        *model.continue_predictor.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        world_parameters, lr=float(training_args["world_lr"])
    )
    optimizer.load_state_dict(payload["world_optimizer"])

    replay_state = payload.get("replay")
    if not isinstance(replay_state, dict):
        raise RuntimeError("parent checkpoint does not contain replay state")
    replay = QuantizedSequenceReplay(
        int(replay_state["capacity"]),
        int(replay_state["agents"]),
        storage_dir=args.replay_dir,
        resume=True,
    )
    observed_replay = replay.state_dict()
    for key in ("schema", "capacity", "agents", "position", "size"):
        if observed_replay[key] != replay_state[key]:
            raise RuntimeError(f"source-locked replay {key} mismatch")

    rng = _restore_rng(payload, device)
    replay_context = int(training_args["replay_context"])
    world_length = int(training_args["world_sequence_length"])
    sample_length = replay_context + world_length
    batch_size = int(training_args["batch_size"])
    microbatch = int(training_args["world_microbatch_size"])
    amp_dtype = str(training_args["amp_dtype"])
    free_nats = (
        float(args.free_nats)
        if args.free_nats is not None
        else float(training_args.get("free_nats", 1.0))
    )
    last_loss = None
    last_gradient = None
    microbatches = 0
    started = time.perf_counter()
    for update in range(args.updates):
        observation, action, reward, continuation = replay.sample(
            batch_size, sample_length, device=device, rng=rng
        )
        last_loss, _, last_gradient, count = _world_model_update(
            model=model,
            optimizer=optimizer,
            observation=observation,
            action=action,
            reward=reward,
            continuation=continuation,
            replay_context=replay_context,
            microbatch_size=microbatch,
            amp_dtype=amp_dtype,
            free_nats=free_nats,
            world_parameters=world_parameters,
        )
        microbatches += count
        if args.progress_every and (update + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "offline_world_updates": update + 1,
                        "world_total": float(last_loss.total.detach().cpu()),
                        "wall_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    assert last_loss is not None and last_gradient is not None
    elapsed = time.perf_counter() - started
    starting_updates = int(payload.get("updates", 0))
    prior_offline_updates = int(payload.get("offline_world_updates", 0))
    report: dict[str, float | int | str] = {
        "parent_checkpoint": str(args.checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "replay_dir": str(args.replay_dir),
        "replay_records_unchanged": 1,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "offline_world_updates": args.updates,
        "cumulative_offline_world_updates": prior_offline_updates + args.updates,
        "starting_world_updates": starting_updates,
        "ending_world_updates": starting_updates + args.updates,
        "world_microbatches_processed": microbatches,
        "wall_seconds": elapsed,
        "last_world_gradient_norm": float(last_gradient.detach().cpu()),
        "device": str(device),
        "amp_dtype": amp_dtype,
        "free_nats": free_nats,
        "distributional_reward": int(
            bool(training_args.get("distributional_reward", False))
        ),
    }
    for field in fields(last_loss):
        report[f"world_{field.name}"] = float(
            getattr(last_loss, field.name).detach().cpu()
        )
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device)
        )

    output = dict(payload)
    output["args"] = dict(training_args)
    output["args"]["free_nats"] = free_nats
    output["model"] = model.state_dict()
    output["world_optimizer"] = optimizer.state_dict()
    output["updates"] = starting_updates + args.updates
    output["offline_world_updates"] = prior_offline_updates + args.updates
    output["offline_parent_checkpoint"] = str(args.checkpoint)
    output["offline_parent_sha256"] = parent_sha256
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
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--free-nats",
        type=float,
        help="optional source-locked override for compact-model KL supervision",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    if args.free_nats is not None and args.free_nats < 0.0:
        parser.error("--free-nats cannot be negative")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_world(parse_args()), indent=2, sort_keys=True))
