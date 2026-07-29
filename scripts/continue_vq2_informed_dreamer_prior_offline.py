#!/usr/bin/env python3
"""Distill only the VQ2 RSSM prior from frozen visual posteriors.

This executable has no native environment, collection, reward fitting, critic,
or actor path. It samples a source-locked replay and applies unclamped dynamics
KL only to ``rssm.prior.*``. All other model tensors and replay files are
verified unchanged before a child checkpoint is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import VQ2InformedDreamer  # noqa: E402
from scripts.train_vq2_informed_dreamer import (  # noqa: E402
    QuantizedSequenceReplay,
    _autocast_context,
    _burn_in_state,
    _finite_clip_grad_norm,
    _save_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): _sha256(path)
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


def continue_prior(args: argparse.Namespace) -> dict:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("prior distillation must not overwrite its parent")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    replay_hashes_before = _replay_hashes(args.replay_dir)
    parent_sha256 = _sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    if not isinstance(training_args, dict):
        raise RuntimeError("parent checkpoint does not contain training arguments")
    if payload.get("actor_updates", 0) != 0:
        raise RuntimeError("prior admission refuses a parent with actor updates")

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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prior_parameters = list(model.rssm.prior.parameters())
    for parameter in prior_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(prior_parameters, lr=args.learning_rate)

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
    microbatch_size = int(training_args["world_microbatch_size"])
    amp_dtype = str(training_args["amp_dtype"])
    last_kl = None
    last_gradient = None
    microbatches = 0
    started = time.perf_counter()

    for update in range(args.updates):
        observation, action, _reward, _continuation = replay.sample(
            batch_size, sample_length, device=device, rng=rng
        )
        optimizer.zero_grad(set_to_none=True)
        logical_kl = torch.zeros((), device=device)
        for start in range(0, batch_size, microbatch_size):
            stop = min(start + microbatch_size, batch_size)
            weight = (stop - start) / batch_size
            with _autocast_context(device, amp_dtype):
                initial_state = _burn_in_state(
                    model,
                    observation[start:stop, :replay_context],
                    action[start:stop, :replay_context],
                )
                output = model.observe_sequence(
                    observation[start:stop, replay_context:],
                    action[start:stop, replay_context:],
                    initial_state=initial_state,
                    deterministic_latent=True,
                )
                kl = model._categorical_kl(
                    output.posterior.logits.detach(), output.prior_logits
                ).mean()
                weighted_kl = kl * weight
            weighted_kl.backward()
            logical_kl += kl.detach().float() * weight
            microbatches += 1
        last_gradient = _finite_clip_grad_norm(
            prior_parameters, 100.0, label="RSSM-prior"
        )
        optimizer.step()
        last_kl = logical_kl
        if args.progress_every and (update + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "offline_prior_updates": update + 1,
                        "prior_raw_kl": float(last_kl.cpu()),
                        "wall_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    assert last_kl is not None and last_gradient is not None
    child_model_state = model.state_dict()
    changed_model_keys = sorted(
        name
        for name, value in child_model_state.items()
        if not torch.equal(value, parent_model_state[name])
    )
    if not changed_model_keys:
        raise RuntimeError("prior distillation did not change any prior tensor")
    unexpected = [
        name for name in changed_model_keys if not name.startswith("rssm.prior.")
    ]
    if unexpected:
        raise RuntimeError(f"prior distillation changed forbidden tensors: {unexpected}")
    replay_hashes_after = _replay_hashes(args.replay_dir)
    if replay_hashes_after != replay_hashes_before:
        raise RuntimeError("source-locked replay changed during prior distillation")

    elapsed = time.perf_counter() - started
    previous_prior_updates = int(payload.get("offline_prior_updates", 0))
    report: dict[str, object] = {
        "parent_checkpoint": str(args.checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "replay_dir": str(args.replay_dir),
        "replay_hashes": replay_hashes_after,
        "replay_records_unchanged": 1,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "world_optimizer_steps": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "offline_prior_updates": args.updates,
        "cumulative_offline_prior_updates": previous_prior_updates + args.updates,
        "prior_raw_kl": float(last_kl.cpu()),
        "last_prior_gradient_norm": float(last_gradient.detach().cpu()),
        "changed_model_keys": changed_model_keys,
        "changed_model_key_count": len(changed_model_keys),
        "forbidden_model_key_changes": 0,
        "world_optimizer_preserved": 1,
        "actor_optimizer_preserved": 1,
        "critic_optimizer_preserved": 1,
        "deterministic_posterior_targets": 1,
        "free_nats": 0.0,
        "learning_rate": args.learning_rate,
        "logical_batch_size": batch_size,
        "world_sequence_length": world_length,
        "microbatches_processed": microbatches,
        "wall_seconds": elapsed,
        "device": str(device),
        "amp_dtype": amp_dtype,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )

    output = dict(payload)
    output["model"] = child_model_state
    output["offline_prior_optimizer"] = optimizer.state_dict()
    output["offline_prior_updates"] = previous_prior_updates + args.updates
    output["offline_prior_parent_checkpoint"] = str(args.checkpoint)
    output["offline_prior_parent_sha256"] = parent_sha256
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
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_prior(parse_args()), indent=2, sort_keys=True))
