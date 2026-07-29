#!/usr/bin/env python3
"""Calibrate only the VQ2 action-conditioned reward head on causal replay.

The RSSM, visual encoder/decoder, continuation model, actor, critic, replay,
and native environment remain frozen. Balanced reset-anchored sequences expose
the reward head to every tested CTBR channel/value without spending simulator
or actor-learning budget.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_dreamer import (  # noqa: E402
    RSSMState,
    VQ2InformedDreamer,
    symlog,
    twohot_cross_entropy,
)
from scripts.finetune_vq2_informed_dreamer_causal_world import (  # noqa: E402
    _anchored_channel_batch,
)
from scripts.train_vq2_informed_dreamer import (  # noqa: E402
    QuantizedSequenceReplay,
    _autocast_context,
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


def _reward_loss(
    model: VQ2InformedDreamer,
    features: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    *,
    target_scale: float = 1.0,
) -> torch.Tensor:
    if target_scale <= 0.0:
        raise ValueError("reward target scale must be positive")
    reward = reward * target_scale
    logits = model.reward_logits(features, action)
    if model.distributional_reward:
        assert model.reward_bins is not None
        return twohot_cross_entropy(logits, reward, model.reward_bins)
    return F.smooth_l1_loss(logits.squeeze(-1), symlog(reward))


def _prior_features(
    model: VQ2InformedDreamer,
    posterior: RSSMState,
    prior_logits: torch.Tensor,
) -> torch.Tensor:
    prior_stochastic = model.rssm._latent(  # noqa: SLF001
        prior_logits, deterministic=True
    )
    return RSSMState(
        posterior.deterministic, prior_stochastic, prior_logits
    ).features


def continue_reward(args: argparse.Namespace) -> dict:
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        raise ValueError("reward calibration must not overwrite its parent")
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
        raise RuntimeError("reward calibration refuses a parent with actor updates")

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
    if not model.action_conditioned_reward:
        raise RuntimeError("causal calibration requires an action-conditioned reward")
    parent_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reward_parameters = list(model.reward_predictor.parameters())
    for parameter in reward_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(reward_parameters, lr=args.learning_rate)

    metadata = json.loads((args.replay_dir / "metadata.json").read_text())
    replay = QuantizedSequenceReplay(
        int(metadata["capacity"]),
        int(metadata["agents"]),
        storage_dir=args.replay_dir,
        resume=True,
    )
    if replay.size != int(metadata["size"]):
        raise RuntimeError("causal replay size mismatch")
    rng = _restore_rng(payload, device)
    last_loss = None
    last_gradient = None
    microbatches = 0
    started = time.perf_counter()

    for update in range(args.updates):
        observation, action, reward, _continuation = _anchored_channel_batch(
            replay,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            steps_per_channel=args.anchored_channel_steps,
            device=device,
            rng=rng,
        )
        optimizer.zero_grad(set_to_none=True)
        logical_loss = torch.zeros((), device=device)
        for start in range(0, args.batch_size, args.microbatch_size):
            stop = min(start + args.microbatch_size, args.batch_size)
            weight = (stop - start) / args.batch_size
            with torch.no_grad(), _autocast_context(device, args.amp_dtype):
                output = model.observe_sequence(
                    observation[start:stop],
                    action[start:stop],
                    deterministic_latent=True,
                )
                posterior_features = output.posterior.features.detach()
                prior_features = _prior_features(
                    model, output.posterior, output.prior_logits
                ).detach()
            with _autocast_context(device, args.amp_dtype):
                loss = _reward_loss(
                    model,
                    posterior_features,
                    action[start:stop],
                    reward[start:stop],
                    target_scale=args.reward_target_scale,
                )
                if args.latent_source == "posterior_and_prior":
                    loss = 0.5 * (
                        loss
                        + _reward_loss(
                            model,
                            prior_features,
                            action[start:stop],
                            reward[start:stop],
                            target_scale=args.reward_target_scale,
                        )
                    )
                weighted_loss = loss * weight
            weighted_loss.backward()
            logical_loss += loss.detach().float() * weight
            microbatches += 1
        last_gradient = _finite_clip_grad_norm(
            reward_parameters, 100.0, label="reward-head"
        )
        optimizer.step()
        last_loss = logical_loss
        if args.progress_every and (update + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "offline_reward_updates": update + 1,
                        "reward_loss": float(last_loss.cpu()),
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
    if not changed_model_keys:
        raise RuntimeError("reward calibration changed no reward tensor")
    unexpected = [
        name
        for name in changed_model_keys
        if not name.startswith("reward_predictor.")
    ]
    if unexpected:
        raise RuntimeError(f"reward calibration changed forbidden tensors: {unexpected}")
    replay_hashes_after = _replay_hashes(args.replay_dir)
    if replay_hashes_after != replay_hashes_before:
        raise RuntimeError("source-locked causal replay changed during calibration")

    elapsed = time.perf_counter() - started
    previous_updates = int(payload.get("offline_reward_updates", 0))
    report: dict[str, object] = {
        "parent_checkpoint": str(args.checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "causal_replay_dir": str(args.replay_dir),
        "causal_replay_hashes": replay_hashes_after,
        "causal_replay_unchanged": 1,
        "native_environment_created": 0,
        "collected_transitions": 0,
        "world_optimizer_steps": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "offline_reward_updates": args.updates,
        "cumulative_offline_reward_updates": previous_updates + args.updates,
        "changed_model_keys": changed_model_keys,
        "changed_model_key_count": len(changed_model_keys),
        "forbidden_model_key_changes": len(unexpected),
        "world_optimizer_preserved": 1,
        "actor_optimizer_preserved": 1,
        "critic_optimizer_preserved": 1,
        "deterministic_latents": 1,
        "latent_source": args.latent_source,
        "distributional_reward": int(model.distributional_reward),
        "action_conditioned_reward": int(model.action_conditioned_reward),
        "anchored_channel_steps": args.anchored_channel_steps,
        "sequence_length": args.sequence_length,
        "logical_batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "microbatches_processed": microbatches,
        "learning_rate": args.learning_rate,
        "reward_target_scale": args.reward_target_scale,
        "last_reward_loss": float(last_loss.cpu()),
        "last_reward_gradient_norm": float(last_gradient.detach().cpu()),
        "wall_seconds": elapsed,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "flightsim_packets": 0,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )

    output = dict(payload)
    output["model"] = child_model_state
    output["offline_reward_optimizer"] = optimizer.state_dict()
    output["offline_reward_updates"] = previous_updates + args.updates
    output["offline_reward_parent_checkpoint"] = str(args.checkpoint)
    output["offline_reward_parent_sha256"] = parent_sha256
    output["offline_reward_replay_dir"] = str(args.replay_dir)
    output["offline_reward_target_scale"] = args.reward_target_scale
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
    parser.add_argument("--sequence-length", type=int, default=17)
    parser.add_argument("--anchored-channel-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument(
        "--latent-source",
        choices=("posterior", "posterior_and_prior"),
        default="posterior_and_prior",
    )
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reward-target-scale", type=float, default=1.0)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.anchored_channel_steps <= 0:
        parser.error("--anchored-channel-steps must be positive")
    if not 1 <= args.microbatch_size <= args.batch_size:
        parser.error("--microbatch-size must lie in [1,batch-size]")
    if args.batch_size % 32:
        parser.error("--batch-size must be a multiple of 32")
    if args.learning_rate <= 0.0 or args.reward_target_scale <= 0.0:
        parser.error("learning rate and reward target scale must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    return args


if __name__ == "__main__":
    print(json.dumps(continue_reward(parse_args()), indent=2, sort_keys=True))
