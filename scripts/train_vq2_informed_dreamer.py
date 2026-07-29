#!/usr/bin/env python3
"""Train the VQ2 visual Puffer policy with compact Informed Dreamer.

The executable expects ``pufferlib._C`` to be built for ``drone_race_vision``.
It collects native rollouts, stores a quantized image replay, trains the world
model from privileged decoder targets, and trains the actor only in latent
imagination. It never connects to FlightSim.
"""

from __future__ import annotations

import argparse
import ctypes
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

from pufferlib import _C, pufferl  # noqa: E402
from pufferlib.vq2_dreamer import (  # noqa: E402
    RSSMState,
    VQ2InformedDreamer,
    WorldModelLoss,
)
from pufferlib.vq2_informed import (  # noqa: E402
    ACTION_SCHEMA,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    MASK_SIZE,
    OBSERVATION_SCHEMA,
    configure_full_start_collection,
)


TRAINING_STATE_SCHEMA = "vq2_dreamer_training_state_v2"


class QuantizedSequenceReplay:
    """Time-major replay retaining continuous masks at uint8 precision."""

    SCHEMA = "vq2_quantized_sequence_replay_v1"

    def __init__(
        self,
        capacity: int,
        agents: int,
        *,
        storage_dir: Path | None = None,
        resume: bool = False,
    ) -> None:
        self.capacity = capacity
        self.agents = agents
        self.storage_dir = storage_dir
        self._metadata_path = (
            None if storage_dir is None else storage_dir / "metadata.json"
        )
        self.position = 0
        self.size = 0
        if resume and storage_dir is None:
            raise ValueError("memory-mapped replay resume requires a storage directory")
        if storage_dir is not None:
            storage_dir.mkdir(parents=True, exist_ok=True)
            if resume:
                self._load_metadata()

        self.mask = self._allocate(
            "mask", (capacity, agents, MASK_SIZE), np.uint8, resume
        )
        self.tail = self._allocate(
            "tail",
            (capacity, agents, ENV_OBS_SIZE - MASK_SIZE),
            np.float16,
            resume,
        )
        self.action = self._allocate(
            "action", (capacity, agents, 4), np.float16, resume
        )
        self.reward = self._allocate(
            "reward", (capacity, agents), np.float16, resume
        )
        self.continuation = self._allocate(
            "continuation", (capacity, agents), np.uint8, resume
        )
        self.valid = self._allocate(
            "valid", (capacity, agents), np.uint8, resume
        )

    def _allocate(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype,
        resume: bool,
    ) -> np.ndarray:
        if self.storage_dir is None:
            return np.empty(shape, dtype=dtype)
        path = self.storage_dir / f"{name}.dat"
        if not resume and path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing replay segment: {path}"
            )
        mode = "r+" if resume else "w+"
        try:
            return np.memmap(path, dtype=dtype, mode=mode, shape=shape)
        except (FileNotFoundError, ValueError) as error:
            raise RuntimeError(
                f"replay segment is missing or has the wrong size: {path}"
            ) from error

    def _load_metadata(self) -> None:
        assert self._metadata_path is not None
        if not self._metadata_path.exists():
            raise RuntimeError(
                f"replay metadata is missing: {self._metadata_path}"
            )
        metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        expected = {
            "schema": self.SCHEMA,
            "capacity": self.capacity,
            "agents": self.agents,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"replay metadata {key} mismatch: expected {value!r}, "
                    f"got {metadata.get(key)!r}"
                )
        self.position = int(metadata["position"])
        self.size = int(metadata["size"])
        if not 0 <= self.position < self.capacity:
            raise RuntimeError("replay metadata position is out of range")
        if not 0 <= self.size <= self.capacity:
            raise RuntimeError("replay metadata size is out of range")

    def state_dict(self) -> dict[str, int | str | None]:
        return {
            "schema": self.SCHEMA,
            "capacity": self.capacity,
            "agents": self.agents,
            "position": self.position,
            "size": self.size,
            "storage_dir": (
                None if self.storage_dir is None else str(self.storage_dir)
            ),
        }

    def flush(self) -> None:
        if self.storage_dir is None:
            return
        for array in (
            self.mask,
            self.tail,
            self.action,
            self.reward,
            self.continuation,
            self.valid,
        ):
            assert isinstance(array, np.memmap)
            array.flush()
        assert self._metadata_path is not None
        temporary = self._metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.state_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._metadata_path)

    @property
    def oldest(self) -> int:
        return (self.position - self.size) % self.capacity

    @property
    def storage_bytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.mask,
                self.tail,
                self.action,
                self.reward,
                self.continuation,
                self.valid,
            )
        )

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        continuation: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        observation_np = observation.detach().cpu().numpy()
        self.mask[self.position] = np.rint(
            np.clip(observation_np[:, :MASK_SIZE], 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        self.tail[self.position] = observation_np[:, MASK_SIZE:].astype(np.float16)
        self.action[self.position] = action.detach().cpu().numpy().astype(np.float16)
        self.reward[self.position] = reward.detach().cpu().numpy().astype(np.float16)
        self.continuation[self.position] = (
            continuation.detach().cpu().numpy() > 0.5
        ).astype(np.uint8)
        self.valid[self.position] = (
            valid.detach().cpu().numpy() > 0.5
        ).astype(np.uint8)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _physical(self, chronological: np.ndarray) -> np.ndarray:
        return (self.oldest + chronological) % self.capacity

    def _candidate_starts(self, sequence_length: int) -> np.ndarray:
        """Return every reset-free ``(start, agent)`` sequence in replay.

        A genuine terminal may occupy the final element because its reward and
        terminal camera observation belong to the preceding episode.  Invalid
        deferred-reset no-ops and terminals before the final element are never
        sampled.
        """

        if self.size < sequence_length:
            return np.empty((0, 2), dtype=np.int64)
        chronological = self._physical(np.arange(self.size))
        valid = self.valid[chronological].astype(np.int32)
        continuation = self.continuation[chronological].astype(np.int32)

        valid_prefix = np.concatenate(
            (np.zeros((1, self.agents), dtype=np.int32), np.cumsum(valid, axis=0)),
            axis=0,
        )
        valid_counts = valid_prefix[sequence_length:] - valid_prefix[:-sequence_length]
        good = valid_counts == sequence_length
        if sequence_length > 1:
            continuation_prefix = np.concatenate(
                (
                    np.zeros((1, self.agents), dtype=np.int32),
                    np.cumsum(continuation, axis=0),
                ),
                axis=0,
            )
            continuation_counts = (
                continuation_prefix[sequence_length - 1 : self.size]
                - continuation_prefix[: self.size - sequence_length + 1]
            )
            good &= continuation_counts == sequence_length - 1
        return np.argwhere(good)

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        device: torch.device,
        rng: random.Random,
        include_info: bool = False,
        recent_steps: int = 0,
    ):
        candidates = self._candidate_starts(sequence_length)
        if not len(candidates):
            raise RuntimeError("replay does not contain a reset-free sequence")
        if recent_steps < 0:
            raise ValueError("recent replay steps cannot be negative")
        observations = []
        actions = []
        rewards = []
        continuations = []
        selected_starts = []
        for _ in range(batch_size):
            start, agent = candidates[rng.randrange(len(candidates))]
            selected_starts.append(int(start))
            logical = np.arange(start, start + sequence_length)
            physical = self._physical(logical)
            continuation = self.continuation[physical, agent]
            mask = self.mask[physical, agent].astype(np.float32) / 255.0
            tail = self.tail[physical, agent].astype(np.float32)
            observations.append(np.concatenate((mask, tail), -1))
            actions.append(self.action[physical, agent].astype(np.float32))
            rewards.append(self.reward[physical, agent].astype(np.float32))
            continuations.append(continuation.astype(np.float32))
        batch = tuple(
            torch.from_numpy(np.stack(value)).to(device)
            for value in (observations, actions, rewards, continuations)
        )
        if not include_info:
            return batch
        selected_starts_np = np.asarray(selected_starts, dtype=np.int64)
        newest_age = self.size - selected_starts_np - sequence_length
        recent_start = max(0, self.size - min(recent_steps, self.size))
        return batch, {
            "newest_age_steps": newest_age,
            "oldest_age_steps": newest_age + sequence_length - 1,
            "fully_recent": selected_starts_np >= recent_start,
        }


def _load_config(name: str) -> dict:
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        return pufferl.load_config(name)
    finally:
        sys.argv = saved_argv


def _tensor_from_pointer(pointer: int, length: int) -> torch.Tensor:
    return torch.frombuffer(
        (ctypes.c_float * length).from_address(pointer), dtype=torch.float32
    )


def _apply_collector_action_hold(
    candidate: torch.Tensor,
    held: torch.Tensor,
    remaining: torch.Tensor,
    duration: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hold exploratory actions long enough to actuate a slow CTBR plant."""

    if duration <= 0:
        raise ValueError("collector action-hold duration must be positive")
    if candidate.shape != held.shape or candidate.shape[0] != remaining.shape[0]:
        raise ValueError("collector action-hold tensors are misaligned")
    resample = remaining <= 0
    action = torch.where(resample[:, None], candidate, held)
    next_remaining = torch.where(
        resample,
        torch.full_like(remaining, duration),
        remaining,
    ) - 1
    return action, action.detach(), next_remaining


def _reset_state_where(
    model: VQ2InformedDreamer,
    state: RSSMState,
    terminal: torch.Tensor,
) -> RSSMState:
    initial = model.rssm.initial(
        terminal.shape[0], device=terminal.device, dtype=state.deterministic.dtype
    )
    mask = terminal.bool()
    values = []
    for current, reset in zip(state, initial, strict=True):
        expanded = mask.reshape(mask.shape[0], *([1] * (current.ndim - 1)))
        values.append(torch.where(expanded, reset, current))
    return RSSMState(*values)


def _last_posterior_state(output) -> RSSMState:
    """Detach the last posterior state for burn-in or imagination."""

    return RSSMState(
        output.posterior.deterministic[:, -1].detach(),
        output.posterior.stochastic[:, -1].detach(),
        output.posterior.logits[:, -1].detach(),
    )


def _select_imagination_starts(
    posterior: RSSMState,
    *,
    mode: str,
    count: int,
) -> RSSMState:
    """Select bounded posterior starts for latent imagination.

    ``all`` matches DreamerV3's default ``imag_last=0`` behavior while
    allowing a hardware-bounded random subset. ``last`` preserves the legacy
    endpoint-only contract for source-locked reproduction.
    """

    if mode == "last":
        if count:
            raise ValueError("last-state imagination does not accept a start count")
        return RSSMState(*(value[:, -1].detach() for value in posterior))
    if mode != "all":
        raise ValueError(f"unsupported imagination start mode: {mode}")
    flat = RSSMState(
        *(value.reshape(-1, *value.shape[2:]).detach() for value in posterior)
    )
    available = flat.deterministic.shape[0]
    selected_count = available if count == 0 else count
    if not 1 <= selected_count <= available:
        raise ValueError(
            f"imagination start count must lie in [1,{available}], got {selected_count}"
        )
    if selected_count == available:
        return flat
    indices = torch.randperm(available, device=flat.deterministic.device)[
        :selected_count
    ]
    return RSSMState(*(value[indices] for value in flat))


def _repeat_imagination_starts(starts: RSSMState, repeats: int) -> RSSMState:
    """Repeat latent starts for independent Monte Carlo imagination samples."""

    if repeats <= 0:
        raise ValueError("imagination repeats must be positive")
    if repeats == 1:
        return starts
    return RSSMState(
        *(value.repeat_interleave(repeats, dim=0) for value in starts)
    )


def _burn_in_state(
    model: VQ2InformedDreamer,
    observation: torch.Tensor,
    action: torch.Tensor,
) -> RSSMState:
    """Build a gradient-free recurrent carry from replay context."""

    if observation.shape[1] == 0:
        return model.rssm.initial(
            observation.shape[0],
            device=observation.device,
            dtype=observation.dtype,
        )
    with torch.no_grad():
        context = model.observe_sequence(
            observation,
            action,
            deterministic_latent=True,
        )
    return _last_posterior_state(context)


def _slice_state(state: RSSMState, start: int, stop: int) -> RSSMState:
    return RSSMState(*(value[start:stop] for value in state))


def _concatenate_states(states: list[RSSMState]) -> RSSMState:
    if not states:
        raise ValueError("cannot concatenate an empty recurrent-state list")
    return RSSMState(
        *(torch.cat([getattr(state, field) for state in states], 0)
          for field in RSSMState._fields)
    )


def _autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    if device.type != "cuda":
        raise RuntimeError("mixed precision is supported only on CUDA")
    if amp_dtype != "bfloat16":
        raise ValueError(f"unsupported AMP dtype: {amp_dtype}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support bfloat16 AMP")
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    )


def _finite_clip_grad_norm(
    parameters: list[torch.nn.Parameter],
    max_norm: float,
    *,
    label: str,
) -> torch.Tensor:
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(norm):
        raise RuntimeError(f"non-finite {label} gradient norm")
    return norm


def _world_model_update(
    *,
    model: VQ2InformedDreamer,
    optimizer: torch.optim.Optimizer,
    observation: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    continuation: torch.Tensor,
    replay_context: int,
    microbatch_size: int,
    amp_dtype: str,
    free_nats: float = 1.0,
    world_parameters: list[torch.nn.Parameter] | None = None,
    deterministic_latent: bool = False,
    imagination_start_mode: str = "last",
    imagination_start_count: int = 0,
) -> tuple[WorldModelLoss, RSSMState, torch.Tensor, int]:
    """Apply one logical world update through weighted microbatches.

    Every loss term is mean-reduced by the model. Weighting each microbatch by
    its share of the logical batch therefore preserves full-batch gradients,
    including a final short microbatch. Recurrent burn-in is also microbatched
    so its temporary activations do not defeat the memory bound.
    """

    batch_size = observation.shape[0]
    if batch_size <= 0:
        raise ValueError("world update requires a non-empty logical batch")
    if microbatch_size <= 0 or microbatch_size > batch_size:
        raise ValueError("world microbatch must lie in [1, batch_size]")
    if replay_context < 0 or replay_context >= observation.shape[1]:
        raise ValueError("replay context must leave a non-empty world sequence")
    if world_parameters is None:
        world_parameters = [
            *model.rssm.parameters(),
            *model.privileged_decoder.parameters(),
            *model.reward_predictor.parameters(),
            *model.continue_predictor.parameters(),
        ]

    optimizer.zero_grad(set_to_none=True)
    aggregate: dict[str, torch.Tensor] = {}
    posteriors: list[RSSMState] = []
    microbatches = 0
    for start in range(0, batch_size, microbatch_size):
        stop = min(start + microbatch_size, batch_size)
        weight = (stop - start) / batch_size
        with _autocast_context(observation.device, amp_dtype):
            initial_state = _burn_in_state(
                model,
                observation[start:stop, :replay_context],
                action[start:stop, :replay_context],
            )
            loss, output = model.world_model_loss(
                observation[start:stop, replay_context:],
                action[start:stop, replay_context:],
                reward[start:stop, replay_context:],
                continuation[start:stop, replay_context:],
                initial_state=initial_state,
                free_nats=free_nats,
                deterministic_latent=deterministic_latent,
            )
            weighted_total = loss.total * weight
        weighted_total.backward()
        for field in fields(loss):
            value = getattr(loss, field.name).detach().float() * weight
            aggregate[field.name] = aggregate.get(field.name, 0.0) + value
        posteriors.append(output.posterior.detached())
        microbatches += 1

    gradient_norm = _finite_clip_grad_norm(
        world_parameters, 100.0, label="world-model"
    )
    optimizer.step()
    posterior = _concatenate_states(posteriors)
    return (
        WorldModelLoss(**aggregate),
        _select_imagination_starts(
            posterior,
            mode=imagination_start_mode,
            count=imagination_start_count,
        ),
        gradient_norm.detach(),
        microbatches,
    )


def _checkpoint_payload(
    *,
    model: VQ2InformedDreamer,
    world_optimizer: torch.optim.Optimizer,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    environment_step: int,
    updates: int,
    actor_updates: int,
    replay: QuantizedSequenceReplay,
    rng: random.Random,
    resume_sessions: int,
    metrics: dict | None = None,
) -> dict:
    return {
        "training_state_schema": TRAINING_STATE_SCHEMA,
        "model": model.state_dict(),
        "world_optimizer": world_optimizer.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "environment_step": environment_step,
        "updates": updates,
        "actor_updates": actor_updates,
        "replay": replay.state_dict(),
        "rng_state": {
            "python": rng.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        },
        "resume_sessions": resume_sessions,
        "metrics": metrics or {},
        "args": vars(args),
        "dreamerv3_return_normalizer": {
            "low": float(model.return_normalizer_low.cpu()),
            "high": float(model.return_normalizer_high.cpu()),
            "rate": 0.01,
            "limit": 1.0,
            "perclo": 5.0,
            "perchi": 95.0,
            "debias": False,
        },
        "legal_observation_size": LEGAL_OBS_SIZE,
        "environment_observation_size": ENV_OBS_SIZE,
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> dict[str, float | int | str]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        cuda_free_at_start, cuda_total_bytes = torch.cuda.mem_get_info(device)
    else:
        cuda_free_at_start = 0
        cuda_total_bytes = 0
    config = _load_config(args.env_name)
    if args.full_start_collection:
        config = configure_full_start_collection(config)
    imagination_control_frequency_hz = 1.0 / float(config["env"]["dt"])
    vec = _C.create_vec(config, gpu=0)
    if vec.obs_size != ENV_OBS_SIZE:
        vec.close()
        raise RuntimeError(
            f"drone_race_vision ABI mismatch: expected {ENV_OBS_SIZE}, got {vec.obs_size}"
        )
    vec.reset()
    observations = _tensor_from_pointer(
        vec.obs_ptr, vec.total_agents * vec.obs_size
    ).reshape(vec.total_agents, vec.obs_size)
    rewards = _tensor_from_pointer(vec.rewards_ptr, vec.total_agents)
    terminals = _tensor_from_pointer(vec.terminals_ptr, vec.total_agents)
    actions_cpu = torch.zeros((vec.total_agents, 4), dtype=torch.float32)

    model = VQ2InformedDreamer(
        deterministic_size=args.deterministic_size,
        stochastic_groups=args.stochastic_groups,
        stochastic_classes=args.stochastic_classes,
        actor_initial_std=args.actor_initial_std,
        actor_distribution_mode=args.actor_distribution_mode,
        distributional_reward=args.distributional_reward,
        action_conditioned_reward=args.action_conditioned_reward,
    ).to(device)
    world_parameters = [
        *model.rssm.parameters(),
        *model.privileged_decoder.parameters(),
        *model.reward_predictor.parameters(),
        *model.continue_predictor.parameters(),
    ]
    world_optimizer = torch.optim.AdamW(
        world_parameters, lr=args.world_lr, weight_decay=args.weight_decay
    )
    actor_parameters = list(model.actor.parameters())
    critic_parameters = list(model.critic.parameters())
    actor_optimizer = torch.optim.AdamW(
        actor_parameters, lr=args.actor_lr, weight_decay=args.weight_decay
    )
    critic_optimizer = torch.optim.AdamW(
        critic_parameters, lr=args.critic_lr, weight_decay=args.weight_decay
    )
    starting_environment_step = 0
    starting_updates = 0
    starting_actor_updates = 0
    resume_payload = None
    resume_sessions = 0
    if args.resume is not None:
        resume_payload = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        if resume_payload.get("training_state_schema") != TRAINING_STATE_SCHEMA:
            raise RuntimeError("resume checkpoint training-state schema mismatch")
        if resume_payload.get("legal_observation_size") != LEGAL_OBS_SIZE:
            raise RuntimeError("resume checkpoint legal-observation ABI mismatch")
        if resume_payload.get("environment_observation_size") != ENV_OBS_SIZE:
            raise RuntimeError("resume checkpoint environment-observation ABI mismatch")
        if resume_payload.get("observation_schema") != OBSERVATION_SCHEMA:
            raise RuntimeError("resume checkpoint observation schema mismatch")
        if resume_payload.get("action_schema") != ACTION_SCHEMA:
            raise RuntimeError("resume checkpoint action schema mismatch")
        model.load_state_dict(resume_payload["model"])
        normalizer = resume_payload.get("dreamerv3_return_normalizer", {})
        if isinstance(normalizer, dict):
            model.return_normalizer_low.fill_(float(normalizer.get("low", 0.0)))
            model.return_normalizer_high.fill_(float(normalizer.get("high", 0.0)))
        if "world_optimizer" in resume_payload:
            world_optimizer.load_state_dict(resume_payload["world_optimizer"])
            if not args.reset_actor_optimizer_on_resume:
                actor_optimizer.load_state_dict(resume_payload["actor_optimizer"])
            if not args.reset_critic_optimizer_on_resume:
                critic_optimizer.load_state_dict(resume_payload["critic_optimizer"])
        starting_environment_step = int(
            resume_payload.get("environment_step", 0)
        )
        starting_updates = int(resume_payload.get("updates", 0))
        starting_actor_updates = int(resume_payload.get("actor_updates", 0))
        resume_sessions = int(resume_payload.get("resume_sessions", 0)) + 1
        if starting_environment_step >= args.environment_steps:
            raise ValueError(
                "resume checkpoint already reached --environment-steps target"
            )
    replay = QuantizedSequenceReplay(
        args.replay_steps,
        vec.total_agents,
        storage_dir=args.replay_dir,
        resume=args.resume is not None,
    )
    if resume_payload is not None:
        checkpoint_replay = resume_payload.get("replay")
        if checkpoint_replay is None:
            raise RuntimeError("resume checkpoint does not contain replay state")
        replay_state = replay.state_dict()
        for key in ("schema", "capacity", "agents", "position", "size"):
            if checkpoint_replay.get(key) != replay_state[key]:
                raise RuntimeError(f"resume checkpoint replay {key} mismatch")
        rng_state = resume_payload.get("rng_state")
        if not isinstance(rng_state, dict):
            raise RuntimeError("resume checkpoint does not contain RNG state")
        rng.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
        if device.type == "cuda":
            cuda_rng = rng_state.get("torch_cuda")
            if not cuda_rng:
                raise RuntimeError("resume checkpoint does not contain CUDA RNG state")
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in cuda_rng]
            )
    replay_size_at_session_start = replay.size
    replay_position_at_session_start = replay.position
    session_replay_vector_steps_written = 0
    # A resumed process has a fresh native vector state, while the retained
    # replay may end inside an episode. Insert one invalid vector record before
    # collection so no sampled sequence can bridge that discontinuity. This is
    # preferable to discarding a real plant step because recurrent action
    # history then remains aligned with the fresh native episode.
    if resume_payload is not None:
        replay.add(
            observations.clone(),
            actions_cpu,
            torch.zeros(vec.total_agents),
            torch.zeros(vec.total_agents),
            torch.zeros(vec.total_agents, dtype=torch.bool),
        )
        session_replay_vector_steps_written += 1
    state = model.rssm.initial(vec.total_agents, device=device)
    previous_action = torch.zeros((vec.total_agents, 4), device=device)
    pending_reset = torch.zeros(vec.total_agents, dtype=torch.bool)
    resume_boundary_invalid_records = vec.total_agents if resume_payload else 0
    last_world = None
    last_imagination = None
    updates = starting_updates
    session_updates = 0
    actor_updates = starting_actor_updates
    session_actor_updates = 0
    skipped_updates_no_sequence = 0
    replayed_context_records = 0
    replayed_world_records = 0
    imagined_records = 0
    world_microbatches_processed = 0
    last_world_gradient_norm = None
    last_actor_gradient_norm = None
    last_critic_gradient_norm = None
    collector_action_sum = torch.zeros(4, dtype=torch.float64)
    collector_action_square_sum = torch.zeros(4, dtype=torch.float64)
    collector_action_abs_max = torch.zeros(4, dtype=torch.float32)
    collector_action_samples = 0
    collector_action_near_limit = 0
    collector_reward_sum = 0.0
    collector_reward_square_sum = 0.0
    collector_reward_min = float("inf")
    collector_reward_max = float("-inf")
    collector_reward_samples = 0
    collector_gate_reward_events = 0
    collector_valid_agent_transitions = 0
    collector_random_agent_transitions = 0
    collector_policy_agent_transitions = 0
    sampled_sequences = 0
    sampled_fully_session_recent_sequences = 0
    sampled_newest_age_steps_sum = 0
    sampled_oldest_age_steps_sum = 0
    sampled_newest_age_steps_max = 0
    sampled_oldest_age_steps_max = 0
    held_action = torch.zeros((vec.total_agents, 4), device=device)
    hold_remaining = torch.zeros(vec.total_agents, dtype=torch.long, device=device)
    started = time.perf_counter()

    replay_sample_length = args.replay_context + args.world_sequence_length

    try:
        for environment_step in range(
            starting_environment_step, args.environment_steps
        ):
            legal = observations[:, :LEGAL_OBS_SIZE].to(device)
            if environment_step < args.random_steps:
                candidate_action = torch.empty(
                    (vec.total_agents, 4), device=device
                ).uniform_(-args.random_action_scale, args.random_action_scale)
                with torch.no_grad():
                    _, state = model.policy_step(legal, previous_action, state)
            else:
                with torch.no_grad():
                    distribution, state = model.policy_distribution_step(
                        legal, previous_action, state
                    )
                    if args.collector_policy_sampling:
                        candidate_action = model.sample_actor_action(distribution)
                    else:
                        candidate_action = model.deterministic_actor_action(distribution)
                        if args.exploration_std > 0.0:
                            candidate_action = candidate_action + (
                                args.exploration_std
                                * torch.randn_like(candidate_action)
                            )
                    candidate_action.clamp_(-1.0, 1.0)
            action, held_action, hold_remaining = _apply_collector_action_hold(
                candidate_action,
                held_action,
                hold_remaining,
                args.collector_action_hold,
            )
            if not torch.isfinite(action).all():
                raise RuntimeError(
                    f"non-finite actor action at environment step {environment_step}"
                )
            active_collection = ~pending_reset
            if active_collection.any():
                selected_action = action[active_collection.to(device)].detach().cpu()
                collector_action_sum += selected_action.double().sum(0)
                collector_action_square_sum += selected_action.double().square().sum(0)
                collector_action_abs_max = torch.maximum(
                    collector_action_abs_max, selected_action.abs().amax(0)
                )
                collector_action_samples += int(selected_action.shape[0])
                collector_action_near_limit += int(
                    (selected_action.abs() >= 0.95).sum().item()
                )
            actions_cpu.copy_(action.cpu())
            vec.cpu_step(actions_cpu.data_ptr())
            reward_copy = rewards.clone()
            terminal_copy = terminals.clone()
            continuation = 1.0 - terminal_copy
            valid_transition = ~pending_reset
            valid_transition_count = int(valid_transition.sum().item())
            collector_valid_agent_transitions += valid_transition_count
            if environment_step < args.random_steps:
                collector_random_agent_transitions += valid_transition_count
            else:
                collector_policy_agent_transitions += valid_transition_count
            selected_reward = reward_copy[valid_transition]
            if selected_reward.numel():
                reward_double = selected_reward.double()
                collector_reward_sum += float(reward_double.sum())
                collector_reward_square_sum += float(reward_double.square().sum())
                collector_reward_min = min(
                    collector_reward_min, float(selected_reward.min())
                )
                collector_reward_max = max(
                    collector_reward_max, float(selected_reward.max())
                )
                collector_reward_samples += int(selected_reward.numel())
                collector_gate_reward_events += int(
                    (selected_reward > 1.0).sum()
                )
            replay.add(
                observations.clone(),
                actions_cpu,
                reward_copy,
                continuation,
                valid_transition,
            )
            session_replay_vector_steps_written += 1
            terminal_device = terminal_copy.to(device)
            reset_device = terminal_device.bool() | pending_reset.to(device)
            state = _reset_state_where(model, state, reset_device)
            held_action = torch.where(
                reset_device[:, None], torch.zeros_like(held_action), held_action
            )
            hold_remaining = torch.where(
                reset_device, torch.zeros_like(hold_remaining), hold_remaining
            )
            previous_action = torch.where(
                reset_device[:, None],
                torch.zeros_like(action),
                action,
            )
            pending_reset = terminal_copy.bool()

            completed_step = environment_step + 1
            should_train = (
                replay.size >= max(args.learning_starts, replay_sample_length)
                and completed_step % args.train_every == 0
                and (
                    resume_payload is None
                    or completed_step - starting_environment_step
                    >= args.resume_refresh_steps
                )
            )
            if should_train:
                for _ in range(args.train_ratio):
                    try:
                        batch, sample_info = replay.sample(
                            args.batch_size,
                            replay_sample_length,
                            device=device,
                            rng=rng,
                            include_info=True,
                            recent_steps=session_replay_vector_steps_written,
                        )
                    except RuntimeError as error:
                        if "reset-free sequence" not in str(error):
                            raise
                        skipped_updates_no_sequence += 1
                        break
                    batch_observation, batch_action, batch_reward, batch_continue = batch
                    sampled_sequences += args.batch_size
                    sampled_fully_session_recent_sequences += int(
                        sample_info["fully_recent"].sum()
                    )
                    sampled_newest_age_steps_sum += int(
                        sample_info["newest_age_steps"].sum()
                    )
                    sampled_oldest_age_steps_sum += int(
                        sample_info["oldest_age_steps"].sum()
                    )
                    sampled_newest_age_steps_max = max(
                        sampled_newest_age_steps_max,
                        int(sample_info["newest_age_steps"].max()),
                    )
                    sampled_oldest_age_steps_max = max(
                        sampled_oldest_age_steps_max,
                        int(sample_info["oldest_age_steps"].max()),
                    )
                    world_loss, start, world_gradient_norm, microbatches = (
                        _world_model_update(
                            model=model,
                            optimizer=world_optimizer,
                            observation=batch_observation,
                            action=batch_action,
                            reward=batch_reward,
                            continuation=batch_continue,
                            replay_context=args.replay_context,
                            microbatch_size=args.world_microbatch_size,
                            amp_dtype=args.amp_dtype,
                            free_nats=args.free_nats,
                            world_parameters=world_parameters,
                            imagination_start_mode=args.imagination_start_mode,
                            imagination_start_count=args.imagination_start_count,
                        )
                    )
                    last_world_gradient_norm = float(world_gradient_norm.cpu())
                    world_microbatches_processed += microbatches

                    with _autocast_context(device, args.amp_dtype):
                        imagination = model.imagination_loss(
                            start,
                            horizon=args.imagination_horizon,
                            gamma=args.gamma,
                            lambda_=args.return_lambda,
                            entropy_weight=args.entropy_weight,
                            smoothness_weight=args.smoothness_weight,
                            reward_source=args.imagination_reward_source,
                            action_effort_weights=tuple(args.action_effort_weights),
                            advantage_normalization=args.advantage_normalization,
                            control_frequency_hz=imagination_control_frequency_hz,
                        )
                    if completed_step >= args.actor_learning_starts:
                        actor_optimizer.zero_grad(set_to_none=True)
                        imagination.actor.backward()
                        actor_gradient_norm = _finite_clip_grad_norm(
                            actor_parameters, 100.0, label="actor"
                        )
                        last_actor_gradient_norm = float(
                            actor_gradient_norm.detach().cpu()
                        )
                        actor_optimizer.step()
                        actor_updates += 1
                        session_actor_updates += 1
                    critic_optimizer.zero_grad(set_to_none=True)
                    imagination.critic.backward()
                    critic_gradient_norm = _finite_clip_grad_norm(
                        critic_parameters, 100.0, label="critic"
                    )
                    last_critic_gradient_norm = float(
                        critic_gradient_norm.detach().cpu()
                    )
                    critic_optimizer.step()
                    model.update_target_critic()
                    last_world = world_loss
                    last_imagination = imagination
                    updates += 1
                    session_updates += 1
                    replayed_context_records += args.batch_size * args.replay_context
                    replayed_world_records += (
                        args.batch_size * args.world_sequence_length
                    )
                    imagined_records += (
                        start.deterministic.shape[0] * args.imagination_horizon
                    )

            if args.progress_every > 0 and completed_step % args.progress_every == 0:
                progress = {
                    "environment_step": completed_step,
                    "updates": updates,
                    "session_updates": session_updates,
                    "actor_updates": actor_updates,
                    "session_actor_updates": session_actor_updates,
                    "wall_seconds": time.perf_counter() - started,
                    "replay_steps": replay.size,
                    "replay_vector_steps_written_this_session": (
                        session_replay_vector_steps_written
                    ),
                    "collector_valid_agent_transitions": (
                        collector_valid_agent_transitions
                    ),
                    "collector_random_agent_transitions": (
                        collector_random_agent_transitions
                    ),
                    "collector_policy_agent_transitions": (
                        collector_policy_agent_transitions
                    ),
                    "replay_context": args.replay_context,
                    "world_sequence_length": args.world_sequence_length,
                    "imagination_horizon": args.imagination_horizon,
                    "imagination_start_mode": args.imagination_start_mode,
                    "imagination_start_count": args.imagination_start_count,
                    "sampled_sequences": sampled_sequences,
                    "sampled_fully_session_recent_sequences": (
                        sampled_fully_session_recent_sequences
                    ),
                    "amp_dtype": args.amp_dtype,
                    "logical_batch_size": args.batch_size,
                    "world_microbatch_size": args.world_microbatch_size,
                    "world_gradient_accumulation_steps": (
                        args.world_gradient_accumulation_steps
                    ),
                    "world_microbatches_processed": world_microbatches_processed,
                    "replayed_context_records": replayed_context_records,
                    "replayed_world_records": replayed_world_records,
                    "imagined_records": imagined_records,
                    "skipped_updates_no_sequence": skipped_updates_no_sequence,
                }
                if last_world is not None:
                    progress["world_total"] = float(last_world.total.detach().cpu())
                if last_imagination is not None:
                    progress["imagination_mean_return"] = float(
                        last_imagination.mean_return.detach().cpu()
                    )
                print(json.dumps(progress, sort_keys=True), flush=True)
            if (
                args.checkpoint is not None
                and args.save_every > 0
                and completed_step % args.save_every == 0
            ):
                replay.flush()
                _save_checkpoint(
                    args.checkpoint,
                    _checkpoint_payload(
                        model=model,
                        world_optimizer=world_optimizer,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        args=args,
                        environment_step=completed_step,
                        updates=updates,
                        actor_updates=actor_updates,
                        replay=replay,
                        rng=rng,
                        resume_sessions=resume_sessions,
                    ),
                )
    finally:
        replay.flush()
        log = dict(vec.log())
        vec.close()

    elapsed = time.perf_counter() - started
    metrics: dict[str, float | int | str] = {
        "environment_steps": args.environment_steps,
        "starting_environment_step": starting_environment_step,
        "session_environment_steps": args.environment_steps - starting_environment_step,
        "agent_steps": (
            (args.environment_steps - starting_environment_step)
            * actions_cpu.shape[0]
        ),
        "collected_agent_transitions": (
            (args.environment_steps - starting_environment_step)
            * actions_cpu.shape[0]
        ),
        "collector_valid_agent_transitions": collector_valid_agent_transitions,
        "collector_random_agent_transitions": collector_random_agent_transitions,
        "collector_policy_agent_transitions": collector_policy_agent_transitions,
        "updates": updates,
        "session_updates": session_updates,
        "world_optimizer_steps": session_updates,
        "cumulative_world_optimizer_steps": updates,
        "actor_updates": actor_updates,
        "session_actor_updates": session_actor_updates,
        "critic_optimizer_steps": session_updates,
        "cumulative_critic_optimizer_steps": updates,
        "skipped_updates_no_sequence": skipped_updates_no_sequence,
        "replay_steps": replay.size,
        "replay_capacity_steps": replay.capacity,
        "replay_size_at_session_start": replay_size_at_session_start,
        "replay_position_at_session_start": replay_position_at_session_start,
        "replay_vector_steps_written_this_session": (
            session_replay_vector_steps_written
        ),
        "replay_unique_vector_steps_refreshed_this_session": min(
            replay.capacity, session_replay_vector_steps_written
        ),
        "replay_refresh_fraction": (
            min(replay.capacity, session_replay_vector_steps_written)
            / max(1, replay.size)
        ),
        "replay_storage_bytes": replay.storage_bytes,
        "replay_storage_backend": (
            "memory" if replay.storage_dir is None else "numpy_memmap"
        ),
        "replay_storage_dir": (
            "" if replay.storage_dir is None else str(replay.storage_dir)
        ),
        "replay_context": args.replay_context,
        "world_sequence_length": args.world_sequence_length,
        "imagination_horizon": args.imagination_horizon,
        "imagination_start_mode": args.imagination_start_mode,
        "imagination_start_count": args.imagination_start_count,
        "action_effort_weight_pitch": args.action_effort_weights[0],
        "action_effort_weight_roll": args.action_effort_weights[1],
        "action_effort_weight_collective": args.action_effort_weights[2],
        "action_effort_weight_yaw": args.action_effort_weights[3],
        "gamma": args.gamma,
        "return_lambda": args.return_lambda,
        "entropy_weight": args.entropy_weight,
        "smoothness_weight": args.smoothness_weight,
        "free_nats": args.free_nats,
        "distributional_reward": int(args.distributional_reward),
        "action_conditioned_reward": int(args.action_conditioned_reward),
        "imagination_reward_source": args.imagination_reward_source,
        "imagination_control_frequency_hz": imagination_control_frequency_hz,
        "collector_policy_sampling": int(args.collector_policy_sampling),
        "collector_action_hold": args.collector_action_hold,
        "actor_initial_std": args.actor_initial_std,
        "actor_distribution_mode": args.actor_distribution_mode,
        "advantage_normalization": args.advantage_normalization,
        "optimizer_weight_decay": args.weight_decay,
        "actor_optimizer_reset_on_resume": int(
            args.reset_actor_optimizer_on_resume
        ),
        "critic_optimizer_reset_on_resume": int(
            args.reset_critic_optimizer_on_resume
        ),
        "return_normalizer_low": float(model.return_normalizer_low.cpu()),
        "return_normalizer_high": float(model.return_normalizer_high.cpu()),
        "amp_dtype": args.amp_dtype,
        "amp_enabled": int(args.amp_dtype != "none"),
        "logical_batch_size": args.batch_size,
        "world_microbatch_size": args.world_microbatch_size,
        "world_gradient_accumulation_steps": (
            args.world_gradient_accumulation_steps
        ),
        "world_microbatches_processed": world_microbatches_processed,
        "resume_sessions": resume_sessions,
        "resume_refresh_steps": args.resume_refresh_steps,
        "reset_actor_optimizer_on_resume": int(
            args.reset_actor_optimizer_on_resume
        ),
        "resume_boundary_invalid_records": resume_boundary_invalid_records,
        "replayed_context_records": replayed_context_records,
        "replayed_world_records": replayed_world_records,
        "replayed_total_records": (
            replayed_context_records + replayed_world_records
        ),
        "imagined_records": imagined_records,
        "wall_seconds": elapsed,
        "agent_steps_per_second": (
            (args.environment_steps - starting_environment_step)
            * actions_cpu.shape[0]
            / max(elapsed, 1e-9)
        ),
        "device": str(device),
        "backend_env_name": "drone_race_vision",
        "full_start_collection": int(args.full_start_collection),
        "sampled_sequences": sampled_sequences,
        "sampled_fully_session_recent_sequences": (
            sampled_fully_session_recent_sequences
        ),
    }
    if sampled_sequences:
        metrics["sampled_fully_session_recent_fraction"] = (
            sampled_fully_session_recent_sequences / sampled_sequences
        )
        metrics["sampled_newest_age_steps_mean"] = (
            sampled_newest_age_steps_sum / sampled_sequences
        )
        metrics["sampled_oldest_age_steps_mean"] = (
            sampled_oldest_age_steps_sum / sampled_sequences
        )
        metrics["sampled_newest_age_steps_max"] = sampled_newest_age_steps_max
        metrics["sampled_oldest_age_steps_max"] = sampled_oldest_age_steps_max
    if device.type == "cuda":
        metrics["cuda_memory_free_at_start_bytes"] = cuda_free_at_start
        metrics["cuda_memory_total_bytes"] = cuda_total_bytes
        metrics["cuda_peak_memory_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device)
        )
        metrics["cuda_peak_memory_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(device)
        )
    if last_world_gradient_norm is not None:
        metrics["last_world_gradient_norm"] = last_world_gradient_norm
    if last_actor_gradient_norm is not None:
        metrics["last_actor_gradient_norm"] = last_actor_gradient_norm
    if last_critic_gradient_norm is not None:
        metrics["last_critic_gradient_norm"] = last_critic_gradient_norm
    if collector_action_samples:
        collector_mean = collector_action_sum / collector_action_samples
        collector_variance = (
            collector_action_square_sum / collector_action_samples
            - collector_mean.square()
        ).clamp_min(0.0)
        metrics["collector_action_samples"] = collector_action_samples
        metrics["collector_action_near_limit_fraction"] = (
            collector_action_near_limit / (4 * collector_action_samples)
        )
        for channel in range(4):
            metrics[f"collector_action_{channel}_mean"] = float(
                collector_mean[channel]
            )
            metrics[f"collector_action_{channel}_std"] = float(
                collector_variance[channel].sqrt()
            )
            metrics[f"collector_action_{channel}_abs_max"] = float(
                collector_action_abs_max[channel]
            )
    if collector_reward_samples:
        collector_reward_mean = collector_reward_sum / collector_reward_samples
        collector_reward_variance = max(
            0.0,
            collector_reward_square_sum / collector_reward_samples
            - collector_reward_mean * collector_reward_mean,
        )
        metrics.update(
            {
                "collector_reward_samples": collector_reward_samples,
                "collector_reward_mean": collector_reward_mean,
                "collector_reward_std": collector_reward_variance**0.5,
                "collector_reward_min": collector_reward_min,
                "collector_reward_max": collector_reward_max,
                "collector_gate_reward_events": collector_gate_reward_events,
            }
        )
    if last_world is not None:
        metrics.update(
            {
                f"world_{field.name}": float(
                    getattr(last_world, field.name).detach().cpu()
                )
                for field in fields(last_world)
            }
        )
    if last_imagination is not None:
        metrics.update(
            {
                f"imagination_{field.name}": float(
                    getattr(last_imagination, field.name).detach().cpu()
                )
                for field in fields(last_imagination)
            }
        )
    metrics.update({f"env_{key}": float(value) for key, value in log.items()})
    if log.get("reset_count", 0.0) > 0.0:
        metrics["env_gate_local_reset_fraction"] = (
            float(log.get("gate_local_reset_count", 0.0))
            / float(log["reset_count"])
        )
    collected = max(1, int(metrics["collected_agent_transitions"]))
    metrics["replayed_world_records_per_collected_transition"] = (
        replayed_world_records / collected
    )
    metrics["replayed_total_records_per_collected_transition"] = (
        (replayed_context_records + replayed_world_records) / collected
    )
    metrics["configured_steady_world_records_per_collected_transition"] = (
        args.batch_size
        * args.world_sequence_length
        * args.train_ratio
        / (actions_cpu.shape[0] * args.train_every)
    )
    metrics["configured_steady_total_records_per_collected_transition"] = (
        args.batch_size
        * (args.replay_context + args.world_sequence_length)
        * args.train_ratio
        / (actions_cpu.shape[0] * args.train_every)
    )
    if args.imagination_start_mode == "last":
        configured_imagination_starts = args.batch_size
    else:
        configured_imagination_starts = (
            args.imagination_start_count
            or args.batch_size * args.world_sequence_length
        )
    metrics["configured_imagination_starts_per_update"] = (
        configured_imagination_starts
    )
    metrics["configured_steady_imagined_records_per_collected_transition"] = (
        configured_imagination_starts
        * args.imagination_horizon
        * args.train_ratio
        / (actions_cpu.shape[0] * args.train_every)
    )

    if args.checkpoint is not None:
        replay.flush()
        _save_checkpoint(
            args.checkpoint,
            _checkpoint_payload(
                model=model,
                world_optimizer=world_optimizer,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                args=args,
                environment_step=args.environment_steps,
                updates=updates,
                actor_updates=actor_updates,
                replay=replay,
                rng=rng,
                resume_sessions=resume_sessions,
                metrics=metrics,
            ),
        )
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="drone_race_vq2_informed_dreamer")
    parser.add_argument("--environment-steps", type=int, default=100_000)
    parser.add_argument("--random-steps", type=int, default=2_000)
    parser.add_argument("--random-action-scale", type=float, default=0.25)
    parser.add_argument(
        "--full-start-collection",
        action="store_true",
        help="disable gate-local/segment curricula for uninterrupted collection",
    )
    parser.add_argument("--exploration-std", type=float, default=0.20)
    parser.add_argument(
        "--collector-policy-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="collect from the learned Gaussian; disable only for legacy mean-plus-noise",
    )
    parser.add_argument("--actor-initial-std", type=float, default=0.20)
    parser.add_argument(
        "--actor-distribution-mode",
        choices=("legacy_tanh_normal", "dreamerv3_bounded_normal"),
        default="dreamerv3_bounded_normal",
    )
    parser.add_argument(
        "--advantage-normalization",
        choices=("legacy_center_std", "dreamerv3_percentile"),
        default="dreamerv3_percentile",
    )
    parser.add_argument(
        "--collector-action-hold",
        type=int,
        default=1,
        help=(
            "training-only exploration hold in native control steps; one "
            "preserves per-step SkyDreamer sampling"
        ),
    )
    parser.add_argument("--replay-steps", type=int, default=2_048)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--learning-starts", type=int, default=128)
    parser.add_argument("--actor-learning-starts", type=int, default=2_048)
    parser.add_argument("--replay-context", type=int, default=16)
    parser.add_argument("--world-sequence-length", type=int)
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="deprecated alias for --world-sequence-length",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--world-microbatch-size",
        type=int,
        default=0,
        help="world-model microbatch; zero uses the full logical batch",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "bfloat16"),
        default="none",
        help="CUDA autocast dtype; bfloat16 avoids fp16 loss scaling",
    )
    parser.add_argument("--train-ratio", type=int, default=1)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--imagination-horizon", type=int, default=16)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="pinned DreamerV3 optimizer uses zero weight decay",
    )
    parser.add_argument(
        "--imagination-start-mode",
        choices=("last", "all"),
        default="all",
        help=(
            "all matches DreamerV3 imag_last=0; last is legacy endpoint-only"
        ),
    )
    parser.add_argument(
        "--imagination-start-count",
        type=int,
        default=0,
        help="bounded all-time posterior starts per update; zero uses all",
    )
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--return-lambda", type=float, default=0.95)
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=3e-4,
        help="SkyDreamer discovery value; lower only in a late source-locked stage",
    )
    parser.add_argument("--smoothness-weight", type=float, default=0.002)
    parser.add_argument(
        "--imagination-reward-source",
        choices=(
            "learned",
            "informed_decoder_progress",
            "informed_decoder_skydreamer",
        ),
        default="informed_decoder_skydreamer",
        help=(
            "learn a reward head, apply the legacy decoded progress term, or "
            "apply the complete paper task reward to the training-only decoder"
        ),
    )
    parser.add_argument(
        "--action-effort-weights",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 0.0),
        metavar=("PITCH", "ROLL", "COLLECTIVE", "YAW"),
        help="training-only centered quadratic effort weights for four CTBR channels",
    )
    parser.add_argument(
        "--resume-refresh-steps",
        type=int,
        default=128,
        help=(
            "fresh native vector steps required before any optimizer update "
            "after fail-closed resume"
        ),
    )
    parser.add_argument(
        "--reset-actor-optimizer-on-resume",
        action="store_true",
        help=(
            "explicitly discard incompatible actor optimizer state; use only "
            "under a source-locked frozen-actor or estimator migration"
        ),
    )
    parser.add_argument(
        "--reset-critic-optimizer-on-resume",
        action="store_true",
        help=(
            "explicitly discard critic optimizer state when a source-locked "
            "resume changes the imagination reward semantics"
        ),
    )
    parser.add_argument(
        "--free-nats",
        type=float,
        default=1.0,
        help=(
            "per-sequence-step KL floor; SkyDreamer/Dreamer default is 1.0, "
            "but compact local-model diagnostics may preregister a lower value"
        ),
    )
    parser.add_argument(
        "--distributional-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use SkyDreamer's pinned DreamerV3 255-bin symexp two-hot "
            "reward head"
        ),
    )
    parser.add_argument(
        "--action-conditioned-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "concatenate the causal action to the training-only reward head; "
            "the deployed actor ABI is unchanged"
        ),
    )
    parser.add_argument("--deterministic-size", type=int, default=256)
    parser.add_argument("--stochastic-groups", type=int, default=16)
    parser.add_argument("--stochastic-classes", type=int, default=16)
    parser.add_argument("--world-lr", type=float, default=4e-5)
    parser.add_argument("--actor-lr", type=float, default=4e-5)
    parser.add_argument("--critic-lr", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    positive = (
        "environment_steps",
        "replay_steps",
        "learning_starts",
        "batch_size",
        "train_ratio",
        "train_every",
        "imagination_horizon",
        "collector_action_hold",
        "deterministic_size",
        "stochastic_groups",
        "stochastic_classes",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.replay_context < 0:
        parser.error("--replay-context cannot be negative")
    if args.world_sequence_length is None:
        args.world_sequence_length = args.sequence_length or 64
    elif (
        args.sequence_length is not None
        and args.sequence_length != args.world_sequence_length
    ):
        parser.error(
            "--sequence-length and --world-sequence-length cannot disagree"
        )
    if args.world_sequence_length <= 0:
        parser.error("--world-sequence-length must be positive")
    if args.world_microbatch_size == 0:
        args.world_microbatch_size = args.batch_size
    if not 1 <= args.world_microbatch_size <= args.batch_size:
        parser.error("--world-microbatch-size must lie in [1, batch-size]")
    args.world_gradient_accumulation_steps = (
        args.batch_size + args.world_microbatch_size - 1
    ) // args.world_microbatch_size
    replay_sample_length = args.replay_context + args.world_sequence_length
    if replay_sample_length > args.replay_steps:
        parser.error(
            "--replay-context plus --world-sequence-length cannot exceed "
            "--replay-steps"
        )
    if not 0 <= args.random_steps <= args.environment_steps:
        parser.error("--random-steps must be between zero and --environment-steps")
    if args.save_every < 0 or args.progress_every < 0:
        parser.error("--save-every and --progress-every cannot be negative")
    if args.actor_learning_starts < 0:
        parser.error("--actor-learning-starts cannot be negative")
    if args.resume_refresh_steps < 1:
        parser.error("--resume-refresh-steps must be positive")
    if args.imagination_start_count < 0:
        parser.error("--imagination-start-count cannot be negative")
    if args.imagination_start_mode == "last" and args.imagination_start_count:
        parser.error(
            "--imagination-start-count must be zero with --imagination-start-mode last"
        )
    maximum_imagination_starts = args.batch_size * args.world_sequence_length
    if (
        args.imagination_start_mode == "all"
        and args.imagination_start_count > maximum_imagination_starts
    ):
        parser.error(
            "--imagination-start-count cannot exceed batch-size times "
            "world-sequence-length"
        )
    if args.free_nats < 0.0:
        parser.error("--free-nats cannot be negative")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must lie in (0,1]")
    if not 0.0 <= args.return_lambda <= 1.0:
        parser.error("--return-lambda must lie in [0,1]")
    if args.entropy_weight < 0.0 or args.smoothness_weight < 0.0:
        parser.error("entropy and smoothness weights cannot be negative")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay cannot be negative")
    if any(weight < 0.0 for weight in args.action_effort_weights):
        parser.error("--action-effort-weights cannot contain negative values")
    if args.actor_distribution_mode == "legacy_tanh_normal":
        if args.actor_initial_std <= 0.05:
            parser.error("--actor-initial-std must exceed 0.05")
    elif not 0.1 < args.actor_initial_std < 1.0:
        parser.error("bounded-normal --actor-initial-std must lie in (0.1,1.0)")
    if args.exploration_std < 0.0:
        parser.error("--exploration-std cannot be negative")
    if args.resume is not None and args.checkpoint is None:
        args.checkpoint = args.resume
    if args.resume is not None and args.replay_dir is None:
        parser.error("--resume requires --replay-dir for fail-closed replay restore")
    return args


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
