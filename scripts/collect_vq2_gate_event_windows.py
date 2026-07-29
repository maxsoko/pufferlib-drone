#!/usr/bin/env python3
"""Collect actor-frozen VQ2 native gate-event windows for world/reward fitting.

Only the checkpoint's recurrent policy acts. Native privileged values are
stored as training targets but never enter the actor. The collector has no
optimizer, never connects to FlightSim, and stops at a source-locked event or
vector-step bound. Gate-local sampled collection remains the default; explicit
flags permit deterministic, uninterrupted policy prefixes for ordered-course
transition diagnosis.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib import _C
from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer
from pufferlib.vq2_informed import (
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    MASK_SIZE,
    configure_full_start_collection,
)
from scripts.continue_vq2_informed_dreamer_actor_offline import _restore_rng, _sha256
from scripts.train_vq2_informed_dreamer import (
    _apply_collector_action_hold,
    _load_config,
    _reset_state_where,
    _tensor_from_pointer,
)


EVENT_WINDOW_SCHEMA = "vq2_actor_frozen_gate_event_windows_v1"
EVENT_WINDOW_BOUNDARY_SCHEMA = "vq2_actor_frozen_gate_event_windows_rssm_boundary_v2"
EVENT_WINDOW_BOUNDARY_FP32_SCHEMA = (
    "vq2_actor_frozen_gate_event_windows_rssm_boundary_fp32_v3"
)
EVENT_WINDOW_EXACT_HISTORY_SCHEMA = (
    "vq2_actor_frozen_gate_event_windows_exact_history_v4"
)


def _configure_event_collection(
    config: dict,
    *,
    offset_min: float,
    offset_max: float,
    full_start_only: bool = False,
    episode_offset: int = 0,
) -> dict:
    if offset_min < 0.0 or offset_max < offset_min:
        raise ValueError("gate-local offsets must satisfy 0 <= min <= max")
    if episode_offset < 0:
        raise ValueError("episode offset cannot be negative")
    configured = copy.deepcopy(config)
    environment = configured.get("env")
    if not isinstance(environment, dict):
        raise RuntimeError("VQ2 config lacks an env section")
    if full_start_only:
        configure_full_start_collection(configured)
    else:
        environment["gate_local_start_curriculum"] = 1
        environment["gate_local_start_probability"] = 1.0
        environment["gate_local_start_offset_min"] = offset_min
        environment["gate_local_start_offset_max"] = offset_max
        environment["mixed_start_curriculum"] = 0
        environment["segment_start_probability"] = 0.0
        environment["evaluation_episode_limit"] = 0
    environment["evaluation_episode_offset"] = episode_offset
    return configured


def _window_indices(write_count: int, window_length: int) -> np.ndarray:
    if write_count < window_length:
        raise ValueError("insufficient history for an event window")
    return np.arange(write_count - window_length, write_count) % window_length


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_sha_before = _sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = payload.get("args")
    if not isinstance(training_args, dict):
        raise RuntimeError("checkpoint lacks training arguments")
    actor_distribution_mode = str(
        training_args.get("actor_distribution_mode", "legacy_tanh_normal")
    )
    model = VQ2InformedDreamer(
        deterministic_size=int(training_args["deterministic_size"]),
        stochastic_groups=int(training_args["stochastic_groups"]),
        stochastic_classes=int(training_args["stochastic_classes"]),
        actor_initial_std=float(training_args.get("actor_initial_std", 0.20)),
        actor_distribution_mode=actor_distribution_mode,
        distributional_reward=bool(training_args.get("distributional_reward", False)),
        action_conditioned_reward=bool(
            training_args.get("action_conditioned_reward", False)
        ),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    source_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _restore_rng(payload, device)

    config = _configure_event_collection(
        _load_config(args.env_name),
        offset_min=args.gate_local_offset_min,
        offset_max=args.gate_local_offset_max,
        full_start_only=args.full_start_only,
        episode_offset=args.episode_offset,
    )
    vec = _C.create_vec(config, gpu=0)
    if vec.obs_size != ENV_OBS_SIZE:
        vec.close()
        raise RuntimeError(
            f"native ABI mismatch: expected {ENV_OBS_SIZE}, got {vec.obs_size}"
        )
    vec.reset()
    observations = _tensor_from_pointer(
        vec.obs_ptr, vec.total_agents * vec.obs_size
    ).reshape(vec.total_agents, vec.obs_size)
    rewards = _tensor_from_pointer(vec.rewards_ptr, vec.total_agents)
    terminals = _tensor_from_pointer(vec.terminals_ptr, vec.total_agents)
    actions_cpu = torch.zeros((vec.total_agents, model.action_size), dtype=torch.float32)

    length = args.context_length + 1
    window_float_dtype = (
        np.float32 if args.window_storage_dtype == "float32" else np.float16
    )
    mask_history = np.zeros((length, vec.total_agents, MASK_SIZE), dtype=np.uint8)
    tail_history = np.zeros(
        (length, vec.total_agents, ENV_OBS_SIZE - MASK_SIZE),
        dtype=window_float_dtype,
    )
    action_history = np.zeros(
        (length, vec.total_agents, model.action_size), dtype=window_float_dtype
    )
    reward_history = np.zeros((length, vec.total_agents), dtype=np.float16)
    continuation_history = np.zeros((length, vec.total_agents), dtype=np.uint8)
    valid_history = np.zeros((length, vec.total_agents), dtype=np.uint8)
    boundary_deterministic_history = None
    boundary_stochastic_index_history = None
    boundary_logits_history = None
    if args.store_initial_rssm_state:
        boundary_float_dtype = (
            np.float32
            if args.boundary_storage_dtype == "float32"
            else np.float16
        )
        boundary_deterministic_history = np.zeros(
            (length, vec.total_agents, model.rssm.deterministic_size),
            dtype=boundary_float_dtype,
        )
        boundary_stochastic_index_history = np.zeros(
            (length, vec.total_agents, model.rssm.stochastic_groups),
            dtype=np.uint8,
        )
        boundary_logits_history = np.zeros(
            (
                length,
                vec.total_agents,
                model.rssm.stochastic_groups,
                model.rssm.stochastic_classes,
            ),
            dtype=boundary_float_dtype,
        )
    state: RSSMState = model.rssm.initial(vec.total_agents, device=device)
    previous_action = torch.zeros(
        (vec.total_agents, model.action_size), device=device
    )
    held_action = torch.zeros_like(previous_action)
    hold_remaining = torch.zeros(
        vec.total_agents, dtype=torch.long, device=device
    )
    pending_reset = torch.zeros(vec.total_agents, dtype=torch.bool)
    captured_mask: list[np.ndarray] = []
    captured_tail: list[np.ndarray] = []
    captured_action: list[np.ndarray] = []
    captured_reward: list[np.ndarray] = []
    captured_continuation: list[np.ndarray] = []
    captured_agent: list[int] = []
    captured_step: list[int] = []
    captured_phase: list[float] = []
    captured_initial_deterministic: list[np.ndarray] = []
    captured_initial_stochastic_index: list[np.ndarray] = []
    captured_initial_logits: list[np.ndarray] = []
    captured_preevent_deterministic: list[np.ndarray] = []
    captured_preevent_stochastic_index: list[np.ndarray] = []
    captured_preevent_logits: list[np.ndarray] = []
    raw_event_count = 0
    skipped_insufficient_history = 0
    skipped_invalid_history = 0
    action_sum = torch.zeros(model.action_size, dtype=torch.float64)
    action_square_sum = torch.zeros(model.action_size, dtype=torch.float64)
    action_abs_max = torch.zeros(model.action_size)
    action_samples = 0
    vector_steps = 0

    try:
        for vector_step in range(args.max_vector_steps):
            legal = observations[:, :LEGAL_OBS_SIZE].to(device)
            with torch.no_grad():
                distribution, state = model.policy_distribution_step(
                    legal, previous_action, state
                )
                if args.deterministic_policy:
                    candidate_action = model.deterministic_actor_action(distribution)
                else:
                    candidate_action = model.sample_actor_action(distribution)
                candidate_action.clamp_(-1.0, 1.0)
            action, held_action, hold_remaining = _apply_collector_action_hold(
                candidate_action,
                held_action,
                hold_remaining,
                args.collector_action_hold,
            )
            if not torch.isfinite(action).all():
                raise RuntimeError("collector actor produced a non-finite action")
            active = ~pending_reset
            if active.any():
                selected = action[active.to(device)].detach().cpu()
                action_sum += selected.double().sum(0)
                action_square_sum += selected.double().square().sum(0)
                action_abs_max = torch.maximum(action_abs_max, selected.abs().amax(0))
                action_samples += int(selected.shape[0])
            actions_cpu.copy_(action.detach().cpu())
            vec.cpu_step(actions_cpu.data_ptr())
            reward_copy = rewards.clone()
            terminal_copy = terminals.clone()
            continuation = 1.0 - terminal_copy
            valid_transition = ~pending_reset

            slot = vector_step % length
            observation_np = observations.detach().cpu().numpy()
            mask_history[slot] = np.rint(
                np.clip(observation_np[:, :MASK_SIZE], 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            tail_history[slot] = observation_np[:, MASK_SIZE:].astype(
                window_float_dtype
            )
            action_history[slot] = actions_cpu.numpy().astype(window_float_dtype)
            reward_history[slot] = reward_copy.numpy().astype(np.float16)
            continuation_history[slot] = (continuation.numpy() > 0.5).astype(np.uint8)
            valid_history[slot] = valid_transition.numpy().astype(np.uint8)
            if args.store_initial_rssm_state:
                assert boundary_deterministic_history is not None
                assert boundary_stochastic_index_history is not None
                assert boundary_logits_history is not None
                boundary_deterministic_history[slot] = (
                    state.deterministic.detach().cpu().numpy().astype(
                        boundary_float_dtype
                    )
                )
                boundary_stochastic_index_history[slot] = (
                    state.stochastic.detach().argmax(-1).cpu().numpy().astype(np.uint8)
                )
                boundary_logits_history[slot] = (
                    state.logits.detach().cpu().numpy().astype(boundary_float_dtype)
                )
            vector_steps = vector_step + 1

            event_agents = torch.nonzero(
                valid_transition & (reward_copy > args.event_threshold),
                as_tuple=False,
            ).flatten().tolist()
            raw_event_count += len(event_agents)
            for agent in event_agents:
                if vector_steps < length:
                    skipped_insufficient_history += 1
                    continue
                indices = _window_indices(vector_steps, length)
                if not valid_history[indices, agent].all() or not (
                    continuation_history[indices[:-1], agent].all()
                ):
                    skipped_invalid_history += 1
                    continue
                captured_mask.append(mask_history[indices, agent].copy())
                captured_tail.append(tail_history[indices, agent].copy())
                captured_action.append(action_history[indices, agent].copy())
                captured_reward.append(reward_history[indices, agent].copy())
                captured_continuation.append(
                    continuation_history[indices, agent].copy()
                )
                captured_agent.append(int(agent))
                captured_step.append(vector_step)
                captured_phase.append(
                    float(tail_history[slot, agent, LEGAL_OBS_SIZE - MASK_SIZE + 32])
                )
                if args.store_initial_rssm_state:
                    assert boundary_deterministic_history is not None
                    assert boundary_stochastic_index_history is not None
                    assert boundary_logits_history is not None
                    first = int(indices[0])
                    captured_initial_deterministic.append(
                        boundary_deterministic_history[first, agent].copy()
                    )
                    captured_initial_stochastic_index.append(
                        boundary_stochastic_index_history[first, agent].copy()
                    )
                    captured_initial_logits.append(
                        boundary_logits_history[first, agent].copy()
                    )
                    last = int(indices[-1])
                    captured_preevent_deterministic.append(
                        boundary_deterministic_history[last, agent].copy()
                    )
                    captured_preevent_stochastic_index.append(
                        boundary_stochastic_index_history[last, agent].copy()
                    )
                    captured_preevent_logits.append(
                        boundary_logits_history[last, agent].copy()
                    )
                if len(captured_mask) >= args.target_events:
                    break

            terminal_device = terminal_copy.to(device).bool()
            reset_device = terminal_device | pending_reset.to(device)
            state = _reset_state_where(model, state, reset_device)
            held_action = torch.where(
                reset_device[:, None], torch.zeros_like(held_action), held_action
            )
            hold_remaining = torch.where(
                reset_device, torch.zeros_like(hold_remaining), hold_remaining
            )
            previous_action = torch.where(
                reset_device[:, None], torch.zeros_like(action), action
            )
            pending_reset = terminal_copy.bool()
            if len(captured_mask) >= args.target_events:
                break
    finally:
        native_log = dict(vec.log())
        vec.close()

    if not captured_mask:
        raise RuntimeError("collector captured no complete gate-event window")
    dataset = {
        "mask": np.stack(captured_mask),
        "tail": np.stack(captured_tail),
        "action": np.stack(captured_action),
        "reward": np.stack(captured_reward),
        "continuation": np.stack(captured_continuation),
        "agent": np.asarray(captured_agent, dtype=np.int16),
        "vector_step": np.asarray(captured_step, dtype=np.int32),
        "phase_after_event": np.asarray(captured_phase, dtype=np.float16),
    }
    if args.store_initial_rssm_state:
        dataset.update(
            {
                "initial_deterministic": np.stack(captured_initial_deterministic),
                "initial_stochastic_index": np.stack(
                    captured_initial_stochastic_index
                ),
                "initial_logits": np.stack(captured_initial_logits),
                "preevent_deterministic": np.stack(
                    captured_preevent_deterministic
                ),
                "preevent_stochastic_index": np.stack(
                    captured_preevent_stochastic_index
                ),
                "preevent_logits": np.stack(captured_preevent_logits),
            }
        )
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    temporary_dataset = args.dataset.with_suffix(args.dataset.suffix + ".tmp")
    with temporary_dataset.open("wb") as stream:
        np.savez_compressed(stream, **dataset)
    temporary_dataset.replace(args.dataset)

    final_model_state = model.state_dict()
    changed_model_keys = [
        name
        for name, value in final_model_state.items()
        if not torch.equal(value, source_model_state[name])
    ]
    checkpoint_sha_after = _sha256(args.checkpoint)
    if changed_model_keys:
        raise RuntimeError(f"collector changed model tensors: {changed_model_keys}")
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("collector changed source checkpoint")
    if args.full_start_only and float(native_log.get("gate_local_reset_count", 0.0)) != 0.0:
        raise RuntimeError("full-start collector observed a gate-local reset")
    action_mean = action_sum / max(1, action_samples)
    action_variance = (
        action_square_sum / max(1, action_samples) - action_mean.square()
    ).clamp_min(0.0)
    boundary_replay = None
    if args.store_initial_rssm_state:
        mask_tensor = torch.from_numpy(dataset["mask"].astype(np.float32)).to(device)
        mask_tensor /= 255.0
        tail_tensor = torch.from_numpy(dataset["tail"].astype(np.float32)).to(device)
        observation_tensor = torch.cat((mask_tensor, tail_tensor), -1)
        action_tensor = torch.from_numpy(dataset["action"].astype(np.float32)).to(
            device
        )
        initial_logits = torch.from_numpy(
            dataset["initial_logits"].astype(np.float32)
        ).to(device)
        initial_index = torch.from_numpy(
            dataset["initial_stochastic_index"].astype(np.int64)
        ).to(device)
        initial_state = RSSMState(
            torch.from_numpy(dataset["initial_deterministic"].astype(np.float32)).to(
                device
            ),
            F.one_hot(
                initial_index, model.rssm.stochastic_classes
            ).to(torch.float32),
            initial_logits,
        )
        with torch.no_grad():
            replayed = model.observe_sequence(
                observation_tensor[:, :-1],
                action_tensor[:, :-1],
                initial_state=initial_state,
                deterministic_latent=True,
            ).posterior
        expected_deterministic = torch.from_numpy(
            dataset["preevent_deterministic"].astype(np.float32)
        ).to(device)
        expected_logits = torch.from_numpy(
            dataset["preevent_logits"].astype(np.float32)
        ).to(device)
        expected_index = torch.from_numpy(
            dataset["preevent_stochastic_index"].astype(np.int64)
        ).to(device)
        boundary_replay = {
            "deterministic_max_abs_error": float(
                (
                    replayed.deterministic[:, -1].float()
                    - expected_deterministic
                ).abs().max().cpu()
            ),
            "logits_max_abs_error": float(
                (replayed.logits[:, -1].float() - expected_logits).abs().max().cpu()
            ),
            "stochastic_index_mismatch_count": int(
                (
                    replayed.stochastic[:, -1].argmax(-1)
                    != expected_index
                ).sum().cpu()
            ),
        }
    report: dict[str, object] = {
        "contract": (
            (
                EVENT_WINDOW_EXACT_HISTORY_SCHEMA
                if args.window_storage_dtype == "float32"
                else (
                    EVENT_WINDOW_BOUNDARY_FP32_SCHEMA
                    if args.boundary_storage_dtype == "float32"
                    else EVENT_WINDOW_BOUNDARY_SCHEMA
                )
            )
            if args.store_initial_rssm_state
            else EVENT_WINDOW_SCHEMA
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha_after,
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256_bytes(args.dataset),
        "dataset_bytes": args.dataset.stat().st_size,
        "target_events": args.target_events,
        "captured_events": len(captured_mask),
        "raw_native_events": raw_event_count,
        "distinct_agents": len(set(captured_agent)),
        "distinct_vector_steps": len(set(captured_step)),
        "captured_agents": captured_agent,
        "captured_vector_steps": captured_step,
        "captured_phase_after_event": captured_phase,
        "context_length": args.context_length,
        "window_length": length,
        "event_at_window_index": args.context_length,
        "initial_rssm_state_stored": int(args.store_initial_rssm_state),
        "boundary_storage_dtype": (
            args.boundary_storage_dtype
            if args.store_initial_rssm_state
            else None
        ),
        "window_storage_dtype": args.window_storage_dtype,
        "initial_rssm_state_semantics": (
            "legal-derived posterior boundary before window index 0"
            if args.store_initial_rssm_state
            else None
        ),
        "preevent_rssm_state_semantics": (
            "legal-derived posterior boundary before event observation"
            if args.store_initial_rssm_state
            else None
        ),
        "boundary_replay": boundary_replay,
        "vector_steps": vector_steps,
        "max_vector_steps": args.max_vector_steps,
        "valid_agent_transitions": action_samples,
        "event_rate_per_valid_transition": raw_event_count / max(1, action_samples),
        "skipped_insufficient_history": skipped_insufficient_history,
        "skipped_invalid_history": skipped_invalid_history,
        "collection_start_contract": (
            "uninterrupted_full_course"
            if args.full_start_only
            else "gate_local"
        ),
        "full_start_only": int(args.full_start_only),
        "evaluation_episode_offset": args.episode_offset,
        "gate_local_start_probability": (
            0.0 if args.full_start_only else 1.0
        ),
        "gate_local_offset_min": (
            None if args.full_start_only else args.gate_local_offset_min
        ),
        "gate_local_offset_max": (
            None if args.full_start_only else args.gate_local_offset_max
        ),
        "collector_policy_sampling": int(not args.deterministic_policy),
        "collector_deterministic_policy": int(args.deterministic_policy),
        "collector_action_hold": args.collector_action_hold,
        "actor_distribution_mode": actor_distribution_mode,
        "collector_action_mean": action_mean.tolist(),
        "collector_action_std": action_variance.sqrt().tolist(),
        "collector_action_abs_max": action_abs_max.tolist(),
        "native_log": {key: float(value) for key, value in native_log.items()},
        "model_tensor_changes": changed_model_keys,
        "checkpoint_unchanged": 1,
        "optimizer_steps": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "world_updates": 0,
        "reward_updates": 0,
        "privileged_actor_input": 0,
        "teacher_actions": 0,
        "analytic_actions": 0,
        "native_environment_created": 1,
        "flightsim_packets": 0,
        "device": str(device),
    }
    if device.type == "cuda":
        report["cuda_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_report.replace(args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-name", default="drone_race_vq2_informed_dreamer")
    parser.add_argument("--target-events", type=int, required=True)
    parser.add_argument("--max-vector-steps", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--event-threshold", type=float, default=1.0)
    parser.add_argument("--gate-local-offset-min", type=float, default=0.75)
    parser.add_argument("--gate-local-offset-max", type=float, default=1.50)
    parser.add_argument("--full-start-only", action="store_true")
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--deterministic-policy", action="store_true")
    parser.add_argument("--collector-action-hold", type=int, default=1)
    parser.add_argument("--store-initial-rssm-state", action="store_true")
    parser.add_argument(
        "--boundary-storage-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--window-storage-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    for name in ("target_events", "max_vector_steps", "context_length", "collector_action_hold"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.event_threshold <= 0.0:
        parser.error("--event-threshold must be positive")
    if args.gate_local_offset_min < 0.0:
        parser.error("--gate-local-offset-min cannot be negative")
    if args.gate_local_offset_max < args.gate_local_offset_min:
        parser.error("--gate-local-offset-max cannot be below the minimum")
    if args.episode_offset < 0:
        parser.error("--episode-offset cannot be negative")
    if args.window_storage_dtype == "float32" and (
        not args.store_initial_rssm_state
        or args.boundary_storage_dtype != "float32"
    ):
        parser.error(
            "float32 window storage requires stored float32 RSSM boundaries"
        )
    if args.dataset.exists() or args.report.exists():
        parser.error("collector refuses to overwrite dataset or report")
    return args


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), indent=2, sort_keys=True))
