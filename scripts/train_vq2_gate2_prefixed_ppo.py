#!/usr/bin/env python3
"""Fine-tune a secondary Puffer policy from the official N295 recurrent state.

Admission and deployment always execute complete Puffer actions.  Offline
training can optionally use bounded teacher assistance to keep exploratory
trajectories recoverable; every resulting checkpoint is screened again with
zero assistance.  An optional frozen Puffer reference can also supply action-
consistency targets from the same legal, unscaled observations.  At deployment
N294 remains in control through Gate 1 while this secondary recurrent policy
observes the same legal stream in parallel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

try:
    from eval_drone_race_checkpoint import _load_config
    from eval_vq2_gate2_prefixed_checkpoint import (
        _buffer,
        _flat_vec_log,
        _overrides,
        advance_forward_gate_rate_world_x,
        LAST_ACTION_START,
        START_PERTURBATION_DIMENSIONS,
        gate2_first_observation,
        gate_rate_world_from_observation,
        linear_gate_features,
        load_policy_prefix,
        scale_gate_observations,
        sha256_file,
        world_frame_gate_features,
    )
    from policy_callable_checkpoint import CheckpointPolicy
    from train_full_policy_bc import SequencePufferNet, save_native_checkpoint
except ModuleNotFoundError:  # Imported as scripts.train_vq2_gate2_prefixed_ppo.
    from scripts.eval_drone_race_checkpoint import _load_config
    from scripts.eval_vq2_gate2_prefixed_checkpoint import (
        _buffer,
        _flat_vec_log,
        _overrides,
        advance_forward_gate_rate_world_x,
        LAST_ACTION_START,
        START_PERTURBATION_DIMENSIONS,
        gate2_first_observation,
        gate_rate_world_from_observation,
        linear_gate_features,
        load_policy_prefix,
        scale_gate_observations,
        sha256_file,
        world_frame_gate_features,
    )
    from scripts.policy_callable_checkpoint import CheckpointPolicy
    from scripts.train_full_policy_bc import SequencePufferNet, save_native_checkpoint


ACTIONS = 4
ACTION_NAMES = ("pitch", "roll", "thrust", "yaw")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(next_value)
    following = next_value
    for step in range(rewards.shape[0] - 1, -1, -1):
        not_done = 1.0 - dones[step]
        delta = rewards[step] + gamma * following * not_done - values[step]
        carry = delta + gamma * gae_lambda * not_done * carry
        advantages[step] = carry
        following = values[step]
    return advantages, advantages + values


def reset_recurrent_state(
    state: torch.Tensor,
    done: torch.Tensor,
    reset_state: torch.Tensor,
) -> torch.Tensor:
    if state.shape != reset_state.shape:
        raise ValueError("state and reset_state shapes differ")
    if done.ndim != 1 or done.shape[0] != state.shape[1]:
        raise ValueError("done mask does not match the agent axis")
    return torch.where(done.view(1, -1, 1), reset_state, state)


def mask_policy_gradients(
    model: SequencePufferNet,
    trainable_action_rows: tuple[int, ...] | None,
) -> None:
    """Restrict an update to selected Puffer action-decoder rows.

    The value row and all encoder/recurrent parameters remain frozen in this
    mode.  This lets a phase-local PPO experiment learn, for example, a
    state-dependent thrust recovery without perturbing an already useful
    steering trajectory.
    """
    if trainable_action_rows is None:
        return
    selected = set(trainable_action_rows)
    if not selected or any(index < 0 or index >= ACTIONS for index in selected):
        raise ValueError("trainable action rows must be nonempty action indices")
    for parameter in model.parameters():
        if parameter is not model.decoder:
            parameter.grad = None
    if model.decoder.grad is not None:
        frozen = torch.ones(
            model.decoder.shape[0], dtype=torch.bool, device=model.decoder.device)
        frozen[list(selected)] = False
        model.decoder.grad[frozen] = 0


def warmed_state(
    model: SequencePufferNet,
    prefix: torch.Tensor,
    agents: int,
) -> torch.Tensor:
    state = model.initial_state(1, prefix.device)
    _, state = model.forward_chunk_outputs(prefix.unsqueeze(0), state)
    return state.repeat(1, agents, 1)


def gate2_initial_state(
    model: SequencePufferNet,
    prefix: torch.Tensor,
    agents: int,
    *,
    reset_recurrent_at_gate2: bool,
) -> torch.Tensor:
    if reset_recurrent_at_gate2:
        return model.initial_state(agents, prefix.device)
    return warmed_state(model, prefix, agents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("official_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--agents", type=int, default=128)
    parser.add_argument("--minibatch-agents", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--teacher-reward", type=float, default=50.0)
    parser.add_argument("--training-teacher-blend", type=float, default=0.0)
    parser.add_argument("--teacher-imitation-coef", type=float, default=0.0)
    parser.add_argument("--gate-camera-alignment-reward", type=float, default=0.0)
    parser.add_argument("--teacher-pitch-speed-target-m-s", type=float, default=1.30)
    parser.add_argument("--teacher-roll-per-m", type=float, default=0.35)
    parser.add_argument("--teacher-roll-rate-per-m-s", type=float, default=0.10)
    parser.add_argument("--reward-scale", type=float, default=0.005)
    parser.add_argument("--log-std", type=float, default=-4.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ppo-policy-coef", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--reference-policy-checkpoint", type=Path)
    parser.add_argument("--reference-policy-consistency-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--train-action-rows", nargs="+", choices=ACTION_NAMES,
        help=(
            "Freeze encoder, MinGRU, value row, and unlisted action rows; "
            "update only these Puffer action-decoder rows."
        ),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=8)
    parser.add_argument("--gate-radius", type=float, default=0.75)
    parser.add_argument("--gate1-x", type=float, default=14.74)
    parser.add_argument("--gate1-y", type=float, default=8.70)
    parser.add_argument("--gate1-z", type=float, default=1.095)
    parser.add_argument("--start-elapsed-time", type=float, default=3.25)
    parser.add_argument("--time-limit-seconds", type=float, default=20.0)
    parser.add_argument("--truncate-prefix-at-gate-advance", action="store_true")
    parser.add_argument("--policy-gate-observation-scale-min", type=float, default=1.0)
    parser.add_argument("--policy-gate-observation-scale-max", type=float, default=1.0)
    parser.add_argument("--policy-world-frame-gate-features", action="store_true")
    parser.add_argument("--policy-linear-gate-features", action="store_true")
    parser.add_argument("--policy-predict-forward-gate-rate", action="store_true")
    parser.add_argument("--carry-prefix-gate-rates", action="store_true")
    parser.add_argument("--perturbed", action="store_true")
    parser.add_argument("--perturbation-scale", type=float, default=1.0)
    parser.add_argument(
        "--perturbation-components", nargs="+",
        choices=("start", "gate", "plant"),
        default=("start", "gate", "plant"),
    )
    parser.add_argument(
        "--start-perturbation-dimensions", nargs="+",
        choices=START_PERTURBATION_DIMENSIONS,
        default=START_PERTURBATION_DIMENSIONS,
    )
    parser.add_argument(
        "--episodic-reset", action="store_true",
        help=(
            "Collect one complete episode per agent and recreate the native vector "
            "environment after every update, avoiding stale recurrent states."
        ),
    )
    parser.add_argument("--reset-recurrent-at-gate2", action="store_true")
    parser.add_argument("--predict-gate-motion-dropout", action="store_true")
    parser.add_argument("--gate-motion-control-accel-gain", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3330)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.updates, args.horizon, args.agents, args.minibatch_agents) <= 0:
        parser.error("updates, horizon, agents, and minibatch agents must be positive")
    if not np.isfinite(args.start_elapsed_time) or args.start_elapsed_time < 0.0:
        parser.error("--start-elapsed-time must be finite and nonnegative")
    if not np.isfinite(args.time_limit_seconds) or args.time_limit_seconds <= 0.0:
        parser.error("--time-limit-seconds must be finite and positive")
    if args.start_elapsed_time >= args.time_limit_seconds:
        parser.error("--start-elapsed-time must be less than --time-limit-seconds")
    if not np.isfinite((args.gate1_x, args.gate1_y, args.gate1_z)).all():
        parser.error("Gate-2 coordinates must be finite")
    if (
        not np.isfinite(args.policy_gate_observation_scale_min)
        or not np.isfinite(args.policy_gate_observation_scale_max)
        or args.policy_gate_observation_scale_min <= 0.0
        or args.policy_gate_observation_scale_max
        < args.policy_gate_observation_scale_min
    ):
        parser.error("invalid gate observation scale range")
    if not np.isfinite(args.perturbation_scale) or args.perturbation_scale <= 0.0:
        parser.error("--perturbation-scale must be finite and positive")
    if not np.isfinite(args.gate_motion_control_accel_gain):
        parser.error("--gate-motion-control-accel-gain must be finite")
    if not np.isfinite(args.gate_camera_alignment_reward) \
            or args.gate_camera_alignment_reward < 0.0:
        parser.error("--gate-camera-alignment-reward must be finite and nonnegative")
    if not 0.0 <= args.training_teacher_blend <= 1.0:
        parser.error("--training-teacher-blend must be in [0, 1]")
    if not np.isfinite(args.teacher_imitation_coef) or args.teacher_imitation_coef < 0.0:
        parser.error("--teacher-imitation-coef must be finite and nonnegative")
    if args.teacher_imitation_coef > 0.0 and args.training_teacher_blend <= 0.0:
        parser.error("teacher imitation requires a positive training teacher blend")
    if (
        not np.isfinite(args.teacher_pitch_speed_target_m_s)
        or args.teacher_pitch_speed_target_m_s <= 0.0
        or not np.isfinite(args.teacher_roll_per_m)
        or not np.isfinite(args.teacher_roll_rate_per_m_s)
    ):
        parser.error("teacher speed and roll gains must be finite and speed positive")
    if not np.isfinite(args.ppo_policy_coef) or args.ppo_policy_coef < 0.0:
        parser.error("--ppo-policy-coef must be finite and nonnegative")
    if (
        not np.isfinite(args.reference_policy_consistency_coef)
        or args.reference_policy_consistency_coef < 0.0
    ):
        parser.error(
            "--reference-policy-consistency-coef must be finite and nonnegative")
    if args.agents % args.minibatch_agents:
        parser.error("--agents must be divisible by --minibatch-agents")

    from pufferlib import _C, pufferl

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    trainable_action_rows = (
        tuple(ACTION_NAMES.index(name) for name in args.train_action_rows)
        if args.train_action_rows is not None else None
    )
    device = torch.device(args.device)
    prefix_observations, prefix_actions = load_policy_prefix(
        args.official_report,
        stop_at_gate_advance=args.truncate_prefix_at_gate_advance,
    )
    prefix = torch.from_numpy(prefix_observations).to(device)
    if args.carry_prefix_gate_rates:
        (
            args.gate_motion_initial_vx,
            args.gate_motion_initial_vy,
            args.gate_motion_initial_vz,
        ) = gate_rate_world_from_observation(prefix_observations[-1])
    source = CheckpointPolicy.load(
        str(args.checkpoint), input_dim=32, num_layers=3,
        layout_precision_bytes=4)
    model = SequencePufferNet(source).to(device)
    reference_model = None
    reference_checkpoint = args.reference_policy_checkpoint
    if args.reference_policy_consistency_coef > 0.0:
        reference_checkpoint = reference_checkpoint or args.checkpoint
        reference_source = CheckpointPolicy.load(
            str(reference_checkpoint), input_dim=32, num_layers=3,
            layout_precision_bytes=4)
        reference_model = SequencePufferNet(reference_source).to(device)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    fixed_log_std = torch.full((ACTIONS,), args.log_std, device=device)

    env_args = argparse.Namespace(
        episodes=args.agents,
        seed=args.seed,
        gate_radius=args.gate_radius,
        perturbed=args.perturbed,
        perturbation_scale=args.perturbation_scale,
        perturbation_components=args.perturbation_components,
        start_perturbation_dimensions=args.start_perturbation_dimensions,
        predict_gate_motion_dropout=args.predict_gate_motion_dropout,
        gate_motion_control_accel_gain=args.gate_motion_control_accel_gain,
        gate_motion_initial_vx=getattr(args, "gate_motion_initial_vx", 0.0),
        gate_motion_initial_vy=getattr(args, "gate_motion_initial_vy", 0.0),
        gate_motion_initial_vz=getattr(args, "gate_motion_initial_vz", 0.0),
        teacher_pitch_speed_target_m_s=args.teacher_pitch_speed_target_m_s,
        teacher_roll_per_m=args.teacher_roll_per_m,
        teacher_roll_rate_per_m_s=args.teacher_roll_rate_per_m_s,
        training_teacher_blend=args.training_teacher_blend,
        start_elapsed_time=args.start_elapsed_time,
        time_limit_seconds=args.time_limit_seconds,
        gate1_x=args.gate1_x,
        gate1_y=args.gate1_y,
        gate1_z=args.gate1_z,
    )
    overrides = _overrides(env_args)
    overrides.extend([
        "--env.w-progress", "35",
        "--env.w-gate", "30",
        "--env.w-finish", "180",
        "--env.w-time", "0.15",
        "--env.w-cross-track", "20",
        "--env.w-gate-camera-alignment", str(args.gate_camera_alignment_reward),
        "--env.gate-camera-alignment-from-gate-index", "1",
        "--env.w-gate-crossing-error", "30",
        "--env.gate-crossing-error-from-gate-index", "1",
        "--env.w-action-teacher", str(args.teacher_reward),
        "--env.teacher-pitch-from-gate-index", "1",
        "--env.teacher-pitch-speed-control", "1",
        "--env.teacher-pitch-speed-target-m-s", "1.30",
        "--env.teacher-pitch-speed-gain", "0.35",
        "--env.teacher-pitch-speed-scale", "0.24",
        "--env.teacher-roll-from-gate-index", "1",
        "--env.teacher-roll-until-gate-index", "2",
        "--env.teacher-thrust-from-gate-index", "1",
        "--env.teacher-yaw-control", "1",
        "--env.teacher-yaw-from-gate-index", "1",
        "--env.teacher-yaw-action", "0",
        "--env.teacher-roll-per-m", "0.35",
        "--env.teacher-roll-rate-per-m-s", "0.10",
        "--env.teacher-thrust-bias", "0",
        "--env.teacher-thrust-per-m", "0.30",
        "--env.teacher-thrust-rate-per-m-s", "0.10",
        "--env.teacher-thrust-world-frame", "1",
        "--env.invalid-penalty", "600",
    ])
    cfg = _load_config(pufferl, "drone_race_full_policy_six_gate_bootstrap", overrides)
    if args.episodic_reset:
        cfg.setdefault("env", {})["evaluation_episode_limit"] = 1
        cfg["env"]["evaluation_episode_offset"] = 0
    if getattr(_C, "precision_bytes", 0) != 4:
        raise RuntimeError("prefixed PPO requires the FP32 native backend")
    vec = _C.create_vec(cfg, gpu=0)
    vec.reset()
    observations = _buffer(
        vec.obs_ptr, vec.total_agents * vec.obs_size).reshape(vec.total_agents, vec.obs_size)
    rewards_buffer = _buffer(vec.rewards_ptr, vec.total_agents)
    terminals_buffer = _buffer(vec.terminals_ptr, vec.total_agents)
    needs_first_patch = np.ones(args.agents, dtype=np.bool_)
    predicted_forward_gate_rate = np.full(
        args.agents, float(args.gate_motion_initial_vx), dtype=np.float32)
    scale_rng = np.random.default_rng(args.seed ^ 0x56413252)
    policy_gate_observation_scales = scale_rng.uniform(
        args.policy_gate_observation_scale_min,
        args.policy_gate_observation_scale_max,
        size=args.agents,
    ).astype(np.float32)
    with torch.no_grad():
        state = gate2_initial_state(
            model, prefix, args.agents,
            reset_recurrent_at_gate2=args.reset_recurrent_at_gate2)
        reference_state = (
            gate2_initial_state(
                reference_model, prefix, args.agents,
                reset_recurrent_at_gate2=args.reset_recurrent_at_gate2)
            if reference_model is not None else None
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    total_transitions = 0
    started = time.time()
    try:
        for update in range(1, args.updates + 1):
            if args.episodic_reset and update > 1:
                vec.close()
                vec = _C.create_vec(cfg, gpu=0)
                vec.reset()
                observations = _buffer(
                    vec.obs_ptr, vec.total_agents * vec.obs_size).reshape(
                        vec.total_agents, vec.obs_size)
                rewards_buffer = _buffer(vec.rewards_ptr, vec.total_agents)
                terminals_buffer = _buffer(vec.terminals_ptr, vec.total_agents)
                needs_first_patch = np.ones(args.agents, dtype=np.bool_)
                predicted_forward_gate_rate.fill(float(args.gate_motion_initial_vx))
                policy_gate_observation_scales = scale_rng.uniform(
                    args.policy_gate_observation_scale_min,
                    args.policy_gate_observation_scale_max,
                    size=args.agents,
                ).astype(np.float32)
                with torch.no_grad():
                    state = gate2_initial_state(
                        model, prefix, args.agents,
                        reset_recurrent_at_gate2=args.reset_recurrent_at_gate2)
                    reference_state = (
                        gate2_initial_state(
                            reference_model, prefix, args.agents,
                            reset_recurrent_at_gate2=args.reset_recurrent_at_gate2)
                        if reference_model is not None else None
                    )
            initial_state = state.detach().clone()
            obs_steps: list[torch.Tensor] = []
            action_steps: list[torch.Tensor] = []
            logprob_steps: list[torch.Tensor] = []
            reward_steps: list[torch.Tensor] = []
            done_steps: list[torch.Tensor] = []
            valid_steps: list[torch.Tensor] = []
            value_steps: list[torch.Tensor] = []
            reference_action_steps: list[torch.Tensor] = []
            teacher_action_steps: list[torch.Tensor] = []
            completed = np.zeros(args.agents, dtype=np.bool_)
            collection_steps = (
                int(cfg["env"]["max_steps"]) + 2
                if args.episodic_reset else args.horizon
            )
            for _ in range(collection_steps):
                active_before_step = ~completed if args.episodic_reset else np.ones(
                    args.agents, dtype=np.bool_)
                first_patch = needs_first_patch.copy()
                current = observations.copy()
                if np.any(needs_first_patch):
                    current[needs_first_patch] = gate2_first_observation(
                        current[needs_first_patch],
                        prefix_actions[-1],
                        (
                            prefix_observations[-1, 0:3]
                            if args.carry_prefix_gate_rates else None
                        ),
                    )
                if reference_model is not None:
                    with torch.no_grad():
                        reference_outputs, reference_state = (
                            reference_model.forward_chunk_outputs(
                                torch.from_numpy(current).to(device).unsqueeze(1),
                                reference_state,
                            )
                        )
                    reference_action_steps.append(reference_outputs[:, 0, :ACTIONS])
                current = scale_gate_observations(
                    current, policy_gate_observation_scales)
                if args.policy_predict_forward_gate_rate and np.any(~first_patch):
                    predicted_forward_gate_rate[~first_patch] = (
                        advance_forward_gate_rate_world_x(
                            predicted_forward_gate_rate[~first_patch],
                            current[~first_patch],
                            dt_s=float(cfg["env"]["dt"]),
                        )
                    )
                if args.policy_world_frame_gate_features:
                    current = world_frame_gate_features(current)
                if args.policy_predict_forward_gate_rate:
                    if not args.policy_world_frame_gate_features:
                        raise RuntimeError(
                            "forward gate-rate prediction requires world-frame features")
                    current[:, 0] = np.tanh(predicted_forward_gate_rate / 5.0)
                if args.policy_linear_gate_features:
                    current = linear_gate_features(current)
                current_tensor = torch.from_numpy(current).to(device)
                with torch.no_grad():
                    outputs, state = model.forward_chunk_outputs(
                        current_tensor.unsqueeze(1), state)
                    mean = outputs[:, 0, :ACTIONS]
                    value = outputs[:, 0, ACTIONS]
                    distribution = torch.distributions.Normal(mean, fixed_log_std.exp())
                    raw_action = distribution.sample()
                    logprob = distribution.log_prob(raw_action).sum(-1)
                plant_action = torch.clamp(raw_action, -1.0, 1.0).float().cpu().contiguous()
                vec.cpu_step(plant_action.data_ptr())
                reward = torch.from_numpy(rewards_buffer.copy()).to(device)
                done = torch.from_numpy((terminals_buffer >= 0.5).copy()).to(device)
                if args.teacher_imitation_coef > 0.0:
                    executed_action = torch.from_numpy(
                        observations[
                            :, LAST_ACTION_START:LAST_ACTION_START + ACTIONS
                        ].copy()
                    ).to(device)
                    teacher_action_steps.append(torch.clamp(
                        (
                            executed_action
                            - (1.0 - args.training_teacher_blend) * plant_action.to(device)
                        )
                        / args.training_teacher_blend,
                        -1.0,
                        1.0,
                    ))
                valid = torch.from_numpy(active_before_step.copy()).to(device)
                if args.episodic_reset:
                    reward = reward * valid
                    done = done | ~valid
                obs_steps.append(current_tensor)
                action_steps.append(raw_action)
                logprob_steps.append(logprob)
                reward_steps.append(reward * args.reward_scale)
                done_steps.append(done.float())
                valid_steps.append(valid)
                value_steps.append(value)
                completed |= terminals_buffer >= 0.5
                if not args.episodic_reset and bool(done.any()):
                    with torch.no_grad():
                        reset_state = gate2_initial_state(
                            model, prefix, args.agents,
                            reset_recurrent_at_gate2=args.reset_recurrent_at_gate2)
                        state = reset_recurrent_state(state, done, reset_state)
                        if reference_model is not None:
                            reference_reset_state = gate2_initial_state(
                                reference_model, prefix, args.agents,
                                reset_recurrent_at_gate2=args.reset_recurrent_at_gate2)
                            reference_state = reset_recurrent_state(
                                reference_state, done, reference_reset_state)
                needs_first_patch = (
                    np.zeros(args.agents, dtype=np.bool_)
                    if args.episodic_reset else done.cpu().numpy()
                )
                if not args.episodic_reset and bool(done.any()):
                    predicted_forward_gate_rate[done.cpu().numpy()] = float(
                        args.gate_motion_initial_vx)
                if args.episodic_reset and bool(np.all(completed)):
                    break

            if args.episodic_reset:
                next_value = torch.zeros(args.agents, device=device)
            else:
                next_observation = observations.copy()
                if np.any(needs_first_patch):
                    next_observation[needs_first_patch] = gate2_first_observation(
                        next_observation[needs_first_patch],
                        prefix_actions[-1],
                        (
                            prefix_observations[-1, 0:3]
                            if args.carry_prefix_gate_rates else None
                        ),
                    )
                next_observation = scale_gate_observations(
                    next_observation, policy_gate_observation_scales)
                next_predicted_forward_gate_rate = predicted_forward_gate_rate
                if args.policy_predict_forward_gate_rate:
                    next_predicted_forward_gate_rate = predicted_forward_gate_rate.copy()
                    if np.any(~needs_first_patch):
                        next_predicted_forward_gate_rate[~needs_first_patch] = (
                            advance_forward_gate_rate_world_x(
                                predicted_forward_gate_rate[~needs_first_patch],
                                next_observation[~needs_first_patch],
                                dt_s=float(cfg["env"]["dt"]),
                            )
                        )
                if args.policy_world_frame_gate_features:
                    next_observation = world_frame_gate_features(next_observation)
                if args.policy_predict_forward_gate_rate:
                    next_observation[:, 0] = np.tanh(
                        next_predicted_forward_gate_rate / 5.0)
                if args.policy_linear_gate_features:
                    next_observation = linear_gate_features(next_observation)
                with torch.no_grad():
                    next_outputs, _ = model.forward_chunk_outputs(
                        torch.from_numpy(next_observation).to(device).unsqueeze(1), state)
                    next_value = next_outputs[:, 0, ACTIONS]

            obs_batch = torch.stack(obs_steps)
            actions_batch = torch.stack(action_steps)
            old_logprob = torch.stack(logprob_steps)
            rewards = torch.stack(reward_steps)
            dones = torch.stack(done_steps)
            valid_mask = torch.stack(valid_steps)
            old_values = torch.stack(value_steps)
            reference_actions = (
                torch.stack(reference_action_steps)
                if reference_model is not None else None
            )
            teacher_actions = (
                torch.stack(teacher_action_steps)
                if args.teacher_imitation_coef > 0.0 else None
            )
            advantages, returns = compute_gae(
                rewards, old_values, dones, next_value,
                gamma=args.gamma, gae_lambda=args.gae_lambda)
            active_advantages = advantages[valid_mask]
            advantages = (
                advantages - active_advantages.mean()
            ) / (active_advantages.std() + 1e-8)

            agent_order = list(range(args.agents))
            policy_loss_total = 0.0
            value_loss_total = 0.0
            consistency_loss_total = 0.0
            teacher_imitation_loss_total = 0.0
            batches = 0
            model.train()
            for _epoch in range(args.update_epochs):
                rng.shuffle(agent_order)
                for start in range(0, args.agents, args.minibatch_agents):
                    indices = torch.tensor(
                        agent_order[start:start + args.minibatch_agents],
                        device=device, dtype=torch.long)
                    train_state = initial_state[:, indices].detach()
                    reset_state = gate2_initial_state(
                        model, prefix, len(indices),
                        reset_recurrent_at_gate2=args.reset_recurrent_at_gate2)
                    sequence_outputs: list[torch.Tensor] = []
                    for step in range(len(obs_steps)):
                        output, train_state = model.forward_chunk_outputs(
                            obs_batch[step, indices].unsqueeze(1), train_state)
                        sequence_outputs.append(output[:, 0])
                        train_state = reset_recurrent_state(
                            train_state, dones[step, indices].bool(), reset_state)
                    outputs = torch.stack(sequence_outputs)
                    means = outputs[..., :ACTIONS]
                    values = outputs[..., ACTIONS]
                    distribution = torch.distributions.Normal(means, fixed_log_std.exp())
                    new_logprob = distribution.log_prob(
                        actions_batch[:, indices]).sum(-1)
                    ratio = (new_logprob - old_logprob[:, indices]).exp()
                    advantage = advantages[:, indices]
                    valid = valid_mask[:, indices]
                    policy_losses = torch.maximum(
                        -advantage * ratio,
                        -advantage * torch.clamp(
                            ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef),
                    )
                    policy_loss = policy_losses[valid].mean()
                    value_loss = 0.5 * torch.square(
                        values[valid] - returns[:, indices][valid]).mean()
                    consistency_loss = torch.zeros((), device=device)
                    if reference_actions is not None:
                        consistency_loss = torch.square(
                            means[valid] - reference_actions[:, indices][valid]
                        ).mean()
                    teacher_imitation_loss = torch.zeros((), device=device)
                    if teacher_actions is not None:
                        teacher_valid = valid & ~dones[:, indices].bool()
                        teacher_imitation_loss = torch.square(
                            means[teacher_valid]
                            - teacher_actions[:, indices][teacher_valid]
                        ).mean()
                    loss = (
                        args.ppo_policy_coef * policy_loss
                        + args.value_coef * value_loss
                        + args.reference_policy_consistency_coef * consistency_loss
                        + args.teacher_imitation_coef * teacher_imitation_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    mask_policy_gradients(model, trainable_action_rows)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    policy_loss_total += float(policy_loss.detach())
                    value_loss_total += float(value_loss.detach())
                    consistency_loss_total += float(consistency_loss.detach())
                    teacher_imitation_loss_total += float(
                        teacher_imitation_loss.detach())
                    batches += 1
            model.eval()
            metrics = _flat_vec_log(pufferl, dict(vec.log()))
            rollout_transitions = int(valid_mask.sum().item())
            total_transitions += rollout_transitions
            row = {
                "update": update,
                "steps": total_transitions,
                "rollout_transitions": rollout_transitions,
                "policy_loss": policy_loss_total / max(batches, 1),
                "value_loss": value_loss_total / max(batches, 1),
                "consistency_loss": consistency_loss_total / max(batches, 1),
                "teacher_imitation_loss": (
                    teacher_imitation_loss_total / max(batches, 1)),
                "episodes": metrics.get("env/n", 0.0),
                "success_rate": metrics.get("env/success_rate", 0.0),
                "crash_rate": metrics.get("env/crash", 0.0),
                "miss_rate": metrics.get("env/missed_gate", 0.0),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if update % args.checkpoint_interval == 0 or update == args.updates:
                checkpoint_name = (
                    f"update_{update:04d}.bin" if args.episodic_reset
                    else f"{update * args.horizon * args.agents:016d}.bin"
                )
                save_native_checkpoint(
                    args.checkpoint,
                    args.output_dir / checkpoint_name,
                    model,
                    layout_precision_bytes=4,
                )
    finally:
        vec.close()

    summary = {
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "official_report": str(args.official_report),
        "official_report_sha256": sha256_file(args.official_report),
        "teacher_action_blend": args.training_teacher_blend,
        "teacher_imitation_coef": args.teacher_imitation_coef,
        "teacher_reward": args.teacher_reward,
        "gate_camera_alignment_reward": args.gate_camera_alignment_reward,
        "teacher_pitch_speed_target_m_s": args.teacher_pitch_speed_target_m_s,
        "teacher_roll_per_m": args.teacher_roll_per_m,
        "teacher_roll_rate_per_m_s": args.teacher_roll_rate_per_m_s,
        "learning_rate": args.learning_rate,
        "log_std": args.log_std,
        "reward_scale": args.reward_scale,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_coef": args.clip_coef,
        "ppo_policy_coef": args.ppo_policy_coef,
        "value_coef": args.value_coef,
        "reference_policy_checkpoint": (
            str(reference_checkpoint) if reference_checkpoint is not None else None),
        "reference_policy_checkpoint_sha256": (
            sha256_file(reference_checkpoint)
            if reference_checkpoint is not None else None),
        "reference_policy_consistency_coef": (
            args.reference_policy_consistency_coef),
        "max_grad_norm": args.max_grad_norm,
        "runtime_controller": "secondary_recurrent_puffer_policy",
        "reset_recurrent_at_gate2": args.reset_recurrent_at_gate2,
        "predict_gate_motion_dropout": args.predict_gate_motion_dropout,
        "gate_motion_control_accel_gain": args.gate_motion_control_accel_gain,
        "truncate_prefix_at_gate_advance": args.truncate_prefix_at_gate_advance,
        "start_elapsed_time": args.start_elapsed_time,
        "time_limit_seconds": args.time_limit_seconds,
        "gate1_xyz": [args.gate1_x, args.gate1_y, args.gate1_z],
        "policy_gate_observation_scale_min": (
            args.policy_gate_observation_scale_min),
        "policy_gate_observation_scale_max": (
            args.policy_gate_observation_scale_max),
        "policy_world_frame_gate_features": (
            args.policy_world_frame_gate_features),
        "policy_linear_gate_features": args.policy_linear_gate_features,
        "policy_predict_forward_gate_rate": (
            args.policy_predict_forward_gate_rate),
        "carry_prefix_gate_rates": args.carry_prefix_gate_rates,
        "train_action_rows": args.train_action_rows,
        "updates": args.updates,
        "horizon": args.horizon,
        "agents": args.agents,
        "episodic_reset": args.episodic_reset,
        "perturbed": args.perturbed,
        "perturbation_scale": args.perturbation_scale,
        "perturbation_components": args.perturbation_components,
        "start_perturbation_dimensions": args.start_perturbation_dimensions,
        "elapsed_seconds": round(time.time() - started, 6),
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
