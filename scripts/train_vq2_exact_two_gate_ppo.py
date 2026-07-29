#!/usr/bin/env python3
"""Teacher-free recurrent PPO for the exact measured two-gate curriculum."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.torch_pufferl import _cpu_tensor
from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE, PRIVILEGED_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.collect_vq2_measured_two_gate_oracle_prefix import MEASURED_GATES, measured_config
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.collect_vq2_public_phase_dagger import (
    PHASE_PRIVILEGED_INDEX,
    STATUS_HOLD_STEPS,
    update_held_phase,
)
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log


TAG = "vq2_sf068_exact_two_gate_ppo_001"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = ROOT / "docs/vq2_sf068_exact_two_gate_ppo_preregistration_2026-07-28.md"
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf066_aggregate_dagger_fit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "f4ee6782de66736110c79a892efaa70635a2ad6c14f0bfaeeaba9812da0ae7a8"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "2ad76bc51417dd2160ef44ec4d4029870d3d9b189efb45d795aacd133e9e226e"
)
REJECTION_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf067_measured_two_gate_teacher_free_512/report.json"
)
REJECTION_REPORT_SHA256 = (
    "572915a2e29e1447d12089aef8a7595c02b8a76617c141b65e7c6c2d0a864cd9"
)
TRAINING_GATE_COUNT = 2
OFFICIAL_GATE_COUNT = 6


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 42068
    agents: int = 128
    updates: int = 48
    horizon: int = 64
    minibatch_agents: int = 16
    update_epochs: int = 2
    actor_learning_rate: float = 3e-6
    critic_learning_rate: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    reward_scale: float = 0.01
    exploration_std: float = 0.06
    max_grad_norm: float = 1.0
    evaluation_interval: int = 4
    evaluation_agents: int = 128
    episode_steps: int = 768


CONFIG = PPOConfig()


class PrivilegedCritic(nn.Module):
    """Training-only critic; it is never part of the deployed actor artifact."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PRIVILEGED_SIZE, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, privileged: torch.Tensor) -> torch.Tensor:
        if privileged.shape[-1] != PRIVILEGED_SIZE:
            raise ValueError("critic accepts only the normalized privileged tail")
        return self.network(privileged).squeeze(-1)


def official_phase_from_training(raw_phase: np.ndarray) -> np.ndarray:
    """Convert native current_gate/2 to public official current_gate/6."""

    phase = np.asarray(raw_phase, dtype=np.float32)
    return np.clip(
        phase * np.float32(TRAINING_GATE_COUNT / OFFICIAL_GATE_COUNT),
        0.0,
        1.0,
    )


def exact_two_gate_config(pufferl_module: Any, *, agents: int, seed: int, evaluation: bool) -> dict:
    config, _ = measured_config(pufferl_module)
    config["seed"] = seed
    config["vec"].update({"total_agents": agents, "num_buffers": 2, "num_threads": 4})
    environment = config["env"]
    environment.update(
        {
            "num_gates": TRAINING_GATE_COUNT,
            "evaluation_episode_limit": 1 if evaluation else 0,
            "evaluation_episode_offset": 0,
            "max_steps": CONFIG.episode_steps,
            "time_limit_seconds": CONFIG.episode_steps * float(environment["dt"]),
            "teacher_action_blend": 0.0,
            "teacher_course_spline": 0,
            "teacher_segment_minimum_jerk": 0,
            "teacher_alignment_governor": 0,
            "w_action_teacher": 0.0,
            "w_progress": 35.0,
            "w_gate": 30.0,
            "w_ordered_gate": 30.0,
            "w_finish": 600.0,
            "w_time": 0.15,
            "w_ctrl": 0.01,
            "w_cross_track": 2.0,
            "w_gate_camera_alignment": 20.0,
            "gate_camera_alignment_from_gate_index": 1,
            "w_gate_crossing_error": 30.0,
            "gate_crossing_error_from_gate_index": 1,
            "invalid_penalty": 600.0,
            "late_invalid_penalty": 600.0,
        }
    )
    for index, (x, y, z) in enumerate(MEASURED_GATES):
        environment[f"gate{index}_x"] = x
        environment[f"gate{index}_y"] = y
        environment[f"gate{index}_z"] = z
    return config


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


def reset_state(
    state: torch.Tensor, reset: torch.Tensor, zero_state: torch.Tensor
) -> torch.Tensor:
    return torch.where(reset.view(1, -1, 1), zero_state, state)


def actor_observation(
    observations: torch.Tensor, held_phase: np.ndarray, device: torch.device
) -> torch.Tensor:
    legal = observations[:, :LEGAL_OBS_SIZE].to(device)
    phase = torch.from_numpy(held_phase[:, None]).to(device)
    return torch.cat((legal, phase), dim=1)


def load_parent(device: torch.device) -> tuple[VQ2PhaseResidualActor, dict[str, Any]]:
    frozen = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        REJECTION_REPORT: REJECTION_REPORT_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    if json.loads(REJECTION_REPORT.read_text()).get("two_gate_milestone_passed"):
        raise RuntimeError("SF068 is unnecessary after a passing SF067")
    payload = torch.load(PARENT_CHECKPOINT, map_location=device, weights_only=False)
    contract = payload["model"]
    actor = VQ2PhaseResidualActor(
        hidden_size=int(contract["hidden_size"]),
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(payload["model_state"])
    with torch.no_grad():
        actor.log_std.fill_(math.log(CONFIG.exploration_std))
    actor.log_std.requires_grad_(False)
    return actor, payload


def deterministic_evaluation(
    actor: VQ2PhaseResidualActor,
    *,
    pufferl_module: Any,
    native_module: Any,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    config = exact_two_gate_config(
        pufferl_module, agents=CONFIG.evaluation_agents, seed=seed, evaluation=True
    )
    vector = native_module.create_vec(config, gpu=0)
    observations = _cpu_tensor(
        vector.obs_ptr, (CONFIG.evaluation_agents, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(
        vector.terminals_ptr, (CONFIG.evaluation_agents,), torch.float32
    )
    actions = torch.zeros((CONFIG.evaluation_agents, ACTION_SIZE), dtype=torch.float32)
    state = actor.initial_state(CONFIG.evaluation_agents, device=device)
    held_phase = np.zeros(CONFIG.evaluation_agents, dtype=np.float32)
    finished = np.zeros(CONFIG.evaluation_agents, dtype=bool)
    steps = 0
    vector.reset()
    actor.eval()
    try:
        with torch.no_grad():
            for step in range(CONFIG.episode_steps + 2):
                active = ~finished
                if not active.any():
                    break
                raw = observations.numpy()[:, PHASE_PRIVILEGED_INDEX]
                official = official_phase_from_training(raw)
                sampled_phase, _ = update_held_phase(official, held_phase, step=step)
                held_phase[active] = sampled_phase[active]
                observation = actor_observation(observations, held_phase, device)
                output, candidate = actor.forward_step(observation, state)
                active_tensor = torch.from_numpy(active).to(device)
                state = torch.where(active_tensor.view(1, -1, 1), candidate, state)
                action = torch.where(
                    active_tensor[:, None], output.mean, torch.zeros_like(output.mean)
                )
                actions.copy_(action.cpu())
                vector.cpu_step(actions.data_ptr())
                terminal = (terminals.numpy() > 0.5) & active
                finished |= terminal
                steps = step + 1
        metrics = flatten_log(pufferl_module, dict(vector.log()))
    finally:
        vector.close()
    return {
        "seed": seed,
        "steps": steps,
        "finished": int(finished.sum()),
        "metrics": metrics,
    }


def evaluation_rank(result: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = result["metrics"]
    # Native logs encode an unsampled terminal crossing as a numeric zero.  Zero
    # must not outrank a real Gate-2 approach, so fall back to closest range
    # unless at least one terminal crossing was actually observed.
    if metrics.get("env/terminal_crossing_sampled", 0.0) > 0.0:
        approach_error = metrics.get("env/terminal_crossing_radial", 1.0e9)
    else:
        approach_error = metrics.get("env/closest_gate_range", 1.0e9)
    return (
        -metrics.get("env/success_rate", 0.0),
        metrics.get("env/crash", 1.0),
        -metrics.get("env/gates_passed", 0.0),
        approach_error,
    )


def evaluation_passes(result: dict[str, Any]) -> bool:
    metrics = result["metrics"]
    return bool(
        result["finished"] == CONFIG.evaluation_agents
        and metrics.get("env/n", 0.0) == float(CONFIG.evaluation_agents)
        and metrics.get("env/success_rate", 0.0) == 1.0
        and metrics.get("env/gates_passed", 0.0) == 2.0
        and metrics.get("env/crash", 0.0) == 0.0
        and metrics.get("env/missed_gate", 0.0) == 0.0
        and metrics.get("env/timeout", 0.0) == 0.0
        and metrics.get("env/out_of_order", 0.0) == 0.0
    )


def recurrent_outputs(
    actor: VQ2PhaseResidualActor,
    observations: torch.Tensor,
    initial_state: torch.Tensor,
    reset_after: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = initial_state
    zero = actor.initial_state(observations.shape[1], device=observations.device)
    means: list[torch.Tensor] = []
    pre_tanh: list[torch.Tensor] = []
    log_stds: list[torch.Tensor] = []
    for step in range(observations.shape[0]):
        output, state = actor.forward_step(observations[step], state)
        means.append(output.mean)
        pre_tanh.append(output.pre_tanh_mean)
        log_stds.append(output.log_std)
        state = reset_state(state, reset_after[step], zero)
    return torch.stack(means), torch.stack(pre_tanh), torch.stack(log_stds)


def save_checkpoint(
    path: Path,
    *,
    actor: VQ2PhaseResidualActor,
    parent: dict[str, Any],
    update: int,
    evaluation: dict[str, Any],
    source_sha256: dict[str, str],
) -> None:
    payload = dict(parent)
    payload["model_state"] = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    payload["best_epoch"] = int(update)
    payload["best_update"] = int(update)
    payload["source_sha256"] = {**parent.get("source_sha256", {}), **source_sha256}
    payload["ppo"] = {
        "training_only_privileged_critic": True,
        "critic_exported": False,
        "teacher_action_blend": 0.0,
        "config": asdict(CONFIG),
        "deterministic_evaluation": evaluation,
    }
    payload["safety"] = {
        **parent.get("safety", {}),
        "flight_sim_packets_sent": 0,
        "sealed_test_accesses": 0,
        "submission_authorized": False,
        "runtime_privileged_values": 0,
    }
    torch.save(payload, path)


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF068 preregisters CUDA training")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled backend is not drone_race_vision")
    device = torch.device(device_name)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    torch.cuda.manual_seed_all(CONFIG.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    actor, parent = load_parent(device)
    critic = PrivilegedCritic().to(device)
    actor_optimizer = torch.optim.AdamW(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=CONFIG.actor_learning_rate,
        weight_decay=0.0,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=CONFIG.critic_learning_rate, weight_decay=0.0
    )
    fixed_log_std = torch.full(
        (ACTION_SIZE,), math.log(CONFIG.exploration_std), device=device
    )

    config = exact_two_gate_config(
        pufferl, agents=CONFIG.agents, seed=CONFIG.seed, evaluation=False
    )
    environment = config["env"]
    if any(
        float(environment[name]) != 0.0
        for name in (
            "teacher_action_blend",
            "teacher_course_spline",
            "teacher_segment_minimum_jerk",
            "teacher_alignment_governor",
            "w_action_teacher",
            "reset_position_noise_xy",
            "reset_position_noise_z",
            "gate_position_domain_randomize",
            "course_geometry_scale_randomize",
            "sitl_plant_domain_randomize",
        )
    ):
        raise RuntimeError("SF068 exact teacher-free environment contract changed")

    vector = _C.create_vec(config, gpu=0)
    observations = _cpu_tensor(
        vector.obs_ptr, (CONFIG.agents, ENV_OBS_SIZE), torch.float32
    )
    rewards_buffer = _cpu_tensor(vector.rewards_ptr, (CONFIG.agents,), torch.float32)
    terminals_buffer = _cpu_tensor(vector.terminals_ptr, (CONFIG.agents,), torch.float32)
    actions_cpu = torch.zeros((CONFIG.agents, ACTION_SIZE), dtype=torch.float32)
    state = actor.initial_state(CONFIG.agents, device=device)
    held_phase = np.zeros(CONFIG.agents, dtype=np.float32)
    reset_only = np.zeros(CONFIG.agents, dtype=bool)
    rng = np.random.default_rng(CONFIG.seed)

    source_paths = (
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "ocean/drone_race/drone_race.c",
        PARENT_CHECKPOINT,
        PARENT_REPORT,
        REJECTION_REPORT,
    )
    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    output.mkdir(parents=True)
    best_path = output / "policy_best.pt"
    history: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    total_valid_steps = 0
    action_delivery_max_error = 0.0
    started = time.perf_counter()

    initial_evaluation = deterministic_evaluation(
        actor,
        pufferl_module=pufferl,
        native_module=_C,
        device=device,
        seed=CONFIG.seed + 1000,
    )
    initial_evaluation["update"] = 0
    evaluations.append(initial_evaluation)
    best_evaluation = initial_evaluation
    best_rank = evaluation_rank(initial_evaluation)
    save_checkpoint(
        best_path,
        actor=actor,
        parent=parent,
        update=0,
        evaluation=initial_evaluation,
        source_sha256=source_sha256,
    )

    vector.reset()
    try:
        for update in range(1, CONFIG.updates + 1):
            actor.eval()
            critic.eval()
            rollout_initial_state = state.detach().clone()
            observation_steps: list[torch.Tensor] = []
            privilege_steps: list[torch.Tensor] = []
            raw_action_steps: list[torch.Tensor] = []
            old_logprob_steps: list[torch.Tensor] = []
            reward_steps: list[torch.Tensor] = []
            done_steps: list[torch.Tensor] = []
            valid_steps: list[torch.Tensor] = []
            value_steps: list[torch.Tensor] = []
            for step in range(CONFIG.horizon):
                valid_np = ~reset_only
                current = observations.numpy()
                official = official_phase_from_training(
                    current[:, PHASE_PRIVILEGED_INDEX]
                )
                sampled, _ = update_held_phase(
                    official, held_phase, step=update * CONFIG.horizon + step
                )
                held_phase[valid_np] = sampled[valid_np]
                policy_observation = actor_observation(observations, held_phase, device)
                privilege = observations[:, LEGAL_OBS_SIZE:].to(device).clone()
                valid = torch.from_numpy(valid_np).to(device)
                with torch.no_grad():
                    actor_output, candidate = actor.forward_step(policy_observation, state)
                    distribution = torch.distributions.Normal(
                        actor_output.pre_tanh_mean, fixed_log_std.exp()
                    )
                    raw_action = distribution.sample()
                    logprob = distribution.log_prob(raw_action).sum(-1)
                    value = critic(privilege)
                state = torch.where(valid.view(1, -1, 1), candidate, state)
                action = torch.where(
                    valid[:, None], torch.tanh(raw_action), torch.zeros_like(raw_action)
                )
                actions_cpu.copy_(action.cpu())
                action_reference = actions_cpu.numpy().copy()
                vector.cpu_step(actions_cpu.data_ptr())
                reward = rewards_buffer.to(device).clone() * CONFIG.reward_scale
                terminal_np = terminals_buffer.numpy() > 0.5
                terminal = torch.from_numpy(terminal_np).to(device)
                effective_done = terminal | ~valid
                executed = observations.numpy()[:, 4103:4107]
                if valid_np.any():
                    action_delivery_max_error = max(
                        action_delivery_max_error,
                        float(np.max(np.abs(executed[valid_np] - action_reference[valid_np]))),
                    )
                observation_steps.append(policy_observation.cpu())
                privilege_steps.append(privilege.cpu())
                raw_action_steps.append(raw_action.cpu())
                old_logprob_steps.append(logprob.cpu())
                reward_steps.append(reward.cpu())
                done_steps.append(effective_done.cpu())
                valid_steps.append(valid.cpu())
                value_steps.append(value.cpu())
                zero = actor.initial_state(CONFIG.agents, device=device)
                state = reset_state(state, effective_done, zero).detach()
                held_phase[terminal_np | ~valid_np] = 0.0
                reset_only = terminal_np

            next_privilege = observations[:, LEGAL_OBS_SIZE:].to(device)
            with torch.no_grad():
                next_value = critic(next_privilege).cpu()
            rollout_observation = torch.stack(observation_steps)
            rollout_privilege = torch.stack(privilege_steps)
            rollout_raw_action = torch.stack(raw_action_steps)
            old_logprob = torch.stack(old_logprob_steps)
            rewards = torch.stack(reward_steps)
            dones = torch.stack(done_steps).float()
            valid_mask = torch.stack(valid_steps).bool()
            old_values = torch.stack(value_steps)
            advantages, returns = compute_gae(
                rewards,
                old_values,
                dones,
                next_value,
                gamma=CONFIG.gamma,
                gae_lambda=CONFIG.gae_lambda,
            )
            active_advantage = advantages[valid_mask]
            advantages = (advantages - active_advantage.mean()) / (
                active_advantage.std() + 1e-8
            )

            actor.train()
            critic.train()
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            entropy_sum = 0.0
            batches = 0
            for _ in range(CONFIG.update_epochs):
                agent_order = rng.permutation(CONFIG.agents)
                for start in range(0, CONFIG.agents, CONFIG.minibatch_agents):
                    indices_np = agent_order[
                        start : start + CONFIG.minibatch_agents
                    ]
                    if len(indices_np) != CONFIG.minibatch_agents:
                        continue
                    indices = torch.from_numpy(indices_np).to(device)
                    obs = rollout_observation[:, indices_np].to(device)
                    privilege = rollout_privilege[:, indices_np].to(device)
                    raw_action = rollout_raw_action[:, indices_np].to(device)
                    old_lp = old_logprob[:, indices_np].to(device)
                    adv = advantages[:, indices_np].to(device)
                    target_return = returns[:, indices_np].to(device)
                    valid = valid_mask[:, indices_np].to(device)
                    reset_after = dones[:, indices_np].bool().to(device)
                    initial = rollout_initial_state[:, indices].detach()
                    _, pre_tanh, log_std = recurrent_outputs(
                        actor, obs, initial, reset_after
                    )
                    distribution = torch.distributions.Normal(
                        pre_tanh, log_std.exp()
                    )
                    new_logprob = distribution.log_prob(raw_action).sum(-1)
                    ratio = (new_logprob - old_lp).exp()
                    clipped = torch.clamp(
                        ratio, 1.0 - CONFIG.clip_coef, 1.0 + CONFIG.clip_coef
                    )
                    policy_loss = torch.maximum(-adv * ratio, -adv * clipped)[valid].mean()
                    entropy = distribution.entropy().sum(-1)[valid].mean()
                    actor_loss = policy_loss - CONFIG.entropy_coef * entropy
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in actor.parameters() if p.requires_grad],
                        CONFIG.max_grad_norm,
                    )
                    actor_optimizer.step()

                    predicted_value = critic(privilege)
                    value_loss = 0.5 * torch.square(
                        predicted_value[valid] - target_return[valid]
                    ).mean()
                    critic_optimizer.zero_grad(set_to_none=True)
                    (CONFIG.value_coef * value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), CONFIG.max_grad_norm)
                    critic_optimizer.step()
                    actor_loss_sum += float(policy_loss.detach())
                    critic_loss_sum += float(value_loss.detach())
                    entropy_sum += float(entropy.detach())
                    batches += 1

            valid_count = int(valid_mask.sum().item())
            total_valid_steps += valid_count
            metrics = flatten_log(pufferl, dict(vector.log()))
            row = {
                "update": update,
                "valid_steps": valid_count,
                "total_valid_steps": total_valid_steps,
                "actor_loss": actor_loss_sum / max(batches, 1),
                "critic_loss": critic_loss_sum / max(batches, 1),
                "entropy": entropy_sum / max(batches, 1),
                "native_metrics": metrics,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

            if update % CONFIG.evaluation_interval == 0 or update == CONFIG.updates:
                evaluation = deterministic_evaluation(
                    actor,
                    pufferl_module=pufferl,
                    native_module=_C,
                    device=device,
                    seed=CONFIG.seed + 1000 + update,
                )
                evaluation["update"] = update
                evaluations.append(evaluation)
                rank = evaluation_rank(evaluation)
                if rank < best_rank:
                    best_rank = rank
                    best_evaluation = evaluation
                    save_checkpoint(
                        best_path,
                        actor=actor,
                        parent=parent,
                        update=update,
                        evaluation=evaluation,
                        source_sha256=source_sha256,
                    )
                print(json.dumps({"evaluation": evaluation}, sort_keys=True), flush=True)
    finally:
        vector.close()

    admitted = evaluation_passes(best_evaluation)
    report = {
        "schema": "vq2_exact_two_gate_ppo_report_v1",
        "tag": TAG,
        "config": asdict(CONFIG),
        "checkpoint_sha256": sha256_path(best_path),
        "best_update": best_evaluation["update"],
        "best_evaluation": best_evaluation,
        "training_admitted": admitted,
        "numerically_admitted": admitted,
        "combined_numerical_admission": admitted,
        "history": history,
        "evaluations": evaluations,
        "total_valid_steps": total_valid_steps,
        "action_delivery_max_error": action_delivery_max_error,
        "source_sha256": source_sha256,
        "wall_time_seconds": time.perf_counter() - started,
        "safety": {
            "runtime_actor_inputs": 4119,
            "runtime_privileged_values": 0,
            "training_only_critic_privileged_values": PRIVILEGED_SIZE,
            "critic_exported": False,
            "teacher_actions_executed": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
