#!/usr/bin/env python3
"""Closed-loop phase-6 PPO updates of the complete LC105 Puffer policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.torch_pufferl import _cpu_tensor
from pufferlib.vq2_informed import ACTION_HISTORY, ENV_OBS_SIZE, LEGAL_OBS_SIZE
from pufferlib.vq2_public_phase import OFFICIAL_PROGRESS_SCALE
from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.eval_vq2_lc001_long_course_oracle import load_config
from scripts.eval_vq2_lc015_restore_vg071_head1 import state_sha256
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc007_converted_vg071_long_course as core
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
from scripts.train_vq2_lc111_phase8_9_full_residual_endpoint import (
    PARAMETER_NAMES,
    parameter_delta_l2,
)
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc127_phase6_onpolicy_ppo_001"
SCHEMA = "vq2_lc127_phase6_onpolicy_ppo_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc127_phase6_onpolicy_ppo_checkpoint_v1"
TOTAL_AGENTS = 256
THREADS = 32
SEED = 432_050
NUM_GATES = 24
MAX_STEPS = 12_000
TARGET_PHASE = 6
TARGET_RAW_INDEX = 10
ROLLOUTS = 4
PPO_EPOCHS = 4
BATCH_SIZE = 4_096
LEARNING_RATE = 5e-5
CLIP_COEFFICIENT = 0.10
MAX_GRADIENT_NORM = 0.5
ANCHOR_COEFFICIENT = 1e-4
EXPLORATION_STD = 0.01
MAXIMUM_UPDATE_DELTA_L2 = 4.0
# Backward-compatible opt-in for later experiments. LC127's source-locked
# sparse gate-rank behavior remains exact at the default zero value.
PHASE_RETURN_ADVANTAGE_WEIGHT = 0.0
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc105_phase7_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "005e5e7929258fd282ab390fd230fda817700afaee790e78144200e67b3e10a4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c614e6929282399cac2f18465084f8337ed8770389a74e48b86ff1fcf4315e7d"
LC126_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc126_phase6_onpolicy_coordinate_es_001/report.json"
)
LC126_REPORT_SHA256 = "7aab360a570cadb868302db711a67f5a9143b097e385dab0bdb40d85c612ef13"
PREREGISTRATION = ROOT / "docs/vq2_lc127_phase6_onpolicy_ppo_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc127_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc127_phase6_onpolicy_ppo.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
ROLLOUT_DTYPE = np.dtype([
    ("hidden", "<f2", (256,)),
    ("base_pre_tanh", "<f4", (4,)),
    ("sample_pre_tanh", "<f4", (4,)),
    ("old_log_probability", "<f4"),
    ("agent_index", "<u2"),
])


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC126_REPORT: LC126_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC127 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC126_REPORT.read_text())
    items = rejected.get("items", [])
    baseline = items[0] if items else {}
    baseline_mean = sum(
        int(index) * count
        for index, count in baseline.get("maximum_raw_index_distribution", {}).items()
    ) / 32.0
    nonzero_means = [
        sum(int(index) * count for index, count in item[
            "maximum_raw_index_distribution"
        ].items()) / 32.0
        for item in items[1:]
    ]
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or rejected.get("schema")
        != "vq2_lc126_phase6_onpolicy_coordinate_es_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or len(items) != 8
        or baseline.get("maximum_raw_index_distribution", {}).get("9") != 1
        or baseline_mean != 3.875
        or not nonzero_means
        or max(nonzero_means) >= baseline_mean
        or any(item.get("target_passes") != 0 for item in items)
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC126 do not authorize LC127")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC126_REPORT,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc007_converted_vg071_long_course.py",
        ROOT / "scripts/eval_vq2_lc058_phase2_bias_milestone.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "src/vecenv.h", ROOT / "src/bindings.cu",
    )
    extension = Path(_C.__file__).resolve()
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {
            **{str(path.relative_to(ROOT)): sha256_path(path) for path in paths},
            "compiled_extension": sha256_path(extension),
        },
        "compiled_extension_path": str(extension),
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
    }


def residual_pre_tanh(
    hidden: torch.Tensor,
    base: torch.Tensor,
    parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    input_weight, input_bias, output_weight, output_bias = parameters
    feature = torch.tanh(hidden @ input_weight.T + input_bias)
    return base + feature @ output_weight.T + output_bias


def normal_log_probability(
    sample: torch.Tensor, mean: torch.Tensor, standard_deviation: float
) -> torch.Tensor:
    variance = float(standard_deviation) ** 2
    constant = math.log(2.0 * math.pi * variance)
    return -0.5 * (((sample - mean).square() / variance) + constant).sum(dim=-1)


def trajectory_advantages(
    maximum_raw_index: np.ndarray,
    passed: np.ndarray,
    agents: np.ndarray,
    phase_return: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    unique = np.unique(agents)
    score = maximum_raw_index[unique].astype(np.float64)
    score += 4.0 * passed[unique].astype(np.float64)
    return_metrics: dict[str, float] = {}
    if PHASE_RETURN_ADVANTAGE_WEIGHT:
        if phase_return is None or phase_return.shape != maximum_raw_index.shape:
            raise ValueError("phase-return advantages require one return per agent")
        selected_return = phase_return[unique].astype(np.float64)
        return_mean = float(selected_return.mean())
        return_standard_deviation = float(selected_return.std())
        normalized_return = (
            (selected_return - return_mean)
            / max(return_standard_deviation, 1e-6)
        )
        score += PHASE_RETURN_ADVANTAGE_WEIGHT * normalized_return
        return_metrics = {
            "queried_phase_return_mean": return_mean,
            "queried_phase_return_std": return_standard_deviation,
            "queried_phase_return_min": float(selected_return.min()),
            "queried_phase_return_max": float(selected_return.max()),
        }
    mean = float(score.mean())
    standard_deviation = float(score.std())
    normalized = (score - mean) / max(standard_deviation, 1e-6)
    advantage = np.zeros(TOTAL_AGENTS, dtype=np.float32)
    advantage[unique] = normalized.astype(np.float32)
    return advantage, {
        "queried_score_mean": mean,
        "queried_score_std": standard_deviation,
        "queried_score_min": float(score.min()),
        "queried_score_max": float(score.max()),
        **return_metrics,
    }


def rollout(
    state: dict[str, torch.Tensor],
    *, iteration: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    from pufferlib import _C, pufferl

    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    actor = milestone.load_actor({**payload, "model_state": state}, device)
    recurrent = actor.initial_state(TOTAL_AGENTS, device=device)
    config, overrides = load_config(
        pufferl, num_gates=NUM_GATES, agents=TOTAL_AGENTS,
        episodes=TOTAL_AGENTS, seed=SEED, threads=THREADS,
    )
    config["env"].update({
        "evaluation_episode_limit": 1,
        "max_steps": MAX_STEPS,
        "time_limit_seconds": MAX_STEPS / 64.0,
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
    })
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != TOTAL_AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("LC127 native vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (TOTAL_AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (TOTAL_AGENTS,), torch.float32)
    rewards = _cpu_tensor(vector.rewards_ptr, (TOTAL_AGENTS,), torch.float32)
    actions_cpu = torch.zeros((TOTAL_AGENTS, ACTION_SIZE), dtype=torch.float32)
    action_weight = state["action_head.weight"].to(device)
    action_bias = state["action_head.bias"].to(device)

    resolved = np.zeros(TOTAL_AGENTS, dtype=bool)
    passed = np.zeros(TOTAL_AGENTS, dtype=bool)
    held = np.zeros(TOTAL_AGENTS, dtype=np.float32)
    previous_held = held.copy()
    maximum_raw_index = np.zeros(TOTAL_AGENTS, dtype=np.int32)
    phase_return = np.zeros(TOTAL_AGENTS, dtype=np.float64)
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_skips = 0
    raw_encoding_max_error = 0.0
    action_envelope_violations = 0
    executed_action_max_error = 0.0
    chunks: list[np.ndarray] = []
    nonfinite_action = False
    vector_steps = 0
    inference_seconds = 0.0
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + 10_000 * iteration)
    started = time.perf_counter()
    try:
        vector.reset()
        with torch.no_grad():
            for step in range(MAX_STEPS):
                if resolved.all():
                    break
                active_np = ~resolved
                current = observations.numpy()
                raw = current[:, core.PHASE_PRIVILEGED_INDEX]
                held, sampled, _ = core.update_held_progress(raw, held, step=step)
                changed = np.abs(held - previous_held) > 1e-7
                delta_index = np.rint(
                    (held - previous_held) * OFFICIAL_PROGRESS_SCALE
                ).astype(np.int32)
                raw_scaled = raw * OFFICIAL_PROGRESS_SCALE
                raw_indices = np.rint(raw_scaled).astype(np.int32)
                encoding_error = np.abs(raw_scaled - raw_indices)
                if changed.any() and not sampled:
                    phase_changes_off_tick += int(changed.sum())
                phase_decreases += int(((delta_index < 0) & active_np).sum())
                phase_skips += int(((delta_index > 1) & active_np).sum())
                raw_encoding_max_error = max(
                    raw_encoding_max_error,
                    float(encoding_error.max(initial=0.0)),
                )
                previous_held = held.copy()
                maximum_raw_index = np.maximum(maximum_raw_index, raw_indices)
                newly_passed = (~resolved) & (raw_indices >= TARGET_RAW_INDEX)
                passed |= newly_passed
                resolved |= newly_passed
                active_np = ~resolved
                if not active_np.any():
                    break

                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                progress = torch.from_numpy(held[:, None]).to(device)
                actor_input = torch.cat((legal, progress), dim=1)
                active_device = torch.from_numpy(active_np).to(device)
                inference_started = time.perf_counter()
                result, next_recurrent = actor.forward_step(actor_input, recurrent)
                recurrent = preserve_frozen_state(
                    recurrent, next_recurrent, active_device
                )
                phase_index = np.rint(
                    held * OFFICIAL_PROGRESS_SCALE
                ).astype(np.int32)
                exploration_mask = active_np & (phase_index == TARGET_PHASE)
                selected_agents = np.flatnonzero(exploration_mask)
                plant = torch.where(
                    active_device[:, None], result.mean, torch.zeros_like(result.mean)
                )
                if selected_agents.size:
                    index = torch.from_numpy(selected_agents).to(device)
                    current_mean = result.pre_tanh_mean.index_select(0, index)
                    noise = torch.randn(
                        current_mean.shape, device=device, generator=generator
                    ) * EXPLORATION_STD
                    sampled_pre = current_mean + noise
                    sampled_action = torch.tanh(sampled_pre)
                    plant.index_copy_(0, index, sampled_action)
                    hidden = next_recurrent[0].index_select(0, index)
                    base_pre_tanh = hidden @ action_weight.T + action_bias
                    old_log_probability = normal_log_probability(
                        sampled_pre, current_mean, EXPLORATION_STD
                    )
                    records = np.empty(selected_agents.size, dtype=ROLLOUT_DTYPE)
                    records["hidden"] = hidden.cpu().numpy().astype(np.float16)
                    records["base_pre_tanh"] = base_pre_tanh.cpu().numpy()
                    records["sample_pre_tanh"] = sampled_pre.cpu().numpy()
                    records["old_log_probability"] = old_log_probability.cpu().numpy()
                    records["agent_index"] = selected_agents.astype(np.uint16)
                    chunks.append(records)
                inference_seconds += time.perf_counter() - inference_started
                plant_np = plant.cpu().numpy().astype(np.float32, copy=False)
                if not np.isfinite(plant_np[active_np]).all():
                    nonfinite_action = True
                    break
                action_envelope_violations += int(
                    (np.abs(plant_np[active_np]) > 1.0 + 1e-6).sum()
                )
                actions_cpu.copy_(torch.from_numpy(plant_np))
                vector.cpu_step(actions_cpu.data_ptr())
                vector_steps = step + 1
                phase_return[exploration_mask] += rewards.numpy()[exploration_mask]
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(np.max(
                        np.abs(executed[active_np] - plant_np[active_np]), initial=0.0
                    )),
                )
                post_raw = observations.numpy()[:, core.PHASE_PRIVILEGED_INDEX]
                post_indices = np.rint(
                    post_raw * OFFICIAL_PROGRESS_SCALE
                ).astype(np.int32)
                maximum_raw_index = np.maximum(maximum_raw_index, post_indices)
                newly_passed = (~resolved) & (post_indices >= TARGET_RAW_INDEX)
                passed |= newly_passed
                resolved |= newly_passed
                terminal_np = (terminals.numpy() > 0.5) & (~resolved)
                resolved |= terminal_np
    finally:
        vector.close()
    records = np.concatenate(chunks) if chunks else np.empty(0, dtype=ROLLOUT_DTYPE)
    query_agents = (
        np.unique(records["agent_index"])
        if records.size else np.empty(0, dtype=np.int64)
    )
    queried_phase_return = phase_return[query_agents]
    clipped = np.minimum(maximum_raw_index, TARGET_RAW_INDEX)
    distribution = {
        str(index): int((clipped == index).sum())
        for index in range(TARGET_RAW_INDEX + 1)
    }
    transport_pass = bool(
        not nonfinite_action
        and action_envelope_violations == 0
        and executed_action_max_error <= core.MAX_EXECUTED_ACTION_ERROR
        and phase_changes_off_tick == 0
        and phase_decreases == 0
        and phase_skips == 0
        and raw_encoding_max_error <= 1e-6
        and np.isfinite(queried_phase_return).all()
        and resolved.all()
    )
    metrics = {
        "iteration": iteration,
        "records": int(records.size),
        "query_agents": int(query_agents.size),
        "target_passes": int(passed.sum()),
        "raw9_or_later": int((maximum_raw_index >= 9).sum()),
        "mean_maximum_raw_index": float(maximum_raw_index.mean()),
        "maximum_raw_index_distribution": distribution,
        "vector_steps": vector_steps,
        "wall_time_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "transport_pass": transport_pass,
        "unresolved": int((~resolved).sum()),
        "nonfinite_action": nonfinite_action,
        "action_envelope_violations": action_envelope_violations,
        "executed_action_max_error": executed_action_max_error,
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases,
        "phase_skips": phase_skips,
        "raw_progress_encoding_max_error": raw_encoding_max_error,
        "queried_phase_return_mean": (
            float(queried_phase_return.mean()) if query_agents.size else None
        ),
        "queried_phase_return_std": (
            float(queried_phase_return.std()) if query_agents.size else None
        ),
        "queried_phase_return_min": (
            float(queried_phase_return.min()) if query_agents.size else None
        ),
        "queried_phase_return_max": (
            float(queried_phase_return.max()) if query_agents.size else None
        ),
        "queried_phase_return_nonfinite": int(
            (~np.isfinite(queried_phase_return)).sum()
        ),
        "loader_overrides": overrides,
    }
    return records, metrics, maximum_raw_index, passed, phase_return


def ppo_update(
    records: np.ndarray,
    maximum_raw_index: np.ndarray,
    passed: np.ndarray,
    phase_return: np.ndarray,
    state: dict[str, torch.Tensor],
    *, iteration: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if records.size == 0:
        raise RuntimeError("LC127 rollout reached no phase-6 states")
    agents = np.asarray(records["agent_index"], dtype=np.int64)
    advantage_by_agent, score_metrics = trajectory_advantages(
        maximum_raw_index, passed, agents, phase_return
    )
    counts = np.bincount(agents, minlength=TOTAL_AGENTS)
    row_weights_np = 1.0 / counts[agents].astype(np.float64)
    row_weights_np /= row_weights_np.sum()
    hidden = torch.from_numpy(
        np.array(records["hidden"], dtype=np.float32, copy=True)
    ).to(device)
    base_pre_tanh = torch.from_numpy(
        np.array(records["base_pre_tanh"], dtype=np.float32, copy=True)
    ).to(device)
    sampled_pre_tanh = torch.from_numpy(
        np.array(records["sample_pre_tanh"], dtype=np.float32, copy=True)
    ).to(device)
    old_log_probability = torch.from_numpy(
        np.array(records["old_log_probability"], dtype=np.float32, copy=True)
    ).to(device)
    advantages = torch.from_numpy(advantage_by_agent[agents]).to(device)
    row_weights = torch.from_numpy(row_weights_np.astype(np.float32)).to(device)
    parents = tuple(
        state[name][TARGET_PHASE].to(device).detach().clone()
        for name in PARAMETER_NAMES
    )
    parameters = tuple(value.clone().requires_grad_(True) for value in parents)
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 20_000 * iteration)
    updates = 0
    maximum_approximate_kl = 0.0
    maximum_clip_fraction = 0.0
    losses: list[float] = []
    for _ in range(PPO_EPOCHS):
        permutation = torch.randperm(records.size, generator=generator)
        for start in range(0, records.size, BATCH_SIZE):
            index = permutation[start : start + BATCH_SIZE].to(device)
            new_mean = residual_pre_tanh(
                hidden.index_select(0, index),
                base_pre_tanh.index_select(0, index),
                parameters,
            )
            new_log_probability = normal_log_probability(
                sampled_pre_tanh.index_select(0, index),
                new_mean,
                EXPLORATION_STD,
            )
            old_log = old_log_probability.index_select(0, index)
            log_ratio = new_log_probability - old_log
            ratio = torch.exp(log_ratio.clamp(-8.0, 8.0))
            advantage = advantages.index_select(0, index)
            unclipped = ratio * advantage
            clipped = ratio.clamp(
                1.0 - CLIP_COEFFICIENT, 1.0 + CLIP_COEFFICIENT
            ) * advantage
            weight = row_weights.index_select(0, index)
            policy_loss = -(
                weight * torch.minimum(unclipped, clipped)
            ).sum() / weight.sum().clamp_min(1e-12)
            anchor = sum(
                (value - parent).square().mean()
                for value, parent in zip(parameters, parents)
            )
            loss = policy_loss + ANCHOR_COEFFICIENT * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, MAX_GRADIENT_NORM)
            optimizer.step()
            with torch.no_grad():
                approximate_kl = float(((ratio - 1.0) - log_ratio).mean())
                clip_fraction = float(
                    ((ratio - 1.0).abs() > CLIP_COEFFICIENT).float().mean()
                )
            maximum_approximate_kl = max(maximum_approximate_kl, approximate_kl)
            maximum_clip_fraction = max(maximum_clip_fraction, clip_fraction)
            losses.append(float(policy_loss.detach()))
            updates += 1
    delta_l2 = float(parameter_delta_l2(parameters, parents))
    finite = bool(
        all(torch.isfinite(value).all() for value in parameters)
        and math.isfinite(delta_l2)
        and all(math.isfinite(value) for value in losses)
    )
    admitted = finite and delta_l2 <= MAXIMUM_UPDATE_DELTA_L2
    next_state = {
        name: value.detach().cpu().clone() for name, value in state.items()
    }
    if admitted:
        for name, value in zip(PARAMETER_NAMES, parameters):
            next_state[name][TARGET_PHASE].copy_(value.detach().cpu().float())
    report = {
        "iteration": iteration,
        "optimizer_updates": updates,
        "policy_loss_mean": float(np.mean(losses)),
        "policy_loss_last": losses[-1],
        "parameter_delta_l2": delta_l2,
        "maximum_approximate_kl": maximum_approximate_kl,
        "maximum_clip_fraction": maximum_clip_fraction,
        "finite": finite,
        "update_admitted": admitted,
        **score_metrics,
    }
    return next_state, report


def rollout_rank(metrics: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(metrics["target_passes"]),
        int(metrics["raw9_or_later"]),
        float(metrics["mean_maximum_raw_index"]),
    )


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    parent = verify_inputs()
    from pufferlib import _C

    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC127 requires the float32 drone_race_vision backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC127 preregisters CUDA inference and fitting")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"LC127 requires OMP_NUM_THREADS={THREADS}")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC127 requires OMP_DYNAMIC=FALSE")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC127 does not resume partial on-policy training")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    device = torch.device(device_name)
    parent_state = {
        name: value.detach().cpu().clone()
        for name, value in parent["model_state"].items()
    }
    state = {
        name: value.detach().cpu().clone() for name, value in parent_state.items()
    }
    parent_state_hash = state_sha256(parent_state)
    rollout_reports: list[dict[str, Any]] = []
    update_reports: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    started = time.perf_counter()
    for iteration in range(1, ROLLOUTS + 1):
        checkpoint = {
            **parent, "schema": CHECKPOINT_SCHEMA,
            "tag": f"{TAG}_rollout_{iteration:02d}",
            "model_state": state,
            "numerically_admitted": False,
            "deployment_candidate": False,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "onpolicy_rollout_iteration": iteration,
        }
        checkpoint_path = output / f"policy_rollout_{iteration:02d}.pt"
        atomic_torch_save(checkpoint_path, checkpoint)
        checkpoint_item = {
            "iteration": iteration,
            "path": checkpoint_path.name,
            "sha256": sha256_path(checkpoint_path),
            "state_sha256": state_sha256(state),
        }
        checkpoints.append(checkpoint_item)
        records, rollout_report, maximum_raw_index, passed, phase_return = rollout(
            state, iteration=iteration, device=device
        )
        rollout_report["checkpoint"] = checkpoint_path.name
        rollout_report["checkpoint_sha256"] = checkpoint_item["sha256"]
        rollout_reports.append(rollout_report)
        if iteration < ROLLOUTS:
            state, update_report = ppo_update(
                records, maximum_raw_index, passed, phase_return, state,
                iteration=iteration, device=device,
            )
            update_reports.append(update_report)
            if not update_report["update_admitted"]:
                break

    training_valid = bool(
        len(rollout_reports) == ROLLOUTS
        and len(update_reports) == ROLLOUTS - 1
        and all(item["transport_pass"] for item in rollout_reports)
        and all(item["records"] > 0 for item in rollout_reports)
        and all(item["update_admitted"] for item in update_reports)
    )
    baseline_rank = rollout_rank(rollout_reports[0])
    eligible = [
        item for item in rollout_reports[1:]
        if item["transport_pass"] and rollout_rank(item) > baseline_rank
    ]
    selected = max(eligible, key=rollout_rank, default=None)
    selected_checkpoint = None
    if training_valid and selected is not None:
        selected_checkpoint = next(
            item for item in checkpoints
            if item["iteration"] == selected["iteration"]
        )
    frozen_exact = True
    for name, value in state.items():
        if name in PARAMETER_NAMES:
            keep = torch.arange(value.shape[0]) != TARGET_PHASE
            frozen_exact &= torch.equal(value[keep], parent_state[name][keep])
        else:
            frozen_exact &= torch.equal(value, parent_state[name])
    training_valid &= frozen_exact
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "training_admitted": training_valid,
        "candidate_selected_for_screen": selected_checkpoint,
        "selected_rollout_metrics": selected,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_state_sha256": parent_state_hash,
        "rollouts": rollout_reports,
        "updates": update_reports,
        "checkpoints": checkpoints,
        "configuration": {
            "rollouts": ROLLOUTS, "ppo_epochs": PPO_EPOCHS,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "clip_coefficient": CLIP_COEFFICIENT,
            "maximum_gradient_norm": MAX_GRADIENT_NORM,
            "anchor_coefficient": ANCHOR_COEFFICIENT,
            "exploration_standard_deviation": EXPLORATION_STD,
            "phase_return_advantage_weight": PHASE_RETURN_ADVANTAGE_WEIGHT,
            "target_phase": TARGET_PHASE, "target_raw_index": TARGET_RAW_INDEX,
            "total_agents": TOTAL_AGENTS, "threads": THREADS,
            "num_proxy_gates": NUM_GATES, "max_steps": MAX_STEPS,
        },
        "frozen_non_phase6_state_exact": frozen_exact,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0,
            "offline_training_teacher_actions": 0,
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Run one reduced deterministic parent-versus-selected-candidate raw-index-10 screen; no FlightSim authority."
            if training_valid and selected_checkpoint is not None else
            "Reject LC127 and retain LC105; diagnose the closed-loop rank or training contract before another update."
        ),
    }
    write_json_once(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
