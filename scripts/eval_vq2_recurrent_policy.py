#!/usr/bin/env python3
"""Teacher-free deterministic native screen for a legal VQ2 recurrent actor."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE, VQ2RecurrentActor
from scripts.eval_vq2_native_oracle import (
    BACKEND_ENV_NAME,
    flatten_log,
    load_fixed_config,
)


CHECKPOINT_SHA256 = (
    "00fd90bae021a2ce0a499e6c1ca5f622967090ab0342227e548565fc2943186a"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf014_recurrent_bc_001/policy_best.pt"
)
TAG = "vq2_sf015_recurrent_bc_teacher_free_64"
DEFAULT_OUTPUT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG
)
AGENTS = 64
EPISODES = 64
SEED = 42015
EPISODE_SECONDS = 180.0
TARGET_SPEED_M_S = 2.0


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_free_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = load_fixed_config(
        pufferl_module,
        agents=AGENTS,
        episodes=EPISODES,
        seed=SEED,
        controller="governed",
        target_speed_m_s=TARGET_SPEED_M_S,
        episode_seconds=EPISODE_SECONDS,
        randomized_course=True,
    )
    environment = config["env"]
    environment.update(
        {
            "teacher_action_blend": 0.0,
            "teacher_course_spline": 0,
            "teacher_segment_minimum_jerk": 0,
            "teacher_alignment_governor": 0,
            "w_action_teacher": 0.0,
        }
    )
    return config, overrides


def load_actor(
    checkpoint: Path, *, device: torch.device
) -> tuple[VQ2RecurrentActor, dict[str, Any]]:
    if sha256_path(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("SF014 checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "vq2_recurrent_bc_checkpoint_v1":
        raise RuntimeError("unsupported recurrent actor checkpoint schema")
    model_contract = payload.get("model", {})
    if (
        model_contract.get("class") != "VQ2RecurrentActor"
        or model_contract.get("legal_observation_size") != LEGAL_OBS_SIZE
        or model_contract.get("action_size") != ACTION_SIZE
    ):
        raise RuntimeError("checkpoint actor ABI changed")
    if payload.get("safety", {}).get(
        "stored_privileged_values_per_actor_record"
    ) != 0:
        raise RuntimeError("checkpoint does not prove a privilege-free actor dataset")
    model = VQ2RecurrentActor(
        hidden_size=int(model_contract["hidden_size"]),
        initial_std=float(model_contract["initial_std"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def policy_screen_passes(metrics: dict[str, float], *, episodes: int) -> bool:
    return (
        metrics.get("env/n", 0.0) == float(episodes)
        and metrics.get("env/success_rate", 0.0) == 1.0
        and metrics.get("env/gates_passed", 0.0) == 6.0
        and metrics.get("env/crash", 0.0) == 0.0
        and metrics.get("env/timeout", 0.0) == 0.0
        and metrics.get("env/missed_gate", 0.0) == 0.0
        and metrics.get("env/out_of_order", 0.0) == 0.0
        and metrics.get("env/valid_run_rate", 0.0) == 1.0
        and metrics.get("env/crossing_margin_violation", 0.0) == 0.0
        and metrics.get("env/action_envelope_violation", 0.0) == 0.0
        and metrics.get("env/wire_rate_envelope_violation", 0.0) == 0.0
        and metrics.get("env/thrust_envelope_violation", 0.0) == 0.0
    )


def preserve_frozen_state(
    previous: torch.Tensor,
    candidate: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    if previous.shape != candidate.shape or active.shape != (previous.shape[1],):
        raise ValueError("recurrent active mask does not align with state")
    return torch.where(active[None, :, None], candidate, previous)


def run_screen(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF015 preregisters CUDA inference but CUDA is unavailable")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled native backend is not drone_race_vision")
    device = torch.device(device_name)
    model, payload = load_actor(CHECKPOINT, device=device)
    config, overrides = teacher_free_config(pufferl)
    environment = config["env"]
    if float(environment["teacher_action_blend"]) != 0.0:
        raise RuntimeError("teacher-free screen refuses a positive teacher blend")

    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native vector ABI differs from the SF015 contract")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions_cpu = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    state = model.initial_state(AGENTS, device=device)
    done = torch.zeros(AGENTS, dtype=torch.bool)
    action_sum = torch.zeros(ACTION_SIZE, dtype=torch.float64)
    action_square_sum = torch.zeros(ACTION_SIZE, dtype=torch.float64)
    action_min = torch.full((ACTION_SIZE,), float("inf"))
    action_max = torch.full((ACTION_SIZE,), float("-inf"))
    action_samples = 0
    nonfinite_action = False
    inference_seconds = 0.0
    vector_steps = 0
    started = time.perf_counter()
    native_log: dict[str, Any] = {}
    try:
        vector.reset()
        with torch.no_grad():
            for step in range(int(environment["max_steps"])):
                active_cpu = ~done
                if not bool(active_cpu.any()):
                    break
                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                active = active_cpu.to(device)
                inference_started = time.perf_counter()
                actor_output, candidate_state = model.forward_step(legal, state)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - inference_started
                state = preserve_frozen_state(state, candidate_state, active)
                action = torch.where(
                    active[:, None], actor_output.mean, torch.zeros_like(actor_output.mean)
                )
                if not bool(torch.isfinite(action[active]).all()):
                    nonfinite_action = True
                    break
                selected = action[active].detach().cpu()
                action_sum += selected.double().sum(0)
                action_square_sum += selected.double().square().sum(0)
                action_min = torch.minimum(action_min, selected.amin(0))
                action_max = torch.maximum(action_max, selected.amax(0))
                action_samples += int(selected.shape[0])
                actions_cpu.copy_(action.detach().cpu())
                vector.cpu_step(actions_cpu.data_ptr())
                done |= terminals > 0.5
                vector_steps = step + 1
        native_log = dict(vector.log())
    finally:
        vector.close()
    wall_time = time.perf_counter() - started
    metrics = flatten_log(pufferl, native_log)
    passed = (
        not nonfinite_action
        and action_samples > 0
        and policy_screen_passes(metrics, episodes=EPISODES)
    )
    action_mean = action_sum / max(action_samples, 1)
    action_variance = torch.clamp(
        action_square_sum / max(action_samples, 1) - action_mean.square(), min=0.0
    )
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "scripts/eval_vq2_native_oracle.py",
        ROOT / "docs/vq2_sf015_recurrent_bc_teacher_free_preregistration_2026-07-28.md",
        CHECKPOINT,
    ]
    report = {
        "schema": "vq2_recurrent_teacher_free_native_screen_v1",
        "tag": TAG,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_best_epoch": payload["best_epoch"],
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "vector_steps": vector_steps,
        "wall_time_seconds": wall_time,
        "inference_seconds": inference_seconds,
        "inference_steps_per_second": (
            vector_steps / inference_seconds if inference_seconds > 0.0 else 0.0
        ),
        "loader_overrides": overrides,
        "teacher_action_blend": float(environment["teacher_action_blend"]),
        "actor_input_width": LEGAL_OBS_SIZE,
        "native_privileged_width_excluded": ENV_OBS_SIZE - LEGAL_OBS_SIZE,
        "action_samples": action_samples,
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": torch.sqrt(action_variance).tolist(),
        "nonfinite_action": nonfinite_action,
        "metrics": metrics,
        "full_course_passed": passed,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
        },
        "safety": {
            "teacher_labels_written": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    report = run_screen(output=args.output.resolve(), device_name=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["full_course_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
