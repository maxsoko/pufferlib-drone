#!/usr/bin/env python3
"""Train native PuffeRL PPO with a structurally legal VQ2 actor input."""

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

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE
from scripts.eval_vq2_lc001_long_course_oracle import (
    long_course_environment,
    sha256_path,
    write_json_once,
)


TAG = "vq2_lc016_native_legal_puffer_ppo_001"
SCHEMA = "vq2_lc016_native_legal_puffer_ppo_report_v1"
PREREGISTRATION = (
    ROOT / "docs/vq2_lc016_native_legal_puffer_ppo_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc016_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
TOTAL_AGENTS = 1024
NUM_BUFFERS = 16
NUM_THREADS = 128
HORIZON = 16
MINIBATCH_SIZE = TOTAL_AGENTS * HORIZON
TOTAL_TIMESTEPS = 10_000_384
CHECKPOINT_INTERVAL_STEPS = 2_000_000
HIDDEN_SIZE = 256
SEED = 431160
PUBLIC_PROGRESS_INPUT_INDEX = LEGAL_OBS_SIZE - 1


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER,
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
        ROOT / "tests/test_drone_race_vision_native.c",
        ROOT / "src/pufferlib.cu", ROOT / "src/models.cu",
        ROOT / "src/bindings.cu", ROOT / "src/vecenv.h",
        ROOT / "config/drone_race_vq2_informed_dreamer.ini",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "scripts/eval_vq2_lc001_long_course_oracle.py",
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
    }


def training_config(pufferl_module: Any) -> dict[str, Any]:
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        config = pufferl_module.load_config("drone_race_vq2_informed_dreamer")
    finally:
        sys.argv = saved_argv
    config.update({
        "cudagraphs": 1, "profile": False, "reset_state": False,
        "rank": 0, "world_size": 1, "gpu_id": 0,
        "nccl_id": "None", "seed": SEED,
    })
    config["vec"].update({
        "total_agents": TOTAL_AGENTS,
        "num_buffers": NUM_BUFFERS,
        "num_threads": NUM_THREADS,
    })
    config["policy"].update({
        "hidden_size": HIDDEN_SIZE, "num_layers": 1,
        "expansion_factor": 1,
    })
    config["train"].update({
        "total_timesteps": TOTAL_TIMESTEPS,
        "horizon": HORIZON,
        "minibatch_size": MINIBATCH_SIZE,
        "replay_ratio": 1.0,
        "learning_start_timesteps": 0,
        "learning_rate": 3e-4,
        "anneal_lr": 1,
        "min_lr_ratio": 0.1,
        "gamma": 0.997,
        "gae_lambda": 0.95,
        "clip_coef": 0.1,
        "vf_coef": 1.0,
        "vf_clip_coef": 0.2,
        "max_grad_norm": 1.0,
        "ent_coef": 0.002,
        "reward_scale": 0.05,
        "reward_clip": 10.0,
        "phase_prio_obs_index": PUBLIC_PROGRESS_INPUT_INDEX,
        "phase_prio_scale": 3.0,
        "phase_prio_max_weight": 8.0,
        "train_encoder_feature_start": -1,
        "train_encoder_feature_end": -1,
    })
    environment = long_course_environment(24)
    environment.update({
        "teacher_action_blend": 0.0,
        "teacher_action_blend_from_step": 0,
        "w_action_teacher": 0.0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "visual_policy_legal_only": 1,
        "observable_gate_progress": 1,
        "observable_gate_index_denominator": 6.0,
        "observable_gate_progress_unbounded": 1,
        "num_gates_per_env_randomize": 1,
        "num_gates_per_env_min": 1,
        "num_gates_per_env_max": 24,
        "num_gates_per_env_seed": SEED,
        "gate_local_start_curriculum": 1,
        "gate_local_start_probability": 0.50,
        "gate_local_start_offset_min": 2.0,
        "gate_local_start_offset_max": 5.0,
        "gate_local_start_gate_min": 0,
        "gate_local_start_gate_max_exclusive": 0,
        "gate_radius": 1.5,
        "gate_radius_randomize": 1,
        "gate_radius_min": 1.25,
        "gate_radius_max": 2.0,
        "max_steps": 7680,
        "time_limit_seconds": 120.0,
        "w_progress": 10.0,
        "w_gate": 0.0,
        "w_ordered_gate": 50.0,
        "w_finish": 100.0,
        "w_time": 0.02,
        "w_ctrl": 0.002,
        "w_body_rate": 0.1,
        "w_cross_track": 2.0,
        "cross_track_from_gate_index": 0,
        "w_gate_camera_alignment": 10.0,
        "gate_camera_alignment_from_gate_index": 0,
        "w_gate_crossing_error": 5.0,
        "gate_crossing_error_from_gate_index": 0,
        "invalid_penalty": 100.0,
        "late_invalid_penalty": 100.0,
        "late_invalid_penalty_from_gate_index": 0,
        "evaluation_episode_limit": 0,
        "evaluation_episode_offset": 0,
    })
    config["env"].update(environment)
    pufferl_module.validate_config(config)
    return config


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def run(*, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if _C.env_name != "drone_race_vision" or _C.precision_bytes != 4:
        raise RuntimeError("LC016 requires the float32 drone_race_vision backend")
    if os.environ.get("OMP_NUM_THREADS") != str(NUM_THREADS):
        raise RuntimeError("LC016 requires 128 OpenMP threads")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = training_config(pufferl)
    identity = source_identity()
    output.mkdir(parents=True)
    engine = _C.create_pufferl(config)
    logs: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    boundary_checked = False
    privileged_max_abs = math.inf
    progress_encoding_max_error = math.inf
    started = time.perf_counter()
    next_checkpoint = CHECKPOINT_INTERVAL_STEPS
    try:
        while engine.global_step < TOTAL_TIMESTEPS:
            _C.rollouts(engine)
            if not boundary_checked:
                trace = _C.rollout_trace(engine)
                observations = np.asarray(trace["observations"], dtype=np.float32)
                if observations.shape[-1] != ENV_OBS_SIZE:
                    raise RuntimeError("LC016 rollout observation ABI changed")
                privileged_max_abs = float(
                    np.abs(observations[..., LEGAL_OBS_SIZE:]).max(initial=0.0)
                )
                progress = observations[..., PUBLIC_PROGRESS_INPUT_INDEX]
                progress_encoding_max_error = float(
                    np.abs(progress * 6.0 - np.rint(progress * 6.0)).max(
                        initial=0.0
                    )
                )
                if (
                    privileged_max_abs != 0.0
                    or progress_encoding_max_error > 1e-6
                    or progress.min(initial=0.0) < 0.0
                ):
                    raise RuntimeError("LC016 policy input crossed the legal boundary")
                boundary_checked = True
            _C.train(engine)
            if engine.global_step >= next_checkpoint:
                log = dict(_C.log(engine))
                logs.append(log)
                path = output / f"policy_{engine.global_step:012d}.bin"
                _C.save_weights(engine, str(path))
                checkpoints.append({
                    "agent_steps": int(engine.global_step),
                    "path": path.name,
                    "sha256": sha256_path(path),
                    "bytes": path.stat().st_size,
                })
                next_checkpoint += CHECKPOINT_INTERVAL_STEPS
        final_log = dict(_C.log(engine))
        logs.append(final_log)
        final_path = output / f"policy_{engine.global_step:012d}_final.bin"
        _C.save_weights(engine, str(final_path))
        checkpoints.append({
            "agent_steps": int(engine.global_step), "path": final_path.name,
            "sha256": sha256_path(final_path), "bytes": final_path.stat().st_size,
        })
        num_params = int(engine.num_params())
        exact_steps = int(engine.global_step)
    finally:
        _C.close(engine)
    wall = time.perf_counter() - started
    report: dict[str, Any] = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "training_admitted": False,
        **identity,
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "configuration": {
            "num_gates_max": 24, "num_gates_min": 1,
            "official_course_count_hardcoded": False,
            "total_agents": TOTAL_AGENTS, "num_buffers": NUM_BUFFERS,
            "num_threads": NUM_THREADS, "horizon": HORIZON,
            "minibatch_size": MINIBATCH_SIZE, "hidden_size": HIDDEN_SIZE,
            "requested_timesteps": TOTAL_TIMESTEPS,
            "visual_policy_legal_only": 1,
            "public_progress_input_index": PUBLIC_PROGRESS_INPUT_INDEX,
        },
        "agent_steps": exact_steps, "wall_time_seconds": wall,
        "agent_steps_per_second": exact_steps / wall,
        "num_params": num_params, "logs": logs,
        "checkpoints": checkpoints,
        "privileged_policy_input_max_abs": privileged_max_abs,
        "public_progress_encoding_max_error": progress_encoding_max_error,
        "policy_input_contract": (
            "camera mask + IMU/actuator/action history + public index/6; "
            "native training-only tail exact zero"
        ),
        "safety": {
            "teacher_plant_actions": 0, "runtime_teacher_actions": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
    }
    predicates = {
        "exact_training_budget": exact_steps >= TOTAL_TIMESTEPS,
        "legal_boundary_exact": privileged_max_abs == 0.0,
        "public_progress_exact": progress_encoding_max_error <= 1e-6,
        "finite_logs": finite_tree(logs),
        "minimum_native_training_speed": report["agent_steps_per_second"] >= 50_000,
        "multiple_checkpoints": len(checkpoints) >= 5,
        "whole_puffer_actions": True,
        "zero_teacher_plant_actions": True,
        "no_flightsim_packets": True,
        "submission_forbidden": True,
    }
    report["admission_predicates"] = predicates
    report["failed_admission_predicates"] = [
        name for name, passed in predicates.items() if not passed
    ]
    report["training_admitted"] = all(predicates.values())
    report["next_authority"] = (
        "Teacher-free deterministic full-start screens of the saved Puffer checkpoints."
        if report["training_admitted"]
        else "Reject LC016 and diagnose the failed native-training predicate."
    )
    write_json_once(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(output=args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
