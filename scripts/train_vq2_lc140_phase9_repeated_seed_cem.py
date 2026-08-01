#!/usr/bin/env python3
"""Search LC105's phase-9 Puffer residual on 512 exact seed-15 copies."""

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
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc140_phase9_repeated_seed_cem_001"
SCHEMA = "vq2_lc140_phase9_repeated_seed_cem_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc140_phase9_repeated_seed_cem_checkpoint_v1"
TOTAL_AGENTS = 512
THREADS = 32
SEED = 432_140
ENV_SEED_INDEX = 15
NUM_GATES = 24
MAX_STEPS = 12_000
TARGET_PHASE = 9
TARGET_RAW_INDEX = 10
GENERATIONS = 4
ELITES = 64
CEM_SMOOTHING = 0.70
INITIAL_STD = np.array((0.08, 0.08, 0.08, 0.01), dtype=np.float32)
MINIMUM_STD = np.array((0.008, 0.008, 0.008, 0.001), dtype=np.float32)
MAXIMUM_ABS_DELTA = np.array((0.35, 0.35, 0.35, 0.05), dtype=np.float32)
PHASE_OUTPUT_BIAS = "indexed_phase_residual_output_bias"
# Backward-compatible report/checkpoint label hook for later phase searches.
FROZEN_STATE_FIELD = "frozen_non_phase9_state_exact"
DELTA_FIELD = "phase9_pre_tanh_output_bias_delta"
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc105_phase7_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "005e5e7929258fd282ab390fd230fda817700afaee790e78144200e67b3e10a4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c614e6929282399cac2f18465084f8337ed8770389a74e48b86ff1fcf4315e7d"
LC139_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc139_phase9_lc105_anchor_rescue_001/report.json"
)
LC139_REPORT_SHA256 = "32882f55943b05786b4c7888a929b5a20098dc01cdffa1fa19a02064eb8ebe00"
PREREGISTRATION = ROOT / "docs/vq2_lc140_phase9_repeated_seed_cem_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc140_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc140_phase9_repeated_seed_cem.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC139_REPORT: LC139_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC140 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC139_REPORT.read_text())
    control, intervention = rejected.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or rejected.get("schema") != "vq2_lc139_phase9_lc105_anchor_rescue_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("training_dataset_admitted")
        or control.get("maximum_raw_index_distribution", {}).get("9") != 1
        or intervention.get("target_passes") != 0
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC139 do not authorize LC140")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC139_REPORT,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
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


def candidate_deltas(
    mean: np.ndarray, standard_deviation: np.ndarray, *, generation: int
) -> np.ndarray:
    if mean.shape != (ACTION_SIZE,) or standard_deviation.shape != mean.shape:
        raise ValueError("LC140 CEM moments must have four channels")
    rng = np.random.default_rng(SEED + generation)
    noise = rng.standard_normal((TOTAL_AGENTS // 2 - 1, ACTION_SIZE)).astype(np.float32)
    positive = mean + noise * standard_deviation
    negative = mean - noise * standard_deviation
    deltas = np.concatenate((np.zeros((1, ACTION_SIZE), dtype=np.float32),
                             mean[None].astype(np.float32), positive, negative), axis=0)
    return np.clip(deltas, -MAXIMUM_ABS_DELTA, MAXIMUM_ABS_DELTA).astype(np.float32)


def checkpoint_with_delta(
    parent: dict[str, Any], delta: np.ndarray
) -> tuple[dict[str, Any], bool]:
    parent_state = parent["model_state"]
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    state[PHASE_OUTPUT_BIAS][TARGET_PHASE].add_(torch.from_numpy(delta).float())
    frozen_exact = True
    for name, value in state.items():
        if name == PHASE_OUTPUT_BIAS:
            keep = torch.arange(value.shape[0]) != TARGET_PHASE
            frozen_exact &= torch.equal(value[keep], parent_state[name][keep])
            expected = parent_state[name][TARGET_PHASE] + torch.from_numpy(delta).float()
            frozen_exact &= torch.equal(value[TARGET_PHASE], expected)
        else:
            frozen_exact &= torch.equal(value, parent_state[name])
    payload = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": TAG,
        "model_state": state,
        "numerically_admitted": True,
        "deployment_candidate": False,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "target_phase": TARGET_PHASE,
        DELTA_FIELD: delta.tolist(),
        FROZEN_STATE_FIELD: frozen_exact,
    }
    return payload, frozen_exact


def run_generation(
    parent: dict[str, Any], deltas: np.ndarray, *, generation: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    from pufferlib import _C, pufferl

    actor = milestone.load_actor(parent, device)
    recurrent = actor.initial_state(TOTAL_AGENTS, device=device)
    config, overrides = load_config(
        pufferl, num_gates=NUM_GATES, agents=TOTAL_AGENTS,
        episodes=TOTAL_AGENTS, seed=SEED, threads=THREADS,
    )
    config["vec"]["env_seed_group_size"] = 1
    config["vec"]["env_seed_index_offset"] = ENV_SEED_INDEX
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
        raise RuntimeError("LC140 native vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (TOTAL_AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (TOTAL_AGENTS,), torch.float32)
    rewards = _cpu_tensor(vector.rewards_ptr, (TOTAL_AGENTS,), torch.float32)
    actions_cpu = torch.zeros((TOTAL_AGENTS, ACTION_SIZE), dtype=torch.float32)
    delta_device = torch.from_numpy(deltas).to(device)

    resolved = np.zeros(TOTAL_AGENTS, dtype=bool)
    passed = np.zeros(TOTAL_AGENTS, dtype=bool)
    queried = np.zeros(TOTAL_AGENTS, dtype=bool)
    held = np.zeros(TOTAL_AGENTS, dtype=np.float32)
    previous_held = held.copy()
    maximum_raw_index = np.zeros(TOTAL_AGENTS, dtype=np.int32)
    phase_return = np.zeros(TOTAL_AGENTS, dtype=np.float64)
    phase_changes_off_tick = phase_decreases = phase_skips = 0
    raw_encoding_max_error = action_envelope_violations = 0.0
    executed_action_max_error = 0.0
    nonfinite_action = False
    initial_states_exact = False
    vector_steps = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        vector.reset()
        initial = observations.numpy()
        initial_states_exact = bool(np.array_equal(
            initial, np.broadcast_to(initial[0], initial.shape)
        ))
        if not initial_states_exact:
            raise RuntimeError("LC140 seed-offset rows do not begin identically")
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
                if changed.any() and not sampled:
                    phase_changes_off_tick += int(changed.sum())
                phase_decreases += int(((delta_index < 0) & active_np).sum())
                phase_skips += int(((delta_index > 1) & active_np).sum())
                raw_encoding_max_error = max(
                    raw_encoding_max_error,
                    float(np.abs(raw_scaled - raw_indices).max(initial=0.0)),
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
                recurrent = preserve_frozen_state(recurrent, next_recurrent, active_device)
                phase_index = np.rint(held * OFFICIAL_PROGRESS_SCALE).astype(np.int32)
                query_np = active_np & (phase_index == TARGET_PHASE)
                queried |= query_np
                query_device = torch.from_numpy(query_np).to(device)
                pre_tanh = result.pre_tanh_mean + query_device[:, None] * delta_device
                plant = torch.where(
                    active_device[:, None], torch.tanh(pre_tanh), torch.zeros_like(pre_tanh)
                )
                inference_seconds += time.perf_counter() - inference_started
                plant_np = plant.cpu().numpy().astype(np.float32, copy=False)
                if not np.isfinite(plant_np[active_np]).all():
                    nonfinite_action = True
                    break
                action_envelope_violations += float(
                    (np.abs(plant_np[active_np]) > 1.0 + 1e-6).sum()
                )
                actions_cpu.copy_(torch.from_numpy(plant_np))
                vector.cpu_step(actions_cpu.data_ptr())
                vector_steps = step + 1
                phase_return[query_np] += rewards.numpy()[query_np]
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(np.max(np.abs(executed[active_np] - plant_np[active_np]), initial=0.0)),
                )
                post_indices = np.rint(
                    observations.numpy()[:, core.PHASE_PRIVILEGED_INDEX]
                    * OFFICIAL_PROGRESS_SCALE
                ).astype(np.int32)
                maximum_raw_index = np.maximum(maximum_raw_index, post_indices)
                newly_passed = (~resolved) & (post_indices >= TARGET_RAW_INDEX)
                passed |= newly_passed
                resolved |= newly_passed
                terminal_np = (terminals.numpy() > 0.5) & (~resolved)
                resolved |= terminal_np
    finally:
        vector.close()

    transport_pass = bool(
        initial_states_exact and queried.all() and resolved.all()
        and not nonfinite_action and action_envelope_violations == 0
        and executed_action_max_error <= core.MAX_EXECUTED_ACTION_ERROR
        and phase_changes_off_tick == 0 and phase_decreases == 0 and phase_skips == 0
        and raw_encoding_max_error <= 1e-6 and np.isfinite(phase_return).all()
    )
    clipped = np.minimum(maximum_raw_index, TARGET_RAW_INDEX)
    score = phase_return + maximum_raw_index.astype(np.float64) * 1_000.0
    score += passed.astype(np.float64) * 1_000_000.0
    order = np.argsort(score)[::-1]
    successes = np.flatnonzero(passed)
    report = {
        "generation": generation,
        "target_passes": int(passed.sum()),
        "query_agents": int(queried.sum()),
        "maximum_raw_index_distribution": {
            str(index): int((clipped == index).sum())
            for index in range(TARGET_RAW_INDEX + 1)
        },
        "phase_return_mean": float(phase_return.mean()),
        "phase_return_std": float(phase_return.std()),
        "phase_return_min": float(phase_return.min()),
        "phase_return_max": float(phase_return.max()),
        "best_agent": int(order[0]),
        "best_delta": deltas[order[0]].tolist(),
        "best_score": float(score[order[0]]),
        "successful_agents": successes[:32].tolist(),
        "successful_deltas": deltas[successes[:32]].tolist(),
        "transport_pass": transport_pass,
        "initial_states_exact": initial_states_exact,
        "unresolved": int((~resolved).sum()),
        "vector_steps": vector_steps,
        "wall_time_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "executed_action_max_error": executed_action_max_error,
        "action_envelope_violations": int(action_envelope_violations),
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases,
        "phase_skips": phase_skips,
        "raw_progress_encoding_max_error": raw_encoding_max_error,
        "loader_overrides": [
            *overrides, "--vec.env-seed-group-size", "1",
            "--vec.env-seed-index-offset", str(ENV_SEED_INDEX),
        ],
    }
    return report, score, passed, order


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    parent = verify_inputs()
    from pufferlib import _C

    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC140 requires rebuilt float32 drone_race_vision backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC140 preregisters CUDA inference")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"LC140 requires OMP_NUM_THREADS={THREADS}")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC140 requires OMP_DYNAMIC=FALSE")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC140 does not resume a partial search")
    output.mkdir(parents=True, exist_ok=True)

    identity = source_identity()
    device = torch.device(device_name)
    mean = np.zeros(ACTION_SIZE, dtype=np.float32)
    standard_deviation = INITIAL_STD.copy()
    generations: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    started = time.perf_counter()
    for generation in range(1, GENERATIONS + 1):
        deltas = candidate_deltas(mean, standard_deviation, generation=generation)
        item, score, passed, order = run_generation(
            parent, deltas, generation=generation, device=device
        )
        item["search_mean"] = mean.tolist()
        item["search_std"] = standard_deviation.tolist()
        generations.append(item)
        if not item["transport_pass"]:
            break
        successful = np.flatnonzero(passed)
        if successful.size:
            best = max(
                successful.tolist(),
                key=lambda index: (float(score[index]), -float(np.linalg.norm(deltas[index]))),
            )
            selected = {
                "generation": generation, "agent": int(best),
                "delta": deltas[best].tolist(), "score": float(score[best]),
                "phase_return": float(score[best] - 1_010_000.0),
            }
            break
        elite = deltas[order[:ELITES]]
        elite_mean = elite.mean(axis=0)
        elite_std = elite.std(axis=0)
        mean = ((1.0 - CEM_SMOOTHING) * mean + CEM_SMOOTHING * elite_mean).astype(np.float32)
        standard_deviation = np.maximum(
            MINIMUM_STD,
            (1.0 - CEM_SMOOTHING) * standard_deviation + CEM_SMOOTHING * elite_std,
        ).astype(np.float32)

    checkpoint_item = None
    frozen_exact = False
    if selected is not None and all(item["transport_pass"] for item in generations):
        delta = np.asarray(selected["delta"], dtype=np.float32)
        checkpoint, frozen_exact = checkpoint_with_delta(parent, delta)
        checkpoint_path = output / "policy_selected.pt"
        atomic_torch_save(checkpoint_path, checkpoint)
        checkpoint_item = {
            "path": checkpoint_path.name,
            "sha256": sha256_path(checkpoint_path),
            "state_sha256": state_sha256(checkpoint["model_state"]),
        }
    admitted = bool(
        selected is not None and checkpoint_item is not None and frozen_exact
        and all(item["transport_pass"] for item in generations)
    )
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "training_admitted": admitted,
        "candidate_selected_for_screen": checkpoint_item,
        "selected_candidate": selected,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_state_sha256": state_sha256(parent["model_state"]),
        FROZEN_STATE_FIELD: frozen_exact,
        "generations": generations,
        "configuration": {
            "total_agents": TOTAL_AGENTS, "threads": THREADS,
            "environment_seed_group_size": 1,
            "environment_seed_index_offset": ENV_SEED_INDEX,
            "num_proxy_gates": NUM_GATES, "max_steps": MAX_STEPS,
            "target_phase": TARGET_PHASE, "target_raw_index": TARGET_RAW_INDEX,
            "maximum_generations": GENERATIONS, "elites": ELITES,
            "cem_smoothing": CEM_SMOOTHING,
            "initial_standard_deviation": INITIAL_STD.tolist(),
            "minimum_standard_deviation": MINIMUM_STD.tolist(),
            "maximum_absolute_delta": MAXIMUM_ABS_DELTA.tolist(),
        },
        "optimization": (
            "512 complete recurrent LC105 Puffer candidates; one constant legal phase-9 pre-tanh residual per trajectory"
        ),
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0, "offline_training_teacher_actions": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Run one deterministic LC105-versus-LC140 raw-10 screen; no FlightSim authority."
            if admitted else
            "Reject LC140 and retain LC105; do not run FlightSim."
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
