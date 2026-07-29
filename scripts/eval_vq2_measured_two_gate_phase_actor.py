#!/usr/bin/env python3
"""Screen SF050 teacher-free through the exact measured Gate-2 milestone."""

from __future__ import annotations

import argparse
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
from pufferlib.vq2_informed import ACTION_HISTORY, ENV_OBS_SIZE, LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.collect_vq2_measured_two_gate_oracle_prefix import (
    MEASURED_GATES,
    measured_config,
)
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.collect_vq2_public_phase_dagger import (
    PHASE_PRIVILEGED_INDEX,
    STATUS_HOLD_STEPS,
    STATUS_RATE_HZ,
    update_held_phase,
)
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state


TAG = "vq2_sf051_measured_two_gate_teacher_free_512"
SCHEMA = "vq2_measured_two_gate_teacher_free_screen_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT
    / "docs/vq2_sf051_measured_two_gate_teacher_free_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf050_measured_two_gate_full_fit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "61631d5bbcfce1eb75522d49804214c91b33930545db6815c2d8c09ea5adaa57"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "d78e6bcbb00e6ca087dee4ed48f5d9de466b08735db63cf073826283a7cfda0c"
)
AGENTS = 512
EPISODES = 512
SEED = 42051
STEP_LIMIT = 2048
MILESTONE_PHASE = np.float32(2.0 / 6.0)
PHASE_EPSILON = 1e-6
MAX_EXECUTED_ACTION_ERROR = 5e-5


def teacher_free_measured_config(
    pufferl_module: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Return the exact SF049 geometry with every teacher path disabled."""

    config, overrides = measured_config(pufferl_module)
    overrides = list(overrides)
    replacements = {
        "--seed": str(SEED),
        "--vec.total-agents": str(AGENTS),
    }
    for option, value in replacements.items():
        position = overrides.index(option)
        overrides[position + 1] = value
    config["seed"] = SEED
    config["vec"]["total_agents"] = AGENTS
    environment = config["env"]
    environment.update(
        {
            "evaluation_episode_limit": 1,
            "max_steps": STEP_LIMIT,
            "time_limit_seconds": STEP_LIMIT * float(environment["dt"]),
            "teacher_action_blend": 0.0,
            "teacher_course_spline": 0,
            "teacher_segment_minimum_jerk": 0,
            "teacher_alignment_governor": 0,
            "w_action_teacher": 0.0,
        }
    )
    return config, overrides


def load_sf050(
    device: torch.device,
) -> tuple[VQ2PhaseResidualActor, dict[str, Any]]:
    """Load only the frozen, numerically admitted SF050 actor."""

    frozen = {
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    train_report = json.loads(TRAIN_REPORT.read_text())
    if not train_report.get("numerically_admitted"):
        raise RuntimeError("SF050 did not pass numerical admission")
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if payload.get("schema") != "vq2_public_phase_recurrent_checkpoint_v1":
        raise RuntimeError("unsupported SF050 checkpoint schema")
    contract = payload.get("model", {})
    expected_contract = {
        "class": "VQ2PhaseResidualActor",
        "legal_observation_size": PHASE_LEGAL_OBS_SIZE,
        "legacy_legal_observation_size": LEGAL_OBS_SIZE,
        "public_status_values": 1,
        "action_size": ACTION_SIZE,
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise RuntimeError("SF050 actor ABI changed")
    safety = payload.get("safety", {})
    if safety.get("stored_training_only_privileged_values_per_actor_record") != 0:
        raise RuntimeError("SF050 does not prove privilege-free actor records")
    actor = VQ2PhaseResidualActor(
        hidden_size=int(contract["hidden_size"]),
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(payload["model_state"])
    actor.eval()
    return actor, payload


def milestone_passes(
    *,
    reached: np.ndarray,
    first_gate_step: np.ndarray,
    second_gate_step: np.ndarray,
    premature_terminal: np.ndarray,
    phase_decreases: int,
    phase_skips: int,
    phase_changes_off_tick: int,
    nonfinite_action: bool,
    action_envelope_violations: int,
    executed_action_max_error: float,
) -> bool:
    """Apply the preregistered exact two-gate admission predicate."""

    return bool(
        reached.shape == (EPISODES,)
        and reached.all()
        and np.all(first_gate_step >= 0)
        and np.all(second_gate_step > first_gate_step)
        and not premature_terminal.any()
        and phase_decreases == 0
        and phase_skips == 0
        and phase_changes_off_tick == 0
        and not nonfinite_action
        and action_envelope_violations == 0
        and executed_action_max_error <= MAX_EXECUTED_ACTION_ERROR
    )


def run_screen(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda"
) -> dict[str, Any]:
    """Run one source-locked, zero-teacher measured-transition screen."""

    from pufferlib import _C, pufferl

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF051 preregisters CUDA inference")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled backend is not drone_race_vision")

    device = torch.device(device_name)
    actor, payload = load_sf050(device)
    config, overrides = teacher_free_measured_config(pufferl)
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
        raise RuntimeError("SF051 exact/teacher-free environment contract changed")

    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native vector ABI differs from SF051")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions_cpu = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    state = actor.initial_state(AGENTS, device=device)

    reached = np.zeros(AGENTS, dtype=bool)
    failed = np.zeros(AGENTS, dtype=bool)
    premature_terminal = np.zeros(AGENTS, dtype=bool)
    first_gate_step = np.full(AGENTS, -1, dtype=np.int32)
    second_gate_step = np.full(AGENTS, -1, dtype=np.int32)
    first_held_gate_step = np.full(AGENTS, -1, dtype=np.int32)
    second_held_gate_step = np.full(AGENTS, -1, dtype=np.int32)
    held_phase = np.zeros(AGENTS, dtype=np.float32)
    previous_raw_phase = np.zeros(AGENTS, dtype=np.float32)
    previous_held_phase = np.zeros(AGENTS, dtype=np.float32)
    maximum_raw_phase = np.zeros(AGENTS, dtype=np.float32)
    maximum_held_phase = np.zeros(AGENTS, dtype=np.float32)
    phase_decreases = 0
    phase_skips = 0
    phase_changes_off_tick = 0
    status_samples = 0
    nonfinite_action = False
    action_envelope_violations = 0
    executed_action_max_error = 0.0
    action_sum = torch.zeros(ACTION_SIZE, dtype=torch.float64)
    action_square_sum = torch.zeros(ACTION_SIZE, dtype=torch.float64)
    action_min = torch.full((ACTION_SIZE,), float("inf"))
    action_max = torch.full((ACTION_SIZE,), float("-inf"))
    action_samples = 0
    inference_seconds = 0.0
    vector_steps = 0
    native_log: dict[str, Any] = {}
    started = time.perf_counter()

    try:
        vector.reset()
        with torch.no_grad():
            for step in range(STEP_LIMIT):
                active_before_status = ~(reached | failed)
                if not active_before_status.any():
                    break
                current = observations.numpy()
                raw_phase = current[:, PHASE_PRIVILEGED_INDEX].astype(
                    np.float32, copy=True
                )
                if not np.isfinite(raw_phase[active_before_status]).all():
                    nonfinite_action = True
                    break
                phase_decreases += int(
                    (
                        raw_phase[active_before_status] + PHASE_EPSILON
                        < previous_raw_phase[active_before_status]
                    ).sum()
                )
                phase_skips += int(
                    (
                        raw_phase[active_before_status]
                        - previous_raw_phase[active_before_status]
                        > np.float32(1.0 / 6.0) + PHASE_EPSILON
                    ).sum()
                )
                previous_raw_phase[active_before_status] = raw_phase[
                    active_before_status
                ]
                maximum_raw_phase = np.maximum(maximum_raw_phase, raw_phase)
                newly_gate1 = active_before_status & (first_gate_step < 0) & (
                    raw_phase >= np.float32(1.0 / 6.0) - PHASE_EPSILON
                )
                newly_gate2 = active_before_status & (second_gate_step < 0) & (
                    raw_phase >= MILESTONE_PHASE - PHASE_EPSILON
                )
                first_gate_step[newly_gate1] = step
                second_gate_step[newly_gate2] = step

                held_phase, sampled = update_held_phase(
                    raw_phase, held_phase, step=step
                )
                held_changed = (
                    np.abs(held_phase - previous_held_phase) > PHASE_EPSILON
                ) & active_before_status
                if held_changed.any() and not sampled:
                    phase_changes_off_tick += int(held_changed.sum())
                if sampled:
                    status_samples += int(active_before_status.sum())
                previous_held_phase[active_before_status] = held_phase[
                    active_before_status
                ]
                maximum_held_phase = np.maximum(maximum_held_phase, held_phase)
                newly_held_gate1 = active_before_status & (
                    first_held_gate_step < 0
                ) & (held_phase >= np.float32(1.0 / 6.0) - PHASE_EPSILON)
                first_held_gate_step[newly_held_gate1] = step
                newly_reached = active_before_status & (
                    held_phase >= MILESTONE_PHASE - PHASE_EPSILON
                )
                second_held_gate_step[newly_reached] = step
                reached |= newly_reached

                active_cpu = ~(reached | failed)
                if not active_cpu.any():
                    break
                phase_tensor = torch.from_numpy(held_phase[:, None]).to(device)
                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                actor_observation = torch.cat((legal, phase_tensor), dim=1)
                active = torch.from_numpy(active_cpu).to(device)
                inference_started = time.perf_counter()
                actor_output, candidate_state = actor.forward_step(
                    actor_observation, state
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - inference_started
                state = preserve_frozen_state(state, candidate_state, active)
                action = torch.where(
                    active[:, None],
                    actor_output.mean,
                    torch.zeros_like(actor_output.mean),
                )
                if not bool(torch.isfinite(action[active]).all()):
                    nonfinite_action = True
                    break
                selected = action[active].detach().cpu()
                action_envelope_violations += int(
                    (selected.abs() > 1.0 + 1e-6).any(dim=1).sum().item()
                )
                action_sum += selected.double().sum(0)
                action_square_sum += selected.double().square().sum(0)
                action_min = torch.minimum(action_min, selected.amin(0))
                action_max = torch.maximum(action_max, selected.amax(0))
                action_samples += int(selected.shape[0])
                actions_cpu.copy_(action.detach().cpu())
                executed_reference = actions_cpu.numpy().copy()
                vector.cpu_step(actions_cpu.data_ptr())
                vector_steps = step + 1
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(
                        np.max(
                            np.abs(
                                executed[active_cpu]
                                - executed_reference[active_cpu]
                            )
                        )
                    ),
                )
                terminal_now = (terminals.numpy() > 0.5) & active_cpu
                premature_terminal |= terminal_now
                failed |= terminal_now
        native_log = dict(vector.log())
    finally:
        vector.close()

    passed = milestone_passes(
        reached=reached,
        first_gate_step=first_gate_step,
        second_gate_step=second_gate_step,
        premature_terminal=premature_terminal,
        phase_decreases=phase_decreases,
        phase_skips=phase_skips,
        phase_changes_off_tick=phase_changes_off_tick,
        nonfinite_action=nonfinite_action,
        action_envelope_violations=action_envelope_violations,
        executed_action_max_error=executed_action_max_error,
    )
    action_mean = action_sum / max(action_samples, 1)
    action_variance = torch.clamp(
        action_square_sum / max(action_samples, 1) - action_mean.square(), min=0.0
    )
    source_paths = (
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_measured_two_gate_oracle_prefix.py",
        ROOT / "scripts/collect_vq2_public_phase_dagger.py",
        ROOT / "scripts/eval_vq2_recurrent_policy.py",
        CHECKPOINT,
        TRAIN_REPORT,
    )
    report = {
        "schema": SCHEMA,
        "tag": TAG,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_best_epoch": payload["best_epoch"],
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "step_limit": STEP_LIMIT,
        "vector_steps": vector_steps,
        "wall_time_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "inference_steps_per_second": (
            action_samples / inference_seconds if inference_seconds > 0.0 else 0.0
        ),
        "measured_gates_ned": [list(gate) for gate in MEASURED_GATES],
        "gate_radius_m": float(environment["gate_radius"]),
        "teacher_action_blend": float(environment["teacher_action_blend"]),
        "actor_input_width": PHASE_LEGAL_OBS_SIZE,
        "legacy_legal_input_width": LEGAL_OBS_SIZE,
        "public_status_values": 1,
        "public_status_source": "4_hz_held_native_active_gate_index_div_6",
        "native_privileged_values_reaching_actor": 0,
        "status_rate_hz": STATUS_RATE_HZ,
        "status_hold_steps": STATUS_HOLD_STEPS,
        "status_samples": status_samples,
        "milestone_phase": float(MILESTONE_PHASE),
        "milestone_reached": int(reached.sum()),
        "premature_terminals": int(premature_terminal.sum()),
        "first_gate_step_min": int(first_gate_step.min()),
        "first_gate_step_max": int(first_gate_step.max()),
        "second_gate_step_min": int(second_gate_step.min()),
        "second_gate_step_max": int(second_gate_step.max()),
        "first_held_gate_step_min": int(first_held_gate_step.min()),
        "first_held_gate_step_max": int(first_held_gate_step.max()),
        "second_held_gate_step_min": int(second_held_gate_step.min()),
        "second_held_gate_step_max": int(second_held_gate_step.max()),
        "maximum_raw_phase_min": float(maximum_raw_phase.min()),
        "maximum_held_phase_min": float(maximum_held_phase.min()),
        "phase_decreases": phase_decreases,
        "phase_skips": phase_skips,
        "phase_changes_off_tick": phase_changes_off_tick,
        "nonfinite_action": nonfinite_action,
        "action_envelope_violations": action_envelope_violations,
        "executed_action_max_error": executed_action_max_error,
        "executed_action_error_threshold": MAX_EXECUTED_ACTION_ERROR,
        "action_samples": action_samples,
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": torch.sqrt(action_variance).tolist(),
        "native_metrics_diagnostic_only": flatten_log(pufferl, native_log),
        "loader_overrides": overrides,
        "two_gate_milestone_passed": passed,
        "official_finish_claimed": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
        },
        "safety": {
            "teacher_actions_executed": 0,
            "student_actions_executed": action_samples,
            "student_updates": 0,
            "labels_written": 0,
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
    return 0 if report["two_gate_milestone_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
