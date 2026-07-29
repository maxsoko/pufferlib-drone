#!/usr/bin/env python3
"""Command-free native diagnostic for the N294-prefix visual suffix composite."""

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
from pufferlib.vq2_n294_visual_composite import (
    ACTION_SIZE,
    LegacyN294ObservationAdapter,
    select_complete_actions,
    visual_suffix_observation,
)
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.collect_vq2_measured_two_gate_oracle_prefix import MEASURED_GATES
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.collect_vq2_public_phase_dagger import (
    PHASE_PRIVILEGED_INDEX,
    update_held_phase,
)
from scripts.eval_vq2_native_oracle import (
    BACKEND_ENV_NAME,
    flatten_log,
    load_fixed_config,
)
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.policy_callable_checkpoint import CheckpointPolicy
from scripts.train_full_policy_bc import SequencePufferNet


TAG = "vq2_c001_n294_visual_suffix_exact_128"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
N294_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_n294_full_blend_refine/alpha_0p60.bin"
)
N294_SHA256 = "a57ca5f4af1bea5d7236b09d6efd9db3fdacbf87114e09a4f3195aeff4169dc6"
SUFFIX_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf066_aggregate_dagger_fit_001/policy_best.pt"
)
SUFFIX_SHA256 = "f4ee6782de66736110c79a892efaa70635a2ad6c14f0bfaeeaba9812da0ae7a8"
SUFFIX_REPORT = SUFFIX_CHECKPOINT.parent / "report.json"
SUFFIX_REPORT_SHA256 = "2ad76bc51417dd2160ef44ec4d4029870d3d9b189efb45d795aacd133e9e226e"
PROMPT = ROOT / "docs/vq2_competitive_completion_execution_prompt.md"
PROMPT_SHA256 = "ffc4986506881746a6a6d4ca537545b585ecddadcdbc51f04a4545ca970ff285"

DT_SECONDS = 1.0 / 64.0
TIME_LIMIT_SECONDS = 14.0
STEP_LIMIT = int(TIME_LIMIT_SECONDS / DT_SECONDS)
FIRST_GATE_PHASE = np.float32(1.0 / 6.0)
SECOND_GATE_PHASE = np.float32(2.0 / 6.0)
PHASE_EPSILON = np.float32(1e-6)


def load_suffix(
    device: torch.device,
) -> tuple[VQ2PhaseResidualActor, dict[str, Any]]:
    frozen = {
        SUFFIX_CHECKPOINT: SUFFIX_SHA256,
        SUFFIX_REPORT: SUFFIX_REPORT_SHA256,
        N294_CHECKPOINT: N294_SHA256,
        PROMPT: PROMPT_SHA256,
    }
    for path, expected in frozen.items():
        actual = sha256_path(path)
        if actual != expected:
            raise RuntimeError(f"source-lock mismatch for {path}: {actual}")
    report = json.loads(SUFFIX_REPORT.read_text())
    if not report.get("combined_numerical_admission"):
        raise RuntimeError("SF066 did not pass its numerical admission")
    payload = torch.load(SUFFIX_CHECKPOINT, map_location=device, weights_only=False)
    if payload.get("schema") != "vq2_public_phase_recurrent_checkpoint_v1":
        raise RuntimeError("unsupported SF066 checkpoint schema")
    contract = payload.get("model", {})
    expected_contract = {
        "class": "VQ2PhaseResidualActor",
        "legal_observation_size": PHASE_LEGAL_OBS_SIZE,
        "legacy_legal_observation_size": LEGAL_OBS_SIZE,
        "public_status_values": 1,
        "action_size": ACTION_SIZE,
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise RuntimeError("SF066 actor ABI changed")
    if payload.get("safety", {}).get(
        "stored_training_only_privileged_values_per_actor_record"
    ) != 0:
        raise RuntimeError("SF066 actor records contain privileged values")
    actor = VQ2PhaseResidualActor(
        hidden_size=int(contract["hidden_size"]),
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(payload["model_state"])
    actor.eval()
    return actor, payload


def load_n294(device: torch.device) -> SequencePufferNet:
    source = CheckpointPolicy.load(
        str(N294_CHECKPOINT),
        input_dim=32,
        num_layers=3,
        layout_precision_bytes=4,
        native_bf16=False,
    )
    model = SequencePufferNet(source, native_bf16=False).to(device)
    model.eval()
    return model


def composite_config(
    pufferl_module: Any,
    *,
    agents: int,
    seed: int,
    camera_pitch_rad: float,
    longitudinal_accel_scale: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    config, overrides = load_fixed_config(
        pufferl_module,
        agents=agents,
        episodes=agents,
        seed=seed,
        controller="governed",
        target_speed_m_s=2.0,
        episode_seconds=TIME_LIMIT_SECONDS,
        randomized_course=False,
    )
    environment = config["env"]
    environment.update(
        {
            "num_gates": 6,
            "use_custom_start": 0,
            "use_custom_gate_layout": 1,
            "reset_position_noise_xy": 0.0,
            "reset_position_noise_z": 0.0,
            "gate_position_domain_randomize": 0,
            "course_geometry_scale_randomize": 0,
            "sitl_plant_domain_randomize": 0,
            "gate_radius": 0.75,
            "gate0_radius": 0.75,
            "gate1_radius": 0.75,
            "max_steps": STEP_LIMIT,
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "crash_height": -10.0,
            "safety_altitude": -8.0,
            "pos_bound": 45.0,
            "visual_camera_roll_rad": 0.0,
            "visual_camera_pitch_rad": camera_pitch_rad,
            "visual_camera_yaw_rad": 0.0,
            "visual_camera_roll_jitter_rad": 0.0,
            "visual_camera_pitch_jitter_rad": 0.0,
            "visual_camera_yaw_jitter_rad": 0.0,
            "visual_camera_dropout_prob": 0.0,
            "visual_edge_dropout_prob": 0.0,
            "visual_edge_corrupt_prob": 0.0,
            "visual_false_segments": 0,
            "visual_rolling_shutter_s": 0.0,
            "teacher_action_blend": 0.0,
            "teacher_course_spline": 0,
            "teacher_segment_minimum_jerk": 0,
            "teacher_alignment_governor": 0,
            "w_action_teacher": 0.0,
            "evaluation_episode_limit": 1,
            "evaluation_episode_offset": 0,
        }
    )
    if longitudinal_accel_scale is not None:
        if not np.isfinite(longitudinal_accel_scale) or longitudinal_accel_scale <= 0:
            raise ValueError("longitudinal_accel_scale must be finite and positive")
        # One coupled training-plant variable: positive-X acceleration and
        # negative-X braking retain the same authority, as in the frozen base
        # configuration. This never modifies a deployed policy action.
        environment["sitl_horizontal_accel_scale"] = longitudinal_accel_scale
        environment["sitl_braking_accel_scale"] = longitudinal_accel_scale
    for index, (x, y, z) in enumerate(MEASURED_GATES):
        environment[f"gate{index}_x"] = x
        environment[f"gate{index}_y"] = y
        environment[f"gate{index}_z"] = z
    return config, overrides


def _first_step(values: np.ndarray, threshold: float) -> list[int]:
    result: list[int] = []
    for row in values:
        indices = np.flatnonzero(row >= threshold - float(PHASE_EPSILON))
        result.append(int(indices[0]) if indices.size else -1)
    return result


def run_diagnostic(
    *,
    output: Path = DEFAULT_OUTPUT,
    agents: int = 128,
    seed: int = 43001,
    device_name: str = "cuda",
    camera_pitch_rad: float = 0.0,
    longitudinal_accel_scale: float | None = None,
    tag: str = TAG,
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if agents <= 0:
        raise ValueError("agents must be positive")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled backend is not drone_race_vision")

    device = torch.device(device_name)
    suffix, suffix_payload = load_suffix(device)
    n294 = load_n294(device)
    config, overrides = composite_config(
        pufferl,
        agents=agents,
        seed=seed,
        camera_pitch_rad=camera_pitch_rad,
        longitudinal_accel_scale=longitudinal_accel_scale,
    )
    environment = config["env"]
    forbidden_teacher_values = (
        "teacher_action_blend",
        "teacher_course_spline",
        "teacher_segment_minimum_jerk",
        "teacher_alignment_governor",
        "w_action_teacher",
    )
    if any(float(environment[name]) != 0.0 for name in forbidden_teacher_values):
        raise RuntimeError("diagnostic refuses any teacher action/reward path")

    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != agents or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native visual vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (agents, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (agents,), torch.float32)
    actions_cpu = torch.zeros((agents, ACTION_SIZE), dtype=torch.float32)
    suffix_state = suffix.initial_state(agents, device=device)
    n294_state = n294.initial_state(agents, device)
    adapter = LegacyN294ObservationAdapter(
        agents,
        dt_s=DT_SECONDS,
        time_limit_s=TIME_LIMIT_SECONDS,
        sample_interval_steps=4,
        dropout_range_m=4.25,
        dropout_from_gate_index=0,
    )

    reached = np.zeros(agents, dtype=bool)
    failed = np.zeros(agents, dtype=bool)
    held_phase = np.zeros(agents, dtype=np.float32)
    raw_history: list[np.ndarray] = []
    held_history: list[np.ndarray] = []
    selected_n294 = np.zeros(agents, dtype=np.int32)
    selected_suffix = np.zeros(agents, dtype=np.int32)
    executed_action_max_error = 0.0
    nonfinite_action = False
    action_envelope_violations = 0
    inference_seconds = 0.0
    vector_steps = 0
    native_log: dict[str, Any] = {}
    trace: dict[str, list[np.ndarray | float | int | bool]] = {
        "legacy_observation": [],
        "visual_observation": [],
        "n294_action": [],
        "suffix_action": [],
        "selected_action": [],
        "raw_phase": [],
        "held_phase": [],
        "suffix_selected": [],
    }
    started = time.perf_counter()

    try:
        vector.reset()
        adapter.reset()
        with torch.no_grad():
            for step in range(STEP_LIMIT):
                current = observations.numpy()
                raw_phase = current[:, PHASE_PRIVILEGED_INDEX].astype(
                    np.float32, copy=True
                )
                if not np.isfinite(raw_phase[~(reached | failed)]).all():
                    nonfinite_action = True
                    break
                held_phase, _ = update_held_phase(
                    raw_phase, held_phase, step=step
                )
                raw_history.append(raw_phase.copy())
                held_history.append(held_phase.copy())
                reached |= held_phase >= SECOND_GATE_PHASE - PHASE_EPSILON
                active_cpu = ~(reached | failed)
                if not active_cpu.any():
                    break

                legacy = adapter.observe(
                    current, raw_phase, held_phase, step=step
                )
                suffix_input = visual_suffix_observation(current, held_phase)
                active = torch.from_numpy(active_cpu).to(device)
                inference_started = time.perf_counter()
                n294_output, n294_candidate = n294.forward_chunk_outputs(
                    torch.from_numpy(legacy).to(device).unsqueeze(1), n294_state
                )
                suffix_output, suffix_candidate = suffix.forward_step(
                    torch.from_numpy(suffix_input).to(device), suffix_state
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - inference_started
                n294_state = preserve_frozen_state(
                    n294_state, n294_candidate, active
                )
                suffix_state = preserve_frozen_state(
                    suffix_state, suffix_candidate, active
                )
                n294_action = torch.clamp(
                    n294_output[:, 0, :ACTION_SIZE], -1.0, 1.0
                ).cpu().numpy()
                suffix_action = suffix_output.mean.cpu().numpy()
                selected, suffix_rows = select_complete_actions(
                    n294_action, suffix_action, held_phase
                )
                selected[~active_cpu] = 0.0
                selected_n294 += (active_cpu & ~suffix_rows).astype(np.int32)
                selected_suffix += (active_cpu & suffix_rows).astype(np.int32)
                if not np.isfinite(selected[active_cpu]).all():
                    nonfinite_action = True
                    break
                action_envelope_violations += int(
                    np.any(np.abs(selected[active_cpu]) > 1.0 + 1e-6, axis=1).sum()
                )

                trace["legacy_observation"].append(legacy[0].copy())
                trace["visual_observation"].append(suffix_input[0].copy())
                trace["n294_action"].append(n294_action[0].copy())
                trace["suffix_action"].append(suffix_action[0].copy())
                trace["selected_action"].append(selected[0].copy())
                trace["raw_phase"].append(float(raw_phase[0]))
                trace["held_phase"].append(float(held_phase[0]))
                trace["suffix_selected"].append(bool(suffix_rows[0]))

                actions_cpu.copy_(torch.from_numpy(selected))
                executed_reference = selected.copy()
                vector.cpu_step(actions_cpu.data_ptr())
                vector_steps = step + 1
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(np.max(np.abs(executed[active_cpu] - executed_reference[active_cpu]))),
                )
                terminal_now = (terminals.numpy() > 0.5) & active_cpu
                failed |= terminal_now
        native_log = dict(vector.log())
    finally:
        vector.close()

    raw_array = np.stack(raw_history, axis=1) if raw_history else np.zeros((agents, 0))
    held_array = (
        np.stack(held_history, axis=1) if held_history else np.zeros((agents, 0))
    )
    first_raw_step = _first_step(raw_array, float(FIRST_GATE_PHASE))
    first_held_step = _first_step(held_array, float(FIRST_GATE_PHASE))
    second_raw_step = _first_step(raw_array, float(SECOND_GATE_PHASE))
    second_held_step = _first_step(held_array, float(SECOND_GATE_PHASE))
    timed_out = ~(reached | failed)
    first_raw_array = np.asarray(first_raw_step, dtype=np.int32)
    first_held_array = np.asarray(first_held_step, dtype=np.int32)
    failed_before_first_gate = failed & (first_raw_array < 0)
    diagnostic_valid = bool(
        not nonfinite_action
        and action_envelope_violations == 0
        and executed_action_max_error <= 5e-5
        and int((selected_n294 > 0).sum()) == agents
        and int((selected_suffix > 0).sum())
        == int((np.asarray(first_held_step) >= 0).sum())
    )
    harness_admitted = bool(
        diagnostic_valid
        and np.all(first_raw_array >= 0)
        and np.all(first_held_array >= 0)
        and not failed_before_first_gate.any()
    )

    output.mkdir(parents=True, exist_ok=False)
    trace_path = output / "trace_agent0.npz"
    np.savez_compressed(
        trace_path,
        **{
            name: np.asarray(values)
            for name, values in trace.items()
        },
    )
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "pufferlib/vq2_n294_visual_composite.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_public_phase_dagger.py",
        N294_CHECKPOINT,
        SUFFIX_CHECKPOINT,
        SUFFIX_REPORT,
        PROMPT,
    )
    metrics = flatten_log(pufferl, native_log)
    report = {
        "schema": "vq2_n294_visual_suffix_composite_diagnostic_v1",
        "tag": tag,
        "diagnostic_valid": diagnostic_valid,
        "harness_admitted": harness_admitted,
        "next_gate_passed": bool(reached.all()),
        "agents": agents,
        "seed": seed,
        "device": device_name,
        "camera_pitch_rad": camera_pitch_rad,
        "longitudinal_accel_scale": float(
            environment["sitl_horizontal_accel_scale"]
        ),
        "braking_accel_scale": float(environment["sitl_braking_accel_scale"]),
        "steps": vector_steps,
        "wall_time_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "composite_inference_steps_per_second": (
            vector_steps / inference_seconds if inference_seconds > 0.0 else 0.0
        ),
        "reached_next_gate": int(reached.sum()),
        "failed_before_next_gate": int(failed.sum()),
        "reached_first_gate_raw": int((first_raw_array >= 0).sum()),
        "reached_first_gate_held": int((first_held_array >= 0).sum()),
        "failed_before_first_gate": int(failed_before_first_gate.sum()),
        "timed_out": int(timed_out.sum()),
        "first_raw_gate_step_min_max": [min(first_raw_step), max(first_raw_step)],
        "first_held_gate_step_min_max": [min(first_held_step), max(first_held_step)],
        "second_raw_gate_step_min_max": [min(second_raw_step), max(second_raw_step)],
        "second_held_gate_step_min_max": [min(second_held_step), max(second_held_step)],
        "selected_n294_steps_min_max": [
            int(selected_n294.min()),
            int(selected_n294.max()),
        ],
        "selected_suffix_steps_min_max": [
            int(selected_suffix.min()),
            int(selected_suffix.max()),
        ],
        "executed_action_max_error": executed_action_max_error,
        "action_envelope_violations": action_envelope_violations,
        "nonfinite_action": nonfinite_action,
        "teacher_action_blend": float(environment["teacher_action_blend"]),
        "runtime_actor_inputs": {
            "n294": 32,
            "visual_suffix": PHASE_LEGAL_OBS_SIZE,
            "visual_suffix_privileged_values": 0,
        },
        "plant_action_contract": {
            "complete_puffer_output_only": True,
            "action_blend": False,
            "channel_override": False,
            "analytic_action": False,
            "n294_owns_held_index_0": True,
            "visual_suffix_owns_held_index_ge_1": True,
        },
        "safety": {
            "flight_sim_packets_sent": 0,
            "teacher_actions_executed": 0,
            "student_updates": 0,
            "submission_actions": 0,
        },
        "suffix_parent": {
            "checkpoint_sha256": SUFFIX_SHA256,
            "best_epoch": suffix_payload["best_epoch"],
        },
        "loader_overrides": overrides,
        "native_metrics_diagnostic_only": metrics,
        "trace": {
            "path": str(trace_path.relative_to(ROOT)),
            "sha256": sha256_path(trace_path),
            "agent": 0,
            "records": len(trace["selected_action"]),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--agents", type=int, default=128)
    parser.add_argument("--seed", type=int, default=43001)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--camera-pitch-rad", type=float, default=0.0)
    parser.add_argument("--longitudinal-accel-scale", type=float)
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args()
    report = run_diagnostic(
        output=args.output,
        agents=args.agents,
        seed=args.seed,
        device_name=args.device,
        camera_pitch_rad=args.camera_pitch_rad,
        longitudinal_accel_scale=args.longitudinal_accel_scale,
        tag=args.tag,
    )
    return 0 if report["harness_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
