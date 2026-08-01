#!/usr/bin/env python3
"""Paired exact test of whether the training-only oracle rescues LC105 phase 9."""

from __future__ import annotations

import argparse
import json
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
from pufferlib.vq2_oracle import alignment_oracle_action
from pufferlib.vq2_public_phase import OFFICIAL_PROGRESS_SCALE
from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.eval_vq2_lc001_long_course_oracle import load_config
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc007_converted_vg071_long_course as core
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone


TAG = "vq2_lc115_phase9_teacher_rescue_001"
SCHEMA = "vq2_lc115_phase9_teacher_rescue_report_v1"
GROUP_SIZE = 128
GROUPS = 2
TOTAL_AGENTS = GROUP_SIZE * GROUPS
EPISODES = TOTAL_AGENTS
THREADS = 32
SEED = 432_050
NUM_GATES = 24
MAX_STEPS = 12_000
TARGET_PHASE = 9
TARGET_RAW_INDEX = 10
PARENT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc105_phase7_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "005e5e7929258fd282ab390fd230fda817700afaee790e78144200e67b3e10a4"
PARENT_REPORT = PARENT / "report.json"
PARENT_REPORT_SHA256 = "c614e6929282399cac2f18465084f8337ed8770389a74e48b86ff1fcf4315e7d"
LC114_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc114_phase9_only_large_scale_screen_001/report.json"
)
LC114_REPORT_SHA256 = "09d730b6dd16ec62aeb475e8b992b5afff0d81dc32c505fce551223b4a0c76e1"
PREREGISTRATION = ROOT / "docs/vq2_lc115_phase9_teacher_rescue_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc115_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc115_phase9_teacher_rescue.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def group_slice(group: int) -> slice:
    if group not in (0, 1):
        raise ValueError("LC115 group index must be baseline or intervention")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def select_plant_actions(
    student: np.ndarray,
    teacher: np.ndarray,
    active: np.ndarray,
    phase_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the oracle only to active candidate rows at held public phase 9."""

    if student.shape != (TOTAL_AGENTS, ACTION_SIZE) or teacher.shape != student.shape:
        raise ValueError("LC115 action batch does not match the exact paired contract")
    if active.shape != (TOTAL_AGENTS,) or phase_index.shape != active.shape:
        raise ValueError("LC115 phase/active batch does not match the paired contract")
    teacher_mask = np.zeros(TOTAL_AGENTS, dtype=bool)
    selected = group_slice(1)
    teacher_mask[selected] = active[selected] & (phase_index[selected] == TARGET_PHASE)
    plant = student.copy()
    plant[teacher_mask] = teacher[teacher_mask]
    return plant, teacher_mask


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC114_REPORT: LC114_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC115 bound input changed: {path}")
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC114_REPORT.read_text())
    rejected_items = rejected.get("items", [])
    if (
        payload.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_report_v1"
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or rejected.get("schema") != "vq2_lc114_phase9_only_large_scale_screen_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or len(rejected_items) != 4
        or any(item.get("target_passes") != 0 for item in rejected_items)
        or any(
            item.get("maximum_raw_index_distribution", {}).get("9") != 1
            for item in rejected_items
        )
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC114 do not authorize the LC115 rescue diagnostic")
    return payload


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC114_REPORT,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc007_converted_vg071_long_course.py",
        ROOT / "scripts/eval_vq2_lc058_phase2_bias_milestone.py",
        ROOT / "src/vecenv.h", ROOT / "src/bindings.cu",
        ROOT / "src/bindings_cpu.cpp", ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h", ROOT / "ocean/drone_race/binding.c",
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
            "numpy": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
    }


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    payload = verify_inputs()
    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC115 requires the float32 drone_race_vision backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC115 preregisters CUDA inference")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"LC115 requires OMP_NUM_THREADS={THREADS}")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC115 requires OMP_DYNAMIC=FALSE")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC115 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)

    identity = source_identity()
    device = torch.device(device_name)
    actor = milestone.load_actor(payload, device)
    recurrent = actor.initial_state(TOTAL_AGENTS, device=device)
    config, overrides = load_config(
        pufferl, num_gates=NUM_GATES, agents=TOTAL_AGENTS,
        episodes=EPISODES, seed=SEED, threads=THREADS,
    )
    config["vec"]["env_seed_group_size"] = GROUP_SIZE
    environment = config["env"]
    environment.update({
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
        raise RuntimeError("LC115 native vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (TOTAL_AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (TOTAL_AGENTS,), torch.float32)
    actions_cpu = torch.zeros((TOTAL_AGENTS, ACTION_SIZE), dtype=torch.float32)

    resolved = np.zeros(TOTAL_AGENTS, dtype=bool)
    passed = np.zeros(TOTAL_AGENTS, dtype=bool)
    pretarget_terminal = np.zeros(TOTAL_AGENTS, dtype=bool)
    held = np.zeros(TOTAL_AGENTS, dtype=np.float32)
    previous_held = held.copy()
    maximum_raw_index = np.zeros(TOTAL_AGENTS, dtype=np.int32)
    resolve_step = np.full(TOTAL_AGENTS, -1, dtype=np.int32)
    teacher_steps_by_agent = np.zeros(TOTAL_AGENTS, dtype=np.int32)
    teacher_min = np.full(ACTION_SIZE, np.inf, dtype=np.float64)
    teacher_max = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
    teacher_below_target = 0
    teacher_outside_candidate = 0
    phase_changes_off_tick = np.zeros(GROUPS, dtype=np.int64)
    phase_decreases = np.zeros(GROUPS, dtype=np.int64)
    phase_skips = np.zeros(GROUPS, dtype=np.int64)
    raw_encoding_max_error = np.zeros(GROUPS, dtype=np.float64)
    action_envelope_violations = np.zeros(GROUPS, dtype=np.int64)
    executed_action_max_error = np.zeros(GROUPS, dtype=np.float64)
    initial_groups_exact = False
    nonfinite_action = False
    vector_steps = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        vector.reset()
        initial = observations.numpy()
        initial_groups_exact = np.array_equal(
            initial[group_slice(0)], initial[group_slice(1)]
        )
        if not initial_groups_exact:
            raise RuntimeError("LC115 paired seed groups do not begin identically")
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
                encoding_errors = np.abs(raw_scaled - raw_indices)
                for group in range(GROUPS):
                    selected = group_slice(group)
                    selected_active = active_np[selected]
                    if changed[selected].any() and not sampled:
                        phase_changes_off_tick[group] += int(changed[selected].sum())
                    phase_decreases[group] += int(
                        ((delta_index[selected] < 0) & selected_active).sum()
                    )
                    phase_skips[group] += int(
                        ((delta_index[selected] > 1) & selected_active).sum()
                    )
                    raw_encoding_max_error[group] = max(
                        raw_encoding_max_error[group],
                        float(encoding_errors[selected].max(initial=0.0)),
                    )
                previous_held = held.copy()
                maximum_raw_index = np.maximum(maximum_raw_index, raw_indices)
                newly_passed = (~resolved) & (raw_indices >= TARGET_RAW_INDEX)
                passed |= newly_passed
                resolve_step[newly_passed] = step
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
                student = torch.where(
                    active_device[:, None], result.mean, torch.zeros_like(result.mean)
                )
                inference_seconds += time.perf_counter() - inference_started
                student_np = student.cpu().numpy().astype(np.float32, copy=False)
                teacher = alignment_oracle_action(current)
                phase_index = np.rint(
                    held * OFFICIAL_PROGRESS_SCALE
                ).astype(np.int32)
                plant, teacher_mask = select_plant_actions(
                    student_np, teacher, active_np, phase_index
                )
                teacher_count = int(teacher_mask.sum())
                if teacher_count:
                    selected_teacher = teacher[teacher_mask]
                    teacher_min = np.minimum(teacher_min, selected_teacher.min(axis=0))
                    teacher_max = np.maximum(teacher_max, selected_teacher.max(axis=0))
                teacher_steps_by_agent += teacher_mask.astype(np.int32)
                teacher_below_target += int(
                    (teacher_mask & (phase_index < TARGET_PHASE)).sum()
                )
                outside = np.ones(TOTAL_AGENTS, dtype=bool)
                outside[group_slice(1)] = False
                teacher_outside_candidate += int((teacher_mask & outside).sum())
                if not np.isfinite(plant[active_np]).all():
                    nonfinite_action = True
                    break
                for group in range(GROUPS):
                    selected = group_slice(group)
                    action_envelope_violations[group] += int(
                        (np.abs(plant[selected][active_np[selected]]) > 1.0 + 1e-6).sum()
                    )
                actions_cpu.copy_(torch.from_numpy(plant))
                vector.cpu_step(actions_cpu.data_ptr())
                vector_steps = step + 1
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                for group in range(GROUPS):
                    selected = group_slice(group)
                    selected_active = active_np[selected]
                    executed_action_max_error[group] = max(
                        executed_action_max_error[group],
                        float(np.max(
                            np.abs(executed[selected][selected_active]
                                   - plant[selected][selected_active]),
                            initial=0.0,
                        )),
                    )
                post_raw = observations.numpy()[:, core.PHASE_PRIVILEGED_INDEX]
                post_indices = np.rint(
                    post_raw * OFFICIAL_PROGRESS_SCALE
                ).astype(np.int32)
                maximum_raw_index = np.maximum(maximum_raw_index, post_indices)
                newly_passed = (~resolved) & (post_indices >= TARGET_RAW_INDEX)
                passed |= newly_passed
                resolve_step[newly_passed] = step + 1
                resolved |= newly_passed
                terminal_np = (terminals.numpy() > 0.5) & (~resolved)
                pretarget_terminal |= terminal_np
                resolve_step[terminal_np] = step + 1
                resolved |= terminal_np
    finally:
        vector.close()
    wall = time.perf_counter() - started

    items: list[dict[str, Any]] = []
    baseline_passed = passed[group_slice(0)]
    for group, name in enumerate(("lc105_puffer_control", "phase9_training_oracle")):
        selected = group_slice(group)
        selected_passed = passed[selected]
        clipped = np.minimum(maximum_raw_index[selected], TARGET_RAW_INDEX)
        pass_steps = resolve_step[selected][selected_passed]
        transport_pass = bool(
            not nonfinite_action
            and action_envelope_violations[group] == 0
            and executed_action_max_error[group] <= core.MAX_EXECUTED_ACTION_ERROR
            and phase_changes_off_tick[group] == 0
            and phase_decreases[group] == 0
            and phase_skips[group] == 0
            and raw_encoding_max_error[group] <= 1e-6
        )
        items.append({
            "group": group, "name": name,
            "plant": "puffer" if group == 0 else "puffer_except_training_oracle_at_phase9",
            "target_raw_index": TARGET_RAW_INDEX,
            "target_passes": int(selected_passed.sum()),
            "target_pass_rate": float(selected_passed.mean()),
            "paired_target_gains_vs_control": int(
                (selected_passed & ~baseline_passed).sum()
            ),
            "paired_target_losses_vs_control": int(
                (~selected_passed & baseline_passed).sum()
            ),
            "pre_target_terminals": int(pretarget_terminal[selected].sum()),
            "unresolved": int((~resolved[selected]).sum()),
            "target_resolve_step_mean": (
                float(pass_steps.mean()) if pass_steps.size else None
            ),
            "target_resolve_step_max": (
                int(pass_steps.max()) if pass_steps.size else None
            ),
            "maximum_raw_index_distribution": {
                str(index): int((clipped == index).sum())
                for index in range(TARGET_RAW_INDEX + 1)
            },
            "offline_teacher_plant_actions": int(teacher_steps_by_agent[selected].sum()),
            "trajectories_with_offline_teacher_actions": int(
                (teacher_steps_by_agent[selected] > 0).sum()
            ),
            "transport_pass": transport_pass,
            "action_envelope_violations": int(action_envelope_violations[group]),
            "executed_action_max_error": float(executed_action_max_error[group]),
            "phase_changes_off_tick": int(phase_changes_off_tick[group]),
            "phase_decreases": int(phase_decreases[group]),
            "phase_skips": int(phase_skips[group]),
            "raw_progress_encoding_max_error": float(raw_encoding_max_error[group]),
        })

    control, intervention = items
    diagnostic_valid = bool(
        initial_groups_exact and not nonfinite_action
        and all(item["transport_pass"] for item in items)
        and all(item["unresolved"] == 0 for item in items)
        and control["offline_teacher_plant_actions"] == 0
        and teacher_below_target == 0
        and teacher_outside_candidate == 0
    )
    rescued = bool(
        diagnostic_valid
        and intervention["target_passes"] >= control["target_passes"] + 1
        and intervention["paired_target_gains_vs_control"] >= 1
        and intervention["paired_target_losses_vs_control"] == 0
        and intervention["pre_target_terminals"] <= control["pre_target_terminals"]
        and intervention["offline_teacher_plant_actions"] > 0
    )
    any_teacher = int(teacher_steps_by_agent.sum()) > 0
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "diagnostic_valid": diagnostic_valid,
        "training_oracle_rescued_phase9": rescued,
        "numerically_admitted": False,
        "deployment_candidate": False,
        "groups": GROUPS, "group_size": GROUP_SIZE,
        "total_agents": TOTAL_AGENTS, "episodes": EPISODES, "threads": THREADS,
        "seed": SEED, "num_proxy_gates": NUM_GATES,
        "official_gate_count_claim": "approximately 20-plus by direct simulator inspection; finish status remains authoritative",
        "target_phase": TARGET_PHASE, "target_raw_index": TARGET_RAW_INDEX,
        "max_steps": MAX_STEPS, "vector_steps": vector_steps,
        "wall_time_seconds": wall, "inference_seconds": inference_seconds,
        "initial_seed_groups_exact": initial_groups_exact,
        "single_cuda_context": True, "single_native_vector": True,
        "actor_execution": "one complete saved-form LC105 Puffer actor over all 256 rows",
        "items": items,
        "offline_teacher": {
            "plant_actions": int(teacher_steps_by_agent.sum()),
            "trajectories": int((teacher_steps_by_agent > 0).sum()),
            "actions_min": teacher_min.tolist() if any_teacher else None,
            "actions_max": teacher_max.tolist() if any_teacher else None,
            "actions_below_target_phase": teacher_below_target,
            "actions_outside_candidate_group": teacher_outside_candidate,
        },
        "loader_overrides": overrides,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0,
            "offline_training_teacher_actions": int(teacher_steps_by_agent.sum()),
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Collect one source-locked exact phase-9 intervention dataset and distill a whole-Puffer candidate; no FlightSim authority."
            if rescued else
            "Reject this phase-9 alignment-oracle intervention and retain LC105 while diagnosing the teacher target offline."
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
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
