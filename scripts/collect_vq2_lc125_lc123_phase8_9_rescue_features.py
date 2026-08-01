#!/usr/bin/env python3
"""Diagnose and capture phase-8/9 oracle rescue on LC123's new states."""

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
from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import (
    FEATURE_DTYPE,
    outcome_agents,
)
from scripts.eval_vq2_lc001_long_course_oracle import load_config
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc007_converted_vg071_long_course as core
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone


TAG = "vq2_lc125_lc123_phase8_9_rescue_features_001"
SCHEMA = "vq2_lc125_lc123_phase8_9_rescue_features_report_v1"
FEATURE_SCHEMA = "vq2_lc125_lc123_phase8_9_rescue_feature_v1"
GROUP_SIZE = 128
GROUPS = 2
TOTAL_AGENTS = GROUP_SIZE * GROUPS
THREADS = 32
SEED = 432_050
NUM_GATES = 24
MAX_STEPS = 12_000
PHASE_MIN = 8
PHASE_MAX_EXCLUSIVE = 10
TARGET_RAW_INDEX = 10
# Backward-compatible hook for later paired anchor/rescue collection. LC125's
# source-locked default continues to record only intervention rows.
CAPTURE_CONTROL_FEATURES = False
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc123_phase6_success_rescue_anchor_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "0d0d9f6ba796dbf1803521214a0c030a79673f24cff538ae05338183ded4f558"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "92f29cdd0f3c76b2c031be3acdcf5bc23fecc9c7e499cafe3cf6edc5758c0bf9"
LC124_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc124_phase6_success_rescue_milestone_001/report.json"
)
LC124_REPORT_SHA256 = "b29a157b1fd4a1ea0d59021eb6bba570e01291899b217b4e753ad3ffcabf0154"
PREREGISTRATION = ROOT / "docs/vq2_lc125_lc123_phase8_9_rescue_features_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc125_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc125_lc123_phase8_9_rescue_features.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def group_slice(group: int) -> slice:
    if group not in (0, 1):
        raise ValueError("LC125 group must be control or intervention")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def intervention_mask(active: np.ndarray, phase: np.ndarray) -> np.ndarray:
    if active.shape != (TOTAL_AGENTS,) or phase.shape != active.shape:
        raise ValueError("LC125 active/phase arrays do not match")
    mask = np.zeros(TOTAL_AGENTS, dtype=bool)
    selected = group_slice(1)
    mask[selected] = (
        active[selected]
        & (phase[selected] >= PHASE_MIN)
        & (phase[selected] < PHASE_MAX_EXCLUSIVE)
    )
    return mask


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC124_REPORT: LC124_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC125 bound input changed: {path}")
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC124_REPORT.read_text())
    control, candidate = rejected.get("items", [{}, {}])
    if (
        payload.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or parent.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_report_v1"
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or rejected.get("schema")
        != "vq2_lc124_phase6_success_rescue_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or control.get("maximum_raw_index_distribution", {}).get("8") != 1
        or control.get("maximum_raw_index_distribution", {}).get("9") != 1
        or candidate.get("maximum_raw_index_distribution", {}).get("8") != 3
        or candidate.get("maximum_raw_index_distribution", {}).get("9") != 0
        or candidate.get("target_passes") != 0
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC123/LC124 do not authorize LC125")
    return payload


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC124_REPORT,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_oracle.py",
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


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    payload = verify_inputs()
    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC125 requires the float32 drone_race_vision backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC125 preregisters CUDA inference")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"LC125 requires OMP_NUM_THREADS={THREADS}")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC125 requires OMP_DYNAMIC=FALSE")
    report_path = output / "report.json"
    feature_path = output / "features.bin"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC125 does not resume a partial collection")
    output.mkdir(parents=True, exist_ok=True)

    identity = source_identity()
    device = torch.device(device_name)
    actor = milestone.load_actor(payload, device)
    recurrent = actor.initial_state(TOTAL_AGENTS, device=device)
    config, overrides = load_config(
        pufferl, num_gates=NUM_GATES, agents=TOTAL_AGENTS,
        episodes=TOTAL_AGENTS, seed=SEED, threads=THREADS,
    )
    config["vec"]["env_seed_group_size"] = GROUP_SIZE
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
        raise RuntimeError("LC125 native vector ABI changed")
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
    teacher_steps = np.zeros(TOTAL_AGENTS, dtype=np.int32)
    phase_changes_off_tick = np.zeros(GROUPS, dtype=np.int64)
    phase_decreases = np.zeros(GROUPS, dtype=np.int64)
    phase_skips = np.zeros(GROUPS, dtype=np.int64)
    raw_encoding_max_error = np.zeros(GROUPS, dtype=np.float64)
    action_envelope_violations = np.zeros(GROUPS, dtype=np.int64)
    executed_action_max_error = np.zeros(GROUPS, dtype=np.float64)
    feature_chunks: list[np.ndarray] = []
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
            raise RuntimeError("LC125 paired groups do not begin identically")
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
                teacher_mask = intervention_mask(active_np, phase_index)
                capture_mask = teacher_mask.copy()
                if CAPTURE_CONTROL_FEATURES:
                    control = group_slice(0)
                    capture_mask[control] = (
                        active_np[control]
                        & (phase_index[control] >= PHASE_MIN)
                        & (phase_index[control] < PHASE_MAX_EXCLUSIVE)
                    )
                selected_agents = np.flatnonzero(capture_mask)
                pending = np.empty(selected_agents.size, dtype=FEATURE_DTYPE)
                if selected_agents.size:
                    index = torch.from_numpy(selected_agents).to(device)
                    pending["hidden"] = next_recurrent[0].index_select(
                        0, index
                    ).cpu().numpy().astype(np.float16)
                    pending["base_pre_tanh"] = result.pre_tanh_mean.index_select(
                        0, index
                    ).cpu().numpy()
                    targets = student_np[selected_agents].copy()
                    selected_teacher = teacher_mask[selected_agents]
                    targets[selected_teacher] = teacher[selected_agents[selected_teacher]]
                    pending["teacher_action"] = targets
                    pending["phase_index"] = phase_index[selected_agents].astype(np.uint8)
                    pending["agent_index"] = selected_agents.astype(np.uint16)
                    pending["step"] = np.uint16(step)
                    pending["terminal"] = 0
                plant = student_np.copy()
                plant[teacher_mask] = teacher[teacher_mask]
                teacher_steps += teacher_mask.astype(np.int32)
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
                terminal_np_all = terminals.numpy() > 0.5
                if pending.size:
                    pending["terminal"] = terminal_np_all[selected_agents].astype(np.uint8)
                    feature_chunks.append(pending)
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
                resolved |= newly_passed
                terminal_np = terminal_np_all & (~resolved)
                pretarget_terminal |= terminal_np
                resolved |= terminal_np
    finally:
        vector.close()
    wall = time.perf_counter() - started

    records = (
        np.concatenate(feature_chunks)
        if feature_chunks else np.empty(0, dtype=FEATURE_DTYPE)
    )
    feature_tmp = output / ".features.bin.tmp"
    records.tofile(feature_tmp)
    with feature_tmp.open("rb") as stream:
        os.fsync(stream.fileno())
    feature_tmp.replace(feature_path)
    feature_sha = sha256_path(feature_path)
    success_agents, failure_agents = outcome_agents(records)
    phase_counts = np.bincount(
        records["phase_index"].astype(np.int64), minlength=33
    )

    items: list[dict[str, Any]] = []
    control_passed = passed[group_slice(0)]
    for group, name in enumerate(("lc123_puffer_control", "lc123_phase8_9_oracle")):
        selected = group_slice(group)
        selected_passed = passed[selected]
        clipped = np.minimum(maximum_raw_index[selected], TARGET_RAW_INDEX)
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
            "target_passes": int(selected_passed.sum()),
            "paired_target_gains_vs_control": int(
                (selected_passed & ~control_passed).sum()
            ),
            "paired_target_losses_vs_control": int(
                (~selected_passed & control_passed).sum()
            ),
            "pre_target_terminals": int(pretarget_terminal[selected].sum()),
            "unresolved": int((~resolved[selected]).sum()),
            "maximum_raw_index_distribution": {
                str(index): int((clipped == index).sum())
                for index in range(TARGET_RAW_INDEX + 1)
            },
            "offline_teacher_plant_actions": int(teacher_steps[selected].sum()),
            "trajectories_with_offline_teacher_actions": int(
                (teacher_steps[selected] > 0).sum()
            ),
            "transport_pass": transport_pass,
            "executed_action_max_error": float(executed_action_max_error[group]),
            "action_envelope_violations": int(action_envelope_violations[group]),
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
    )
    rescued = bool(
        diagnostic_valid
        and intervention["target_passes"] >= control["target_passes"] + 1
        and intervention["paired_target_gains_vs_control"] >= 1
        and intervention["paired_target_losses_vs_control"] == 0
        and intervention["pre_target_terminals"] <= control["pre_target_terminals"]
    )
    finite_features = bool(
        np.isfinite(records["hidden"]).all()
        and np.isfinite(records["base_pre_tanh"]).all()
        and np.isfinite(records["teacher_action"]).all()
    )
    feature_envelope_violations = int(
        (np.abs(records["teacher_action"]) > 1.0 + 1e-6).any(axis=1).sum()
    )
    predicates = {
        "diagnostic_rescue": rescued,
        "exact_feature_and_teacher_count": (
            len(records) == intervention["offline_teacher_plant_actions"]
        ),
        "only_intervention_group": bool(
            len(records) and records["agent_index"].min() >= GROUP_SIZE
        ),
        "only_phase8_9": bool(
            phase_counts[PHASE_MIN:PHASE_MAX_EXCLUSIVE].sum() == len(records)
            and phase_counts[:PHASE_MIN].sum() == 0
            and phase_counts[PHASE_MAX_EXCLUSIVE:].sum() == 0
        ),
        "finite_in_envelope_features": (
            finite_features and feature_envelope_violations == 0
        ),
        "splittable_outcomes": (
            len(success_agents) >= 2 and len(failure_agents) >= 2
        ),
        "training_only_authority": True,
    }
    admitted = all(predicates.values())
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "diagnostic_valid": diagnostic_valid,
        "training_oracle_rescued_phase8_9": rescued,
        "training_dataset_admitted": admitted,
        "admission_predicates": predicates,
        "failed_admission_predicates": [
            name for name, passed_predicate in predicates.items()
            if not passed_predicate
        ],
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "groups": GROUPS, "group_size": GROUP_SIZE,
        "total_agents": TOTAL_AGENTS, "threads": THREADS,
        "seed": SEED, "num_proxy_gates": NUM_GATES,
        "target_raw_index": TARGET_RAW_INDEX,
        "teacher_phase_min": PHASE_MIN,
        "teacher_phase_max_exclusive": PHASE_MAX_EXCLUSIVE,
        "max_steps": MAX_STEPS, "vector_steps": vector_steps,
        "wall_time_seconds": wall, "inference_seconds": inference_seconds,
        "initial_seed_groups_exact": initial_groups_exact,
        "items": items,
        "feature_schema": FEATURE_SCHEMA,
        "feature_dtype": FEATURE_DTYPE.descr,
        "feature_itemsize": FEATURE_DTYPE.itemsize,
        "feature_records": len(records), "feature_sha256": feature_sha,
        "feature_phase_records": phase_counts.tolist(),
        "query_agents": int(np.unique(records["agent_index"]).size),
        "query_outcome_success_agents": success_agents.tolist(),
        "query_outcome_failure_agents": failure_agents.tolist(),
        "teacher_action_envelope_violations": feature_envelope_violations,
        "loader_overrides": overrides,
        "single_cuda_context": True, "single_native_vector": True,
        "actor_execution": "one complete saved-form LC123 Puffer actor over all 256 rows",
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0,
            "offline_training_teacher_actions": int(teacher_steps.sum()),
            "student_updates": 0, "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Fit one source-locked whole-Puffer phase-8/9 endpoint on this LC123-owned distribution; no FlightSim authority."
            if admitted else
            "Do not fit LC125; retain LC105 and pivot to on-policy phase-6 optimization."
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
    report = collect(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
