#!/usr/bin/env python3
"""Collect compact teacher-intervention features after a Puffer-warmed Gate 1."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
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
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP, PUBLIC_STATUS_INTERVAL_STEPS
from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.collect_vq2_variable_gate_oracle_bc_dataset import (
    PHASE_PRIVILEGED_INDEX,
    update_held_phase,
)
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import (
    current_runtime_manifest,
    load_variable_config,
    mixed_variable_environment,
    sha256_path,
    write_json_atomic,
)
import scripts.eval_vq2_variable_gate_recurrent_policy as recurrent_evaluator


TAG = "vq2_vg062_warmed_teacher_intervention_features_001"
SCHEMA = "vq2_vg062_warmed_teacher_intervention_report_v1"
STATE_SCHEMA = "vq2_vg062_warmed_teacher_intervention_state_v1"
FEATURE_SCHEMA = "vq2_indexed_residual_hidden_feature_v1"
AGENTS = 512
EPISODES = 512
SEED = 429197
STEP_LIMIT = 4096
INTERVENTION_PHASE_MIN = 1
MINIMUM_SUCCESS_RATE = 0.90
MINIMUM_RECORDS = 100_000
MAX_EXECUTED_ACTION_ERROR = 1e-7
HIDDEN_SIZE = 256
PARENT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg033_variable_gate_eight_source_refit_001"
)
PARENT_CHECKPOINT = PARENT / "policy_best.pt"
PARENT_CHECKPOINT_SHA256 = (
    "56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26"
)
PARENT_REPORT = PARENT / "report.json"
PARENT_REPORT_SHA256 = (
    "033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60"
)
PARENT_ADMISSION = ROOT / "docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
PARENT_ADMISSION_SHA256 = (
    "dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44"
)
ORACLE_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf016_alignment_oracle_query_parity_64/report.json"
)
ORACLE_REPORT_SHA256 = (
    "a368916dadf058bbb0ede21838d8de7db4b2b1bba209d3e9ac78ccd3ea6e301d"
)
ORACLE_SOURCE_SHA256 = (
    "877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694"
)
REJECTION = ROOT / "docs/vq2_vg061_direct_phase4_residual_bracket_rejection_2026-07-31.json"
REJECTION_SHA256 = (
    "dbd93998ce5935a26f3b58ff0b15d25cb718f9c338778ba52d1a80e4430cebec"
)
GOAL = ROOT / "docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"
GOAL_SHA256 = (
    "03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1"
)
PREREGISTRATION = ROOT / "docs/vq2_vg062_warmed_teacher_intervention_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg062_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()


FEATURE_DTYPE = np.dtype([
    ("hidden", "<f2", (HIDDEN_SIZE,)),
    ("base_pre_tanh", "<f4", (ACTION_SIZE,)),
    ("teacher_action", "<f4", (ACTION_SIZE,)),
    ("phase_index", "u1"),
    ("agent_index", "<u2"),
    ("step", "<u2"),
    ("terminal", "u1"),
])


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, GOAL,
        PARENT_CHECKPOINT, PARENT_REPORT, PARENT_ADMISSION,
        ORACLE_REPORT, REJECTION, ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_informed.py", ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent.py", ROOT / "pufferlib/vq2_recurrent_phase.py",
        ROOT / "scripts/collect_vq2_variable_gate_oracle_bc_dataset.py",
        ROOT / "scripts/eval_vq2_variable_gate_oracle.py",
        ROOT / "scripts/eval_vq2_variable_gate_recurrent_policy.py",
        ROOT / "tests/test_collect_vq2_vg062_warmed_teacher_intervention_features.py",
        *EXTRA_SOURCE_PATHS,
    )


def verify_inputs() -> None:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        PARENT_ADMISSION: PARENT_ADMISSION_SHA256,
        ORACLE_REPORT: ORACLE_REPORT_SHA256,
        ROOT / "pufferlib/vq2_oracle.py": ORACLE_SOURCE_SHA256,
        REJECTION: REJECTION_SHA256,
        GOAL: GOAL_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG062 bound input changed: {path}")
    if not PREREGISTRATION.is_file() or not RUNNER.is_file():
        raise RuntimeError("VG062 source-lock surface is incomplete")
    parent_admission = json.loads(PARENT_ADMISSION.read_text())
    oracle = json.loads(ORACLE_REPORT.read_text())
    rejection = json.loads(REJECTION.read_text())
    if (
        not parent_admission.get("numerically_admitted")
        or parent_admission.get("artifact_sha256", {}).get("checkpoint")
        != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG062 parent is not admitted VG033")
    if not oracle.get("parity_passed"):
        raise RuntimeError("VG062 oracle query is not admitted")
    if (
        rejection.get("schema")
        != "vq2_vg061_direct_phase4_residual_bracket_rejection_v1"
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("live_authority")
    ):
        raise RuntimeError("VG061 does not authorize the intervention collection")


def source_identity() -> tuple[str, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return commit, {
        str(path.resolve().relative_to(ROOT)): sha256_path(path)
        for path in source_paths()
    }


def intervention_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = load_variable_config(
        pufferl_module, agents=AGENTS, episodes=EPISODES, seed=SEED, num_gates=12,
    )
    config["env"].update(mixed_variable_environment(seed=SEED))
    config["env"].update({
        "evaluation_episode_limit": 1,
        "max_steps": STEP_LIMIT,
        "time_limit_seconds": STEP_LIMIT / 64.0,
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
        "observable_gate_index_denominator": float(ENGINE_GATE_CAP),
    })
    return config, overrides


def select_intervention_plant_actions(
    student: np.ndarray,
    teacher: np.ndarray,
    active: np.ndarray,
    phase_index: np.ndarray,
    *,
    phase_min: int = INTERVENTION_PHASE_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a training-only teacher plant after the held public phase gate."""

    student = np.asarray(student, dtype=np.float32)
    teacher = np.asarray(teacher, dtype=np.float32)
    active = np.asarray(active, dtype=bool)
    phase_index = np.asarray(phase_index, dtype=np.int32)
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("student and teacher actions must share a 2D shape")
    if student.shape[0] != active.size or active.shape != phase_index.shape:
        raise ValueError("plant selector batch shapes changed")
    if phase_min < 0 or phase_min > ENGINE_GATE_CAP:
        raise ValueError("intervention phase is outside the public range")
    teacher_mask = active & (phase_index >= phase_min)
    plant = student.copy()
    plant[teacher_mask] = teacher[teacher_mask]
    return plant, teacher_mask


def collection_predicates(report: dict[str, Any]) -> dict[str, bool]:
    metrics = report.get("metrics", {})
    phases = report.get("feature_phase_records", [])
    exact_counts = all(
        abs(metrics.get(f"env/gate_count{count}_episode", -1.0) - 0.125)
        <= 1e-12
        for count in range(5, 13)
    ) and all(
        metrics.get(f"env/gate_count{count}_episode", 0.0) == 0.0
        for count in (*range(1, 5), *range(13, 17))
    )
    required_zero = (
        "out_of_order", "action_envelope_violation",
        "wire_rate_envelope_violation", "thrust_envelope_violation",
    )
    return {
        "exact_episode_count": metrics.get("env/n") == float(EPISODES),
        "exact_uniform_gate_counts": exact_counts,
        "minimum_success_rate": (
            metrics.get("env/success_rate", 0.0) >= MINIMUM_SUCCESS_RATE
        ),
        "zero_crash": metrics.get("env/crash", math.inf) == 0.0,
        "zero_hard_native_fault": not any(
            metrics.get(f"env/{name}", math.inf) != 0.0 for name in required_zero
        ),
        "gate_1_warmup_reliable": (
            metrics.get("env/ordered_gate0_sampled", 0.0) >= 0.95
        ),
        "minimum_feature_records": report.get("feature_records", 0) >= MINIMUM_RECORDS,
        "feature_records_match_teacher_actions": (
            report.get("feature_records")
            == report.get("teacher_plant_actions_executed")
        ),
        "all_later_public_heads_observed": (
            len(phases) == ENGINE_GATE_CAP + 1
            and all(phases[index] > 0 for index in range(1, 12))
        ),
        "teacher_begins_only_at_public_phase_1": (
            report.get("minimum_teacher_phase_index") == INTERVENTION_PHASE_MIN
        ),
        "plant_action_history_exact": (
            report.get("executed_action_max_error", math.inf)
            <= MAX_EXECUTED_ACTION_ERROR
        ),
        "phase_changes_only_on_public_ticks": report.get("phase_changes_off_tick") == 0,
        "phase_never_decreases": report.get("phase_decreases") == 0,
        "phase_never_skips": report.get("phase_skips") == 0,
        "phase_encoding_exact": report.get("raw_phase_encoding_max_error", math.inf) <= 1e-6,
        "finite_actions_and_features": report.get("nonfinite_values") == 0,
        "teacher_actions_in_envelope": report.get("teacher_action_envelope_violations") == 0,
        "feature_file_hash_present": len(report.get("feature_sha256", "")) == 64,
        "training_only_authority": (
            report.get("safety", {}).get("flight_sim_packets_sent") == 0
            and report.get("safety", {}).get("submission_authorized") is False
            and report.get("safety", {}).get("runtime_teacher_authorized") is False
        ),
    }


def collection_passes(report: dict[str, Any]) -> bool:
    return all(collection_predicates(report).values())


def _load_actor(device: torch.device):
    recurrent_evaluator.CHECKPOINT = PARENT_CHECKPOINT
    recurrent_evaluator.CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    recurrent_evaluator.TRAIN_REPORT = PARENT_REPORT
    recurrent_evaluator.TRAIN_REPORT_SHA256 = PARENT_REPORT_SHA256
    return recurrent_evaluator.load_actor(device)


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False,
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    verify_inputs()
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("VG062 requires the drone_race_vision native backend")
    if getattr(_C, "precision_bytes", None) != 4:
        raise RuntimeError("VG062 requires the float32 native backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG062 preregisters CUDA actor inference")
    report_path = output / "report.json"
    state_path = output.with_name(f"{output.name}_state.json")
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if not resume:
            raise RuntimeError("VG062 terminal output already exists; use --resume")
        return report
    if resume and state_path.is_file():
        old = json.loads(state_path.read_text())
        if old.get("status") not in ("running", "interrupted"):
            raise RuntimeError("VG062 state is terminal without a report")

    device = torch.device(device_name)
    actor, payload = _load_actor(device)
    commit, hashes = source_identity()
    extension = Path(_C.__file__).resolve()
    hashes["compiled_extension"] = sha256_path(extension)
    state = {
        "schema": STATE_SCHEMA, "tag": TAG, "status": "running",
        "source_commit": commit, "source_sha256": hashes,
        "runtime": current_runtime_manifest(),
        "agents": AGENTS, "episodes": EPISODES, "seed": SEED,
        "step_limit": STEP_LIMIT, "intervention_phase_min": INTERVENTION_PHASE_MIN,
        "safety": {
            "training_only_teacher_plant_actions": True,
            "runtime_teacher_authorized": False,
            "student_updates": 0, "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0, "submission_authorized": False,
        },
    }
    write_json_atomic(state_path, state)

    staging = output.with_name(f".{output.name}_staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    partial = staging / "features.bin"
    config, overrides = intervention_config(pufferl)
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("VG062 native vector ABI changed")
    observations = _cpu_tensor(vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32)
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions_cpu = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    recurrent = actor.initial_state(AGENTS, device=device)
    done = np.zeros(AGENTS, dtype=bool)
    held_phase = np.zeros(AGENTS, dtype=np.float32)
    previous_phase = held_phase.copy()
    phase_records = np.zeros(ENGINE_GATE_CAP + 1, dtype=np.int64)
    lengths = np.zeros(AGENTS, dtype=np.int32)
    teacher_steps_by_agent = np.zeros(AGENTS, dtype=np.int32)
    phase_changes_off_tick = phase_decreases = phase_skips = 0
    raw_phase_encoding_max_error = executed_action_max_error = 0.0
    feature_records = nonfinite_values = teacher_action_envelope_violations = 0
    minimum_teacher_phase_index = ENGINE_GATE_CAP + 1
    maximum_teacher_phase_index = -1
    teacher_min = np.full(ACTION_SIZE, np.inf, dtype=np.float64)
    teacher_max = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
    started = time.perf_counter()
    native_log: dict[str, Any] = {}
    vector_steps = 0
    try:
        vector.reset()
        with partial.open("wb") as feature_file, torch.no_grad():
            for step in range(STEP_LIMIT):
                active = ~done
                if not active.any():
                    break
                current = observations.numpy()
                raw_phase = current[:, PHASE_PRIVILEGED_INDEX]
                held_phase, sampled, encoding_error = update_held_phase(
                    raw_phase, held_phase, step=step
                )
                raw_phase_encoding_max_error = max(raw_phase_encoding_max_error, encoding_error)
                changed = np.abs(held_phase - previous_phase) > 1e-7
                if changed.any() and not sampled:
                    phase_changes_off_tick += int(changed.sum())
                delta = held_phase[active] - previous_phase[active]
                phase_decreases += int((delta < -1e-7).sum())
                phase_skips += int((delta > 1.0 / ENGINE_GATE_CAP + 1e-7).sum())
                previous_phase = held_phase.copy()
                phase_index = np.rint(held_phase * ENGINE_GATE_CAP).astype(np.int32)

                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                phase_tensor = torch.from_numpy(held_phase[:, None]).to(device)
                actor_input = torch.cat((legal, phase_tensor), dim=1)
                active_device = torch.from_numpy(active).to(device)
                result, candidate = actor.forward_step(actor_input, recurrent)
                recurrent = preserve_frozen_state(recurrent, candidate, active_device)
                student = torch.where(
                    active_device[:, None], result.mean, torch.zeros_like(result.mean)
                )
                student_np = student.cpu().numpy().astype(np.float32, copy=False)
                teacher = alignment_oracle_action(current)
                plant, teacher_mask = select_intervention_plant_actions(
                    student_np, teacher, active, phase_index
                )
                selected = np.flatnonzero(teacher_mask)
                if selected.size:
                    selected_teacher = teacher[selected]
                    teacher_min = np.minimum(teacher_min, selected_teacher.min(axis=0))
                    teacher_max = np.maximum(teacher_max, selected_teacher.max(axis=0))
                    teacher_action_envelope_violations += int(
                        np.any(np.abs(selected_teacher) > 1.0 + 1e-6, axis=1).sum()
                    )
                    minimum_teacher_phase_index = min(
                        minimum_teacher_phase_index, int(phase_index[selected].min())
                    )
                    maximum_teacher_phase_index = max(
                        maximum_teacher_phase_index, int(phase_index[selected].max())
                    )

                actions_cpu.copy_(torch.from_numpy(plant))
                vector.cpu_step(actions_cpu.data_ptr())
                vector_steps += 1
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(np.max(np.abs(executed[active] - plant[active]), initial=0.0)),
                )
                terminal = (terminals.numpy() > 0.5) & active
                lengths[active] += 1
                teacher_steps_by_agent += teacher_mask.astype(np.int32)
                if selected.size:
                    records = np.empty(selected.size, dtype=FEATURE_DTYPE)
                    selected_device = torch.from_numpy(selected).to(device)
                    records["hidden"] = candidate[0].index_select(
                        0, selected_device
                    ).cpu().numpy().astype(np.float16)
                    records["base_pre_tanh"] = result.pre_tanh_mean.index_select(
                        0, selected_device
                    ).cpu().numpy()
                    records["teacher_action"] = teacher[selected]
                    records["phase_index"] = phase_index[selected].astype(np.uint8)
                    records["agent_index"] = selected.astype(np.uint16)
                    records["step"] = np.uint16(step)
                    records["terminal"] = terminal[selected].astype(np.uint8)
                    nonfinite_values += int(
                        (~np.isfinite(records["hidden"])).sum()
                        + (~np.isfinite(records["base_pre_tanh"])).sum()
                        + (~np.isfinite(records["teacher_action"])).sum()
                    )
                    records.tofile(feature_file)
                    feature_records += int(selected.size)
                    phase_records += np.bincount(
                        phase_index[selected], minlength=ENGINE_GATE_CAP + 1
                    )
                done |= terminal
            feature_file.flush()
            os.fsync(feature_file.fileno())
        native_log = dict(vector.log())
    except BaseException:
        state["status"] = "interrupted"
        write_json_atomic(state_path, state)
        raise
    finally:
        vector.close()

    if not done.all():
        raise RuntimeError("VG062 did not observe one terminal episode per agent")
    expected_bytes = feature_records * FEATURE_DTYPE.itemsize
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError("VG062 feature file size does not match the record count")
    feature_sha = sha256_path(partial)
    metrics = flatten_log(pufferl, native_log)
    wall = time.perf_counter() - started
    report: dict[str, Any] = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "training_dataset_admitted": False,
        "source_commit": commit, "source_sha256": hashes,
        "runtime": {**current_runtime_manifest(), "compiled_extension": str(extension)},
        "config_overrides": overrides, "agents": AGENTS, "episodes": EPISODES,
        "seed": SEED, "step_limit": STEP_LIMIT, "vector_steps": vector_steps,
        "wall_time_seconds": wall, "metrics": metrics,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_best_epoch": payload.get("best_epoch"),
        "feature_schema": FEATURE_SCHEMA, "feature_dtype": FEATURE_DTYPE.descr,
        "feature_itemsize": FEATURE_DTYPE.itemsize,
        "feature_records": feature_records, "feature_sha256": feature_sha,
        "feature_phase_records": phase_records.tolist(),
        "episode_length_min": int(lengths.min()),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_max": int(lengths.max()),
        "teacher_steps_by_agent_min": int(teacher_steps_by_agent.min()),
        "teacher_steps_by_agent_mean": float(teacher_steps_by_agent.mean()),
        "teacher_steps_by_agent_max": int(teacher_steps_by_agent.max()),
        "teacher_plant_actions_executed": int(teacher_steps_by_agent.sum()),
        "student_plant_actions_executed": int(lengths.sum() - teacher_steps_by_agent.sum()),
        "intervention_phase_min": INTERVENTION_PHASE_MIN,
        "minimum_teacher_phase_index": minimum_teacher_phase_index,
        "maximum_teacher_phase_index": maximum_teacher_phase_index,
        "teacher_action_min": teacher_min.tolist(),
        "teacher_action_max": teacher_max.tolist(),
        "teacher_action_envelope_violations": teacher_action_envelope_violations,
        "executed_action_max_error": executed_action_max_error,
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases, "phase_skips": phase_skips,
        "raw_phase_encoding_max_error": raw_phase_encoding_max_error,
        "nonfinite_values": nonfinite_values,
        "plant_action_contract": (
            "VG033 deterministic mean through held public phase 0; "
            "training-only SF016 oracle query at held public phases 1+"
        ),
        "safety": {
            "training_only_teacher_plant_actions": int(teacher_steps_by_agent.sum()),
            "runtime_teacher_authorized": False, "student_updates": 0,
            "flight_sim_packets_sent": 0, "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    report["admission_predicates"] = collection_predicates(report)
    report["failed_admission_predicates"] = [
        name for name, passed in report["admission_predicates"].items() if not passed
    ]
    report["training_dataset_admitted"] = collection_passes(report)
    report["next_authority"] = (
        "One source-locked indexed recurrent Puffer distillation from these "
        "training-only intervention features; no live authority."
        if report["training_dataset_admitted"]
        else "Reject VG062 and diagnose the failed intervention predicates."
    )
    report["source_sha256"] = hashes
    output.mkdir(parents=True)
    shutil.move(str(partial), str(output / "features.bin"))
    write_json_atomic(output / "report.json", report)
    shutil.rmtree(staging)
    state.update({
        "status": "completed" if report["training_dataset_admitted"] else "rejected",
        "training_dataset_admitted": report["training_dataset_admitted"],
        "feature_records": feature_records, "feature_sha256": feature_sha,
        "report_sha256": sha256_path(output / "report.json"),
        "failed_admission_predicates": report["failed_admission_predicates"],
    })
    write_json_atomic(state_path, state)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_dataset_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
