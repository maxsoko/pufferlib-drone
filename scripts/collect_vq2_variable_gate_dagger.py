#!/usr/bin/env python3
"""Collect VG005-visited variable-course states with oracle query labels."""

from __future__ import annotations

import argparse
import json
import math
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
from pufferlib.vq2_informed import (
    ACTION_HISTORY,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    MASK_SIZE,
)
from pufferlib.vq2_oracle import alignment_oracle_action
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP, PUBLIC_STATUS_INTERVAL_STEPS
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.collect_vq2_variable_gate_oracle_bc_dataset import (
    PHASE_PRIVILEGED_INDEX,
    PHASE_TAIL_SIZE,
    VariableGateDatasetWriter,
    phase_storage_parts,
    prepare_collection_state,
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
from scripts.eval_vq2_variable_gate_recurrent_policy import (
    CHECKPOINT_SHA256,
    load_actor,
)


TAG = "vq2_vg007_variable_gate_dagger_round1_256"
SCHEMA = "vq2_variable_gate_dagger_time_major_v1"
AGENTS = 256
EPISODES = 256
SEED = 429047
COLLECTION_STEP_LIMIT = 2048
MINIMUM_RECORDS = 100_000
MINIMUM_GATE1_RATE = 0.90
MINIMUM_GATE2_RATE = 0.01
MAXIMUM_CRASH_RATE = 0.05
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
PREREGISTRATION = (
    ROOT / "docs/vq2_vg007_variable_gate_dagger_round1_preregistration_2026-07-29.md"
)
VG006_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg006_variable_gate_recurrent_teacher_free_256/report.json"
)
VG006_REPORT_SHA256 = (
    "1f848400806e2ba8c353f557db1e30099ea2c570fbb6c16f0d7664c1c1f057f0"
)
SF016_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf016_alignment_oracle_query_parity_64/report.json"
)
SF016_REPORT_SHA256 = (
    "a368916dadf058bbb0ede21838d8de7db4b2b1bba209d3e9ac78ccd3ea6e301d"
)
ORACLE_QUERY_SHA256 = (
    "877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694"
)


def dagger_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = load_variable_config(
        pufferl_module,
        agents=AGENTS,
        episodes=EPISODES,
        seed=SEED,
        num_gates=12,
    )
    config["env"].update(mixed_variable_environment(seed=SEED))
    config["env"].update({
        "evaluation_episode_limit": 1,
        "max_steps": COLLECTION_STEP_LIMIT,
        "time_limit_seconds": COLLECTION_STEP_LIMIT / 64.0,
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
        "observable_gate_index_denominator": float(ENGINE_GATE_CAP),
    })
    return config, overrides


def dagger_collection_passes(
    metrics: dict[str, float],
    *,
    lengths: np.ndarray,
    terminal_count: np.ndarray,
    terminal_is_last: np.ndarray,
    labels: int,
    executed_action_max_error: float,
    phase_changes_off_tick: int,
    phase_decreases: int,
    phase_skips: int,
    raw_phase_encoding_max_error: float,
    phase_records: np.ndarray,
) -> bool:
    if metrics.get("env/n") != float(EPISODES):
        return False
    required_zero = (
        "out_of_order",
        "crossing_margin_violation",
        "action_envelope_violation",
        "wire_rate_envelope_violation",
        "thrust_envelope_violation",
    )
    if any(metrics.get(f"env/{name}", math.inf) != 0.0 for name in required_zero):
        return False
    if metrics.get("env/crash", math.inf) > MAXIMUM_CRASH_RATE:
        return False
    for count in range(1, 17):
        expected = 0.125 if 5 <= count <= 12 else 0.0
        if abs(metrics.get(f"env/gate_count{count}_episode", -1.0) - expected) > 1e-12:
            return False
    if metrics.get("env/ordered_gate0_sampled", 0.0) < MINIMUM_GATE1_RATE:
        return False
    if metrics.get("env/ordered_gate1_sampled", 0.0) < MINIMUM_GATE2_RATE:
        return False
    return bool(
        lengths.shape == (AGENTS,)
        and np.all(lengths > 0)
        and np.all(lengths <= COLLECTION_STEP_LIMIT)
        and np.all(terminal_count == 1)
        and np.all(terminal_is_last)
        and labels == int(lengths.sum())
        and labels >= MINIMUM_RECORDS
        and executed_action_max_error <= 1e-7
        and phase_changes_off_tick == 0
        and phase_decreases == 0
        and phase_skips == 0
        and raw_phase_encoding_max_error <= 1e-6
        and phase_records.shape == (ENGINE_GATE_CAP + 1,)
        and phase_records[1] > 0
    )


def verify_inputs() -> None:
    if sha256_path(VG006_REPORT) != VG006_REPORT_SHA256:
        raise RuntimeError("VG006 rejection report hash mismatch")
    vg006 = json.loads(VG006_REPORT.read_text())
    if vg006.get("admitted") or vg006.get("successes") != 0:
        raise RuntimeError("VG006 is not the source-locked Gate-2 failure")
    if sha256_path(SF016_REPORT) != SF016_REPORT_SHA256:
        raise RuntimeError("SF016 oracle-query parity report hash mismatch")
    if not json.loads(SF016_REPORT.read_text()).get("parity_passed"):
        raise RuntimeError("SF016 did not admit the oracle query")
    if sha256_path(ROOT / "pufferlib/vq2_oracle.py") != ORACLE_QUERY_SHA256:
        raise RuntimeError("admitted oracle-query source changed")


def collect(
    output: Path = DEFAULT_OUTPUT,
    *,
    device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    verify_inputs()
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG007 preregistration is missing")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG007 preregisters CUDA student inference")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("VG007 requires drone_race_vision backend")
    if getattr(_C, "precision_bytes", None) != 4:
        raise RuntimeError("VG007 requires a float32 native binding")
    device = torch.device(device_name)
    actor, _checkpoint = load_actor(device)
    extension = Path(_C.__file__).resolve()
    source_paths = (
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "scripts/run_vq2_vg007_vast.sh",
        ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        ROOT / "scripts/collect_vq2_variable_gate_oracle_bc_dataset.py",
        ROOT / "scripts/eval_vq2_variable_gate_recurrent_policy.py",
        VG006_REPORT,
        SF016_REPORT,
    )
    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    source_sha256["compiled_extension"] = sha256_path(extension)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state_identity = {
        "schema": "vq2_variable_gate_dagger_collection_state_v1",
        "tag": TAG,
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "collection_step_limit": COLLECTION_STEP_LIMIT,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": current_runtime_manifest(),
        "compiled_extension_path": str(extension),
        "safety": {
            "teacher_actions_executed": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    state_path = output.with_name(f"{output.name}_state.json")
    staging = output.with_name(f".{output.name}_staging")
    state, existing = prepare_collection_state(
        output=output,
        state_path=state_path,
        staging=staging,
        state_identity=state_identity,
        source_sha256=source_sha256,
        resume=resume,
    )
    if existing is not None:
        return existing

    config, overrides = dagger_config(pufferl)
    environment = config["env"]
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("VG007 native vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions_cpu = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    recurrent = actor.initial_state(AGENTS, device=device)
    done = np.zeros(AGENTS, dtype=bool)
    lengths = np.zeros(AGENTS, dtype=np.int32)
    tracked_terminal_count = np.zeros(AGENTS, dtype=np.int32)
    held_phase = np.zeros(AGENTS, dtype=np.float32)
    previous_held_phase = np.zeros(AGENTS, dtype=np.float32)
    phase_records = np.zeros(ENGINE_GATE_CAP + 1, dtype=np.int64)
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_skips = 0
    raw_phase_encoding_max_error = 0.0
    executed_action_max_error = 0.0
    label_count = 0
    query_square_error = np.zeros(ACTION_SIZE, dtype=np.float64)
    query_absolute_error = np.zeros(ACTION_SIZE, dtype=np.float64)
    query_min = np.full(ACTION_SIZE, np.inf, dtype=np.float64)
    query_max = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
    student_min = np.full(ACTION_SIZE, np.inf, dtype=np.float64)
    student_max = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
    native_log: dict[str, Any] = {}
    manifest: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    try:
        vector.reset()
        writer = VariableGateDatasetWriter(
            staging, max_steps=COLLECTION_STEP_LIMIT, agents=AGENTS
        )
        with torch.no_grad():
            for step in range(COLLECTION_STEP_LIMIT):
                active = ~done
                if not active.any():
                    break
                current = observations.numpy()
                raw_phase = current[:, PHASE_PRIVILEGED_INDEX]
                held_phase, sampled, encoding_error = update_held_phase(
                    raw_phase, held_phase, step=step
                )
                raw_phase_encoding_max_error = max(
                    raw_phase_encoding_max_error, encoding_error
                )
                changed = np.abs(held_phase - previous_held_phase) > 1e-7
                if changed.any() and not sampled:
                    phase_changes_off_tick += int(changed.sum())
                delta = held_phase[active] - previous_held_phase[active]
                phase_decreases += int((delta < -1e-7).sum())
                phase_skips += int((delta > 1.0 / ENGINE_GATE_CAP + 1e-7).sum())
                previous_held_phase = held_phase.copy()
                phase_index = np.rint(held_phase * ENGINE_GATE_CAP).astype(np.int32)
                phase_records += np.bincount(
                    phase_index[active], minlength=ENGINE_GATE_CAP + 1
                )

                mask, tail = phase_storage_parts(current, held_phase)
                query = alignment_oracle_action(current)
                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                phase_tensor = torch.from_numpy(held_phase[:, None]).to(device)
                actor_input = torch.cat((legal, phase_tensor), dim=1)
                active_device = torch.from_numpy(active).to(device)
                output, candidate = actor.forward_step(actor_input, recurrent)
                recurrent = preserve_frozen_state(
                    recurrent, candidate, active_device
                )
                student = torch.where(
                    active_device[:, None],
                    output.mean,
                    torch.zeros_like(output.mean),
                )
                if not bool(torch.isfinite(student[active_device]).all()):
                    raise RuntimeError("VG007 student emitted a non-finite action")
                student_np = student.detach().cpu().numpy().astype(np.float32, copy=False)
                selected_query = query[active].astype(np.float64)
                selected_student = student_np[active].astype(np.float64)
                error = selected_student - selected_query
                query_square_error += np.square(error).sum(axis=0)
                query_absolute_error += np.abs(error).sum(axis=0)
                query_min = np.minimum(query_min, selected_query.min(axis=0))
                query_max = np.maximum(query_max, selected_query.max(axis=0))
                student_min = np.minimum(student_min, selected_student.min(axis=0))
                student_max = np.maximum(student_max, selected_student.max(axis=0))

                actions_cpu.copy_(student.detach().cpu())
                vector.cpu_step(actions_cpu.data_ptr())
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(
                        np.max(
                            np.abs(executed[active] - student_np[active]),
                            initial=0.0,
                        )
                    ),
                )
                terminal = (terminals.numpy() > 0.5) & active
                lengths[active] += 1
                tracked_terminal_count += terminal.astype(np.int32)
                label_count += int(active.sum())
                mask[~active] = 0
                tail[~active] = 0.0
                query[~active] = 0.0
                writer.append(
                    mask=mask,
                    tail=tail,
                    action=query,
                    terminal=terminal.astype(np.uint8),
                    valid=active.astype(np.uint8),
                )
                done |= terminal
        native_log = dict(vector.log())
        terminal_count, terminal_is_last = writer.validate_episode_layout(lengths)
        if not np.array_equal(terminal_count, tracked_terminal_count):
            raise RuntimeError("VG007 staged/tracked terminal counts differ")
        metrics = flatten_log(pufferl, native_log)
        admitted = dagger_collection_passes(
            metrics,
            lengths=lengths,
            terminal_count=terminal_count,
            terminal_is_last=terminal_is_last,
            labels=label_count,
            executed_action_max_error=executed_action_max_error,
            phase_changes_off_tick=phase_changes_off_tick,
            phase_decreases=phase_decreases,
            phase_skips=phase_skips,
            raw_phase_encoding_max_error=raw_phase_encoding_max_error,
            phase_records=phase_records,
        )
        if not admitted:
            state.update({
                "status": "rejected",
                "records": int(label_count),
                "vector_steps": int(lengths.max(initial=0)),
            })
            write_json_atomic(state_path, state)
            raise RuntimeError("VG007 DAgger collection failed admission")
        manifest = writer.finalize(output)
    finally:
        vector.close()

    metadata = {
        "schema": SCHEMA,
        "tag": TAG,
        "time_steps": int(lengths.max()),
        "agents": AGENTS,
        "episodes": EPISODES,
        "records": int(label_count),
        "episode_lengths": lengths.tolist(),
        "observation": {
            "schema": "vq2_visual_ctbr_public_phase_v1",
            "stored_legal_width": PHASE_LEGAL_OBS_SIZE,
            "mask_width": MASK_SIZE,
            "mask_dtype": "uint8",
            "mask_decode_scale": 1.0 / 255.0,
            "legal_sensor_history_width": LEGAL_OBS_SIZE - MASK_SIZE,
            "public_phase_width": 1,
            "public_phase_tail_index": PHASE_TAIL_SIZE - 1,
            "public_phase_encoding": "clamp(active_gate_index,0,16)/16",
            "public_phase_rate_hz": 4,
            "stored_training_only_privileged_values_per_record": 0,
            "stored_total_gate_count_values_per_record": 0,
        },
        "action": {
            "schema": "normalized_attitude_ctbr_v1",
            "width": ACTION_SIZE,
            "dtype": "float32",
            "source": "sf016_admitted_alignment_oracle_query_at_vg005_state",
            "plant_action_source": "vg005_recurrent_actor_deterministic_mean",
        },
        "files": manifest,
        "source_sha256": source_sha256,
    }
    write_json_atomic(output / "metadata.json", metadata)
    report = {
        "schema": "vq2_variable_gate_dagger_collection_report_v1",
        "tag": TAG,
        "admitted": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "vg006_report_sha256": VG006_REPORT_SHA256,
        "sf016_report_sha256": SF016_REPORT_SHA256,
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "collection_step_limit": COLLECTION_STEP_LIMIT,
        "records": int(label_count),
        "vector_steps": int(lengths.max()),
        "episode_length_min": int(lengths.min()),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_max": int(lengths.max()),
        "phase_records": phase_records.tolist(),
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases,
        "phase_skips": phase_skips,
        "raw_phase_encoding_max_error": raw_phase_encoding_max_error,
        "executed_action_max_error": executed_action_max_error,
        "student_query_mse": (query_square_error / label_count).tolist(),
        "student_query_mae": (query_absolute_error / label_count).tolist(),
        "query_action_min": query_min.tolist(),
        "query_action_max": query_max.tolist(),
        "student_action_min": student_min.tolist(),
        "student_action_max": student_max.tolist(),
        "stored_training_only_privileged_values_per_record": 0,
        "stored_total_gate_count_values_per_record": 0,
        "metadata_sha256": sha256_path(output / "metadata.json"),
        "files": manifest,
        "metrics": flatten_log(pufferl, native_log),
        "loader_overrides": overrides,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": state_identity["runtime"],
        "wall_time_seconds": time.perf_counter() - started,
        "safety": {
            "student_actions_executed": int(label_count),
            "teacher_actions_executed": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    write_json_atomic(output / "report.json", report)
    state.update({
        "status": "admitted",
        "records": int(label_count),
        "vector_steps": int(lengths.max()),
        "metadata_sha256": report["metadata_sha256"],
        "report_sha256": sha256_path(output / "report.json"),
        "files": manifest,
    })
    write_json_atomic(state_path, state)
    shutil.rmtree(staging)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(
        args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
