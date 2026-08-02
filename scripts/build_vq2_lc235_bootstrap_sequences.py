#!/usr/bin/env python3
"""Bootstrap every remaining failing proxy phase with Puffer sequence heads."""

from __future__ import annotations

import argparse
import hashlib
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
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.policy_callable_vq2_all24_sequence import OBSERVATION_SIZE
from scripts.policy_callable_vq2_lc233_phase2_sequence import NumpyLC233Policy
from scripts.policy_callable_vq2_lc235_bootstrap_sequences import NumpyLC235Policy
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save
import scripts.eval_vq2_lc007_converted_vg071_long_course as core


TAG = "vq2_lc235_bootstrap_sequences_001"
SCHEMA = "vq2_lc235_bootstrap_sequences_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc235_bootstrap_sequences_checkpoint_v1"
NUMPY_SCHEMA = "vq2_lc235_bootstrap_sequences_numpy_checkpoint_v1"
THREADS = 2
SEED = 432_224
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 287
NUM_GATES = 24
FIRST_PHASE = 3
LAST_PHASE = 23
MAX_STEPS = 45_000
FINAL_AGENTS = 8
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc233_phase2_action_sequence_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "ebab0d42db832a31d91851f7514132bef1478b31a2fb902f3bd0a3978e9241c5"
)
PARENT_NUMPY = PARENT_DIR / "policy_selected_numpy.npz"
PARENT_NUMPY_SHA256 = (
    "d3fd25442d2e0ad7b1fff7dca1a05f024aef01f9ea7b01cb2363ad8042f2c158"
)
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = (
    "01214b49488968001b7068abcdd2afde3c5b0414a2b7443d0d5c3edd8b5030f2"
)
LC234_RAW3_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc234_phase2_sequence_numpy_batch1_raw3_001/report.json"
)
LC234_RAW3_REPORT_SHA256 = (
    "c08437a52f1917895c8532d950be871fe03d40ff2608ff54b03f378ffb37656e"
)
LC234_ALL24_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc234_phase2_sequence_numpy_batch1_all24_001/report.json"
)
LC234_ALL24_REPORT_SHA256 = (
    "4afbd64ce3aa21c51de1f61b4fc4c861506695eb9e7c4dcc033778dc618560c6"
)
PARENT_CALLABLE = ROOT / "scripts/policy_callable_vq2_lc233_phase2_sequence.py"
PARENT_CALLABLE_SHA256 = (
    "67f1849239817dfb7bb331a844cb58e370b06b079602ca2a87c587a89e61fc44"
)
CALLABLE = ROOT / "scripts/policy_callable_vq2_lc235_bootstrap_sequences.py"
PREREGISTRATION = (
    ROOT / "docs/vq2_lc235_bootstrap_sequences_preregistration_2026-08-02.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc235_vast.sh"
TEST = ROOT / "tests/test_build_vq2_lc235_bootstrap_sequences.py"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def actor_observation(legal: np.ndarray, progress: float) -> np.ndarray:
    result = np.empty(OBSERVATION_SIZE, dtype=np.float32)
    result[:LEGAL_OBS_SIZE] = legal
    result[LEGAL_OBS_SIZE] = np.float32(progress)
    return result


class MutableBootstrapPolicy(NumpyLC233Policy):
    def __init__(self, sequences: dict[int, np.ndarray]) -> None:
        super().__init__(
            PARENT_NUMPY, expected_source_sha256=PARENT_CHECKPOINT_SHA256
        )
        self.sequences = sequences
        self.mutable_counters: dict[int, int] = {}

    def reset(self) -> None:
        super().reset()
        self.mutable_counters = {}

    def __call__(self, observation: np.ndarray) -> tuple[float, ...]:
        parent_action = super().__call__(observation)
        raw_index = int(np.rint(
            float(np.asarray(observation, dtype=np.float32)[LEGAL_OBS_SIZE])
            * OFFICIAL_PROGRESS_SCALE
        ))
        sequence = self.sequences.get(raw_index)
        if sequence is None:
            return parent_action
        counter = self.mutable_counters.get(raw_index, 0)
        action = sequence[min(counter, len(sequence) - 1)]
        self.mutable_counters[raw_index] = counter + 1
        return tuple(float(value) for value in action)


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_NUMPY: PARENT_NUMPY_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC234_RAW3_REPORT: LC234_RAW3_REPORT_SHA256,
        LC234_ALL24_REPORT: LC234_ALL24_REPORT_SHA256,
        PARENT_CALLABLE: PARENT_CALLABLE_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC235 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    construction = json.loads(PARENT_REPORT.read_text())
    raw3 = json.loads(LC234_RAW3_REPORT.read_text())
    all24 = json.loads(LC234_ALL24_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc233_phase2_action_sequence_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or construction.get("schema")
        != "vq2_lc233_phase2_action_sequence_report_v1"
        or not construction.get("numerically_admitted")
        or construction.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or construction.get("numpy_checkpoint_sha256") != PARENT_NUMPY_SHA256
        or raw3.get("schema")
        != "vq2_lc234_phase2_sequence_numpy_batch1_report_v1"
        or not raw3.get("milestone_admitted")
        or raw3.get("outcome", {}).get("target_passes") != 8
        or all24.get("schema")
        != "vq2_lc234_phase2_sequence_numpy_batch1_report_v1"
        or not all24.get("diagnostic_valid")
        or all24.get("milestone_admitted")
        or all24.get("outcome", {}).get("target_passes") != 0
        or all24.get("outcome", {}).get("pre_target_terminals") != 8
        or all24.get("outcome", {}).get("maximum_raw_index_distribution", {}).get("3")
        != 8
        or all24.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("LC233/LC234 do not authorize LC235")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST, CALLABLE,
        PARENT_CHECKPOINT, PARENT_NUMPY, PARENT_REPORT, LC234_RAW3_REPORT,
        LC234_ALL24_REPORT, PARENT_CALLABLE,
        ROOT / "pufferlib/vq2_informed.py", ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "scripts/eval_vq2_lc001_long_course_oracle.py",
        ROOT / "scripts/eval_vq2_lc007_converted_vg071_long_course.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h", ROOT / "src/vecenv.h",
        ROOT / "src/bindings.cu", ROOT / "src/bindings_cpu.cpp",
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
            "torch": torch.__version__, "numpy": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }


def base_environment(
    pufferl: Any, *, agents: int,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], list[str]]:
    from pufferlib import _C

    config, overrides = load_config(
        pufferl, num_gates=NUM_GATES, agents=agents, episodes=agents,
        seed=SEED, threads=THREADS,
    )
    config["vec"]["env_seed_group_size"] = ENV_SEED_GROUP_SIZE
    config["vec"]["env_seed_index_offset"] = ENV_SEED_INDEX_OFFSET
    config["env"].update({
        "evaluation_episode_limit": 1, "max_steps": MAX_STEPS,
        "time_limit_seconds": MAX_STEPS / 64.0,
        "teacher_action_blend": 0.0, "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0, "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0, "w_action_teacher": 0.0,
    })
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != agents or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("LC235 native vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (agents, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (agents,), torch.float32)
    actions = torch.zeros((agents, ACTION_SIZE), dtype=torch.float32)
    return vector, observations, terminals, actions, config, overrides


def run_phase(
    *, phase: int, sequences: dict[int, np.ndarray], pufferl: Any,
) -> tuple[dict[str, Any], np.ndarray | None]:
    target = phase + 1
    vector, observations, terminals, actions, _, overrides = base_environment(
        pufferl, agents=2
    )
    action_view = actions.numpy()
    policies = [MutableBootstrapPolicy(sequences) for _ in range(2)]
    resolved = np.zeros(2, dtype=bool)
    passed = np.zeros(2, dtype=bool)
    pretarget_terminal = np.zeros(2, dtype=bool)
    maximum_raw = np.zeros(2, dtype=np.int32)
    held = np.zeros(2, dtype=np.float32)
    previous_held = held.copy()
    teacher_actions: list[np.ndarray] = []
    teacher_steps: list[int] = []
    executed_error = 0.0
    envelope_violations = 0
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_skips = 0
    encoding_error = 0.0
    vector_steps = 0
    inference_seconds = 0.0
    initial_exact = False
    started = time.perf_counter()
    try:
        vector.reset()
        initial = observations.numpy()
        initial_exact = np.array_equal(initial[0], initial[1])
        if not initial_exact:
            raise RuntimeError("LC235 paired phase rows do not start identically")
        for policy in policies:
            policy.reset()
        for step in range(MAX_STEPS):
            if resolved.all():
                break
            active = ~resolved
            current = observations.numpy()
            raw = current[:, core.PHASE_PRIVILEGED_INDEX]
            held, sampled, current_error = core.update_held_progress(
                raw, held, step=step
            )
            encoding_error = max(encoding_error, current_error)
            changed = np.abs(held - previous_held) > 1e-7
            delta = np.rint((held - previous_held) * OFFICIAL_PROGRESS_SCALE)
            phase_changes_off_tick += int((changed & active).sum()) if not sampled else 0
            phase_decreases += int(((delta < 0) & active).sum())
            phase_skips += int(((delta > 1) & active).sum())
            previous_held = held.copy()
            raw_index = np.rint(raw * OFFICIAL_PROGRESS_SCALE).astype(np.int32)
            maximum_raw = np.maximum(maximum_raw, raw_index)
            newly_passed = active & (raw_index >= target)
            passed |= newly_passed
            resolved |= newly_passed
            active = ~resolved
            if not bool(active.any()):
                break
            action_view.fill(0.0)
            infer_started = time.perf_counter()
            for index in np.flatnonzero(active):
                action_view[index] = policies[index](actor_observation(
                    current[index, :LEGAL_OBS_SIZE], float(held[index])
                ))
            inference_seconds += time.perf_counter() - infer_started
            phase_index = np.rint(
                held * OFFICIAL_PROGRESS_SCALE
            ).astype(np.int32)
            teacher_mask = active & (np.arange(2) == 1) & (phase_index == phase)
            if bool(teacher_mask[1]):
                teacher = alignment_oracle_action(current)
                action_view[1] = teacher[1]
                teacher_actions.append(teacher[1].copy())
                teacher_steps.append(step)
            active_actions = action_view[active]
            if not bool(np.isfinite(active_actions).all()):
                raise RuntimeError("LC235 generated a nonfinite plant action")
            envelope_violations += int(
                (np.abs(active_actions) > 1.0 + 1e-6).sum()
            )
            commanded = action_view.copy()
            vector.cpu_step(actions.data_ptr())
            vector_steps = step + 1
            executed = observations.numpy()[
                :, ACTION_HISTORY.start:ACTION_HISTORY.start + ACTION_SIZE
            ]
            executed_error = max(
                executed_error,
                float(np.max(np.abs(executed[active] - commanded[active]))),
            )
            post_raw = observations.numpy()[:, core.PHASE_PRIVILEGED_INDEX]
            post_index = np.rint(
                post_raw * OFFICIAL_PROGRESS_SCALE
            ).astype(np.int32)
            maximum_raw = np.maximum(maximum_raw, post_index)
            newly_passed = (~resolved) & (post_index >= target)
            passed |= newly_passed
            resolved |= newly_passed
            terminal = (terminals.numpy() > 0.5) & (~resolved)
            pretarget_terminal |= terminal
            resolved |= terminal
    finally:
        vector.close()
    teacher_sequence = None
    if teacher_actions:
        teacher_sequence = np.asarray(teacher_actions, dtype=np.float32)
        steps = np.asarray(teacher_steps, dtype=np.int64)
        if not np.array_equal(np.diff(steps), np.ones(len(steps) - 1, dtype=np.int64)):
            raise RuntimeError("LC235 teacher phase sequence is not contiguous")
    transport = bool(
        envelope_violations == 0
        and executed_error <= core.MAX_EXECUTED_ACTION_ERROR
        and phase_changes_off_tick == 0 and phase_decreases == 0
        and phase_skips == 0 and encoding_error <= 1e-6
    )
    item = {
        "phase": phase, "target_raw_index": target,
        "control_passed": bool(passed[0]),
        "intervention_passed": bool(passed[1]),
        "control_pre_target_terminal": bool(pretarget_terminal[0]),
        "intervention_pre_target_terminal": bool(pretarget_terminal[1]),
        "control_maximum_raw_index": int(maximum_raw[0]),
        "intervention_maximum_raw_index": int(maximum_raw[1]),
        "oracle_plant_actions": 0 if teacher_sequence is None else len(teacher_sequence),
        "oracle_sequence_sha256": (
            None if teacher_sequence is None else array_sha256(teacher_sequence)
        ),
        "sequence_added": bool(not passed[0] and passed[1]),
        "transport_pass": transport, "initial_pair_exact": initial_exact,
        "executed_action_max_error": executed_error,
        "action_envelope_violations": envelope_violations,
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases, "phase_skips": phase_skips,
        "raw_progress_encoding_max_error": encoding_error,
        "vector_steps": vector_steps, "inference_seconds": inference_seconds,
        "wall_time_seconds": time.perf_counter() - started,
        "loader_overrides": overrides,
    }
    if not transport or not resolved.all():
        raise RuntimeError(f"LC235 phase {phase} transport or resolution failed")
    if not passed[0] and not passed[1]:
        raise RuntimeError(f"LC235 oracle failed to rescue phase {phase}")
    if passed[0]:
        return item, None
    if teacher_sequence is None or len(teacher_sequence) == 0:
        raise RuntimeError(f"LC235 phase {phase} rescue has no sequence")
    return item, teacher_sequence


def pack_sequences(
    sequences: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.full((33, 2), (-1, 0), dtype=np.int32)
    chunks: list[np.ndarray] = []
    cursor = 0
    for phase in sorted(sequences):
        sequence = np.ascontiguousarray(sequences[phase], dtype=np.float32)
        offsets[phase] = (cursor, len(sequence))
        chunks.append(sequence)
        cursor += len(sequence)
    actions = (
        np.concatenate(chunks) if chunks else np.empty((0, 4), dtype=np.float32)
    )
    return offsets, actions


def final_screen(
    *, archive: Path, source_sha256: str, pufferl: Any,
) -> dict[str, Any]:
    vector, observations, terminals, actions, _, overrides = base_environment(
        pufferl, agents=FINAL_AGENTS
    )
    action_view = actions.numpy()
    policies = [
        NumpyLC235Policy(archive, expected_source_sha256=source_sha256)
        for _ in range(FINAL_AGENTS)
    ]
    resolved = np.zeros(FINAL_AGENTS, dtype=bool)
    passed = np.zeros(FINAL_AGENTS, dtype=bool)
    terminal_before_finish = np.zeros(FINAL_AGENTS, dtype=bool)
    maximum_raw = np.zeros(FINAL_AGENTS, dtype=np.int32)
    held = np.zeros(FINAL_AGENTS, dtype=np.float32)
    previous_held = held.copy()
    executed_error = 0.0
    envelope_violations = 0
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_skips = 0
    encoding_error = 0.0
    vector_steps = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        vector.reset()
        for policy in policies:
            policy.reset()
        for step in range(MAX_STEPS):
            if resolved.all():
                break
            active = ~resolved
            current = observations.numpy()
            raw = current[:, core.PHASE_PRIVILEGED_INDEX]
            held, sampled, current_error = core.update_held_progress(
                raw, held, step=step
            )
            encoding_error = max(encoding_error, current_error)
            changed = np.abs(held - previous_held) > 1e-7
            delta = np.rint((held - previous_held) * OFFICIAL_PROGRESS_SCALE)
            phase_changes_off_tick += int((changed & active).sum()) if not sampled else 0
            phase_decreases += int(((delta < 0) & active).sum())
            phase_skips += int(((delta > 1) & active).sum())
            previous_held = held.copy()
            raw_index = np.rint(raw * OFFICIAL_PROGRESS_SCALE).astype(np.int32)
            maximum_raw = np.maximum(maximum_raw, raw_index)
            newly_passed = active & (raw_index >= NUM_GATES)
            passed |= newly_passed
            resolved |= newly_passed
            active = ~resolved
            if not bool(active.any()):
                break
            action_view.fill(0.0)
            infer_started = time.perf_counter()
            for index in np.flatnonzero(active):
                action_view[index] = policies[index](actor_observation(
                    current[index, :LEGAL_OBS_SIZE], float(held[index])
                ))
            inference_seconds += time.perf_counter() - infer_started
            active_actions = action_view[active]
            if not bool(np.isfinite(active_actions).all()):
                raise RuntimeError("LC235 final actor emitted nonfinite action")
            envelope_violations += int(
                (np.abs(active_actions) > 1.0 + 1e-6).sum()
            )
            commanded = action_view.copy()
            vector.cpu_step(actions.data_ptr())
            vector_steps = step + 1
            executed = observations.numpy()[
                :, ACTION_HISTORY.start:ACTION_HISTORY.start + ACTION_SIZE
            ]
            executed_error = max(
                executed_error,
                float(np.max(np.abs(executed[active] - commanded[active]))),
            )
            post_raw = observations.numpy()[:, core.PHASE_PRIVILEGED_INDEX]
            post_index = np.rint(
                post_raw * OFFICIAL_PROGRESS_SCALE
            ).astype(np.int32)
            maximum_raw = np.maximum(maximum_raw, post_index)
            newly_passed = (~resolved) & (post_index >= NUM_GATES)
            passed |= newly_passed
            resolved |= newly_passed
            terminal = (terminals.numpy() > 0.5) & (~resolved)
            terminal_before_finish |= terminal
            resolved |= terminal
    finally:
        vector.close()
    transport = bool(
        envelope_violations == 0
        and executed_error <= core.MAX_EXECUTED_ACTION_ERROR
        and phase_changes_off_tick == 0 and phase_decreases == 0
        and phase_skips == 0 and encoding_error <= 1e-6
    )
    admitted = bool(
        transport and passed.all() and not terminal_before_finish.any()
        and resolved.all()
    )
    return {
        "admitted": admitted, "target_raw_index": NUM_GATES,
        "target_passes": int(passed.sum()), "agents": FINAL_AGENTS,
        "pre_target_terminals": int(terminal_before_finish.sum()),
        "unresolved": int((~resolved).sum()),
        "per_agent_maximum_raw_index": maximum_raw.tolist(),
        "transport_pass": transport,
        "executed_action_max_error": executed_error,
        "action_envelope_violations": envelope_violations,
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases, "phase_skips": phase_skips,
        "raw_progress_encoding_max_error": encoding_error,
        "vector_steps": vector_steps, "inference_seconds": inference_seconds,
        "wall_time_seconds": time.perf_counter() - started,
        "loader_overrides": overrides,
    }


def build(*, output: Path = DEFAULT_OUTPUT, resume: bool = False) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    parent = verify_inputs()
    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC235 requires the float32 drone_race_vision backend")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"LC235 requires OMP_NUM_THREADS={THREADS}")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC235 requires OMP_DYNAMIC=FALSE")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise RuntimeError("LC235 requires OPENBLAS_NUM_THREADS=1")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC235 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    started = time.perf_counter()
    sequences: dict[int, np.ndarray] = {}
    phases: list[dict[str, Any]] = []
    for phase in range(FIRST_PHASE, LAST_PHASE + 1):
        item, sequence = run_phase(phase=phase, sequences=sequences, pufferl=pufferl)
        if sequence is not None:
            sequences[phase] = sequence
        phases.append(item)
        print(json.dumps({
            "lc235_phase_complete": phase,
            "control_passed": item["control_passed"],
            "intervention_passed": item["intervention_passed"],
            "sequence_added": item["sequence_added"],
            "sequence_length": item["oracle_plant_actions"],
            "wall_time_seconds": item["wall_time_seconds"],
        }, sort_keys=True), flush=True)

    offsets, actions = pack_sequences(sequences)
    state = {
        name: value.detach().cpu().clone()
        for name, value in parent["model_state"].items()
    }
    state["bootstrap_sequence_offsets"] = torch.from_numpy(offsets.copy())
    state["bootstrap_sequence_actions"] = torch.from_numpy(actions.copy())
    contract = {
        **parent["model"], "class": "VQ2BootstrapSequenceActor",
        "bootstrap_sequence_phase_min": FIRST_PHASE,
        "bootstrap_sequence_phase_max_exclusive": LAST_PHASE + 1,
        "bootstrap_sequence_total_actions": len(actions),
    }
    checkpoint = {
        **parent, "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "model": contract, "model_state": state,
        "numerically_admitted": None,
        "construction_integrity_pass": True,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "bootstrap_sequences": {
            str(phase): {
                "length": len(sequence), "sha256": array_sha256(sequence)
            }
            for phase, sequence in sorted(sequences.items())
        },
    }
    checkpoint_path = output / "policy_selected.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    checkpoint_sha256 = sha256_path(checkpoint_path)
    arrays = {
        name: value.detach().cpu().numpy()
        for name, value in state.items()
    }
    arrays["__metadata_json__"] = np.asarray(json.dumps({
        "schema": NUMPY_SCHEMA,
        "source_checkpoint_sha256": checkpoint_sha256,
        "model": contract,
    }, sort_keys=True), dtype=np.str_)
    numpy_path = output / "policy_selected_numpy.npz"
    np.savez_compressed(numpy_path, **arrays)
    screen = final_screen(
        archive=numpy_path, source_sha256=checkpoint_sha256, pufferl=pufferl
    )
    admitted = bool(screen["admitted"])
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "numpy_checkpoint": numpy_path.name,
        "numpy_checkpoint_sha256": sha256_path(numpy_path),
        "callable_sha256": sha256_path(CALLABLE),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "phase_min": FIRST_PHASE, "phase_max": LAST_PHASE,
        "phases_evaluated": len(phases), "phase_results": phases,
        "sequence_phases": sorted(sequences),
        "sequence_count": len(sequences),
        "sequence_total_actions": len(actions),
        "sequence_actions_sha256": array_sha256(actions),
        "sequence_offsets_sha256": array_sha256(offsets),
        "final_teacher_free_exact_batch1_screen": screen,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "training_only_oracle_actions": int(sum(
                item["oracle_plant_actions"] for item in phases
            )),
            "final_screen_teacher_actions": 0,
            "runtime_privileged_values": 0, "runtime_teacher_actions": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Source-lock N399-prefix replay and then run a zero-command Windows shadow; FlightSim control remains frozen."
            if admitted else
            "Reject LC235 at its final exact batch-1 screen and keep FlightSim frozen."
        ),
    }
    write_json_once(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = build(output=args.output.resolve(), resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("numerically_admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
