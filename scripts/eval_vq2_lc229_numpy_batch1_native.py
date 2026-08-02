#!/usr/bin/env python3
"""Run LC216 in native closed loop through the exact NumPy batch-1 callable."""

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
from pufferlib.vq2_public_phase import OFFICIAL_PROGRESS_SCALE
from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.eval_vq2_lc001_long_course_oracle import load_config
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.policy_callable_vq2_all24_sequence import (
    OBSERVATION_SIZE,
    NumpyLC216Policy,
)
import scripts.eval_vq2_lc007_converted_vg071_long_course as core


TAG_PREFIX = "vq2_lc229_numpy_batch1_native"
SCHEMA = "vq2_lc229_numpy_batch1_native_report_v1"
AGENTS = 8
EPISODES = AGENTS
THREADS = 8
SEED = 432_224
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 287
NUM_GATES = 24
RAW3_MAX_STEPS = 10_000
ALL24_MAX_STEPS = 45_000
CHECKPOINT = ROOT / "checkpoints/vq2_lc216_all24_action_sequence_numpy.npz"
CHECKPOINT_SHA256 = (
    "248d3574d3354862891b27942d239fb0d123dc8832e8c1e04db5fe12c0f3da1f"
)
SOURCE_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc216_all24_action_sequence_001/policy_selected.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "672af0c4b014bb7d7399e2d709f268a487a83294b7b2a72ebfc79a48896a95b9"
)
SOURCE_REPORT = SOURCE_CHECKPOINT.parent / "report.json"
SOURCE_REPORT_SHA256 = (
    "b112413c778d984f369717643cdd8d69e0ec984e28c47e1b36b20aed18503617"
)
LC227_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc227_raw3_factor_isolation_control_001/report.json"
)
LC227_REPORT_SHA256 = (
    "d2994cb13d7bccc37e1b6ae13e075b67ff06a7d9921dd82f709b183a8359150e"
)
LC228R_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc228r_phase2_broad_rescue_001/report.json"
)
LC228R_REPORT_SHA256 = (
    "247aef21940af1e6b6d17a9d7588fb030edba6a0024b378259d2ac2ac4af08f7"
)
CALLABLE = ROOT / "scripts/policy_callable_vq2_all24_sequence.py"
CALLABLE_SHA256 = (
    "ce64fc1422fbe29a40c239ab921c023e73cac008fa9511ee0dbc7c9e6aa73773"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc229_numpy_batch1_native_preregistration_2026-08-02.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc229_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc229_numpy_batch1_native.py"
LOG_ROOT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"


def tag_for_target(target_raw_index: int) -> str:
    suffix = "raw3_001" if target_raw_index == 3 else "all24_001"
    return f"{TAG_PREFIX}_{suffix}"


def default_output(target_raw_index: int) -> Path:
    return LOG_ROOT / tag_for_target(target_raw_index)


def max_steps_for_target(target_raw_index: int) -> int:
    if target_raw_index == 3:
        return RAW3_MAX_STEPS
    if target_raw_index == 24:
        return ALL24_MAX_STEPS
    raise ValueError("LC229 target must be raw index 3 or 24")


def deployment_observation(legal: np.ndarray, held_progress: float) -> np.ndarray:
    values = np.asarray(legal, dtype=np.float32)
    if values.shape != (LEGAL_OBS_SIZE,):
        raise ValueError("LC229 legal observation shape changed")
    result = np.empty(OBSERVATION_SIZE, dtype=np.float32)
    result[:LEGAL_OBS_SIZE] = values
    result[LEGAL_OBS_SIZE] = np.float32(held_progress)
    return result


def verify_inputs() -> None:
    expected = {
        CHECKPOINT: CHECKPOINT_SHA256,
        SOURCE_CHECKPOINT: SOURCE_CHECKPOINT_SHA256,
        SOURCE_REPORT: SOURCE_REPORT_SHA256,
        LC227_REPORT: LC227_REPORT_SHA256,
        LC228R_REPORT: LC228R_REPORT_SHA256,
        CALLABLE: CALLABLE_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC229 bound input changed: {path}")
    construction = json.loads(SOURCE_REPORT.read_text())
    small_batch = json.loads(LC227_REPORT.read_text())
    broad_batch = json.loads(LC228R_REPORT.read_text())
    small_candidate = small_batch.get("items", [{}, {}])[1]
    broad_control = broad_batch.get("items", [{}])[0]
    if (
        construction.get("schema")
        != "vq2_lc216_all24_action_sequence_report_v1"
        or not construction.get("numerically_admitted")
        or construction.get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
        or small_batch.get("schema")
        != "vq2_lc227_raw3_factor_isolation_report_v1"
        or not small_batch.get("diagnostic_valid")
        or small_candidate.get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
        or small_candidate.get("target_passes") != 0
        or broad_batch.get("schema")
        != "vq2_lc228r_phase2_broad_rescue_report_v1"
        or not broad_batch.get("diagnostic_valid")
        or broad_control.get("target_passes") != 256
        or broad_batch.get("training_dataset_admitted")
        or broad_batch.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("LC216/LC227/LC228R do not authorize LC229")


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST, CALLABLE,
        CHECKPOINT, SOURCE_CHECKPOINT, SOURCE_REPORT, LC227_REPORT,
        LC228R_REPORT, ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "scripts/eval_vq2_lc001_long_course_oracle.py",
        ROOT / "scripts/eval_vq2_lc007_converted_vg071_long_course.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c", ROOT / "src/vecenv.h",
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


def summarize(
    *, target_raw_index: int, maximum_raw_index: np.ndarray,
    passed: np.ndarray, resolved: np.ndarray, pretarget_terminal: np.ndarray,
    resolve_step: np.ndarray,
) -> dict[str, Any]:
    capped = np.minimum(maximum_raw_index, target_raw_index)
    steps = resolve_step[passed]
    return {
        "target_raw_index": target_raw_index,
        "target_passes": int(passed.sum()),
        "target_pass_rate": float(passed.mean()),
        "pre_target_terminals": int(pretarget_terminal.sum()),
        "unresolved": int((~resolved).sum()),
        "maximum_raw_index_distribution": {
            str(index): int((capped == index).sum())
            for index in range(target_raw_index + 1)
            if bool((capped == index).any())
        },
        "target_resolve_step_mean": float(steps.mean()) if steps.size else None,
        "target_resolve_step_max": int(steps.max()) if steps.size else None,
        "per_agent_maximum_raw_index": maximum_raw_index.tolist(),
        "per_agent_resolve_step": resolve_step.tolist(),
    }


def run(
    *, target_raw_index: int, output: Path, resume: bool = False,
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    max_steps = max_steps_for_target(target_raw_index)
    verify_inputs()
    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC229 requires the float32 drone_race_vision backend")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"LC229 requires OMP_NUM_THREADS={THREADS}")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC229 requires OMP_DYNAMIC=FALSE")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise RuntimeError("LC229 requires OPENBLAS_NUM_THREADS=1")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC229 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)

    identity = source_identity()
    policies = [NumpyLC216Policy(CHECKPOINT) for _ in range(AGENTS)]
    config, overrides = load_config(
        pufferl, num_gates=NUM_GATES, agents=AGENTS, episodes=EPISODES,
        seed=SEED, threads=THREADS,
    )
    config["vec"]["env_seed_group_size"] = ENV_SEED_GROUP_SIZE
    config["vec"]["env_seed_index_offset"] = ENV_SEED_INDEX_OFFSET
    environment = config["env"]
    environment.update({
        "evaluation_episode_limit": 1,
        "max_steps": max_steps,
        "time_limit_seconds": max_steps / 64.0,
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
    })
    evaluated_environment = {
        name: environment.get(name)
        for name in (
            "reset_position_noise_xy", "reset_position_noise_z",
            "visual_camera_dropout_prob", "visual_edge_dropout_prob",
            "visual_rolling_shutter_s", "sitl_plant_domain_randomize",
            "sitl_rate_gain_jitter_frac", "sitl_hover_thrust_jitter",
            "sitl_rate_lag_jitter_frac", "sitl_linear_drag_jitter_frac",
        )
    }
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("LC229 native vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    action_view = actions.numpy()
    done = np.zeros(AGENTS, dtype=bool)
    resolved = np.zeros(AGENTS, dtype=bool)
    passed = np.zeros(AGENTS, dtype=bool)
    pretarget_terminal = np.zeros(AGENTS, dtype=bool)
    held = np.zeros(AGENTS, dtype=np.float32)
    previous_held = held.copy()
    maximum_raw_index = np.zeros(AGENTS, dtype=np.int32)
    resolve_step = np.full(AGENTS, -1, dtype=np.int32)
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_skips = 0
    raw_encoding_max_error = 0.0
    action_envelope_violations = 0
    executed_action_max_error = 0.0
    nonfinite_action = False
    vector_steps = 0
    policy_calls = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        vector.reset()
        for policy in policies:
            policy.reset()
        for step in range(max_steps):
            active = ~done
            if resolved.all() or not bool(active.any()):
                break
            current = observations.numpy()
            raw = current[:, core.PHASE_PRIVILEGED_INDEX]
            held, sampled, encoding_error = core.update_held_progress(
                raw, held, step=step
            )
            raw_encoding_max_error = max(raw_encoding_max_error, encoding_error)
            changed = np.abs(held - previous_held) > 1e-7
            delta = np.rint((held - previous_held) * OFFICIAL_PROGRESS_SCALE)
            phase_changes_off_tick += int((changed & active).sum()) if not sampled else 0
            phase_decreases += int(((delta < 0) & active).sum())
            phase_skips += int(((delta > 1) & active).sum())
            previous_held = held.copy()
            raw_indices = np.rint(raw * OFFICIAL_PROGRESS_SCALE).astype(np.int32)
            maximum_raw_index = np.maximum(maximum_raw_index, raw_indices)
            newly_passed = (~resolved) & (raw_indices >= target_raw_index)
            passed |= newly_passed
            resolve_step[newly_passed] = step
            resolved |= newly_passed

            action_view.fill(0.0)
            inference_started = time.perf_counter()
            for index in np.flatnonzero(active & ~resolved):
                actor_input = deployment_observation(
                    current[index, :LEGAL_OBS_SIZE], float(held[index])
                )
                action_view[index] = policies[index](actor_input)
                policy_calls += 1
            inference_seconds += time.perf_counter() - inference_started
            active_actions = action_view[active & ~resolved]
            if active_actions.size and not bool(np.isfinite(active_actions).all()):
                nonfinite_action = True
                break
            action_envelope_violations += int(
                (np.abs(active_actions) > 1.0 + 1e-6).sum()
            )
            commanded = action_view.copy()
            vector.cpu_step(actions.data_ptr())
            executed = observations.numpy()[
                :, ACTION_HISTORY.start:ACTION_HISTORY.start + ACTION_SIZE
            ]
            active_command = active & ~resolved
            if bool(active_command.any()):
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(np.max(np.abs(
                        executed[active_command] - commanded[active_command]
                    ))),
                )
            post_raw = observations.numpy()[:, core.PHASE_PRIVILEGED_INDEX]
            post_indices = np.rint(
                post_raw * OFFICIAL_PROGRESS_SCALE
            ).astype(np.int32)
            maximum_raw_index = np.maximum(maximum_raw_index, post_indices)
            newly_passed = (~resolved) & (post_indices >= target_raw_index)
            passed |= newly_passed
            resolve_step[newly_passed] = step + 1
            resolved |= newly_passed
            terminal = (terminals.numpy() > 0.5) & active
            newly_terminal = (~resolved) & terminal
            pretarget_terminal |= newly_terminal
            resolve_step[newly_terminal] = step + 1
            resolved |= newly_terminal
            done |= terminal
            vector_steps = step + 1
    finally:
        vector.close()
    wall = time.perf_counter() - started

    transport_pass = bool(
        not nonfinite_action
        and action_envelope_violations == 0
        and executed_action_max_error <= core.MAX_EXECUTED_ACTION_ERROR
        and phase_changes_off_tick == 0
        and phase_decreases == 0
        and phase_skips == 0
        and raw_encoding_max_error <= 1e-6
    )
    outcome = summarize(
        target_raw_index=target_raw_index,
        maximum_raw_index=maximum_raw_index,
        passed=passed,
        resolved=resolved,
        pretarget_terminal=pretarget_terminal,
        resolve_step=resolve_step,
    )
    milestone_admitted = bool(
        transport_pass and outcome["target_passes"] == AGENTS
        and outcome["pre_target_terminals"] == 0 and outcome["unresolved"] == 0
    )
    report = {
        "schema": SCHEMA, "tag": tag_for_target(target_raw_index),
        "completed": True, "diagnostic_valid": transport_pass,
        "milestone_admitted": milestone_admitted,
        "all24_proxy_admitted": milestone_admitted and target_raw_index == 24,
        "actor_execution": (
            "eight independent NumpyLC216Policy objects; each call receives one "
            "4119-value observation and performs no batched policy arithmetic"
        ),
        "deployment_callable_exact": True,
        "agents": AGENTS, "episodes": EPISODES, "threads": THREADS,
        "seed": SEED, "environment_seed_group_size": ENV_SEED_GROUP_SIZE,
        "environment_seed_index_offset": ENV_SEED_INDEX_OFFSET,
        "num_gates": NUM_GATES, "max_steps": max_steps,
        "vector_steps": vector_steps, "policy_calls": policy_calls,
        "wall_time_seconds": wall, "inference_seconds": inference_seconds,
        "outcome": outcome, "transport_pass": transport_pass,
        "action_envelope_violations": action_envelope_violations,
        "executed_action_max_error": executed_action_max_error,
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases, "phase_skips": phase_skips,
        "raw_progress_encoding_max_error": raw_encoding_max_error,
        "nonfinite_action": nonfinite_action,
        "environment": evaluated_environment, "loader_overrides": overrides,
        "numpy_checkpoint_sha256": CHECKPOINT_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Run the exact-callable all-24 native stage."
            if milestone_admitted and target_raw_index == 3 else
            "Proceed to N399-prefix replay; FlightSim remains frozen."
            if milestone_admitted else
            "Reject deployment-context admission and repair offline; keep FlightSim frozen."
        ),
    }
    write_json_once(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, choices=(3, 24), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = default_output(args.target) if args.output is None else args.output.resolve()
    report = run(target_raw_index=args.target, output=output, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("diagnostic_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
