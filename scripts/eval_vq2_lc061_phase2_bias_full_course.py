#!/usr/bin/env python3
"""Paired full 24-gate screen of LC048 versus the confirmed phase-2 bias."""

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
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save
import scripts.eval_vq2_lc007_converted_vg071_long_course as core
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone


TAG = "vq2_lc061_phase2_bias_full_course_001"
SCHEMA = "vq2_lc061_phase2_bias_full_course_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc061_phase2_bias_full_course_checkpoint_v1"
GROUP_SIZE = 64
GROUPS = 2
TOTAL_AGENTS = GROUP_SIZE * GROUPS
EPISODES = TOTAL_AGENTS
THREADS = 32
SEED = 431610
NUM_GATES = 24
MAX_STEPS = 12_000
TARGET_PHASE = 2
BIAS = (-0.0025, 0.0, 0.0, 0.0)
MINIMUM_MEAN_GATE_GAIN = 0.05
MINIMUM_GATE3_PASS_GAIN = 1
PARENT_CHECKPOINT = milestone.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = milestone.PARENT_CHECKPOINT_SHA256
LC060_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc060_phase2_bias_confirmation_001/report.json"
)
LC060_REPORT_SHA256 = "ea070c0485a1000cbe3ef90eeffb4a1911c7eb60e26af7df0eee72bc125a4342"
PREREGISTRATION = ROOT / "docs/vq2_lc061_phase2_bias_full_course_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc061_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc061_phase2_bias_full_course.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()
CANDIDATE_NAME = "pitch_m0p0025"
NEXT_AUTHORITY_SELECTED = (
    "Retain the whole-Puffer checkpoint as the offline phase-2 frontier and "
    "rediagnose its next bottleneck."
)
NEXT_AUTHORITY_NONE = "Reject the candidate and retain the parent checkpoint."
PROMOTION_TARGET_RAW_INDEX = 3
SURGERY_PAYLOAD_KEY = "phase2_surgery"
ACTOR_EXECUTION_CONTRACT = (
    "one parent actor plus vectorized post-forward candidate action surgery"
)


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> torch.Tensor:
    del payload
    deltas = torch.zeros((TOTAL_AGENTS, ACTION_SIZE), dtype=torch.float32, device=device)
    deltas[group_slice(1)] = torch.tensor(BIAS, dtype=torch.float32, device=device)
    return deltas


def apply_candidate_actions(
    actor_output: Any,
    next_recurrent: torch.Tensor,
    held_progress: torch.Tensor,
    context: Any,
) -> torch.Tensor:
    del next_recurrent
    return milestone.apply_phase2_bias(actor_output.pre_tanh_mean, held_progress, context)


def initialize_actor_execution(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    actor = milestone.load_actor(payload, device)
    return {
        "actor": actor,
        "recurrent": actor.initial_state(TOTAL_AGENTS, device=device),
    }


def execute_actor_actions(
    execution: dict[str, Any],
    actor_input: torch.Tensor,
    active: torch.Tensor,
    held_progress: torch.Tensor,
    candidate_context: Any,
) -> torch.Tensor:
    actor_output, next_recurrent = execution["actor"].forward_step(
        actor_input, execution["recurrent"]
    )
    execution["recurrent"] = preserve_frozen_state(
        execution["recurrent"], next_recurrent, active
    )
    return apply_candidate_actions(
        actor_output, next_recurrent, held_progress, candidate_context
    )


def build_selected_candidate_state(
    parent_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return milestone.candidate_state(parent_state, BIAS)


def selected_candidate_metadata() -> dict[str, Any]:
    return {"bias": list(BIAS)}


def checkpoint_surgery_metadata(candidate_state_sha256: str) -> dict[str, Any]:
    return {
        "operation": "phase2_pre_tanh_output_bias",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "target_phase": TARGET_PHASE,
        "pre_tanh_output_bias_delta": list(BIAS),
        "candidate_state_sha256": candidate_state_sha256,
    }


def group_slice(group: int) -> slice:
    if not 0 <= group < GROUPS:
        raise ValueError("candidate group is outside the LC061 pair")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def choose_candidate(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return bool(
        parent["transport_pass"] and candidate["transport_pass"]
        and candidate["mean_gates_passed"]
        >= parent["mean_gates_passed"] + MINIMUM_MEAN_GATE_GAIN
        and candidate["gate3_passes"]
        >= parent["gate3_passes"] + MINIMUM_GATE3_PASS_GAIN
        and candidate["crash_rate"] <= parent["crash_rate"]
        and candidate["maximum_raw_index"] >= parent["maximum_raw_index"]
    )


def verify_inputs() -> dict[str, Any]:
    payload = milestone.verify_inputs()
    if sha256_path(LC060_REPORT) != LC060_REPORT_SHA256:
        raise RuntimeError("LC061 bound LC060 report changed")
    report = json.loads(LC060_REPORT.read_text())
    selected = report.get("causal_screen_selected", {})
    if (
        report.get("schema") != "vq2_lc060_phase2_bias_confirmation_report_v1"
        or not report.get("diagnostic_valid")
        or selected.get("bias") != list(BIAS)
        or selected.get("gate3_passes") != 12
        or selected.get("paired_gate3_gains_vs_baseline") != 4
        or selected.get("paired_gate3_losses_vs_baseline") != 2
        or report.get("items", [{}])[0].get("gate3_passes") != 10
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC060 does not authorize the full-course pair")
    return payload


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, milestone.PARENT_REPORT, milestone.PARENT_CHILD_REPORT,
        LC060_REPORT,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc007_converted_vg071_long_course.py",
        ROOT / "scripts/eval_vq2_lc058_phase2_bias_milestone.py",
        ROOT / "src/vecenv.h", ROOT / "src/bindings.cu", ROOT / "src/bindings_cpu.cpp",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
        *EXTRA_SOURCE_PATHS,
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


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    payload = verify_inputs()
    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC061 requires the float32 drone_race_vision backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC061 preregisters CUDA inference")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"LC061 requires OMP_NUM_THREADS={THREADS}")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC061 requires OMP_DYNAMIC=FALSE")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC061 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)

    identity = source_identity()
    device = torch.device(device_name)
    actor_execution = initialize_actor_execution(payload, device=device)
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
    if (
        vector.total_agents != TOTAL_AGENTS or vector.obs_size != ENV_OBS_SIZE
        or not hasattr(vector, "eval_log_range")
    ):
        vector.close()
        raise RuntimeError("LC061 native vector ABI changed")
    observations = _cpu_tensor(vector.obs_ptr, (TOTAL_AGENTS, ENV_OBS_SIZE), torch.float32)
    terminals = _cpu_tensor(vector.terminals_ptr, (TOTAL_AGENTS,), torch.float32)
    actions_cpu = torch.zeros((TOTAL_AGENTS, ACTION_SIZE), dtype=torch.float32)
    candidate_context = build_candidate_context(payload, device=device)
    done = torch.zeros(TOTAL_AGENTS, dtype=torch.bool)
    held = np.zeros(TOTAL_AGENTS, dtype=np.float32)
    previous_held = held.copy()
    maximum_raw_index = np.zeros(TOTAL_AGENTS, dtype=np.int32)
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
    group_logs: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        vector.reset()
        initial = observations.numpy()
        initial_groups_exact = np.array_equal(initial[group_slice(0)], initial[group_slice(1)])
        if not initial_groups_exact:
            raise RuntimeError("LC061 paired seed groups do not begin identically")
        with torch.no_grad():
            for step in range(MAX_STEPS):
                active_cpu = ~done
                if not bool(active_cpu.any()):
                    break
                current = observations.numpy()
                raw = current[:, core.PHASE_PRIVILEGED_INDEX]
                held, sampled, _ = core.update_held_progress(raw, held, step=step)
                active_np = active_cpu.numpy()
                changed = np.abs(held - previous_held) > 1e-7
                delta_index = np.rint((held - previous_held) * OFFICIAL_PROGRESS_SCALE)
                raw_scaled = raw * OFFICIAL_PROGRESS_SCALE
                raw_indices = np.rint(raw_scaled).astype(np.int32)
                encoding_errors = np.abs(raw_scaled - raw_indices)
                for group in range(GROUPS):
                    selected = group_slice(group)
                    group_active = active_np[selected]
                    if changed[selected].any() and not sampled:
                        phase_changes_off_tick[group] += int(changed[selected].sum())
                    phase_decreases[group] += int(((delta_index[selected] < 0) & group_active).sum())
                    phase_skips[group] += int(((delta_index[selected] > 1) & group_active).sum())
                    raw_encoding_max_error[group] = max(
                        raw_encoding_max_error[group],
                        float(encoding_errors[selected].max(initial=0.0)),
                    )
                previous_held = held.copy()
                maximum_raw_index = np.maximum(maximum_raw_index, raw_indices)

                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                progress = torch.from_numpy(held[:, None]).to(device)
                actor_input = torch.cat((legal, progress), dim=1)
                active = active_cpu.to(device)
                inference_started = time.perf_counter()
                action = execute_actor_actions(
                    actor_execution, actor_input, active, progress, candidate_context
                )
                action = torch.where(active[:, None], action, torch.zeros_like(action))
                inference_seconds += time.perf_counter() - inference_started
                if not bool(torch.isfinite(action[active]).all()):
                    nonfinite_action = True
                    break
                actions_cpu.copy_(action.detach().cpu())
                vector.cpu_step(actions_cpu.data_ptr())
                executed = observations.numpy()[:, ACTION_HISTORY.start:ACTION_HISTORY.start + ACTION_SIZE]
                action_np = actions_cpu.numpy()
                for group in range(GROUPS):
                    selected = group_slice(group)
                    group_active = active_np[selected]
                    action_envelope_violations[group] += int(
                        (np.abs(action_np[selected][group_active]) > 1.0 + 1e-6).sum()
                    )
                    executed_action_max_error[group] = max(
                        executed_action_max_error[group],
                        float(np.max(np.abs(executed[selected][group_active] - action_np[selected][group_active]), initial=0.0)),
                    )
                post_raw = observations.numpy()[:, core.PHASE_PRIVILEGED_INDEX]
                maximum_raw_index = np.maximum(
                    maximum_raw_index,
                    np.rint(post_raw * OFFICIAL_PROGRESS_SCALE).astype(np.int32),
                )
                done |= terminals > 0.5
                vector_steps = step + 1
        group_logs = [
            dict(vector.eval_log_range(group * GROUP_SIZE, GROUP_SIZE))
            for group in range(GROUPS)
        ]
    finally:
        vector.close()
    wall = time.perf_counter() - started

    items: list[dict[str, Any]] = []
    for group, name in enumerate(("parent", CANDIDATE_NAME)):
        selected = group_slice(group)
        metrics = flatten_log(pufferl, group_logs[group])
        distribution = {
            str(index): int((maximum_raw_index[selected] == index).sum())
            for index in range(NUM_GATES + 1)
        }
        maximum_index = max(int(index) for index, count in distribution.items() if count)
        completed = bool(
            metrics.get("env/n") == float(GROUP_SIZE)
            and metrics.get(f"env/gate_count{NUM_GATES}_episode") == 1.0
            and all(math.isfinite(float(metrics.get(key, math.nan))) for key in (
                "env/gates_passed", "env/crash", "env/missed_gate", "env/timeout",
            ))
        )
        transport = bool(
            completed and not nonfinite_action
            and action_envelope_violations[group] == 0
            and executed_action_max_error[group] <= core.MAX_EXECUTED_ACTION_ERROR
            and phase_changes_off_tick[group] == 0
            and phase_decreases[group] == 0 and phase_skips[group] == 0
            and raw_encoding_max_error[group] <= 1e-6
            and metrics.get("env/out_of_order") == 0.0
        )
        state = (
            {key: value.detach().cpu().clone() for key, value in payload["model_state"].items()}
            if group == 0 else build_selected_candidate_state(payload["model_state"])
        )
        metadata = {"bias": [0.0, 0.0, 0.0, 0.0]} if group == 0 else selected_candidate_metadata()
        items.append({
            "candidate_index": group, "name": name, **metadata,
            "candidate_state_sha256": milestone.state_sha256(state),
            "mean_gates_passed": float(metrics["env/gates_passed"]),
            "maximum_raw_index": int(maximum_index),
            "maximum_raw_index_distribution": distribution,
            "gate3_passes": int((maximum_raw_index[selected] >= 3).sum()),
            "promotion_target_raw_index": PROMOTION_TARGET_RAW_INDEX,
            "promotion_target_passes": int(
                (maximum_raw_index[selected] >= PROMOTION_TARGET_RAW_INDEX).sum()
            ),
            "crash_rate": float(metrics["env/crash"]),
            "miss_rate": float(metrics["env/missed_gate"]),
            "timeout_rate": float(metrics["env/timeout"]),
            "transport_pass": transport,
            "action_envelope_violations": int(action_envelope_violations[group]),
            "executed_action_max_error": float(executed_action_max_error[group]),
            "phase_changes_off_tick": int(phase_changes_off_tick[group]),
            "phase_decreases": int(phase_decreases[group]),
            "phase_skips": int(phase_skips[group]),
            "raw_progress_encoding_max_error": float(raw_encoding_max_error[group]),
            "metrics": metrics,
        })
    selected = choose_candidate(items[0], items[1])
    candidate_checkpoint_sha256 = None
    if selected:
        checkpoint_state = build_selected_candidate_state(payload["model_state"])
        checkpoint_payload = {
            **payload,
            "schema": CHECKPOINT_SCHEMA,
            "tag": TAG,
            "model_state": checkpoint_state,
            "best_epoch": 0,
            "optimizer_updates": 0,
            "numerically_admitted": True,
            SURGERY_PAYLOAD_KEY: checkpoint_surgery_metadata(
                items[1]["candidate_state_sha256"]
            ),
        }
        checkpoint_path = output / "policy_selected.pt"
        atomic_torch_save(checkpoint_path, checkpoint_payload)
        candidate_checkpoint_sha256 = sha256_path(checkpoint_path)
    diagnostic_valid = bool(initial_groups_exact and all(item["transport_pass"] for item in items))
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "diagnostic_valid": diagnostic_valid,
        "numerically_admitted": bool(selected and diagnostic_valid),
        "selected_candidate": items[1] if selected else None,
        "checkpoint": "policy_selected.pt" if selected else None,
        "checkpoint_sha256": candidate_checkpoint_sha256,
        "mean_gate_gain": items[1]["mean_gates_passed"] - items[0]["mean_gates_passed"],
        "gate3_pass_gain": items[1]["gate3_passes"] - items[0]["gate3_passes"],
        "promotion_target_raw_index": PROMOTION_TARGET_RAW_INDEX,
        "promotion_target_pass_gain": (
            items[1]["promotion_target_passes"]
            - items[0]["promotion_target_passes"]
        ),
        "items": items, "group_size": GROUP_SIZE, "total_agents": TOTAL_AGENTS,
        "seed": SEED, "num_gates": NUM_GATES, "max_steps": MAX_STEPS,
        "vector_steps": vector_steps, "wall_time_seconds": wall,
        "inference_seconds": inference_seconds, "initial_seed_groups_exact": initial_groups_exact,
        "single_cuda_context": True, "single_native_vector": True,
        "actor_execution_contract": ACTOR_EXECUTION_CONTRACT,
        "loader_overrides": overrides, "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            NEXT_AUTHORITY_SELECTED if selected else NEXT_AUTHORITY_NONE
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
