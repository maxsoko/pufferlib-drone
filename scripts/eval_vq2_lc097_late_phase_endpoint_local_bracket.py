#!/usr/bin/env python3
"""Teacher-free local-start bracket of the admissible LC096 phase heads."""

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
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save
import scripts.eval_vq2_lc007_converted_vg071_long_course as core
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone


TAG = "vq2_lc097_late_phase_endpoint_local_bracket_001"
SCHEMA = "vq2_lc097_late_phase_endpoint_local_bracket_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc097_late_phase_endpoint_local_bracket_checkpoint_v1"
ALPHAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
GROUP_SIZE = 64
GROUPS = len(ALPHAS)
TOTAL_AGENTS = GROUP_SIZE * GROUPS
EPISODES = TOTAL_AGENTS
THREADS = 32
SEED = 431_970
NUM_GATES = 24
PHASE_MIN = 6
PHASE_MAX_EXCLUSIVE = 24
OFFSET_MIN = 2.0
OFFSET_MAX = 5.0
MAX_STEPS = 2_048
MINIMUM_MEAN_ADVANCE_GAIN = 0.05
MINIMUM_ONE_GATE_PASS_GAIN = 2
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc094_full_batch_serialized_checkpoint_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "14a5f90e8a7a80d3535ef0d86d3f9b5d75d42317c387a0554cf59ecf283e340a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "639edd98ec0d99c981d16837c802d758adc6622736314b3e6afe09a58af21139"
ENDPOINT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc096_late_phase_multihead_endpoint_001"
)
ENDPOINT = ENDPOINT_DIR / "policy_endpoint.pt"
ENDPOINT_SHA256 = "912d8f0b9692ecf2cef3ed41c4f379476e8e58aee3e48d7007a90183502e1bfb"
ENDPOINT_REPORT = ENDPOINT_DIR / "report.json"
ENDPOINT_REPORT_SHA256 = "75b65b85340eaefd7f84948fa75bb3246a45b6047a54eedc2ada34f6c0880f04"
ADMISSIBLE_PHASES = (6, 7, *range(10, 24))
INTERPOLATED_NAMES = (
    "indexed_phase_residual_output",
    "indexed_phase_residual_output_bias",
)
PREREGISTRATION = ROOT / "docs/vq2_lc097_late_phase_endpoint_local_bracket_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc097_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc097_late_phase_endpoint_local_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def group_slice(group: int) -> slice:
    if not 0 <= group < GROUPS:
        raise ValueError("LC097 group index is outside the bracket")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        ENDPOINT: ENDPOINT_SHA256,
        ENDPOINT_REPORT: ENDPOINT_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC097 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    endpoint = torch.load(ENDPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    fit = json.loads(ENDPOINT_REPORT.read_text())
    admitted = tuple(sorted(
        int(phase) for phase, report in fit.get("phase_reports", {}).items()
        if report.get("numerically_admitted")
    ))
    rejected = tuple(sorted(
        int(phase) for phase, report in fit.get("phase_reports", {}).items()
        if not report.get("numerically_admitted")
    ))
    if (
        parent.get("schema") != "vq2_lc094_full_batch_serialized_checkpoint_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or endpoint.get("schema") != "vq2_lc096_late_phase_multihead_endpoint_checkpoint_v1"
        or endpoint.get("numerically_admitted")
        or fit.get("schema") != "vq2_lc096_late_phase_multihead_endpoint_report_v1"
        or fit.get("numerically_admitted")
        or admitted != ADMISSIBLE_PHASES
        or rejected != (8, 9)
        or fit.get("checkpoint_sha256") != ENDPOINT_SHA256
        or fit.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC094/LC096 do not authorize LC097")
    return parent, endpoint


def candidate_state(
    parent: dict[str, torch.Tensor], endpoint: dict[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent.items()}
    for name in INTERPOLATED_NAMES:
        for phase in ADMISSIBLE_PHASES:
            state[name][phase].lerp_(endpoint[name][phase], float(alpha))
    return state


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, ENDPOINT, ENDPOINT_REPORT,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc007_converted_vg071_long_course.py",
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
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline = items[0]
    eligible = [
        item for item in items[1:]
        if item["transport_pass"]
        and item["mean_gate_advance"]
        >= baseline["mean_gate_advance"] + MINIMUM_MEAN_ADVANCE_GAIN
        and item["one_gate_passes"]
        >= baseline["one_gate_passes"] + MINIMUM_ONE_GATE_PASS_GAIN
        and item["crash_rate"] <= baseline["crash_rate"]
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            item["mean_gate_advance"], item["one_gate_passes"],
            item["two_gate_passes"], -item["crash_rate"], -item["alpha"],
        ),
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    parent, endpoint = verify_inputs()
    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC097 requires the float32 drone_race_vision backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC097 preregisters CUDA inference")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError("LC097 requires OMP_NUM_THREADS=32")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC097 requires OMP_DYNAMIC=FALSE")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC097 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)

    identity = source_identity()
    device = torch.device(device_name)
    states = [
        candidate_state(parent["model_state"], endpoint["model_state"], alpha)
        for alpha in ALPHAS
    ]
    payloads = [{**parent, "model_state": state} for state in states]
    actors = [milestone.load_actor(payload, device) for payload in payloads]
    recurrents = [
        actor.initial_state(TOTAL_AGENTS, device=device) for actor in actors
    ]
    config, overrides = load_config(
        pufferl, num_gates=NUM_GATES, agents=TOTAL_AGENTS,
        episodes=EPISODES, seed=SEED, threads=THREADS,
    )
    config["vec"]["env_seed_group_size"] = GROUP_SIZE
    config["env"].update({
        "evaluation_episode_limit": 1,
        "max_steps": MAX_STEPS,
        "time_limit_seconds": MAX_STEPS / 64.0,
        "gate_local_start_curriculum": 1,
        "gate_local_start_probability": 1.0,
        "gate_local_start_offset_min": OFFSET_MIN,
        "gate_local_start_offset_max": OFFSET_MAX,
        "gate_local_start_gate_min": PHASE_MIN,
        "gate_local_start_gate_max_exclusive": PHASE_MAX_EXCLUSIVE,
        "mixed_start_curriculum": 0, "segment_start_probability": 0.0,
        "use_custom_start": 0,
        "teacher_action_blend": 0.0, "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0, "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0, "w_action_teacher": 0.0,
    })
    vector = _C.create_vec(config, gpu=0)
    observations = _cpu_tensor(
        vector.obs_ptr, (TOTAL_AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (TOTAL_AGENTS,), torch.float32)
    actions_cpu = torch.zeros((TOTAL_AGENTS, ACTION_SIZE), dtype=torch.float32)
    done = torch.zeros(TOTAL_AGENTS, dtype=torch.bool)
    held = np.zeros(TOTAL_AGENTS, dtype=np.float32)
    previous_held = held.copy()
    start_index = np.zeros(TOTAL_AGENTS, dtype=np.int32)
    maximum_index = np.zeros(TOTAL_AGENTS, dtype=np.int32)
    phase_changes_off_tick = np.zeros(GROUPS, dtype=np.int64)
    phase_decreases = np.zeros(GROUPS, dtype=np.int64)
    phase_skips = np.zeros(GROUPS, dtype=np.int64)
    raw_encoding_max_error = np.zeros(GROUPS, dtype=np.float64)
    action_envelope_violations = np.zeros(GROUPS, dtype=np.int64)
    executed_action_max_error = np.zeros(GROUPS, dtype=np.float64)
    nonfinite_action = False
    vector_steps = 0
    initial_groups_exact = False
    group_logs: list[dict[str, Any]] = []
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        vector.reset()
        initial = observations.numpy()
        initial_groups_exact = all(
            np.array_equal(initial[group_slice(0)], initial[group_slice(group)])
            for group in range(1, GROUPS)
        )
        if not initial_groups_exact:
            raise RuntimeError("LC097 paired local-start groups differ initially")
        initial_raw = initial[:, core.PHASE_PRIVILEGED_INDEX]
        start_index = np.rint(initial_raw * OFFICIAL_PROGRESS_SCALE).astype(np.int32)
        maximum_index = start_index.copy()
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
                raw_index = np.rint(raw_scaled).astype(np.int32)
                maximum_index = np.maximum(maximum_index, raw_index)
                for group in range(GROUPS):
                    selected = group_slice(group)
                    group_active = active_np[selected]
                    if changed[selected].any() and not sampled:
                        phase_changes_off_tick[group] += int(changed[selected].sum())
                    phase_decreases[group] += int(
                        ((delta_index[selected] < 0) & group_active).sum()
                    )
                    phase_skips[group] += int(
                        ((delta_index[selected] > 1) & group_active).sum()
                    )
                    raw_encoding_max_error[group] = max(
                        raw_encoding_max_error[group],
                        float(np.abs(raw_scaled[selected] - raw_index[selected]).max(initial=0.0)),
                    )
                previous_held = held.copy()
                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                progress = torch.from_numpy(held[:, None]).to(device)
                actor_input = torch.cat((legal, progress), dim=1)
                active = active_cpu.to(device)
                inference_started = time.perf_counter()
                outputs = []
                for group, actor in enumerate(actors):
                    actor_output, candidate_recurrent = actor.forward_step(
                        actor_input, recurrents[group]
                    )
                    recurrents[group] = preserve_frozen_state(
                        recurrents[group], candidate_recurrent, active
                    )
                    outputs.append(actor_output.mean)
                action = torch.zeros_like(outputs[0])
                for group in range(GROUPS):
                    selected = group_slice(group)
                    action[selected] = outputs[group][selected]
                action = torch.where(active[:, None], action, torch.zeros_like(action))
                inference_seconds += time.perf_counter() - inference_started
                if not bool(torch.isfinite(action[active]).all()):
                    nonfinite_action = True
                    break
                actions_cpu.copy_(action.cpu())
                action_np = actions_cpu.numpy()
                for group in range(GROUPS):
                    selected = group_slice(group)
                    group_active = active_np[selected]
                    action_envelope_violations[group] += int(
                        (np.abs(action_np[selected][group_active]) > 1.0 + 1e-6).sum()
                    )
                vector.cpu_step(actions_cpu.data_ptr())
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start:ACTION_HISTORY.start + ACTION_SIZE
                ]
                for group in range(GROUPS):
                    selected = group_slice(group)
                    group_active = active_np[selected]
                    executed_action_max_error[group] = max(
                        executed_action_max_error[group],
                        float(np.max(np.abs(
                            executed[selected][group_active] - action_np[selected][group_active]
                        ), initial=0.0)),
                    )
                post_raw = observations.numpy()[:, core.PHASE_PRIVILEGED_INDEX]
                maximum_index = np.maximum(
                    maximum_index,
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

    items = []
    for group, (alpha, native_log, state) in enumerate(zip(ALPHAS, group_logs, states)):
        metrics = flatten_log(pufferl, native_log)
        selected = group_slice(group)
        advances = maximum_index[selected] - start_index[selected]
        transport = bool(
            metrics.get("env/n") == float(GROUP_SIZE)
            and metrics.get("env/gate_local_start_rate") == 1.0
            and not nonfinite_action
            and action_envelope_violations[group] == 0
            and executed_action_max_error[group] <= core.MAX_EXECUTED_ACTION_ERROR
            and phase_changes_off_tick[group] == 0
            and phase_decreases[group] == 0
            and phase_skips[group] == GROUP_SIZE
            and raw_encoding_max_error[group] <= 1e-6
            and metrics.get("env/out_of_order") == 0.0
        )
        items.append({
            "candidate_index": group, "alpha": alpha,
            "candidate_state_sha256": state_sha256(state),
            "mean_gate_advance": float(advances.mean()),
            "one_gate_passes": int((advances >= 1).sum()),
            "two_gate_passes": int((advances >= 2).sum()),
            "maximum_gate_advance": int(advances.max(initial=0)),
            "gate_advance_distribution": {
                str(value): int((advances == value).sum())
                for value in range(int(advances.max(initial=0)) + 1)
            },
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
    selected = choose_candidate(items)
    checkpoint_sha256 = None
    if selected is not None:
        selected_state = states[selected["candidate_index"]]
        checkpoint = {
            **parent, "schema": CHECKPOINT_SCHEMA, "tag": TAG,
            "model_state": selected_state, "best_epoch": 0,
            "optimizer_updates": 0, "numerically_admitted": True,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "endpoint_checkpoint_sha256": ENDPOINT_SHA256,
            "late_phase_interpolation": {
                "alpha": selected["alpha"],
                "phases": list(ADMISSIBLE_PHASES),
                "candidate_state_sha256": selected["candidate_state_sha256"],
            },
        }
        checkpoint_path = output / "policy_selected.pt"
        atomic_torch_save(checkpoint_path, checkpoint)
        checkpoint_sha256 = sha256_path(checkpoint_path)
    diagnostic_valid = bool(
        initial_groups_exact and not nonfinite_action
        and all(item["transport_pass"] for item in items)
    )
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "diagnostic_valid": diagnostic_valid,
        "numerically_admitted": bool(selected is not None and diagnostic_valid),
        "selected_candidate": selected,
        "checkpoint": "policy_selected.pt" if selected is not None else None,
        "checkpoint_sha256": checkpoint_sha256,
        "alphas": list(ALPHAS), "admissible_phases": list(ADMISSIBLE_PHASES),
        "group_size": GROUP_SIZE, "candidate_groups": GROUPS,
        "total_agents": TOTAL_AGENTS, "threads": THREADS, "seed": SEED,
        "num_gates": NUM_GATES, "max_steps": MAX_STEPS,
        "vector_steps": vector_steps, "wall_time_seconds": wall,
        "inference_seconds": inference_seconds,
        "initial_seed_groups_exact": initial_groups_exact,
        "single_cuda_context": True, "single_native_vector": True,
        "actor_execution_contract": (
            "seven independently loaded complete saved Puffer checkpoints, each executed over the full 448-agent batch"
        ),
        "loader_overrides": overrides, "items": items,
        "minimum_mean_advance_gain": MINIMUM_MEAN_ADVANCE_GAIN,
        "minimum_one_gate_pass_gain": MINIMUM_ONE_GATE_PASS_GAIN,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Run one full-start serialization-exact 24-gate screen of the selected Puffer checkpoint."
            if selected is not None else "Reject the admitted-head interpolation family and retain LC094."
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
