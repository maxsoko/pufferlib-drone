#!/usr/bin/env python3
"""Teacher-free 20/24-gate baseline for the converted VG071 visual Puffer."""

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
from pufferlib.vq2_informed import (
    ACTION_HISTORY,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    PRIVILEGED_NAMES,
)
from pufferlib.vq2_public_phase import (
    OFFICIAL_PROGRESS_SCALE,
    PUBLIC_STATUS_INTERVAL_STEPS,
)
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import (
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_lc001_long_course_oracle import (
    load_config,
    sha256_path,
    write_json_once,
)
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state


TAG = "vq2_lc007_converted_vg071_long_course_001"
SCHEMA = "vq2_lc007_converted_vg071_long_course_report_v1"
COUNTS = (20, 24)
AGENTS = 32
EPISODES = 32
THREADS = 32
SEEDS = {20: 431070, 24: 431074}
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg071_nonlinear_residual_count5_multi_offset_001/policy_selected.pt"
)
CHECKPOINT_SHA256 = (
    "60677e385cefb6d6af5c957071cb846de8396d8872880a90832d987b7494f010"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc007_converted_vg071_long_course_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc007_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
PHASE_PRIVILEGED_INDEX = LEGAL_OBS_SIZE + PRIVILEGED_NAMES.index(
    "ordered_gate_phase"
)
MAX_EXECUTED_ACTION_ERROR = 5e-5
MAX_STEPS_OVERRIDE: int | None = None
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()


def conversion_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    del payload
    return {
        "source_encoding": "clamp(active_gate_index,0,16)/16",
        "target_encoding": "active_gate_index/6",
        "legacy_indices_0_through_16_preserved": True,
    }


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, CHECKPOINT,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc001_long_course_oracle.py",
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
            "numpy": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
    }


def load_converted_actor(
    device: torch.device,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any]]:
    if sha256_path(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("LC007 VG071 checkpoint hash changed")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != "vq2_vg071_paired_synthetic_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or contract.get("class") != "VQ2IndexedPhaseMLPResidualActor"
        or contract.get("hidden_size") != 256
        or contract.get("residual_size") != 64
    ):
        raise RuntimeError("LC007 VG071 actor contract changed")
    actor = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_converted_state(payload["model_state"])
    actor.eval()
    return actor, payload


def update_held_progress(
    raw_progress: np.ndarray,
    held_progress: np.ndarray,
    *,
    step: int,
) -> tuple[np.ndarray, bool, float]:
    raw = np.asarray(raw_progress, dtype=np.float32)
    held = np.asarray(held_progress, dtype=np.float32).copy()
    if raw.shape != held.shape:
        raise ValueError("raw and held progress do not align")
    raw_indices = np.rint(raw * OFFICIAL_PROGRESS_SCALE)
    encoding_error = float(np.max(
        np.abs(raw - raw_indices / OFFICIAL_PROGRESS_SCALE), initial=0.0
    ))
    sampled = step % PUBLIC_STATUS_INTERVAL_STEPS == 0
    if sampled:
        held[:] = raw
    if not np.isfinite(held).all() or held.min(initial=0.0) < -1e-6:
        raise RuntimeError("held public progress is invalid")
    return held, sampled, encoding_error


def run_count(
    *, num_gates: int, device: torch.device, identity: dict[str, Any]
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    actor, payload = load_converted_actor(device)
    config, overrides = load_config(
        pufferl, num_gates=num_gates, agents=AGENTS, episodes=EPISODES,
        seed=SEEDS[num_gates], threads=THREADS,
    )
    environment = config["env"]
    environment.update({
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
    })
    if MAX_STEPS_OVERRIDE is not None:
        if MAX_STEPS_OVERRIDE <= 0:
            raise RuntimeError("long-course step override must be positive")
        environment["max_steps"] = MAX_STEPS_OVERRIDE
        environment["time_limit_seconds"] = MAX_STEPS_OVERRIDE / 64.0
    if any(float(environment[name]) != 0.0 for name in (
        "teacher_action_blend", "teacher_course_spline",
        "teacher_segment_minimum_jerk", "teacher_alignment_governor",
        "w_action_teacher",
    )):
        raise RuntimeError("LC007 inherited a runtime teacher path")
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("LC007 vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions_cpu = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    state = actor.initial_state(AGENTS, device=device)
    done = torch.zeros(AGENTS, dtype=torch.bool)
    held = np.zeros(AGENTS, dtype=np.float32)
    previous_held = held.copy()
    maximum_raw_index = np.zeros(AGENTS, dtype=np.int32)
    maximum_held_index = np.zeros(AGENTS, dtype=np.int32)
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_skips = 0
    raw_encoding_max_error = 0.0
    action_envelope_violations = 0
    executed_action_max_error = 0.0
    nonfinite_action = False
    vector_steps = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    native_log: dict[str, Any] = {}
    try:
        vector.reset()
        with torch.no_grad():
            for step in range(int(environment["max_steps"])):
                active_cpu = ~done
                if not bool(active_cpu.any()):
                    break
                current = observations.numpy()
                raw = current[:, PHASE_PRIVILEGED_INDEX]
                held, sampled, encoding_error = update_held_progress(
                    raw, held, step=step
                )
                raw_encoding_max_error = max(raw_encoding_max_error, encoding_error)
                active_np = active_cpu.numpy()
                changed = np.abs(held - previous_held) > 1e-7
                if changed.any() and not sampled:
                    phase_changes_off_tick += int(changed.sum())
                delta_index = np.rint(
                    (held[active_np] - previous_held[active_np])
                    * OFFICIAL_PROGRESS_SCALE
                )
                phase_decreases += int((delta_index < 0).sum())
                phase_skips += int((delta_index > 1).sum())
                previous_held = held.copy()
                raw_index = np.rint(raw * OFFICIAL_PROGRESS_SCALE).astype(np.int32)
                held_index = np.rint(held * OFFICIAL_PROGRESS_SCALE).astype(np.int32)
                maximum_raw_index = np.maximum(maximum_raw_index, raw_index)
                maximum_held_index = np.maximum(maximum_held_index, held_index)

                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                progress = torch.from_numpy(held[:, None]).to(device)
                actor_input = torch.cat((legal, progress), dim=1)
                active = active_cpu.to(device)
                inference_started = time.perf_counter()
                output, candidate_state = actor.forward_step(actor_input, state)
                state = preserve_frozen_state(state, candidate_state, active)
                action = torch.where(
                    active[:, None], output.mean, torch.zeros_like(output.mean)
                )
                if not bool(torch.isfinite(action[active]).all()):
                    nonfinite_action = True
                    break
                inference_seconds += time.perf_counter() - inference_started
                selected = action[active].detach().cpu()
                action_envelope_violations += int(
                    (selected.abs() > 1.0 + 1e-6).sum()
                )
                actions_cpu.copy_(action.detach().cpu())
                vector.cpu_step(actions_cpu.data_ptr())
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(np.max(np.abs(
                        executed[active_np] - actions_cpu.numpy()[active_np]
                    ), initial=0.0)),
                )
                done |= terminals > 0.5
                vector_steps = step + 1
        native_log = dict(vector.log())
    finally:
        vector.close()
    metrics = flatten_log(pufferl, native_log)
    completed = bool(
        metrics.get("env/n") == float(EPISODES)
        and metrics.get(f"env/gate_count{num_gates}_episode") == 1.0
        and all(math.isfinite(float(metrics.get(name, math.nan))) for name in (
            "env/success_rate", "env/gates_passed", "env/crash",
            "env/missed_gate", "env/timeout",
        ))
    )
    transport_pass = bool(
        completed and not nonfinite_action and action_envelope_violations == 0
        and executed_action_max_error <= MAX_EXECUTED_ACTION_ERROR
        and phase_changes_off_tick == 0 and phase_decreases == 0
        and phase_skips == 0 and raw_encoding_max_error <= 1e-6
        and metrics.get("env/out_of_order") == 0.0
    )
    return {
        "schema": "vq2_lc007_long_course_component_v1",
        "tag": f"{TAG}_{num_gates}g", "completed": completed,
        "transport_pass": transport_pass, "num_gates": num_gates,
        "agents": AGENTS, "episodes": EPISODES, "seed": SEEDS[num_gates],
        "vector_steps": vector_steps,
        "wall_time_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds, "loader_overrides": overrides,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_best_epoch": payload.get("best_epoch"),
        "conversion": conversion_metadata(payload),
        "actor_input_width": PHASE_LEGAL_OBS_SIZE,
        "actor_input_privileged_values": 0,
        "teacher_action_blend": float(environment["teacher_action_blend"]),
        "status_hold_steps": PUBLIC_STATUS_INTERVAL_STEPS,
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases, "phase_skips": phase_skips,
        "raw_progress_encoding_max_error": raw_encoding_max_error,
        "maximum_raw_index_distribution": {
            str(index): int((maximum_raw_index == index).sum())
            for index in range(num_gates + 1)
        },
        "maximum_held_index_distribution": {
            str(index): int((maximum_held_index == index).sum())
            for index in range(num_gates + 1)
        },
        "action_envelope_violations": action_envelope_violations,
        "executed_action_max_error": executed_action_max_error,
        "nonfinite_action": nonfinite_action, "metrics": metrics,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
    }


def run(*, output: Path, device_name: str, resume: bool) -> dict[str, Any]:
    from pufferlib import _C

    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC007 requires the float32 drone_race_vision backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC007 preregisters CUDA actor inference")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError("LC007 requires OMP_NUM_THREADS=32")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC007 requires OMP_DYNAMIC=FALSE")
    identity = source_identity()
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        report = json.loads(report_path.read_text())
        if report.get("source_identity") != identity:
            raise RuntimeError("LC007 completed report source changed")
        return report
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC007 output exists without an aggregate report")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    components = []
    for count in COUNTS:
        component_path = output / f"count_{count}.json"
        component = run_count(num_gates=count, device=device, identity=identity)
        write_json_once(component_path, component)
        components.append(component)
    report = {
        "schema": SCHEMA, "tag": TAG,
        "completed": all(item["completed"] for item in components),
        "diagnostic_valid": all(item["transport_pass"] for item in components),
        "counts": list(COUNTS),
        "components": {
            str(item["num_gates"]): {
                "path": f"count_{item['num_gates']}.json",
                "sha256": sha256_path(output / f"count_{item['num_gates']}.json"),
                "success_rate": item["metrics"]["env/success_rate"],
                "crash_rate": item["metrics"]["env/crash"],
                "miss_rate": item["metrics"]["env/missed_gate"],
                "mean_gates_passed": item["metrics"]["env/gates_passed"],
                "maximum_raw_index_distribution": item[
                    "maximum_raw_index_distribution"
                ],
            }
            for item in components
        },
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_identity": identity,
        "safety": {
            "teacher_plant_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Use the observed first-failure phases to generate local-start teacher "
            "records; this converted checkpoint has no live authority."
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
    report = run(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
