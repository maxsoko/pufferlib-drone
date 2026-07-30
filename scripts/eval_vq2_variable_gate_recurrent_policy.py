#!/usr/bin/env python3
"""Teacher-free held-out variable-course screen for the VG005 actor."""

from __future__ import annotations

import argparse
import json
import math
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
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP, PUBLIC_STATUS_INTERVAL_STEPS
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import (
    PHASE_LEGAL_OBS_SIZE,
    VQ2PhaseRecurrentActor,
)
from scripts.collect_vq2_variable_gate_oracle_bc_dataset import (
    PHASE_PRIVILEGED_INDEX,
    update_held_phase,
)
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import (
    load_variable_config,
    sha256_path,
    write_json_atomic,
    write_json_once,
)


TAG = "vq2_vg006_variable_gate_recurrent_teacher_free_256"
SCHEMA = "vq2_variable_gate_recurrent_teacher_free_count_screen_v1"
COUNTS = (5, 8, 11, 12)
AGENTS = 64
EPISODES_PER_COUNT = 64
TOTAL_EPISODES = EPISODES_PER_COUNT * len(COUNTS)
SEEDS = {count: 429032 + count for count in COUNTS}
MINIMUM_SUCCESS_RATE = 0.90
MAX_EXECUTED_ACTION_ERROR = 5e-5
REQUIRE_ZERO_CROSSING_MARGIN = True
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg005_variable_gate_recurrent_bc_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "f686a35dc923137bc40fd6feca459e52e5585d5b6fa047b7f77a171e2832a6c3"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "bdcd2b38ea3537f2c250281c63f2e6aeca0ad87c97521339f098eec7ff474e5b"
)
PREREGISTRATION = (
    ROOT
    / "docs/vq2_vg006_variable_gate_recurrent_teacher_free_preregistration_2026-07-29.md"
)
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
RUNNER = ROOT / "scripts/run_vq2_vg006_vast.sh"
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()


def runtime_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "numpy": np.__version__,
    }


def load_actor(device: torch.device) -> tuple[VQ2PhaseRecurrentActor, dict[str, Any]]:
    if not CHECKPOINT_SHA256 or sha256_path(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("VG005 checkpoint hash mismatch")
    if not TRAIN_REPORT_SHA256 or sha256_path(TRAIN_REPORT) != TRAIN_REPORT_SHA256:
        raise RuntimeError("VG005 training report hash mismatch")
    train_report = json.loads(TRAIN_REPORT.read_text())
    if (
        not train_report.get("completed")
        or train_report.get("minimum_transition_window_exposure", 0.0) < 3.0
    ):
        raise RuntimeError("VG005 training did not complete its fixed contract")
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if payload.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1":
        raise RuntimeError("unsupported VG005 checkpoint schema")
    contract = payload.get("model", {})
    expected = {
        "class": "VQ2PhaseRecurrentActor",
        "legal_observation_size": PHASE_LEGAL_OBS_SIZE,
        "legacy_legal_observation_size": LEGAL_OBS_SIZE,
        "public_status_values": 1,
        "public_phase_encoding": "clamp(active_gate_index,0,16)/16",
        "public_phase_rate_hz": 4,
        "action_size": ACTION_SIZE,
        "hidden_size": 256,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise RuntimeError("VG005 checkpoint actor contract changed")
    safety = payload.get("safety", {})
    if (
        safety.get("actor_input_privileged_values") != 0
        or safety.get("teacher_blend") != 0.0
        or payload.get("minimum_transition_window_exposure", 0.0) < 3.0
    ):
        raise RuntimeError("VG005 checkpoint safety contract changed")
    actor = VQ2PhaseRecurrentActor(
        hidden_size=int(contract["hidden_size"]),
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(payload["model_state"])
    actor.eval()
    return actor, payload


def teacher_free_config(
    pufferl_module: Any,
    *,
    num_gates: int,
) -> tuple[dict[str, Any], list[str]]:
    seed = SEEDS[num_gates]
    config, overrides = load_variable_config(
        pufferl_module,
        agents=AGENTS,
        episodes=EPISODES_PER_COUNT,
        seed=seed,
        num_gates=num_gates,
    )
    environment = config["env"]
    environment.update({
        "num_gates": num_gates,
        "num_gates_per_env_randomize": 0,
        "num_gates_per_env_min": num_gates,
        "num_gates_per_env_max": num_gates,
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
        "observable_gate_index_denominator": float(ENGINE_GATE_CAP),
    })
    return config, overrides


def completed_count_screen(metrics: dict[str, float], *, num_gates: int) -> bool:
    required_finite = (
        "env/n",
        "env/success_rate",
        "env/gates_passed",
        "env/crash",
        "env/timeout",
        "env/missed_gate",
        "env/out_of_order",
        "env/valid_run_rate",
    )
    return bool(
        metrics.get("env/n") == float(EPISODES_PER_COUNT)
        and all(np.isfinite(metrics.get(name, math.nan)) for name in required_finite)
        and metrics.get(f"env/gate_count{num_gates}_episode") == 1.0
    )


def count_safety_passes(report: dict[str, Any]) -> bool:
    metrics = report["metrics"]
    return bool(
        report.get("completed")
        and report.get("teacher_action_blend") == 0.0
        and not report.get("nonfinite_action")
        and report.get("action_envelope_violations") == 0
        and report.get("executed_action_max_error", math.inf)
        <= MAX_EXECUTED_ACTION_ERROR
        and report.get("phase_changes_off_tick") == 0
        and report.get("phase_decreases") == 0
        and report.get("phase_skips") == 0
        and report.get("raw_phase_encoding_max_error", math.inf) <= 1e-6
        and metrics.get("env/crash") == 0.0
        and metrics.get("env/out_of_order") == 0.0
        and (
            not REQUIRE_ZERO_CROSSING_MARGIN
            or metrics.get("env/crossing_margin_violation") == 0.0
        )
        and metrics.get("env/action_envelope_violation") == 0.0
        and metrics.get("env/wire_rate_envelope_violation") == 0.0
        and metrics.get("env/thrust_envelope_violation") == 0.0
    )


def aggregate_admission_passes(reports: list[dict[str, Any]]) -> bool:
    if [report.get("num_gates") for report in reports] != list(COUNTS):
        return False
    successes = sum(
        report["metrics"]["env/success_rate"] * EPISODES_PER_COUNT
        for report in reports
    )
    return bool(
        all(count_safety_passes(report) for report in reports)
        and successes / TOTAL_EPISODES + 1e-12 >= MINIMUM_SUCCESS_RATE
    )


def run_count(
    *,
    num_gates: int,
    device: torch.device,
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    actor, payload = load_actor(device)
    config, overrides = teacher_free_config(pufferl, num_gates=num_gates)
    environment = config["env"]
    if any(
        float(environment[name]) != 0.0
        for name in (
            "teacher_action_blend",
            "teacher_course_spline",
            "teacher_segment_minimum_jerk",
            "teacher_alignment_governor",
            "w_action_teacher",
        )
    ):
        raise RuntimeError("VG006 screen inherited a teacher path")
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("VG006 native vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions_cpu = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    state = actor.initial_state(AGENTS, device=device)
    done = torch.zeros(AGENTS, dtype=torch.bool)
    held_phase = np.zeros(AGENTS, dtype=np.float32)
    previous_held_phase = np.zeros(AGENTS, dtype=np.float32)
    maximum_raw_phase = np.zeros(AGENTS, dtype=np.float32)
    maximum_held_phase = np.zeros(AGENTS, dtype=np.float32)
    action_sum = torch.zeros(ACTION_SIZE, dtype=torch.float64)
    action_square_sum = torch.zeros(ACTION_SIZE, dtype=torch.float64)
    action_min = torch.full((ACTION_SIZE,), float("inf"))
    action_max = torch.full((ACTION_SIZE,), float("-inf"))
    action_samples = 0
    action_envelope_violations = 0
    executed_action_max_error = 0.0
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_skips = 0
    phase_samples = 0
    raw_phase_encoding_max_error = 0.0
    nonfinite_action = False
    inference_seconds = 0.0
    vector_steps = 0
    native_log: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        vector.reset()
        with torch.no_grad():
            for step in range(int(environment["max_steps"])):
                active_cpu = ~done
                if not bool(active_cpu.any()):
                    break
                current = observations.numpy()
                raw_phase = current[:, PHASE_PRIVILEGED_INDEX]
                held_phase, sampled, encoding_error = update_held_phase(
                    raw_phase, held_phase, step=step
                )
                raw_phase_encoding_max_error = max(
                    raw_phase_encoding_max_error, encoding_error
                )
                active_np = active_cpu.numpy()
                changed = np.abs(held_phase - previous_held_phase) > 1e-7
                if changed.any() and not sampled:
                    phase_changes_off_tick += int(changed.sum())
                delta = held_phase[active_np] - previous_held_phase[active_np]
                phase_decreases += int((delta < -1e-7).sum())
                phase_skips += int((delta > 1.0 / ENGINE_GATE_CAP + 1e-7).sum())
                if sampled:
                    phase_samples += int(active_np.sum())
                previous_held_phase = held_phase.copy()
                maximum_raw_phase = np.maximum(maximum_raw_phase, raw_phase)
                maximum_held_phase = np.maximum(maximum_held_phase, held_phase)

                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                phase_tensor = torch.from_numpy(held_phase[:, None]).to(device)
                actor_input = torch.cat((legal, phase_tensor), dim=1)
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
                selected = action[active].detach().cpu()
                inference_seconds += time.perf_counter() - inference_started
                action_envelope_violations += int((selected.abs() > 1.0 + 1e-6).sum())
                action_sum += selected.double().sum(0)
                action_square_sum += selected.double().square().sum(0)
                action_min = torch.minimum(action_min, selected.amin(0))
                action_max = torch.maximum(action_max, selected.amax(0))
                action_samples += int(selected.shape[0])
                actions_cpu.copy_(action.detach().cpu())
                vector.cpu_step(actions_cpu.data_ptr())
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(
                        np.max(
                            np.abs(executed[active_np] - actions_cpu.numpy()[active_np]),
                            initial=0.0,
                        )
                    ),
                )
                done |= terminals > 0.5
                vector_steps = step + 1
        native_log = dict(vector.log())
    finally:
        vector.close()
    metrics = flatten_log(pufferl, native_log)
    completed = completed_count_screen(metrics, num_gates=num_gates)
    action_mean = action_sum / max(action_samples, 1)
    action_variance = torch.clamp(
        action_square_sum / max(action_samples, 1) - action_mean.square(), min=0.0
    )
    return {
        "schema": SCHEMA,
        "tag": f"{TAG}_{num_gates}g",
        "completed": completed,
        "num_gates": num_gates,
        "agents": AGENTS,
        "episodes": EPISODES_PER_COUNT,
        "seed": SEEDS[num_gates],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_best_epoch": payload["best_epoch"],
        "crossing_margin_admission_predicate": REQUIRE_ZERO_CROSSING_MARGIN,
        "vector_steps": vector_steps,
        "wall_time_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "loader_overrides": overrides,
        "teacher_action_blend": float(environment["teacher_action_blend"]),
        "actor_input_width": PHASE_LEGAL_OBS_SIZE,
        "actor_input_privileged_values": 0,
        "native_public_phase_mirror_index": PHASE_PRIVILEGED_INDEX,
        "status_hold_steps": PUBLIC_STATUS_INTERVAL_STEPS,
        "phase_samples": phase_samples,
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases,
        "phase_skips": phase_skips,
        "raw_phase_encoding_max_error": raw_phase_encoding_max_error,
        "maximum_raw_public_index_distribution": {
            str(index): int(
                np.isclose(maximum_raw_phase * ENGINE_GATE_CAP, index, atol=1e-5).sum()
            )
            for index in range(ENGINE_GATE_CAP + 1)
        },
        "maximum_held_public_index_distribution": {
            str(index): int(
                np.isclose(maximum_held_phase * ENGINE_GATE_CAP, index, atol=1e-5).sum()
            )
            for index in range(ENGINE_GATE_CAP + 1)
        },
        "action_samples": action_samples,
        "action_envelope_violations": action_envelope_violations,
        "executed_action_max_error": executed_action_max_error,
        "nonfinite_action": nonfinite_action,
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": torch.sqrt(action_variance).tolist(),
        "metrics": metrics,
        "source_identity": source_identity,
        "safety": {
            "teacher_labels_written": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("VG006 requires drone_race_vision backend")
    if getattr(_C, "precision_bytes", None) != 4:
        raise RuntimeError("VG006 requires float32 native binding")
    extension = Path(_C.__file__).resolve()
    paths = (
        Path(__file__).resolve(),
        PREREGISTRATION,
        RUNNER,
        ROOT / "scripts/collect_vq2_variable_gate_oracle_bc_dataset.py",
        ROOT / "scripts/eval_vq2_variable_gate_oracle.py",
        ROOT / "scripts/eval_vq2_recurrent_policy.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
        ROOT / "config/drone_race_vq2_informed_dreamer.ini",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        CHECKPOINT,
        TRAIN_REPORT,
        *EXTRA_SOURCE_PATHS,
    )
    sources = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in paths
    }
    sources["compiled_extension"] = sha256_path(extension)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "source_commit": commit,
        "source_sha256": sources,
        "runtime": runtime_manifest(),
        "compiled_extension_path": str(extension),
    }


def run_admission(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG006 preregistration is missing")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG006 preregisters CUDA inference")
    identity = source_identity()
    state_path = output / "state.json"
    report_path = output / "report.json"
    state_identity = {
        "schema": "vq2_variable_gate_recurrent_screen_state_v1",
        "tag": TAG,
        "counts": list(COUNTS),
        "agents_per_count": AGENTS,
        "episodes_per_count": EPISODES_PER_COUNT,
        "seeds": {str(key): value for key, value in SEEDS.items()},
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "admission_contract": {
            "minimum_success_rate": MINIMUM_SUCCESS_RATE,
            "require_zero_crossing_margin": REQUIRE_ZERO_CROSSING_MARGIN,
        },
        **identity,
        "safety": {
            "teacher_labels_written": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        report = json.loads(report_path.read_text())
        if report.get("source_identity") != identity:
            raise RuntimeError("VG006 completed report source mismatch")
        return report
    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"VG006 state exists at {state_path}")
        state = json.loads(state_path.read_text())
        for key, expected in state_identity.items():
            if state.get(key) != expected:
                raise RuntimeError(f"VG006 resume mismatch for {key}")
        if state.get("status") != "screening":
            raise RuntimeError("VG006 terminal state has no aggregate report")
    else:
        if output.exists():
            raise RuntimeError("VG006 output exists without source-locked state")
        output.mkdir(parents=True)
        state = {**state_identity, "status": "screening", "completed_counts": []}
        write_json_atomic(state_path, state)

    device = torch.device(device_name)
    reports: list[dict[str, Any]] = []
    for count in COUNTS:
        count_path = output / f"count_{count}.json"
        if count_path.is_file():
            count_report = json.loads(count_path.read_text())
            if count_report.get("source_identity") != identity:
                raise RuntimeError(f"VG006 count {count} source mismatch")
        else:
            count_report = run_count(
                num_gates=count,
                device=device,
                source_identity=identity,
            )
            write_json_once(count_path, count_report)
        reports.append(count_report)
        completed = [report["num_gates"] for report in reports]
        state.update({"status": "screening", "completed_counts": completed})
        write_json_atomic(state_path, state)

    successes = sum(
        report["metrics"]["env/success_rate"] * EPISODES_PER_COUNT
        for report in reports
    )
    admitted = aggregate_admission_passes(reports)
    aggregate = {
        "schema": "vq2_variable_gate_recurrent_teacher_free_admission_v1",
        "tag": TAG,
        "completed": True,
        "admitted": admitted,
        "counts": list(COUNTS),
        "episodes_per_count": EPISODES_PER_COUNT,
        "total_episodes": TOTAL_EPISODES,
        "successes": int(round(successes)),
        "success_rate": successes / TOTAL_EPISODES,
        "minimum_success_rate": MINIMUM_SUCCESS_RATE,
        "zero_crash": all(report["metrics"]["env/crash"] == 0.0 for report in reports),
        "crossing_margin_admission_predicate": REQUIRE_ZERO_CROSSING_MARGIN,
        "crossing_margin_violation_rate": sum(
            report["metrics"]["env/crossing_margin_violation"]
            * EPISODES_PER_COUNT
            for report in reports
        )
        / TOTAL_EPISODES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "count_reports": {
            str(report["num_gates"]): {
                "path": f"count_{report['num_gates']}.json",
                "sha256": sha256_path(output / f"count_{report['num_gates']}.json"),
                "success_rate": report["metrics"]["env/success_rate"],
                "crash": report["metrics"]["env/crash"],
                "gates_passed": report["metrics"]["env/gates_passed"],
            }
            for report in reports
        },
        "source_identity": identity,
        "safety": state_identity["safety"],
    }
    write_json_once(report_path, aggregate)
    state.update({
        "status": "admitted" if admitted else "rejected",
        "completed_counts": list(COUNTS),
        "success_rate": aggregate["success_rate"],
        "report_sha256": sha256_path(report_path),
    })
    write_json_atomic(state_path, state)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run_admission(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
