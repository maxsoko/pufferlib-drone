#!/usr/bin/env python3
"""Reproduce LC050's paired bracket concurrently in one CUDA context."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.torch_pufferl import _cpu_tensor
from pufferlib.vq2_informed import ACTION_HISTORY, ENV_OBS_SIZE, LEGAL_OBS_SIZE
from pufferlib.vq2_public_phase import OFFICIAL_PROGRESS_SCALE, PUBLIC_STATUS_INTERVAL_STEPS
from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.eval_vq2_lc001_long_course_oracle import load_config
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc007_converted_vg071_long_course as core
import scripts.eval_vq2_lc050_phase7_reduced_interpolation_bracket as lc050


TAG = "vq2_lc056_single_context_concurrent_bracket_001"
SCHEMA = "vq2_lc056_single_context_concurrent_bracket_report_v1"
COMPONENT_SCHEMA = "vq2_lc056_concurrent_component_v1"
ALPHAS = lc050.ALPHAS
AGENTS_PER_CANDIDATE = EPISODES_PER_CANDIDATE = 32
THREADS_PER_CANDIDATE = 32
NUM_GATES = 24
SEED = lc050.SEED
MAX_STEPS = 12_000
MINIMUM_SPEEDUP = 2.0
LC050_REPORT = lc050.DEFAULT_OUTPUT / "report.json"
LC050_REPORT_SHA256 = (
    "8bef58764fda323f9d4c61701bc21821b3c147a5f2edc12c56e440710ec2641b"
)
LC055_REJECTION = ROOT / "docs/vq2_lc055_chunked_collection_rejection_2026-07-31.json"
LC055_REJECTION_SHA256 = (
    "5a939c6030bdd0c7844a201aae30beb3bbcb857412c3aac3defed4df7a2731ae"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc056_single_context_concurrent_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc056_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc056_single_context_concurrent_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


@dataclass
class Candidate:
    alpha: float
    actor: Any
    payload: dict[str, Any]
    state_sha256: str


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        LC050_REPORT, LC055_REJECTION,
        lc050.PARENT_CHECKPOINT, lc050.PARENT_REPORT,
        lc050.LC040_REPORT, lc050.FIT_CHECKPOINT, lc050.FIT_REPORT,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc007_converted_vg071_long_course.py",
        ROOT / "scripts/eval_vq2_lc050_phase7_reduced_interpolation_bracket.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
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


def verify_inputs() -> dict[str, Any]:
    lc050.verify_inputs()
    if sha256_path(LC050_REPORT) != LC050_REPORT_SHA256:
        raise RuntimeError("LC056 bound LC050 report changed")
    if sha256_path(LC055_REJECTION) != LC055_REJECTION_SHA256:
        raise RuntimeError("LC056 bound LC055 rejection changed")
    prior = json.loads(LC050_REPORT.read_text())
    rejection = json.loads(LC055_REJECTION.read_text())
    if (
        prior.get("schema")
        != "vq2_lc050_phase7_reduced_interpolation_bracket_report_v1"
        or not prior.get("diagnostic_valid")
        or prior.get("wall_time_seconds") != 111.75675652897917
        or tuple(item.get("alpha") for item in prior.get("items", [])) != ALPHAS
        or rejection.get("schema") != "vq2_lc055_chunked_collection_rejection_v1"
        or not rejection.get("rejected")
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("feature_bytes") != 0
        or rejection.get("flight_sim_packets_sent") != 0
        or rejection.get("submission_authorized")
    ):
        raise RuntimeError("LC050/LC055 do not authorize LC056")
    return prior


def candidate_result_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    exact_fields = (
        "alpha", "candidate_state_sha256", "mean_gates_passed",
        "maximum_raw_index", "maximum_raw_index_distribution",
        "crash_rate", "miss_rate", "timeout_rate", "diagnostic_valid",
    )
    return all(actual.get(field) == expected.get(field) for field in exact_fields)


def build_candidates(device: torch.device) -> list[Candidate]:
    candidates = []
    for alpha in ALPHAS:
        lc050.configure_alpha(alpha)
        actor, payload, state_sha = lc050.build_candidate(device)
        candidates.append(Candidate(alpha, actor, payload, state_sha))
    return candidates


def build_vector():
    from pufferlib import _C, pufferl

    config, overrides = load_config(
        pufferl, num_gates=NUM_GATES, agents=AGENTS_PER_CANDIDATE,
        episodes=EPISODES_PER_CANDIDATE, seed=SEED,
        threads=THREADS_PER_CANDIDATE,
    )
    environment = config["env"]
    environment.update({
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
        "max_steps": MAX_STEPS,
        "time_limit_seconds": MAX_STEPS / 64.0,
    })
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS_PER_CANDIDATE or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("LC056 vector ABI changed")
    return vector, environment, overrides


def run_candidate(
    candidate: Candidate,
    vector: Any,
    environment: dict[str, Any],
    overrides: list[str],
    device: torch.device,
) -> dict[str, Any]:
    from pufferlib import pufferl

    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS_PER_CANDIDATE, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(
        vector.terminals_ptr, (AGENTS_PER_CANDIDATE,), torch.float32
    )
    actions_cpu = torch.zeros(
        (AGENTS_PER_CANDIDATE, ACTION_SIZE), dtype=torch.float32
    )
    state = candidate.actor.initial_state(AGENTS_PER_CANDIDATE, device=device)
    done = torch.zeros(AGENTS_PER_CANDIDATE, dtype=torch.bool)
    held = np.zeros(AGENTS_PER_CANDIDATE, dtype=np.float32)
    previous_held = held.copy()
    maximum_raw_index = np.zeros(AGENTS_PER_CANDIDATE, dtype=np.int32)
    maximum_held_index = np.zeros(AGENTS_PER_CANDIDATE, dtype=np.int32)
    phase_changes_off_tick = phase_decreases = phase_skips = 0
    raw_encoding_max_error = executed_action_max_error = 0.0
    action_envelope_violations = 0
    nonfinite_action = False
    vector_steps = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    native_log: dict[str, Any] = {}
    try:
        vector.reset()
        with torch.no_grad():
            for step in range(MAX_STEPS):
                active_cpu = ~done
                if not bool(active_cpu.any()):
                    break
                current = observations.numpy()
                raw = current[:, core.PHASE_PRIVILEGED_INDEX]
                held, sampled, encoding_error = core.update_held_progress(
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
                output, candidate_state = candidate.actor.forward_step(actor_input, state)
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
        metrics.get("env/n") == float(EPISODES_PER_CANDIDATE)
        and metrics.get(f"env/gate_count{NUM_GATES}_episode") == 1.0
        and all(math.isfinite(float(metrics.get(name, math.nan))) for name in (
            "env/success_rate", "env/gates_passed", "env/crash",
            "env/missed_gate", "env/timeout",
        ))
    )
    transport_pass = bool(
        completed and not nonfinite_action and action_envelope_violations == 0
        and executed_action_max_error <= core.MAX_EXECUTED_ACTION_ERROR
        and phase_changes_off_tick == 0 and phase_decreases == 0
        and phase_skips == 0 and raw_encoding_max_error <= 1e-6
        and metrics.get("env/out_of_order") == 0.0
    )
    distribution = {
        str(index): int((maximum_raw_index == index).sum())
        for index in range(NUM_GATES + 1)
    }
    maximum_index = max(int(index) for index, count in distribution.items() if count)
    return {
        "schema": COMPONENT_SCHEMA,
        "alpha": candidate.alpha,
        "completed": completed,
        "diagnostic_valid": transport_pass,
        "candidate_state_sha256": candidate.state_sha256,
        "mean_gates_passed": float(metrics["env/gates_passed"]),
        "maximum_raw_index": maximum_index,
        "maximum_raw_index_distribution": distribution,
        "maximum_held_index_distribution": {
            str(index): int((maximum_held_index == index).sum())
            for index in range(NUM_GATES + 1)
        },
        "crash_rate": float(metrics["env/crash"]),
        "miss_rate": float(metrics["env/missed_gate"]),
        "timeout_rate": float(metrics["env/timeout"]),
        "vector_steps": vector_steps,
        "wall_time_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "loader_overrides": overrides,
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases,
        "phase_skips": phase_skips,
        "raw_progress_encoding_max_error": raw_encoding_max_error,
        "action_envelope_violations": action_envelope_violations,
        "executed_action_max_error": executed_action_max_error,
        "nonfinite_action": nonfinite_action,
        "metrics": metrics,
        "safety": {
            "runtime_teacher_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
    }


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    from pufferlib import _C

    prior = verify_inputs()
    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC056 requires the float32 drone_race_vision backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC056 preregisters CUDA inference")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS_PER_CANDIDATE):
        raise RuntimeError("LC056 requires OMP_NUM_THREADS=32")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC056 requires OMP_DYNAMIC=FALSE")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC056 output exists without an aggregate report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    device = torch.device(device_name)
    candidates = build_candidates(device)
    vector_inputs = [build_vector() for _ in candidates]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = [
            pool.submit(run_candidate, candidate, vector, environment, overrides, device)
            for candidate, (vector, environment, overrides)
            in zip(candidates, vector_inputs)
        ]
        components = [future.result() for future in futures]
    wall = time.perf_counter() - started
    expected = prior["items"]
    parity = [
        candidate_result_matches(actual, reference)
        for actual, reference in zip(components, expected)
    ]
    speedup = float(prior["wall_time_seconds"]) / wall
    for component in components:
        slug = f"a{component['alpha']:.3f}".replace(".", "p")
        write_json_once(output / f"{slug}.json", component)
    admitted = bool(
        all(parity)
        and all(component["diagnostic_valid"] for component in components)
        and speedup >= MINIMUM_SPEEDUP
    )
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted,
        "exact_lc050_candidate_parity": parity,
        "all_candidate_parity": all(parity),
        "single_cuda_context": True,
        "concurrent_native_vectors": len(candidates),
        "threads_per_vector": THREADS_PER_CANDIDATE,
        "sequential_lc050_wall_time_seconds": prior["wall_time_seconds"],
        "concurrent_wall_time_seconds": wall,
        "wall_time_speedup": speedup,
        "minimum_speedup": MINIMUM_SPEEDUP,
        "items": components,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Use this one-context concurrent layout for future four-candidate offline brackets."
            if admitted else "Reject LC056 and retain sequential one-context brackets."
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
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
