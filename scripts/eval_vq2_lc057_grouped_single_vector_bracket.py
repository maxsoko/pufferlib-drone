#!/usr/bin/env python3
"""Reproduce LC050 in one 128-env vector with four paired seed groups."""

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
from pufferlib.vq2_public_phase import OFFICIAL_PROGRESS_SCALE, PUBLIC_STATUS_INTERVAL_STEPS
from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.eval_vq2_lc001_long_course_oracle import load_config
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc007_converted_vg071_long_course as core
import scripts.eval_vq2_lc050_phase7_reduced_interpolation_bracket as lc050
import scripts.eval_vq2_lc056_single_context_concurrent_bracket as lc056


TAG = "vq2_lc057_grouped_single_vector_bracket_001"
SCHEMA = "vq2_lc057_grouped_single_vector_bracket_report_v1"
ALPHAS = lc050.ALPHAS
GROUP_SIZE = 32
GROUPS = len(ALPHAS)
TOTAL_AGENTS = GROUP_SIZE * GROUPS
EPISODES = TOTAL_AGENTS
THREADS = 128
NUM_GATES = 24
SEED = lc050.SEED
MAX_STEPS = 12_000
MINIMUM_SPEEDUP = 2.0
LC050_REPORT = lc050.DEFAULT_OUTPUT / "report.json"
LC050_REPORT_SHA256 = lc056.LC050_REPORT_SHA256
LC056_REJECTION = ROOT / "docs/vq2_lc056_concurrent_vector_rejection_2026-07-31.json"
LC056_REJECTION_SHA256 = (
    "4e9f4dc2544e8029776752e8a8359c2efa703e270b8c59f558982405c8a5be73"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc057_grouped_single_vector_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc057_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc057_grouped_single_vector_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        LC050_REPORT, LC056_REJECTION,
        lc050.PARENT_CHECKPOINT, lc050.PARENT_REPORT,
        lc050.LC040_REPORT, lc050.FIT_CHECKPOINT, lc050.FIT_REPORT,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc050_phase7_reduced_interpolation_bracket.py",
        ROOT / "scripts/eval_vq2_lc056_single_context_concurrent_bracket.py",
        ROOT / "src/vecenv.h", ROOT / "src/bindings.cu",
        ROOT / "src/bindings_cpu.cpp",
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
    prior = lc056.verify_inputs()
    if sha256_path(LC050_REPORT) != LC050_REPORT_SHA256:
        raise RuntimeError("LC057 bound LC050 report changed")
    if sha256_path(LC056_REJECTION) != LC056_REJECTION_SHA256:
        raise RuntimeError("LC057 bound LC056 rejection changed")
    rejection = json.loads(LC056_REJECTION.read_text())
    if (
        rejection.get("schema") != "vq2_lc056_concurrent_vector_rejection_v1"
        or not rejection.get("rejected")
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("minimum_observed_wall_time_seconds", 0) < 235
        or rejection.get("flight_sim_packets_sent") != 0
        or rejection.get("submission_authorized")
    ):
        raise RuntimeError("LC056 does not authorize LC057")
    return prior


def group_slice(group: int) -> slice:
    if group < 0 or group >= GROUPS:
        raise ValueError("candidate group is outside the bracket")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def build_vector():
    from pufferlib import _C, pufferl

    config, overrides = load_config(
        pufferl, num_gates=NUM_GATES, agents=TOTAL_AGENTS,
        episodes=EPISODES, seed=SEED, threads=THREADS,
    )
    config["vec"]["env_seed_group_size"] = GROUP_SIZE
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
    if (
        vector.total_agents != TOTAL_AGENTS
        or vector.obs_size != ENV_OBS_SIZE
        or not hasattr(vector, "eval_log_range")
    ):
        vector.close()
        raise RuntimeError("LC057 grouped vector ABI changed")
    return vector, environment, overrides


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    prior = verify_inputs()
    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC057 requires the float32 drone_race_vision backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC057 preregisters CUDA inference")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError("LC057 requires OMP_NUM_THREADS=128")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC057 requires OMP_DYNAMIC=FALSE")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC057 output exists without an aggregate report")
    output.mkdir(parents=True, exist_ok=True)

    identity = source_identity()
    device = torch.device(device_name)
    candidates = lc056.build_candidates(device)
    vector, environment, overrides = build_vector()
    observations = _cpu_tensor(
        vector.obs_ptr, (TOTAL_AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (TOTAL_AGENTS,), torch.float32)
    actions_cpu = torch.zeros((TOTAL_AGENTS, ACTION_SIZE), dtype=torch.float32)
    states = [candidate.actor.initial_state(GROUP_SIZE, device=device)
              for candidate in candidates]
    done = torch.zeros(TOTAL_AGENTS, dtype=torch.bool)
    held = np.zeros(TOTAL_AGENTS, dtype=np.float32)
    previous_held = held.copy()
    maximum_raw_index = np.zeros(TOTAL_AGENTS, dtype=np.int32)
    maximum_held_index = np.zeros(TOTAL_AGENTS, dtype=np.int32)
    phase_changes_off_tick = np.zeros(GROUPS, dtype=np.int64)
    phase_decreases = np.zeros(GROUPS, dtype=np.int64)
    phase_skips = np.zeros(GROUPS, dtype=np.int64)
    raw_encoding_max_error = np.zeros(GROUPS, dtype=np.float64)
    action_envelope_violations = np.zeros(GROUPS, dtype=np.int64)
    executed_action_max_error = np.zeros(GROUPS, dtype=np.float64)
    nonfinite_action = np.zeros(GROUPS, dtype=bool)
    inference_seconds = np.zeros(GROUPS, dtype=np.float64)
    vector_steps = 0
    group_logs: list[dict[str, Any]] = []
    aggregate_log: dict[str, Any] = {}
    initial_groups_exact = False
    started = time.perf_counter()
    try:
        vector.reset()
        initial = observations.numpy()
        initial_groups_exact = all(
            np.array_equal(initial[group_slice(0)], initial[group_slice(group)])
            for group in range(1, GROUPS)
        )
        if not initial_groups_exact:
            raise RuntimeError("LC057 seed groups do not begin identically")
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
                delta_index = np.rint(
                    (held - previous_held) * OFFICIAL_PROGRESS_SCALE
                )
                raw_indices = np.rint(raw * OFFICIAL_PROGRESS_SCALE)
                encoding_errors = np.abs(raw - raw_indices / OFFICIAL_PROGRESS_SCALE)
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
                        float(encoding_errors[selected].max(initial=0.0)),
                    )
                previous_held = held.copy()
                raw_index = raw_indices.astype(np.int32)
                held_index = np.rint(held * OFFICIAL_PROGRESS_SCALE).astype(np.int32)
                maximum_raw_index = np.maximum(maximum_raw_index, raw_index)
                maximum_held_index = np.maximum(maximum_held_index, held_index)

                for group, candidate in enumerate(candidates):
                    selected = group_slice(group)
                    active_group_cpu = active_cpu[selected]
                    legal = observations[selected, :LEGAL_OBS_SIZE].to(device)
                    progress = torch.from_numpy(held[selected, None]).to(device)
                    actor_input = torch.cat((legal, progress), dim=1)
                    active_group = active_group_cpu.to(device)
                    inference_started = time.perf_counter()
                    actor_output, candidate_state = candidate.actor.forward_step(
                        actor_input, states[group]
                    )
                    states[group] = preserve_frozen_state(
                        states[group], candidate_state, active_group
                    )
                    action = torch.where(
                        active_group[:, None], actor_output.mean,
                        torch.zeros_like(actor_output.mean),
                    )
                    if not bool(torch.isfinite(action[active_group]).all()):
                        nonfinite_action[group] = True
                    inference_seconds[group] += time.perf_counter() - inference_started
                    active_actions = action[active_group].detach().cpu()
                    action_envelope_violations[group] += int(
                        (active_actions.abs() > 1.0 + 1e-6).sum()
                    )
                    actions_cpu[selected].copy_(action.detach().cpu())
                if nonfinite_action.any():
                    break
                vector.cpu_step(actions_cpu.data_ptr())
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                action_np = actions_cpu.numpy()
                for group in range(GROUPS):
                    selected = group_slice(group)
                    group_active = active_np[selected]
                    executed_action_max_error[group] = max(
                        executed_action_max_error[group],
                        float(np.max(np.abs(
                            executed[selected][group_active]
                            - action_np[selected][group_active]
                        ), initial=0.0)),
                    )
                done |= terminals > 0.5
                vector_steps = step + 1
        group_logs = [
            dict(vector.eval_log_range(group * GROUP_SIZE, GROUP_SIZE))
            for group in range(GROUPS)
        ]
        aggregate_log = dict(vector.log())
    finally:
        vector.close()

    wall = time.perf_counter() - started
    items = []
    for group, (candidate, native_log) in enumerate(zip(candidates, group_logs)):
        metrics = flatten_log(pufferl, native_log)
        selected = group_slice(group)
        distribution = {
            str(index): int((maximum_raw_index[selected] == index).sum())
            for index in range(NUM_GATES + 1)
        }
        maximum_index = max(int(index) for index, count in distribution.items() if count)
        completed = bool(
            metrics.get("env/n") == float(GROUP_SIZE)
            and metrics.get(f"env/gate_count{NUM_GATES}_episode") == 1.0
            and all(math.isfinite(float(metrics.get(name, math.nan))) for name in (
                "env/success_rate", "env/gates_passed", "env/crash",
                "env/missed_gate", "env/timeout",
            ))
        )
        transport = bool(
            completed and not nonfinite_action[group]
            and action_envelope_violations[group] == 0
            and executed_action_max_error[group] <= core.MAX_EXECUTED_ACTION_ERROR
            and phase_changes_off_tick[group] == 0
            and phase_decreases[group] == 0 and phase_skips[group] == 0
            and raw_encoding_max_error[group] <= 1e-6
            and metrics.get("env/out_of_order") == 0.0
        )
        items.append({
            "alpha": candidate.alpha,
            "candidate_state_sha256": candidate.state_sha256,
            "mean_gates_passed": float(metrics["env/gates_passed"]),
            "maximum_raw_index": maximum_index,
            "maximum_raw_index_distribution": distribution,
            "crash_rate": float(metrics["env/crash"]),
            "miss_rate": float(metrics["env/missed_gate"]),
            "timeout_rate": float(metrics["env/timeout"]),
            "diagnostic_valid": transport,
            "inference_seconds": float(inference_seconds[group]),
            "executed_action_max_error": float(executed_action_max_error[group]),
            "phase_changes_off_tick": int(phase_changes_off_tick[group]),
            "phase_decreases": int(phase_decreases[group]),
            "phase_skips": int(phase_skips[group]),
            "raw_progress_encoding_max_error": float(raw_encoding_max_error[group]),
        })
    parity = [
        lc056.candidate_result_matches(actual, expected)
        for actual, expected in zip(items, prior["items"])
    ]
    speedup = float(prior["wall_time_seconds"]) / wall
    aggregate_metrics = flatten_log(pufferl, aggregate_log)
    admitted = bool(
        initial_groups_exact and all(parity)
        and all(item["diagnostic_valid"] for item in items)
        and aggregate_metrics.get("env/n") == float(EPISODES)
        and speedup >= MINIMUM_SPEEDUP
    )
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted,
        "exact_lc050_candidate_parity": parity,
        "all_candidate_parity": all(parity),
        "initial_seed_groups_exact": initial_groups_exact,
        "single_cuda_context": True,
        "single_native_vector": True,
        "seed_group_size": GROUP_SIZE,
        "candidate_groups": GROUPS,
        "total_agents": TOTAL_AGENTS,
        "threads": THREADS,
        "vector_steps": vector_steps,
        "loader_overrides": overrides,
        "sequential_lc050_wall_time_seconds": prior["wall_time_seconds"],
        "grouped_wall_time_seconds": wall,
        "wall_time_speedup": speedup,
        "minimum_speedup": MINIMUM_SPEEDUP,
        "items": items,
        "aggregate_metrics": aggregate_metrics,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Use grouped single-vector execution for future paired four-candidate brackets."
            if admitted else "Reject LC057 and retain the sequential single-vector layout."
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
