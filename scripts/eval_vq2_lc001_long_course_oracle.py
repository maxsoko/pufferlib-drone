#!/usr/bin/env python3
"""Prove the native teacher and unbounded public phase on 20--24+ gates."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
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

from pufferlib.vq2_informed import configure_full_start_evaluation
from pufferlib.vq2_public_phase import LONG_COURSE_GATE_CAP, OFFICIAL_PROGRESS_SCALE
from scripts.eval_vq2_native_oracle import (
    BACKEND_ENV_NAME,
    ENV_NAME,
    fixed_environment,
    flatten_log,
    native_step_budget,
)
from scripts.eval_vq2_variable_gate_oracle import variable_oracle_passes


TAG = "vq2_lc002_long_course_oracle_throughput"
MIN_GATES = 20
DEFAULT_COUNTS = (20, 24)
DT_SECONDS = 1.0 / 64.0
EPISODE_SECONDS = 600.0
DEFAULT_SEED = 431001
DEFAULT_AGENTS = 64
DEFAULT_THREADS = 32
MINIMUM_PROJECTED_SPEEDUP = 10.0
# The vision binding keeps native gate phase training-only at the penultimate
# observation value. Deployment samples the corresponding official status at
# 4 Hz and appends the same unsaturated index/6 scalar outside this binding.
NATIVE_PROGRESS_INDEX_FROM_END = -2
DEFAULT_OUTPUT_ROOT = (
    ROOT / "logs" / "drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc002_long_course_oracle_throughput_001"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "config/drone_race_vq2_informed_dreamer.ini",
        ROOT / "scripts/eval_vq2_native_oracle.py",
        ROOT / "scripts/eval_vq2_variable_gate_oracle.py",
        Path(__file__).resolve(),
    )
    extension = Path(_C.__file__).resolve()
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_sha256": {
            **{str(path.relative_to(ROOT)): sha256_path(path) for path in paths},
            "compiled_extension": sha256_path(extension),
        },
        "compiled_extension_path": str(extension),
    }


def runtime_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
    }


def long_course_environment(num_gates: int) -> dict[str, int | float]:
    """Return the source-locked LC001 teacher course."""

    if num_gates < MIN_GATES or num_gates > LONG_COURSE_GATE_CAP:
        raise ValueError(
            f"LC001 gate count must be in [{MIN_GATES}, {LONG_COURSE_GATE_CAP}]"
        )
    values = fixed_environment(
        "governed",
        target_speed_m_s=2.0,
        episode_seconds=EPISODE_SECONDS,
        randomized_course=True,
    )
    values.update({
        "num_gates": num_gates,
        "num_gates_per_env_randomize": 0,
        "num_gates_per_env_min": num_gates,
        "num_gates_per_env_max": num_gates,
        "num_gates_per_env_seed": DEFAULT_SEED,
        "teacher_roll_until_gate_index": num_gates,
        "gate_position_require_valid_course": 1,
        "gate_position_min_forward_gap_m": 8.0,
        "gate_position_max_segment_distance_m": 40.0,
        "gate_position_resample_attempts": 64,
        "pos_bound": max(800.0, num_gates * 40.0),
        "max_steps": int(round(EPISODE_SECONDS / DT_SECONDS)),
        "time_limit_seconds": EPISODE_SECONDS,
        "observable_gate_progress": 1,
        "observable_gate_index_denominator": OFFICIAL_PROGRESS_SCALE,
        "observable_gate_progress_unbounded": 1,
    })
    return values


def loader_overrides(*, agents: int, seed: int, threads: int) -> list[str]:
    buffers = min(8, agents)
    while agents % buffers:
        buffers -= 1
    return [
        "--seed", str(seed),
        "--vec.total-agents", str(agents),
        "--vec.num-buffers", str(buffers),
        "--vec.num-threads", str(threads),
    ]


def load_config(
    pufferl_module: Any,
    *,
    num_gates: int,
    agents: int,
    episodes: int,
    seed: int,
    threads: int,
) -> tuple[dict[str, Any], list[str]]:
    if agents <= 0 or episodes <= 0 or episodes % agents:
        raise ValueError("episodes must be a positive multiple of agents")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if os.environ.get("OMP_NUM_THREADS") != str(threads):
        raise RuntimeError(
            "LC002 requires OMP_NUM_THREADS to equal the source-locked thread count"
        )
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC002 requires OMP_DYNAMIC=FALSE")
    overrides = loader_overrides(agents=agents, seed=seed, threads=threads)
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *overrides]
        config = pufferl_module.load_config(ENV_NAME)
    finally:
        sys.argv = saved_argv
    if config.get("backend_env_name") != BACKEND_ENV_NAME:
        raise RuntimeError("LC001 requires the drone_race_vision backend")
    config.setdefault("env", {}).update(long_course_environment(num_gates))
    configure_full_start_evaluation(
        config,
        episodes_per_agent=episodes // agents,
        episode_offset=0,
    )
    return config, overrides


def observation_view(vector: Any) -> np.ndarray:
    count = vector.total_agents * vector.obs_size
    raw = (ctypes.c_float * count).from_address(vector.obs_ptr)
    return np.ctypeslib.as_array(raw).reshape(vector.total_agents, vector.obs_size)


def run_count(
    *,
    num_gates: int,
    agents: int,
    episodes: int,
    seed: int,
    threads: int,
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError(f"compiled backend is {_C.env_name!r}, not {BACKEND_ENV_NAME!r}")
    if getattr(_C, "precision_bytes", None) != 4:
        raise RuntimeError("LC001 requires the float32 vision binding")
    identity = source_identity()
    config, overrides = load_config(
        pufferl,
        num_gates=num_gates,
        agents=agents,
        episodes=episodes,
        seed=seed,
        threads=threads,
    )
    torch.set_num_threads(threads)
    vector = _C.create_vec(config, gpu=0)
    actions = torch.zeros((vector.total_agents, vector.num_atns), dtype=torch.float32)
    max_public_progress = -math.inf
    native_steps = native_step_budget(
        max_steps=int(config["env"]["max_steps"]),
        episodes=episodes,
        agents=agents,
    )
    started = time.perf_counter()
    try:
        vector.reset()
        observations = observation_view(vector)
        for _ in range(native_steps):
            max_public_progress = max(
                max_public_progress,
                float(np.max(observations[:, NATIVE_PROGRESS_INDEX_FROM_END])),
            )
            vector.cpu_step(actions.data_ptr())
        max_public_progress = max(
            max_public_progress,
            float(np.max(observations[:, NATIVE_PROGRESS_INDEX_FROM_END])),
        )
        metrics = flatten_log(pufferl, dict(vector.log()))
    finally:
        vector.close()
    wall_time = time.perf_counter() - started
    agent_steps = native_steps * agents
    expected_progress = (num_gates - 1) / OFFICIAL_PROGRESS_SCALE
    progress_proved = max_public_progress >= expected_progress - 1e-5
    teacher_admitted = variable_oracle_passes(
        metrics, episodes=episodes, num_gates=num_gates
    )
    return {
        "schema": "vq2_lc002_long_course_oracle_count_v1",
        "tag": f"{TAG}_{num_gates}g_{agents}a_{episodes}e",
        "num_gates": num_gates,
        "agents": agents,
        "episodes": episodes,
        "seed": seed,
        "threads": threads,
        "loader_overrides": overrides,
        "fixed_environment": long_course_environment(num_gates),
        "native_steps": native_steps,
        "agent_steps": agent_steps,
        "wall_time_seconds": wall_time,
        "agent_steps_per_second": agent_steps / wall_time,
        "simulated_realtime_factor": agent_steps * DT_SECONDS / wall_time,
        "max_native_progress_source": max_public_progress,
        "deployment_progress_encoding": "held official active_gate_index / 6",
        "minimum_required_progress": expected_progress,
        "unbounded_progress_proved": progress_proved,
        "teacher_admitted": teacher_admitted,
        "admitted": teacher_admitted and progress_proved,
        "metrics": metrics,
        **identity,
        "runtime": runtime_manifest(),
        "safety": {
            "flight_sim_packets_sent": 0,
            "student_updates": 0,
            "submission_authorized": False,
        },
    }


def aggregate_reports(
    *, baseline_path: Path, vector20_path: Path, vector24_path: Path
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text())
    vector20 = json.loads(vector20_path.read_text())
    vector24 = json.loads(vector24_path.read_text())
    if baseline["num_gates"] != 20 or baseline["agents"] != 1:
        raise RuntimeError("baseline must be the one-agent 20-gate report")
    if vector20["num_gates"] != 20 or vector20["agents"] <= 1:
        raise RuntimeError("vector20 must be a multi-agent 20-gate report")
    if vector24["num_gates"] != 24 or vector24["agents"] <= 1:
        raise RuntimeError("vector24 must be a multi-agent 24-gate report")
    if baseline["native_steps"] != vector20["native_steps"]:
        raise RuntimeError("20-gate baseline/vector horizons differ")
    speedup = (
        baseline["wall_time_seconds"] * vector20["episodes"]
        / vector20["wall_time_seconds"]
    )
    static_runtime = lambda report: {
        key: value for key, value in report["runtime"].items()
        if key not in {"omp_num_threads", "omp_dynamic"}
    }
    identity_exact = all(
        baseline[key] == report[key]
        for report in (vector20, vector24)
        for key in ("source_commit", "source_sha256")
    ) and all(
        static_runtime(baseline) == static_runtime(report)
        for report in (vector20, vector24)
    )
    omp_contract_exact = (
        baseline["runtime"].get("omp_num_threads") == "1"
        and baseline["runtime"].get("omp_dynamic") == "FALSE"
        and all(
            report["runtime"].get("omp_num_threads") == "32"
            and report["runtime"].get("omp_dynamic") == "FALSE"
            for report in (vector20, vector24)
        )
    )
    reports_admitted = all(
        report["admitted"] for report in (baseline, vector20, vector24)
    )
    return {
        "schema": "vq2_lc002_long_course_oracle_aggregate_v1",
        "tag": TAG,
        "report_sha256": {
            "baseline_20g": sha256_path(baseline_path),
            "vector_20g": sha256_path(vector20_path),
            "vector_24g": sha256_path(vector24_path),
        },
        "projected_serial_wall_time_for_vector20_episodes": (
            baseline["wall_time_seconds"] * vector20["episodes"]
        ),
        "vector20_wall_time_seconds": vector20["wall_time_seconds"],
        "projected_wall_clock_speedup": speedup,
        "minimum_projected_speedup": MINIMUM_PROJECTED_SPEEDUP,
        "throughput_gate_passed": speedup >= MINIMUM_PROJECTED_SPEEDUP,
        "identity_exact": identity_exact,
        "omp_contract_exact": omp_contract_exact,
        "reports_admitted": reports_admitted,
        "admitted": identity_exact and omp_contract_exact and reports_admitted,
        "puffer_throughput_admitted": (
            identity_exact and omp_contract_exact and reports_admitted
            and speedup >= MINIMUM_PROJECTED_SPEEDUP
        ),
        "safety": baseline["safety"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--vector20", type=Path)
    parser.add_argument("--vector24", type=Path)
    args = parser.parse_args()
    output = (args.output or (DEFAULT_OUTPUT_ROOT / "report.json")).resolve()
    if args.aggregate:
        if not all((args.baseline, args.vector20, args.vector24)):
            parser.error("--aggregate requires --baseline, --vector20, and --vector24")
        report = aggregate_reports(
            baseline_path=args.baseline.resolve(),
            vector20_path=args.vector20.resolve(),
            vector24_path=args.vector24.resolve(),
        )
    else:
        if args.count is None:
            parser.error("--count is required unless --aggregate is used")
        episodes = args.episodes if args.episodes is not None else args.agents
        report = run_count(
            num_gates=args.count,
            agents=args.agents,
            episodes=episodes,
            seed=args.seed,
            threads=args.threads,
        )
    write_json_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
