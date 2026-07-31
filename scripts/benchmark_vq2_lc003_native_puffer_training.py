#!/usr/bin/env python3
"""Measure native PuffeRL rollout-plus-training throughput on a long course."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_lc001_long_course_oracle import (
    long_course_environment,
    sha256_path,
    write_json_once,
)


TAG = "vq2_lc003_native_puffer_throughput_001"
PREREGISTRATION = (
    ROOT / "docs/vq2_lc003_native_puffer_throughput_preregistration_2026-07-31.md"
)
DEFAULT_BASELINE = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc001_long_course_oracle_001/baseline_20g.json"
)
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG / "report.json"
)
EXPECTED_BASELINE_SHA256 = (
    "877b1d3086d9ed68fc17f10f31f58221492dde9a0183561e0e0b43d8f0a432e1"
)
TOTAL_AGENTS = 1024
NUM_BUFFERS = 16
NUM_THREADS = 128
HORIZON = 16
MINIBATCH_SIZE = TOTAL_AGENTS * HORIZON
HIDDEN_SIZE = 256
WARMUP_CYCLES = 2
MEASURED_CYCLES = 64
MINIMUM_SPEEDUP = 10.0
MINIMUM_MAX_GPU_PERCENT = 50.0


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
        ROOT / "src/vecenv.h",
        ROOT / "src/pufferlib.cu",
        ROOT / "src/bindings.cu",
        ROOT / "config/default.ini",
        ROOT / "config/drone_race_vq2_informed_dreamer.ini",
        ROOT / "scripts/eval_vq2_lc001_long_course_oracle.py",
        PREREGISTRATION,
        Path(__file__).resolve(),
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
    }


def benchmark_config(pufferl_module: Any) -> dict[str, Any]:
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        config = pufferl_module.load_config("drone_race_vq2_informed_dreamer")
    finally:
        sys.argv = saved_argv
    config.update({
        "cudagraphs": 1,
        "profile": False,
        "reset_state": False,
        "rank": 0,
        "world_size": 1,
        "gpu_id": 0,
        "nccl_id": "None",
        "seed": 431030,
    })
    config["vec"].update({
        "total_agents": TOTAL_AGENTS,
        "num_buffers": NUM_BUFFERS,
        "num_threads": NUM_THREADS,
    })
    config["policy"].update({
        "hidden_size": HIDDEN_SIZE,
        "num_layers": 1,
        "expansion_factor": 1,
    })
    config["train"].update({
        "total_timesteps": (
            TOTAL_AGENTS * HORIZON * (WARMUP_CYCLES + MEASURED_CYCLES + 4)
        ),
        "horizon": HORIZON,
        "minibatch_size": MINIBATCH_SIZE,
        "replay_ratio": 1.0,
        "learning_start_timesteps": 0,
    })
    config["env"].update(long_course_environment(24))
    pufferl_module.validate_config(config)
    return config


def _poll_gpu(stop: threading.Event, samples: list[dict[str, float]]) -> None:
    while not stop.is_set():
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        first = result.stdout.strip().splitlines()[0]
        utilization, memory_mib = (float(value.strip()) for value in first.split(","))
        samples.append({
            "time_monotonic": time.monotonic(),
            "gpu_percent": utilization,
            "memory_mib": memory_mib,
        })
        stop.wait(0.10)


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def admission_predicates(report: dict[str, Any]) -> dict[str, bool]:
    return {
        "baseline_hash_exact": (
            report.get("baseline_sha256") == EXPECTED_BASELINE_SHA256
        ),
        "step_accounting_exact": (
            report.get("measured_agent_steps")
            == TOTAL_AGENTS * HORIZON * MEASURED_CYCLES
        ),
        "finite_logs": finite_tree(report.get("puffer_log", {})),
        "speedup_at_least_10x": (
            report.get("end_to_end_speedup", 0.0) >= MINIMUM_SPEEDUP
        ),
        "gpu_reached_50_percent": (
            report.get("gpu_sampling", {}).get("maximum_percent", 0.0)
            >= MINIMUM_MAX_GPU_PERCENT
        ),
        "float32_backend": report.get("precision_bytes") == 4,
        "throughput_only_privileged_policy_not_saved": (
            report.get("deployment_legal") is False
            and report.get("checkpoints_written") == 0
        ),
        "no_flightsim_packets": report.get("flight_sim_packets_sent") == 0,
        "submission_forbidden": report.get("submission_authorized") is False,
    }


def run_benchmark(*, baseline_path: Path) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if _C.env_name != "drone_race_vision" or _C.precision_bytes != 4:
        raise RuntimeError("LC003 requires the float32 drone_race_vision backend")
    if sha256_path(baseline_path) != EXPECTED_BASELINE_SHA256:
        raise RuntimeError("LC003 baseline hash changed")
    baseline = json.loads(baseline_path.read_text())
    baseline_sps = float(baseline["agent_steps_per_second"])
    config = benchmark_config(pufferl)
    identity = source_identity()
    initialization_started = time.perf_counter()
    engine = _C.create_pufferl(config)
    initialization_seconds = time.perf_counter() - initialization_started
    try:
        for _ in range(WARMUP_CYCLES):
            _C.rollouts(engine)
            _C.train(engine)
        warmup_log = dict(_C.log(engine))

        gpu_samples: list[dict[str, float]] = []
        stop = threading.Event()
        sampler = threading.Thread(
            target=_poll_gpu, args=(stop, gpu_samples), daemon=True
        )
        sampler.start()
        measured_started_unix = time.time()
        measured_started = time.perf_counter()
        for _ in range(MEASURED_CYCLES):
            _C.rollouts(engine)
            _C.train(engine)
        measured_seconds = time.perf_counter() - measured_started
        measured_finished_unix = time.time()
        stop.set()
        sampler.join(timeout=5.0)
        puffer_log = dict(_C.log(engine))
        num_params = int(engine.num_params())
    finally:
        _C.close(engine)

    measured_agent_steps = TOTAL_AGENTS * HORIZON * MEASURED_CYCLES
    end_to_end_sps = measured_agent_steps / measured_seconds
    gpu_percentages = [sample["gpu_percent"] for sample in gpu_samples]
    gpu_memory = [sample["memory_mib"] for sample in gpu_samples]
    report: dict[str, Any] = {
        "schema": "vq2_lc003_native_puffer_throughput_report_v1",
        "tag": TAG,
        "configuration": {
            "num_gates": 24,
            "total_agents": TOTAL_AGENTS,
            "num_buffers": NUM_BUFFERS,
            "num_threads": NUM_THREADS,
            "horizon": HORIZON,
            "minibatch_size": MINIBATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "warmup_cycles": WARMUP_CYCLES,
            "measured_cycles": MEASURED_CYCLES,
            "cudagraphs": config["cudagraphs"],
        },
        "precision_bytes": int(_C.precision_bytes),
        "num_params": num_params,
        "initialization_seconds": initialization_seconds,
        "measured_seconds": measured_seconds,
        "measured_started_unix": measured_started_unix,
        "measured_finished_unix": measured_finished_unix,
        "measured_agent_steps": measured_agent_steps,
        "end_to_end_agent_steps_per_second": end_to_end_sps,
        "baseline_path": str(baseline_path),
        "baseline_sha256": sha256_path(baseline_path),
        "baseline_agent_steps_per_second": baseline_sps,
        "end_to_end_speedup": end_to_end_sps / baseline_sps,
        "warmup_log": warmup_log,
        "puffer_log": puffer_log,
        "gpu_sampling": {
            "samples": len(gpu_samples),
            "mean_percent": statistics.fmean(gpu_percentages) if gpu_percentages else 0.0,
            "maximum_percent": max(gpu_percentages, default=0.0),
            "maximum_memory_mib": max(gpu_memory, default=0.0),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity_count": len(os.sched_getaffinity(0)),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
        **identity,
        "deployment_legal": False,
        "deployment_rejection_reason": (
            "throughput-only native actor consumes privileged training suffix"
        ),
        "optimizer_cycles": WARMUP_CYCLES + MEASURED_CYCLES,
        "checkpoints_written": 0,
        "flight_sim_packets_sent": 0,
        "submission_authorized": False,
    }
    predicates = admission_predicates(report)
    report["admission_predicates"] = predicates
    report["admitted"] = all(predicates.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_benchmark(baseline_path=args.baseline.resolve())
    write_json_once(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
