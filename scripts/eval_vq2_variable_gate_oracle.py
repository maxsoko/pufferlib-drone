#!/usr/bin/env python3
"""Evaluate the admitted alignment oracle on variable-count VQ2 courses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_native_oracle import (
    BACKEND_ENV_NAME,
    ENV_NAME,
    fixed_environment,
    fixed_overrides,
    flatten_log,
    native_step_budget,
)
from pufferlib.vq2_informed import configure_full_start_evaluation
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP


TAG_PREFIX = "vq2_vg002_variable_gate_oracle"
ACCEPTANCE_COUNTS = (5, 8, 11, 12)
DT_SECONDS = 1.0 / 64.0
EPISODE_SECONDS = 360.0
TARGET_SPEED_M_S = 2.0
MINIMUM_SUCCESS_RATE = 0.99
DEFAULT_AGENTS = 512
DEFAULT_EPISODES = 512
DEFAULT_SEED = 429020
DEFAULT_OUTPUT_ROOT = (
    ROOT / "logs" / "drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg002_variable_gate_oracle_admission"
)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace a mutable run-state file without exposing a partial JSON."""

    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish immutable evidence and refuse an overwrite."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    write_json_atomic(path, payload)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_source_sha256() -> tuple[dict[str, str], str, str]:
    """Return all executable VG002 sources, extension path, and Git commit."""

    from pufferlib import _C

    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError(
            f"compiled backend is {getattr(_C, 'env_name', None)!r}; "
            f"expected {BACKEND_ENV_NAME!r}"
        )
    if getattr(_C, "precision_bytes", None) != 4:
        raise RuntimeError("VG002 requires a float32 drone_race_vision binding")
    extension_path = str(Path(_C.__file__).resolve())
    sources = {
        str(path.relative_to(ROOT)): sha256_path(path)
        for path in (
            ROOT / "ocean/drone_race/drone_race.c",
            ROOT / "ocean/drone_race/drone_race.h",
            ROOT / "ocean/drone_race/binding.c",
            ROOT / "pufferlib/vq2_public_phase.py",
            ROOT / "pufferlib/vq2_informed.py",
            ROOT / "config/drone_race_vq2_informed_dreamer.ini",
            ROOT / "scripts/eval_vq2_native_oracle.py",
            ROOT / "scripts/run_vq2_vg002_vast.sh",
            Path(__file__).resolve(),
        )
    }
    sources["compiled_extension"] = sha256_path(Path(extension_path))
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return sources, extension_path, source_commit


def current_runtime_manifest() -> dict[str, str]:
    """Record package/platform versions that can affect native evaluation."""

    import numpy
    import pybind11
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "numpy": numpy.__version__,
        "pybind11": pybind11.__version__,
    }


def variable_environment(num_gates: int) -> dict[str, int | float]:
    """Return the fixed VG002 environment for one explicit gate count."""

    if num_gates < 5 or num_gates > 12:
        raise ValueError("VG002 gate count must be in [5, 12]")
    values = fixed_environment(
        "governed",
        target_speed_m_s=TARGET_SPEED_M_S,
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
        # Gate 11 is nominally at x=242 m before +-3 m jitter. Preserve ample
        # bounds without changing plant or teacher dynamics.
        "pos_bound": 320.0,
        # Any native/public progress diagnostic uses the same fixed encoding
        # as collection, policy screening, export, and deployment.
        "observable_gate_index_denominator": float(ENGINE_GATE_CAP),
    })
    return values


def mixed_variable_environment(*, seed: int = DEFAULT_SEED) -> dict[str, int | float]:
    """Return the exact-uniform 5..12 per-vector-instance training profile."""

    values = variable_environment(12)
    values.update({
        # Six remains the explicit fallback/regression anchor. The binding
        # replaces it once per vector env and never again on reset.
        "num_gates": 6,
        "num_gates_per_env_randomize": 1,
        "num_gates_per_env_min": 5,
        "num_gates_per_env_max": 12,
        "num_gates_per_env_seed": seed,
        "teacher_roll_until_gate_index": ENGINE_GATE_CAP,
    })
    return values


def load_variable_config(
    pufferl_module: Any,
    *,
    agents: int,
    episodes: int,
    seed: int,
    num_gates: int,
):
    if agents <= 0 or episodes <= 0 or episodes % agents:
        raise ValueError("episodes must be a positive multiple of agents")
    saved_argv = sys.argv
    overrides = fixed_overrides(agents=agents, seed=seed)
    try:
        sys.argv = [sys.argv[0], *overrides]
        config = pufferl_module.load_config(ENV_NAME)
    finally:
        sys.argv = saved_argv

    if config.get("backend_env_name") != BACKEND_ENV_NAME:
        raise RuntimeError("VG002 requires the drone_race_vision backend")
    if int(config["vec"]["total_agents"]) != agents:
        raise RuntimeError("vector agent count differs from the fixed override")
    config.setdefault("env", {}).update(variable_environment(num_gates))
    configure_full_start_evaluation(
        config,
        episodes_per_agent=episodes // agents,
        episode_offset=0,
    )
    return config, overrides


def variable_oracle_passes(
    metrics: dict[str, float],
    *,
    episodes: int,
    num_gates: int,
) -> bool:
    if metrics.get("env/n", 0.0) != float(episodes):
        return False
    success_rate = metrics.get("env/success_rate", 0.0)
    if success_rate + 1e-12 < MINIMUM_SUCCESS_RATE:
        return False
    required_zero = (
        "crash",
        "out_of_order",
        "crossing_margin_violation",
        "action_envelope_violation",
        "wire_rate_envelope_violation",
        "thrust_envelope_violation",
    )
    if any(metrics.get(f"env/{name}", math.inf) != 0.0 for name in required_zero):
        return False
    if metrics.get(f"env/gate_count{num_gates}_episode", 0.0) != 1.0:
        return False
    if metrics.get(f"env/gate_count{num_gates}_success", 0.0) != success_rate:
        return False
    for gate in range(num_gates):
        sampled = metrics.get(f"env/ordered_gate{gate}_sampled", 0.0)
        radial = metrics.get(f"env/ordered_gate{gate}_radial", math.inf)
        if sampled + 1e-12 < success_rate or radial > 0.10:
            return False
    return True


def run_count(
    *,
    num_gates: int,
    agents: int,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    sources, extension_path, source_commit = current_source_sha256()
    runtime = current_runtime_manifest()
    config, loader_overrides = load_variable_config(
        pufferl,
        agents=agents,
        episodes=episodes,
        seed=seed,
        num_gates=num_gates,
    )
    vector = _C.create_vec(config, gpu=0)
    actions = torch.zeros(
        (vector.total_agents, vector.num_atns), dtype=torch.float32
    )
    started = time.perf_counter()
    try:
        vector.reset()
        native_steps = native_step_budget(
            max_steps=int(config["env"]["max_steps"]),
            episodes=episodes,
            agents=agents,
        )
        for _ in range(native_steps):
            vector.cpu_step(actions.data_ptr())
        metrics = flatten_log(pufferl, dict(vector.log()))
    finally:
        vector.close()

    return {
        "schema": "vq2_variable_gate_oracle_count_report_v1",
        "tag": f"{TAG_PREFIX}_{num_gates}g_{episodes}",
        "num_gates": num_gates,
        "agents": agents,
        "episodes": episodes,
        "seed": seed,
        "precision_bytes": int(_C.precision_bytes),
        "compiled_extension_path": extension_path,
        "source_commit": source_commit,
        "runtime": runtime,
        "native_steps": native_steps,
        "wall_time_seconds": time.perf_counter() - started,
        "minimum_success_rate": MINIMUM_SUCCESS_RATE,
        "admitted": variable_oracle_passes(
            metrics, episodes=episodes, num_gates=num_gates
        ),
        "loader_overrides": loader_overrides,
        "fixed_environment": variable_environment(num_gates),
        "metrics": metrics,
        "source_sha256": sources,
        "safety": {
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "student_updates": 0,
            "student_checkpoints_written": 0,
            "submission_authorized": False,
        },
    }


def run_admission(
    *,
    output_root: Path,
    counts: tuple[int, ...] = ACCEPTANCE_COUNTS,
    agents: int = DEFAULT_AGENTS,
    episodes: int = DEFAULT_EPISODES,
    seed: int = DEFAULT_SEED,
    resume: bool = False,
) -> dict[str, Any]:
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("counts must be a nonempty sequence without duplicates")
    if any(count < 5 or count > 12 for count in counts):
        raise ValueError("all VG002 counts must be in [5, 12]")
    if agents <= 0 or episodes <= 0 or episodes % agents:
        raise ValueError("episodes must be a positive multiple of agents")

    sources, extension_path, source_commit = current_source_sha256()
    runtime = current_runtime_manifest()
    state_identity = {
        "schema": "vq2_variable_gate_oracle_run_state_v1",
        "tag": f"vq2_vg002_variable_gate_oracle_admission_{episodes * len(counts)}",
        "required_counts": list(counts),
        "agents": agents,
        "episodes_per_count": episodes,
        "base_seed": seed,
        "count_seeds": {
            str(count): seed + offset for offset, count in enumerate(counts)
        },
        "source_sha256": sources,
        "compiled_extension_path": extension_path,
        "source_commit": source_commit,
        "runtime": runtime,
        "safety": {
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "student_updates": 0,
            "student_checkpoints_written": 0,
            "submission_authorized": False,
        },
    }
    state_path = output_root / "state.json"
    if output_root.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if not output_root.exists():
        output_root.mkdir(parents=True)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for key, expected in state_identity.items():
            if state.get(key) != expected:
                raise RuntimeError(f"resume state mismatch for {key}")
    else:
        material = [path for path in output_root.iterdir() if not path.name.startswith(".")]
        if material:
            raise RuntimeError("cannot resume an evidence directory without state.json")
        state = dict(state_identity)
        state.update({
            "status": "pending",
            "running_count": None,
            "completed_counts": [],
            "count_report_sha256": {},
        })
        write_json_atomic(state_path, state)

    reports: list[dict[str, Any]] = []
    for offset, count in enumerate(counts):
        report_path = output_root / f"count_{count}.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
        else:
            state.update({
                "status": "running",
                "running_count": count,
                "completed_counts": [item["num_gates"] for item in reports],
                "count_report_sha256": {
                    str(item["num_gates"]): sha256_path(
                        output_root / f"count_{item['num_gates']}.json"
                    )
                    for item in reports
                },
            })
            write_json_atomic(state_path, state)
            report = run_count(
                num_gates=count,
                agents=agents,
                episodes=episodes,
                seed=seed + offset,
            )
            write_json_once(report_path, report)

        expected_admission = variable_oracle_passes(
            report.get("metrics", {}), episodes=episodes, num_gates=count
        )
        expected_fields = {
            "schema": "vq2_variable_gate_oracle_count_report_v1",
            "tag": f"{TAG_PREFIX}_{count}g_{episodes}",
            "num_gates": count,
            "agents": agents,
            "episodes": episodes,
            "seed": seed + offset,
            "precision_bytes": 4,
            "compiled_extension_path": extension_path,
            "source_commit": source_commit,
            "runtime": runtime,
            "native_steps": native_step_budget(
                max_steps=int(variable_environment(count)["max_steps"]),
                episodes=episodes,
                agents=agents,
            ),
            "minimum_success_rate": MINIMUM_SUCCESS_RATE,
            "admitted": expected_admission,
            "loader_overrides": fixed_overrides(agents=agents, seed=seed + offset),
            "fixed_environment": variable_environment(count),
            "source_sha256": sources,
            "safety": state_identity["safety"],
        }
        for key, expected in expected_fields.items():
            if report.get(key) != expected:
                raise RuntimeError(
                    f"count {count} evidence mismatch for {key}"
                )
        reports.append(report)
        state.update({
            "status": "count_complete",
            "running_count": None,
            "completed_counts": [item["num_gates"] for item in reports],
            "count_report_sha256": {
                str(item["num_gates"]): sha256_path(
                    output_root / f"count_{item['num_gates']}.json"
                )
                for item in reports
            },
        })
        write_json_atomic(state_path, state)
        if not report["admitted"]:
            break

    expected_count_paths = {
        output_root / f"count_{report['num_gates']}.json" for report in reports
    }
    actual_count_paths = set(output_root.glob("count_*.json"))
    if actual_count_paths != expected_count_paths:
        raise RuntimeError("count evidence is not one ordered completed prefix")

    aggregate = {
        "schema": "vq2_variable_gate_oracle_admission_report_v1",
        "tag": f"vq2_vg002_variable_gate_oracle_admission_{episodes * len(counts)}",
        "required_counts": list(counts),
        "completed_counts": [report["num_gates"] for report in reports],
        "agents": agents,
        "episodes_per_count": episodes,
        "total_required_episodes": episodes * len(counts),
        "base_seed": seed,
        "admitted": len(reports) == len(counts)
        and all(report["admitted"] for report in reports),
        "count_report_sha256": {
            str(report["num_gates"]): sha256_path(
                output_root / f"count_{report['num_gates']}.json"
            )
            for report in reports
        },
        "source_sha256": sources,
        "compiled_extension_path": extension_path,
        "source_commit": source_commit,
        "runtime": runtime,
        "safety": {
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "student_updates": 0,
            "student_checkpoints_written": 0,
            "submission_authorized": False,
        },
    }
    aggregate_path = output_root / "report.json"
    if aggregate_path.exists():
        existing_aggregate = json.loads(aggregate_path.read_text())
        if existing_aggregate != aggregate:
            raise RuntimeError("existing aggregate report does not match run evidence")
    else:
        write_json_once(aggregate_path, aggregate)
    state.update({
        "status": "admitted" if aggregate["admitted"] else "rejected",
        "running_count": None,
        "completed_counts": aggregate["completed_counts"],
        "count_report_sha256": aggregate["count_report_sha256"],
        "aggregate_report_sha256": sha256_path(aggregate_path),
    })
    write_json_atomic(state_path, state)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only source-identical completed count evidence",
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=list(ACCEPTANCE_COUNTS),
    )
    args = parser.parse_args()
    report = run_admission(
        output_root=args.output_root.resolve(),
        counts=tuple(args.counts),
        agents=args.agents,
        episodes=args.episodes,
        seed=args.seed,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
