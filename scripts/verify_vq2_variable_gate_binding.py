#!/usr/bin/env python3
"""Verify fixed-per-instance 5..12 assignment through the compiled binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import configure_full_start_evaluation
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, ENV_NAME, flatten_log
from scripts.eval_vq2_variable_gate_oracle import mixed_variable_environment


TAG = "vq2_vg001b_variable_gate_float_binding_smoke"
DEFAULT_OUTPUT = (
    ROOT / "logs" / "drone_race_full_policy_six_gate_bootstrap"
    / TAG / "report.json"
)
AGENTS = 64
EPISODES_PER_AGENT = 2
SEED = 429001


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding_distribution_passes(metrics: dict[str, float]) -> bool:
    if metrics.get("env/n") != float(AGENTS * EPISODES_PER_AGENT):
        return False
    for count in range(1, 17):
        expected = 0.125 if 5 <= count <= 12 else 0.0
        if abs(metrics.get(f"env/gate_count{count}_episode", -1.0) - expected) > 1e-12:
            return False
        if metrics.get(f"env/gate_count{count}_success", -1.0) != 0.0:
            return False
    return True


def load_smoke_config(pufferl_module: Any) -> dict[str, Any]:
    saved_argv = sys.argv
    try:
        sys.argv = [
            sys.argv[0],
            "--seed", str(SEED),
            "--vec.total-agents", str(AGENTS),
            "--vec.num-buffers", "1",
            "--vec.num-threads", "4",
        ]
        config = pufferl_module.load_config(ENV_NAME)
    finally:
        sys.argv = saved_argv
    if config.get("backend_env_name") != BACKEND_ENV_NAME:
        raise RuntimeError("variable-gate smoke requires drone_race_vision")
    config.setdefault("env", {}).update(mixed_variable_environment(seed=SEED))
    configure_full_start_evaluation(
        config,
        episodes_per_agent=EPISODES_PER_AGENT,
        episode_offset=0,
    )
    config["env"].update({
        # Force two quick timeout episodes. The count diagnostics are written
        # at terminal and prove the construction-time assignment survives a
        # reset; this smoke does not assess the oracle.
        "max_steps": 1,
        "time_limit_seconds": 1.0 / 64.0,
        "teacher_action_blend": 0.0,
        "teacher_alignment_governor": 0,
        "gate_position_domain_randomize": 0,
        "course_geometry_scale_randomize": 0,
        "visual_camera_dropout_prob": 0.0,
        "visual_edge_dropout_prob": 0.0,
        "visual_edge_corrupt_prob": 0.0,
        "visual_false_segments": 0,
    })
    return config


def run_smoke(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError(
            f"compiled backend is {getattr(_C, 'env_name', None)!r}; "
            f"expected {BACKEND_ENV_NAME!r}"
        )
    if getattr(_C, "precision_bytes", None) != 4:
        raise RuntimeError("variable-gate legal ABI requires float32 precision")
    config = load_smoke_config(pufferl)
    vector = _C.create_vec(config, gpu=0)
    actions = torch.zeros((AGENTS, vector.num_atns), dtype=torch.float32)
    try:
        vector.reset()
        for _ in range(6):
            vector.cpu_step(actions.data_ptr())
        metrics = flatten_log(pufferl, dict(vector.log()))
    finally:
        vector.close()

    sources = {
        str(path.relative_to(ROOT)): sha256_path(path)
        for path in (
            ROOT / "ocean/drone_race/drone_race.c",
            ROOT / "ocean/drone_race/drone_race.h",
            ROOT / "ocean/drone_race/binding.c",
            ROOT / "scripts/eval_vq2_variable_gate_oracle.py",
            Path(__file__).resolve(),
        )
    }
    sources[str(Path(_C.__file__).resolve())] = sha256_path(Path(_C.__file__).resolve())
    report = {
        "schema": "vq2_variable_gate_binding_smoke_v1",
        "tag": TAG,
        "admitted": binding_distribution_passes(metrics),
        "agents": AGENTS,
        "episodes_per_agent": EPISODES_PER_AGENT,
        "seed": SEED,
        "precision_bytes": int(_C.precision_bytes),
        "fixed_environment": config["env"],
        "metrics": metrics,
        "source_sha256": sources,
        "safety": {
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "student_updates": 0,
            "submission_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_smoke(args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
