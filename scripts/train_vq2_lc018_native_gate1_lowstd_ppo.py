#!/usr/bin/env python3
"""Train the legal native Puffer on Gate 1 with low exploration variance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_lc016_native_legal_puffer_ppo as base


TAG = "vq2_lc018_native_gate1_lowstd_ppo_001"
SCHEMA = "vq2_lc018_native_gate1_lowstd_ppo_report_v1"
TOTAL_TIMESTEPS = 10_000_384
SEED = 431180
LC017_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc017_native_legal_puffer_checkpoint_screen_001/report.json"
)
LC017_REPORT_SHA256 = (
    "8e8fcff9a70ce27ed0e5c55ec3934191fd6c061ad85fddc409a14e95c0b6dec3"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc018_native_gate1_lowstd_ppo_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc018_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
ORIGINAL_TRAINING_CONFIG = base.training_config


def verify_inputs() -> None:
    if sha256_path(LC017_REPORT) != LC017_REPORT_SHA256:
        raise RuntimeError("LC018 bound LC017 rejection changed")
    report = json.loads(LC017_REPORT.read_text())
    if (
        report.get("schema")
        != "vq2_lc017_native_legal_puffer_checkpoint_screen_report_v1"
        or not report.get("diagnostic_valid")
        or report.get("numerically_admitted")
        or any(item.get("mean_gates_passed") != 0.0 for item in report.get("items", []))
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC017 does not authorize the low-variance restart")


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, LC017_REPORT,
        ROOT / "scripts/train_vq2_lc016_native_legal_puffer_ppo.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
        ROOT / "src/pufferlib.cu", ROOT / "src/models.cu",
        ROOT / "src/bindings.cu", ROOT / "src/vecenv.h",
        ROOT / "config/drone_race_vq2_informed_dreamer.ini",
        ROOT / "pufferlib/vq2_informed.py",
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


def gate1_training_config(pufferl_module: Any) -> dict[str, Any]:
    config = ORIGINAL_TRAINING_CONFIG(pufferl_module)
    config["train"].update({
        "total_timesteps": TOTAL_TIMESTEPS,
        "learning_rate": 2e-4,
        "min_lr_ratio": 0.2,
        "ent_coef": 0.0,
        "phase_prio_obs_index": -1,
        "phase_prio_scale": 0.0,
        "phase_prio_max_weight": 1.0,
    })
    config["env"].update({
        "num_gates": 1,
        "num_gates_per_env_randomize": 0,
        "num_gates_per_env_min": 1,
        "num_gates_per_env_max": 1,
        "num_gates_per_env_seed": SEED,
        "gate_local_start_curriculum": 0,
        "gate_local_start_probability": 0.0,
        "gate_radius": 1.25,
        "gate_radius_randomize": 1,
        "gate_radius_min": 0.75,
        "gate_radius_max": 2.0,
        "gate_radius_randomize_gate_index": 0,
        "max_steps": 2560,
        "time_limit_seconds": 40.0,
        "pos_bound": 120.0,
        "w_progress": 15.0,
        "w_ordered_gate": 75.0,
        "w_finish": 150.0,
        "w_time": 0.01,
        "w_ctrl": 0.001,
        "w_body_rate": 0.1,
        "w_cross_track": 4.0,
        "w_gate_camera_alignment": 20.0,
        "w_gate_crossing_error": 10.0,
        "invalid_penalty": 150.0,
        "late_invalid_penalty": 150.0,
    })
    pufferl_module.validate_config(config)
    return config


def configure() -> None:
    base.TAG = TAG
    base.SCHEMA = SCHEMA
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.TOTAL_TIMESTEPS = TOTAL_TIMESTEPS
    base.SEED = SEED
    base.INITIAL_LOG_STD = -2.0
    base.TRAIN_NUM_GATES_MIN = 1
    base.TRAIN_NUM_GATES_MAX = 1
    base.NEXT_AUTHORITY = (
        "Teacher-free deterministic Gate-1 screens of the saved low-variance Puffers."
    )
    base.training_config = gate1_training_config
    base.source_identity = source_identity


def run(*, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    verify_inputs()
    configure()
    return base.run(output=output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(output=args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
