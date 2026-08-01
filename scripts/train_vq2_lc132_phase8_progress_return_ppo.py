#!/usr/bin/env python3
"""Phase-8 PPO with a telescoping training-only native progress return."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_lc131_phase8_dense_return_ppo as dense


BASE_WRITE_JSON_ONCE = dense.BASE_WRITE_JSON_ONCE

TAG = "vq2_lc132_phase8_progress_return_ppo_001"
SCHEMA = "vq2_lc132_phase8_progress_return_ppo_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc132_phase8_progress_return_ppo_checkpoint_v1"
ROLLOUTS = 6
PPO_EPOCHS = 2
LEARNING_RATE = 1e-5
REWARD_OVERRIDES: dict[str, int | float] = {
    "w_progress": 5.0,
    "w_gate": 0.0,
    "w_ordered_gate": 30.0,
    "w_finish": 0.0,
    "w_time": 0.0,
    "w_ctrl": 0.0,
    "w_body_rate": 0.0,
    "w_cross_track": 0.0,
    "w_gate_camera_alignment": 0.0,
    "w_gate_crossing_error": 0.0,
    "w_gate_exit_lateral_velocity": 0.0,
    "w_gate_exit_velocity": 0.0,
    "w_forward_speed_excess": 0.0,
    "w_altitude_floor": 0.0,
    "w_descent_floor": 0.0,
    "w_action_teacher": 0.0,
    "invalid_penalty": 0.0,
    "late_invalid_penalty": 0.0,
}
PARENT_CHECKPOINT = dense.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = dense.PARENT_CHECKPOINT_SHA256
PARENT_REPORT = dense.PARENT_REPORT
PARENT_REPORT_SHA256 = dense.PARENT_REPORT_SHA256
LC131_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc131_phase8_dense_return_ppo_001/report.json"
)
LC131_REPORT_SHA256 = "fe676fcb2e05bf6b439163f3830877600b92ce8889ca1af384d756c27984d421"
PREREGISTRATION = ROOT / "docs/vq2_lc132_phase8_progress_return_ppo_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc132_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc132_phase8_progress_return_ppo.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC131_REPORT: LC131_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC132 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC131_REPORT.read_text())
    rollouts = rejected.get("rollouts", [])
    updates = rejected.get("updates", [])
    if (
        parent.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or rejected.get("schema")
        != "vq2_lc131_phase8_dense_return_ppo_report_v1"
        or rejected.get("candidate_selected_for_screen") is not None
        or not rejected.get("nonzero_updates")
        or not rejected.get("nonzero_phase_return_variance")
        or len(rollouts) != 4
        or any(item.get("raw9_or_later") != 0 for item in rollouts)
        or rollouts[3].get("records", 0) >= rollouts[0].get("records", 0)
        or len(updates) != 3
        or any(item.get("parameter_delta_l2", 0.0) <= 0.0 for item in updates)
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC123/LC131 do not authorize LC132")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC131_REPORT,
        ROOT / "scripts/train_vq2_lc127_phase6_onpolicy_ppo.py",
        ROOT / "scripts/train_vq2_lc131_phase8_dense_return_ppo.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "src/vecenv.h", ROOT / "src/bindings.cu",
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
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
    }


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        frozen = corrected.pop("frozen_non_phase6_state_exact")
        corrected["frozen_non_target_phase_state_exact"] = frozen
        corrected["actor_execution"] = (
            "one complete 256-row Puffer actor over two repeated 128-seed groups"
        )
        corrected["native_seed_group_size"] = dense.SEED_GROUP_SIZE
        corrected["training_reward_overrides"] = REWARD_OVERRIDES
        updates = corrected.get("updates", [])
        nonzero_updates = bool(
            updates and all(item.get("parameter_delta_l2", 0.0) > 0.0 for item in updates)
        )
        return_variance = bool(
            updates and all(item.get("queried_phase_return_std", 0.0) > 0.0 for item in updates)
        )
        corrected["nonzero_updates"] = nonzero_updates
        corrected["nonzero_phase_return_variance"] = return_variance
        corrected["training_admitted"] = bool(
            corrected.get("training_admitted") and nonzero_updates and return_variance
        )
        corrected["next_authority"] = (
            "Run one reduced deterministic LC123-versus-selected-candidate raw-9 screen; no FlightSim authority."
            if corrected["training_admitted"]
            and corrected.get("candidate_selected_for_screen") is not None
            else "Reject LC132 and retain LC105 as the safe frontier; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        dense.TAG, dense.SCHEMA, dense.CHECKPOINT_SCHEMA,
        dense.ROLLOUTS, dense.PPO_EPOCHS, dense.LEARNING_RATE,
        dense.PHASE_RETURN_ENV_OVERRIDES,
        dense.PREREGISTRATION, dense.RUNNER, dense.TEST, dense.DEFAULT_OUTPUT,
        dense.verify_inputs, dense.source_identity, dense.corrected_writer,
    )
    dense.TAG, dense.SCHEMA, dense.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    dense.ROLLOUTS, dense.PPO_EPOCHS, dense.LEARNING_RATE = (
        ROLLOUTS, PPO_EPOCHS, LEARNING_RATE,
    )
    dense.PHASE_RETURN_ENV_OVERRIDES = dict(REWARD_OVERRIDES)
    dense.PREREGISTRATION, dense.RUNNER, dense.TEST = PREREGISTRATION, RUNNER, TEST
    dense.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    dense.verify_inputs = verify_inputs
    dense.source_identity = source_identity
    dense.corrected_writer = corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        dense.TAG, dense.SCHEMA, dense.CHECKPOINT_SCHEMA,
        dense.ROLLOUTS, dense.PPO_EPOCHS, dense.LEARNING_RATE,
        dense.PHASE_RETURN_ENV_OVERRIDES,
        dense.PREREGISTRATION, dense.RUNNER, dense.TEST, dense.DEFAULT_OUTPUT,
        dense.verify_inputs, dense.source_identity, dense.corrected_writer,
    ) = originals


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        return dense.train(output=output, device_name=device_name, resume=resume)
    finally:
        restore(originals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
