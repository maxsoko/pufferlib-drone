#!/usr/bin/env python3
"""Cheap teacher-free 24-gate screen for the LC013 phase-1 DAgger actor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase_residual import (
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc007_converted_vg071_long_course as core


TAG = "vq2_lc014_bounded_student_dagger_screen_001"
SCHEMA = "vq2_lc014_bounded_student_dagger_screen_report_v1"
COUNTS = (24,)
AGENTS = 32
EPISODES = 32
THREADS = 32
SEEDS = {24: 431140}
MAX_STEPS = 12000
CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc013_index1_student_dagger_head_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "a66bbb1b9505b8d08166e2ce12c7bef29ee2fd824b82032ef67d9925d426c4ef"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "f45f08c5089bac61d82a9439c43d3dca654a8248d843aa1ffb164b8993940087"
)
LC011_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc011_bounded_prefix_screen_001/report.json"
)
LC011_REPORT_SHA256 = (
    "ed82daf9a06babafffc7236820045a924ce3100faba5e948807aed27fdccffd6"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc014_bounded_student_dagger_screen_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc014_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    expected = {
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        LC011_REPORT: LC011_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC014 bound input changed: {path}")
    train = json.loads(TRAIN_REPORT.read_text())
    baseline = json.loads(LC011_REPORT.read_text())
    if (
        train.get("schema")
        != "vq2_lc013_index1_student_dagger_head_report_v1"
        or not train.get("numerically_admitted")
        or train.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or train.get("best_epoch") != 10
        or not train.get("base_parameters_exact")
        or not train.get("non_target_outputs_zero")
        or train.get("safety", {}).get("flight_sim_packets_sent") != 0
        or baseline.get("schema")
        != "vq2_lc011_bounded_prefix_screen_report_v1"
        or not baseline.get("diagnostic_valid")
        or baseline.get("components", {}).get("24", {}).get("mean_gates_passed")
        != 1.09375
        or baseline.get("safety", {}).get("flight_sim_packets_sent") != 0
        or baseline.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC013 or its rejected LC011 baseline changed")


def load_actor(
    device: torch.device,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any]]:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema")
        != "vq2_lc013_index1_student_dagger_head_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or contract.get("class") != "VQ2UnboundedProgressMLPResidualActor"
        or contract.get("hidden_size") != 256
        or contract.get("residual_size") != 64
        or contract.get("residual_heads") != 33
    ):
        raise RuntimeError("LC014 checkpoint contract changed")
    actor = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(payload["model_state"])
    actor.eval()
    return actor, payload


def conversion_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": "load LC013 native long-course actor state",
        "checkpoint_schema": payload["schema"],
        "public_progress": "active_gate_index/6 without saturation",
        "whole_output_recurrent_puffer": True,
        "teacher_runtime_actions": 0,
    }


def configure() -> None:
    core.TAG = TAG
    core.SCHEMA = SCHEMA
    core.COUNTS = COUNTS
    core.AGENTS = AGENTS
    core.EPISODES = EPISODES
    core.THREADS = THREADS
    core.SEEDS = SEEDS
    core.CHECKPOINT = CHECKPOINT
    core.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    core.PREREGISTRATION = PREREGISTRATION
    core.RUNNER = RUNNER
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.MAX_STEPS_OVERRIDE = MAX_STEPS
    core.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), TRAIN_REPORT, LC011_REPORT,
    )
    core.load_converted_actor = load_actor
    core.conversion_metadata = conversion_metadata


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    verify_inputs()
    configure()
    return core.run(output=output, device_name=device_name, resume=resume)


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
