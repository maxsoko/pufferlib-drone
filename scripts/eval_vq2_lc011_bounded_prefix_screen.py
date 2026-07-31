#!/usr/bin/env python3
"""Cheap teacher-free 24-gate prefix screen for the LC010S visual Puffer."""

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


TAG = "vq2_lc011_bounded_prefix_screen_001"
SCHEMA = "vq2_lc011_bounded_prefix_screen_report_v1"
COUNTS = (24,)
AGENTS = 32
EPISODES = 32
THREADS = 32
SEEDS = {24: 431110}
MAX_STEPS = 12000
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc010s_phase_independent_early_stop_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "34747327c2f431a6153d200891bad45b8431a5a5d0fe2418108ae822aa716d4d"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "afae8fa795159824bd447472b7eb5047167f19a4848264c95593c08ab43af21f"
)
BASELINE_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc007_converted_vg071_long_course_001/report.json"
)
BASELINE_REPORT_SHA256 = (
    "0f55347e761631d02bb3f7fb670316fd4e3253146b2e4e3bda3997b64cb79a0b"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc011_bounded_prefix_screen_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc011_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    expected = {
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        BASELINE_REPORT: BASELINE_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC011 bound input changed: {path}")
    report = json.loads(TRAIN_REPORT.read_text())
    if (
        report.get("schema") != "vq2_lc010s_phase_independent_early_stop_report_v1"
        or not report.get("completed")
        or not report.get("numerically_admitted")
        or report.get("vg068_history_replay_max_abs_error") != 0.0
        or report.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or not report.get("base_parameters_exact")
        or not report.get("non_target_outputs_zero")
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC010S does not authorize LC011")


def load_actor(
    device: torch.device,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any]]:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema")
        != "vq2_lc010s_phase_independent_early_stop_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or contract.get("class") != "VQ2UnboundedProgressMLPResidualActor"
        or contract.get("hidden_size") != 256
        or contract.get("residual_size") != 64
    ):
        raise RuntimeError("LC011 checkpoint contract changed")
    actor = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(payload["model_state"])
    actor.eval()
    return actor, payload


def conversion_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": "load native long-course actor state",
        "checkpoint_schema": payload["schema"],
        "public_progress": "active_gate_index/6 without saturation",
        "whole_output_recurrent_puffer": True,
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
    core.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), TRAIN_REPORT, BASELINE_REPORT)
    core.load_converted_actor = load_actor
    core.conversion_metadata = conversion_metadata


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
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
