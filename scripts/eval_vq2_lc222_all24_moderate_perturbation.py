#!/usr/bin/env python3
"""Teacher-free all-24 LC216 screen under moderate deployment perturbations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc217_all24_action_sequence_milestone as base


BASE_CONFIGURE = base.configure
BASE_VERIFY_INPUTS = base.verify_inputs
MILESTONE = base.base.milestone
BASE_LOAD_CONFIG = MILESTONE.load_config
TAG = "vq2_lc222_all24_moderate_perturbation_001"
SCHEMA = "vq2_lc222_all24_moderate_perturbation_report_v1"
GROUP_SIZE = 32
PAIR_SIZE = 64
SEED = 432_222
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 143
MINIMUM_PASS_GAIN = GROUP_SIZE
LC218_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc218_all24_action_sequence_confirmation_001/report.json"
LC218_REPORT_SHA256 = "3ae103609608cd090b9e0ccbf00f73e38bd3e2aa5ab21e9930046171da2d5800"
LC221_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc221_all24_numpy_shadow_001/report.json"
LC221_REPORT_SHA256 = "7c8bd6f5b6e2b9267f9d9d33b68da2659566fab74049c6e1d8ca49af55c153c7"
PREREGISTRATION = ROOT / "docs/vq2_lc222_all24_moderate_perturbation_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc222_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc222_all24_moderate_perturbation.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG

PERTURBATIONS: dict[str, int | float] = {
    "reset_position_noise_xy": 0.15,
    "reset_position_noise_z": 0.075,
    "visual_camera_roll_jitter_rad": 0.005,
    "visual_camera_pitch_jitter_rad": 0.005,
    "visual_camera_yaw_jitter_rad": 0.005,
    "visual_camera_dropout_prob": 0.01,
    "visual_edge_dropout_prob": 0.005,
    "visual_rolling_shutter_s": 0.001,
    "sitl_plant_domain_randomize": 1,
    "sitl_rate_gain_jitter_frac": 0.01,
    "sitl_hover_thrust_jitter": 0.002,
    "sitl_rate_lag_jitter_frac": 0.02,
    "sitl_linear_drag_jitter_frac": 0.02,
}


def verify_inputs() -> dict[str, Any]:
    parent = BASE_VERIFY_INPUTS()
    for path, digest in {
        LC218_REPORT: LC218_REPORT_SHA256,
        LC221_REPORT: LC221_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC222 bound prerequisite changed: {path}")
    confirmation = json.loads(LC218_REPORT.read_text())
    shadow = json.loads(LC221_REPORT.read_text())
    if (
        confirmation.get("schema")
        != "vq2_lc218_all24_action_sequence_confirmation_report_v1"
        or not confirmation.get("diagnostic_valid")
        or confirmation.get("causal_screen_selected", {}).get("target_passes") != 128
        or confirmation.get("causal_screen_selected", {}).get(
            "maximum_raw_index_distribution", {}
        ).get("24") != 128
        or not shadow.get("acceptance_passed")
        or shadow.get("invalid_run")
        or shadow.get("crash_detected")
        or shadow.get("sitl", {}).get("commands_sent") != 0
        or shadow.get("control_inputs", {}).get("arm_commands_sent") != 0
        or shadow.get("control_inputs", {}).get("disarm_commands_sent") != 0
        or shadow.get("control_inputs", {}).get("official_reset_start", {}).get(
            "reset_sent"
        )
        or shadow.get("policy_trace", {}).get("inference_ticks", 0) < 600
    ):
        raise RuntimeError("LC218/LC221 do not authorize LC222")
    return parent


def perturbed_load_config(*args: Any, **kwargs: Any) -> Any:
    config, overrides = BASE_LOAD_CONFIG(*args, **kwargs)
    config["env"].update(PERTURBATIONS)
    return config, overrides


def configure() -> None:
    BASE_CONFIGURE()
    MILESTONE.load_config = perturbed_load_config
    MILESTONE.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), LC218_REPORT, LC221_REPORT,
        base.CANDIDATE_CHECKPOINT, base.CANDIDATE_REPORT,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    MILESTONE.NEXT_AUTHORITY_SELECTED = (
        "Scale LC216's moderate all-24 perturbation screen to 128 disjoint episodes; FlightSim control remains frozen."
    )
    MILESTONE.NEXT_AUTHORITY_NONE = (
        "Reject perturbation admission and retain LC216 only as an exact offline result; do not run FlightSim."
    )


def configure_outer() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "GROUP_SIZE", "PAIR_SIZE", "SEED",
        "ENV_SEED_GROUP_SIZE", "ENV_SEED_INDEX_OFFSET", "MINIMUM_PASS_GAIN",
        "PREREGISTRATION", "RUNNER", "TEST", "DEFAULT_OUTPUT",
        "verify_inputs", "configure",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG, SCHEMA, GROUP_SIZE, PAIR_SIZE, SEED, ENV_SEED_GROUP_SIZE,
        ENV_SEED_INDEX_OFFSET, MINIMUM_PASS_GAIN, PREREGISTRATION, RUNNER,
        TEST, DEFAULT_OUTPUT, verify_inputs, configure,
    )
    for name, value in zip(names, values):
        setattr(base, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(base, name, value)
    MILESTONE.load_config = BASE_LOAD_CONFIG


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_outer()
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("diagnostic_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
