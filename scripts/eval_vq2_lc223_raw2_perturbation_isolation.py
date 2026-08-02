#!/usr/bin/env python3
"""Isolate which LC222 perturbation prevents the early raw-2 milestone."""

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
SCHEMA = "vq2_lc223_raw2_perturbation_isolation_report_v1"
GROUP_SIZE = 32
PAIR_SIZE = 64
TARGET_PHASE = 1
TARGET_RAW_INDEX = 2
MAX_STEPS = 5_000
SEED = 432_223
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 143
MINIMUM_PASS_GAIN = 0
PROFILES: dict[str, dict[str, int | float]] = {
    "control": {},
    "reset": {
        "reset_position_noise_xy": 0.15,
        "reset_position_noise_z": 0.075,
    },
    "camera_jitter": {
        "visual_camera_roll_jitter_rad": 0.005,
        "visual_camera_pitch_jitter_rad": 0.005,
        "visual_camera_yaw_jitter_rad": 0.005,
    },
    "camera_dropout": {"visual_camera_dropout_prob": 0.01},
    "edge_dropout": {"visual_edge_dropout_prob": 0.005},
    "rolling_shutter": {"visual_rolling_shutter_s": 0.001},
    "plant_gain_hover": {
        "sitl_plant_domain_randomize": 1,
        "sitl_rate_gain_jitter_frac": 0.01,
        "sitl_hover_thrust_jitter": 0.002,
    },
    "plant_lag_drag": {
        "sitl_plant_domain_randomize": 1,
        "sitl_rate_lag_jitter_frac": 0.02,
        "sitl_linear_drag_jitter_frac": 0.02,
    },
    "combined": {
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
    },
}
PROFILE = "control"
TAG = "vq2_lc223_raw2_perturbation_isolation_control_001"
LC222_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc222_all24_moderate_perturbation_001/report.json"
LC222_REPORT_SHA256 = "4cd63e0e1fa5ee66236248067097e8ae8ba39669518f60da9ddcf41c41138aea"
PREREGISTRATION = ROOT / "docs/vq2_lc223_raw2_perturbation_isolation_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc223_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc223_raw2_perturbation_isolation.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def set_profile(profile: str) -> None:
    global PROFILE, TAG, DEFAULT_OUTPUT
    if profile not in PROFILES:
        raise ValueError(f"unknown LC223 profile {profile!r}")
    PROFILE = profile
    TAG = f"vq2_lc223_raw2_perturbation_isolation_{profile}_001"
    DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    parent = BASE_VERIFY_INPUTS()
    if sha256_path(LC222_REPORT) != LC222_REPORT_SHA256:
        raise RuntimeError("LC223 bound LC222 report changed")
    failed = json.loads(LC222_REPORT.read_text())
    if (
        failed.get("schema") != "vq2_lc222_all24_moderate_perturbation_report_v1"
        or not failed.get("diagnostic_valid")
        or failed.get("causal_screen_selected") is not None
        or any(
            item.get("maximum_raw_index_distribution", {}).get("1") != GROUP_SIZE
            for item in failed.get("items", [])
        )
        or failed.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("LC222 does not authorize component isolation")
    return parent


def profiled_load_config(*args: Any, **kwargs: Any) -> Any:
    config, overrides = BASE_LOAD_CONFIG(*args, **kwargs)
    config["env"].update(PROFILES[PROFILE])
    return config, overrides


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(items) != 2:
        return None
    if all(
        item.get("target_passes") == GROUP_SIZE
        and item.get("pre_target_terminals") == 0
        and item.get("transport_pass")
        for item in items
    ):
        return items[1]
    return None


def configure() -> None:
    BASE_CONFIGURE()
    MILESTONE.load_config = profiled_load_config
    MILESTONE.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), LC222_REPORT,
        base.CANDIDATE_CHECKPOINT, base.CANDIDATE_REPORT,
    )
    MILESTONE.NEXT_AUTHORITY_SELECTED = (
        "Use the LC223 component table to define the smallest defensible all-24 robustness screen; FlightSim control remains frozen."
    )
    MILESTONE.NEXT_AUTHORITY_NONE = (
        "Treat this profile as an early-course failure and keep FlightSim control frozen."
    )


def configure_outer() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "GROUP_SIZE", "PAIR_SIZE", "TARGET_PHASE",
        "TARGET_RAW_INDEX", "MAX_STEPS", "SEED", "ENV_SEED_GROUP_SIZE",
        "ENV_SEED_INDEX_OFFSET", "MINIMUM_PASS_GAIN", "PREREGISTRATION",
        "RUNNER", "TEST", "DEFAULT_OUTPUT", "verify_inputs", "choose_candidate",
        "configure",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG, SCHEMA, GROUP_SIZE, PAIR_SIZE, TARGET_PHASE, TARGET_RAW_INDEX,
        MAX_STEPS, SEED, ENV_SEED_GROUP_SIZE, ENV_SEED_INDEX_OFFSET,
        MINIMUM_PASS_GAIN, PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
        verify_inputs, choose_candidate, configure,
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
    *, output: Path, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_outer()
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    set_profile(args.profile)
    output = DEFAULT_OUTPUT if args.output is None else args.output.resolve()
    report = run(output=output, device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("diagnostic_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
