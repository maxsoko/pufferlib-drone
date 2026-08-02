#!/usr/bin/env python3
"""Split LC226 Gate-3 failures into individual visual and reset/plant factors."""

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
import scripts.eval_vq2_lc226_raw3_interaction_isolation as base


BASE_CONFIGURE = base.configure
BASE_VERIFY_INPUTS = base.verify_inputs
MILESTONE = base.MILESTONE
BASE_LOAD_CONFIG = base.BASE_LOAD_CONFIG
SCHEMA = "vq2_lc227_raw3_factor_isolation_report_v1"
GROUP_SIZE = 8
PAIR_SIZE = 16
TARGET_PHASE = 2
TARGET_RAW_INDEX = 3
MAX_STEPS = 10_000
SEED = 432_224
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 287
MINIMUM_PASS_GAIN = 0
RESET = base.base.RESET
PROFILES = {
    "control": {},
    "reset": RESET,
    "camera_dropout": {"visual_camera_dropout_prob": 0.01},
    "edge_dropout": {"visual_edge_dropout_prob": 0.005},
    "rolling_shutter": {"visual_rolling_shutter_s": 0.001},
    "reset_gain_hover": {
        **RESET,
        "sitl_plant_domain_randomize": 1,
        "sitl_rate_gain_jitter_frac": 0.01,
        "sitl_hover_thrust_jitter": 0.002,
    },
    "reset_lag_drag": {
        **RESET,
        "sitl_plant_domain_randomize": 1,
        "sitl_rate_lag_jitter_frac": 0.02,
        "sitl_linear_drag_jitter_frac": 0.02,
    },
}
PROFILE = "control"
TAG = "vq2_lc227_raw3_factor_isolation_control_001"
LC226_REPORT_SHA256 = {
    "perception": "e6b23de7ce93f035fb9f5d7bab9f8667bb59ab180cc3eeeccb71f9ed6429a75b",
    "plant": "d62652030f8f1b218ebdbf5f16db3ecd2290153d41904f33755e2ac632a7bfe6",
    "reset_perception": "af44313a8e800839c059386f5f7745bfb64785b4068c22103d1baedc9b25e35a",
    "reset_plant": "6f7109eb76b99eb45b373b73b4ab862b42fecb4904ad2ba4a251e427a17d48be",
    "perception_plant": "ab6acb97535d9b63fa14a3d316c7075400f669b85b76e99295ca0c800011c8e6",
    "combined_no_camera": "e631ba22e40bc86d343363c08188e8b392f5046440e9ef2cd59fa111911a4cf4",
}
PASSING_LC226 = {"plant", "perception_plant"}
PREREGISTRATION = ROOT / "docs/vq2_lc227_raw3_factor_isolation_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc227_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc227_raw3_factor_isolation.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def lc226_report(profile: str) -> Path:
    return ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / (
        f"vq2_lc226_raw3_interaction_isolation_{profile}_001"
    ) / "report.json"


def set_profile(profile: str) -> None:
    global PROFILE, TAG, DEFAULT_OUTPUT
    if profile not in PROFILES:
        raise ValueError(f"unknown LC227 profile {profile!r}")
    PROFILE = profile
    TAG = f"vq2_lc227_raw3_factor_isolation_{profile}_001"
    DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    parent = BASE_VERIFY_INPUTS()
    for profile, digest in LC226_REPORT_SHA256.items():
        path = lc226_report(profile)
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC227 bound LC226 report changed: {profile}")
        report = json.loads(path.read_text())
        selected = report.get("causal_screen_selected")
        passed = selected is not None and selected.get("target_passes") == GROUP_SIZE
        if (
            report.get("schema") != "vq2_lc226_raw3_interaction_isolation_report_v1"
            or not report.get("diagnostic_valid")
            or passed != (profile in PASSING_LC226)
            or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        ):
            raise RuntimeError(f"LC226 result is inconsistent: {profile}")
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
        Path(__file__).resolve(),
        *(lc226_report(profile) for profile in LC226_REPORT_SHA256),
        base.base.base.CANDIDATE_CHECKPOINT, base.base.base.CANDIDATE_REPORT,
    )
    MILESTONE.NEXT_AUTHORITY_SELECTED = (
        "Use LC227 to define the smallest deployment-relevant all-24 robustness screen; FlightSim remains frozen."
    )
    MILESTONE.NEXT_AUTHORITY_NONE = (
        "Reject this Gate-3 factor under the 10,000-step diagnostic bound."
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
