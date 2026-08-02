#!/usr/bin/env python3
"""Locate the LC224 failure approaching raw Gate 3 with a small paired screen."""

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
import scripts.eval_vq2_lc225_raw2_interaction_isolation as base


BASE_CONFIGURE = base.configure
BASE_VERIFY_INPUTS = base.verify_inputs
MILESTONE = base.MILESTONE
BASE_LOAD_CONFIG = base.BASE_LOAD_CONFIG
SCHEMA = "vq2_lc226_raw3_interaction_isolation_report_v1"
GROUP_SIZE = 8
PAIR_SIZE = 16
TARGET_PHASE = 2
TARGET_RAW_INDEX = 3
MAX_STEPS = 45_000
SEED = 432_224
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 287
MINIMUM_PASS_GAIN = 0
PROFILES = base.PROFILES
PROFILE = "perception"
TAG = "vq2_lc226_raw3_interaction_isolation_perception_001"
LC225_REPORT_SHA256 = {
    "perception": "6a4f88129a60f008e6ac4b10505bf05e9f6a920f8cffb7a808b9760be8eb73b4",
    "plant": "270a1a95b0d2f2afc94a46d5a27fbd92933bbba90d02b5fbcb153bca7f3c0d08",
    "reset_perception": "72b91250a1fd26cc43c4b518b7978356aa230fe9bf7457309eddaa4ca3eb91bd",
    "reset_plant": "b056f71482af200b92703daa3e8432253e1ec9f39b0618aa91478e56a828c9c5",
    "perception_plant": "1c65f814bc28ac5cfebcc819a8f77463a1975f8fc53e339b17a7b64c5405d189",
    "combined_no_camera": "b890e208a799a9d597f4d2fd0815c5f26c3e01620ad2d82409393e1dec7ed686",
}
PREREGISTRATION = ROOT / "docs/vq2_lc226_raw3_interaction_isolation_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc226_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc226_raw3_interaction_isolation.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def lc225_report(profile: str) -> Path:
    return ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / (
        f"vq2_lc225_raw2_interaction_isolation_{profile}_001"
    ) / "report.json"


def set_profile(profile: str) -> None:
    global PROFILE, TAG, DEFAULT_OUTPUT
    if profile not in PROFILES:
        raise ValueError(f"unknown LC226 profile {profile!r}")
    PROFILE = profile
    TAG = f"vq2_lc226_raw3_interaction_isolation_{profile}_001"
    DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    original_group_size = base.GROUP_SIZE
    base.GROUP_SIZE = 32
    try:
        parent = BASE_VERIFY_INPUTS()
    finally:
        base.GROUP_SIZE = original_group_size
    for profile, digest in LC225_REPORT_SHA256.items():
        path = lc225_report(profile)
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC226 bound LC225 report changed: {profile}")
        report = json.loads(path.read_text())
        selected = report.get("causal_screen_selected", {})
        if (
            report.get("schema") != "vq2_lc225_raw2_interaction_isolation_report_v1"
            or not report.get("diagnostic_valid")
            or selected.get("target_passes") != 32
            or selected.get("pre_target_terminals") != 0
            or selected.get("maximum_raw_index_distribution", {}).get("2") != 32
            or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        ):
            raise RuntimeError(f"LC225 raw-2 gate failed: {profile}")
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
        *(lc225_report(profile) for profile in LC225_REPORT_SHA256),
        base.base.CANDIDATE_CHECKPOINT, base.base.CANDIDATE_REPORT,
    )
    MILESTONE.NEXT_AUTHORITY_SELECTED = (
        "Use the LC226 Gate-3 interaction table to choose one full-24 robustness profile; FlightSim remains frozen."
    )
    MILESTONE.NEXT_AUTHORITY_NONE = (
        "Treat this interaction as the LC224 Gate-3 failure and keep FlightSim frozen."
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
