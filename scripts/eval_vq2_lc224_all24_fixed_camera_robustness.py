#!/usr/bin/env python3
"""All-24 LC216 screen with fixed camera calibration and runtime perturbations."""

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
TAG = "vq2_lc224_all24_fixed_camera_robustness_001"
SCHEMA = "vq2_lc224_all24_fixed_camera_robustness_report_v1"
GROUP_SIZE = 32
PAIR_SIZE = 64
SEED = 432_224
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 287
MINIMUM_PASS_GAIN = GROUP_SIZE
PREREGISTRATION = ROOT / "docs/vq2_lc224_all24_fixed_camera_robustness_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc224_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc224_all24_fixed_camera_robustness.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG

PERTURBATIONS: dict[str, int | float] = {
    "reset_position_noise_xy": 0.15,
    "reset_position_noise_z": 0.075,
    "visual_camera_dropout_prob": 0.01,
    "visual_edge_dropout_prob": 0.005,
    "visual_rolling_shutter_s": 0.001,
    "sitl_plant_domain_randomize": 1,
    "sitl_rate_gain_jitter_frac": 0.01,
    "sitl_hover_thrust_jitter": 0.002,
    "sitl_rate_lag_jitter_frac": 0.02,
    "sitl_linear_drag_jitter_frac": 0.02,
}

LC223_REPORT_SHA256 = {
    "control": "5609abbbe9a09c144c255291b05e5a3fc37e99ae94af53a857e137fff8f2da4d",
    "reset": "8bbb506ca956d248ae5e5aabe03b87e4a62cb42c77c70a8273454e4f4f45ca39",
    "camera_jitter": "c00e52d8c67a9bb3be445311dfc4aecea8f28e3f1e127159762fe9b170a2a30c",
    "camera_dropout": "307d90ae903365449528366234c220e4f9fbe961abc524bf8ee4ff8568173e2a",
    "edge_dropout": "5cba06f5fe86d934ac936fd13cbe1d760bd05608cd39a574de08e78445b0ce2b",
    "rolling_shutter": "b3b76e7763784773f0fe73a319f14ad01ed4411ab829585cc2e9b3f59337fa16",
    "plant_gain_hover": "10f87a6c5410a29ce7967c26fc4c7766783aa8a1f2e85e72666ab85d681fd20e",
    "plant_lag_drag": "664170644d4c384b749240274c1b00e1b24078875f6ef30e94cef303734c9a20",
    "combined": "c2e2af3f6e041012f9a9a0fd2d0b0e93a68888d9990b10be3d85e3a102f0a6bd",
}
PASSING_PROFILES = {
    "control", "reset", "camera_dropout", "edge_dropout", "rolling_shutter",
    "plant_gain_hover", "plant_lag_drag",
}


def lc223_report(profile: str) -> Path:
    return ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / (
        f"vq2_lc223_raw2_perturbation_isolation_{profile}_001"
    ) / "report.json"


def verify_inputs() -> dict[str, Any]:
    parent = BASE_VERIFY_INPUTS()
    for profile, digest in LC223_REPORT_SHA256.items():
        path = lc223_report(profile)
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC224 bound LC223 report changed: {profile}")
        report = json.loads(path.read_text())
        selected = report.get("causal_screen_selected")
        passed = (
            selected is not None
            and selected.get("target_passes") == GROUP_SIZE
            and selected.get("pre_target_terminals") == 0
            and selected.get("maximum_raw_index_distribution", {}).get("2")
            == GROUP_SIZE
        )
        if (
            report.get("schema")
            != "vq2_lc223_raw2_perturbation_isolation_report_v1"
            or not report.get("diagnostic_valid")
            or report.get("safety", {}).get("flight_sim_packets_sent") != 0
            or passed != (profile in PASSING_PROFILES)
        ):
            raise RuntimeError(f"LC223 profile result is inconsistent: {profile}")
    return parent


def perturbed_load_config(*args: Any, **kwargs: Any) -> Any:
    config, overrides = BASE_LOAD_CONFIG(*args, **kwargs)
    config["env"].update(PERTURBATIONS)
    return config, overrides


def configure() -> None:
    BASE_CONFIGURE()
    MILESTONE.load_config = perturbed_load_config
    MILESTONE.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        *(lc223_report(profile) for profile in LC223_REPORT_SHA256),
        base.CANDIDATE_CHECKPOINT, base.CANDIDATE_REPORT,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    MILESTONE.NEXT_AUTHORITY_SELECTED = (
        "Audit deployment match and the N399-prefix requirement before preregistering one bounded VQ2 Training attempt; Submission stays forbidden."
    )
    MILESTONE.NEXT_AUTHORITY_NONE = (
        "Reject LC216 robustness admission and keep FlightSim control frozen."
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
