#!/usr/bin/env python3
"""Collect oracle labels on SF050's exact measured-course failure states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_public_phase_dagger_next as collector
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.collect_vq2_measured_two_gate_oracle_prefix import measured_config


TAG = "vq2_sf052_measured_two_gate_dagger_64"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf052_measured_two_gate_dagger_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf050_measured_two_gate_full_fit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "61631d5bbcfce1eb75522d49804214c91b33930545db6815c2d8c09ea5adaa57"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "d78e6bcbb00e6ca087dee4ed48f5d9de466b08735db63cf073826283a7cfda0c"
)
SF051_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf051_measured_two_gate_teacher_free_512/report.json"
)
SF051_REPORT_SHA256 = (
    "56a260f04d9204f12e82ec5d4c3cee0035cea88a0e76224ea5bb7617b2efe6c9"
)
AGENTS = 64
EPISODES = 64
SEED = 42052
COLLECTION_STEP_LIMIT = 768


def exact_dagger_config(pufferl_module):
    config, overrides = measured_config(pufferl_module)
    overrides = list(overrides)
    seed_position = overrides.index("--seed")
    overrides[seed_position + 1] = str(SEED)
    config["seed"] = SEED
    environment = config["env"]
    environment.update(
        {
            "teacher_action_blend": 0.0,
            "teacher_course_spline": 0,
            "teacher_segment_minimum_jerk": 0,
            "teacher_alignment_governor": 0,
            "w_action_teacher": 0.0,
        }
    )
    return config, overrides


def exact_collection_passes(
    metrics: dict[str, float],
    *,
    lengths: np.ndarray,
    terminal_count: np.ndarray,
    terminal_is_last: np.ndarray,
    labels: int,
    executed_action_max_error: float,
) -> bool:
    return bool(
        metrics.get("env/n", 0.0) == float(EPISODES)
        and metrics.get("env/timeout", 0.0) == 0.0
        and metrics.get("env/out_of_order", 0.0) == 0.0
        and metrics.get("env/action_envelope_violation", 0.0) == 0.0
        and metrics.get("env/wire_rate_envelope_violation", 0.0) == 0.0
        and metrics.get("env/thrust_envelope_violation", 0.0) == 0.0
        and lengths.shape == (AGENTS,)
        and np.all((lengths > 0) & (lengths <= COLLECTION_STEP_LIMIT))
        and np.all(terminal_count == 1)
        and np.all(terminal_is_last)
        and labels == int(lengths.sum())
        and executed_action_max_error <= 1e-7
    )


def configure_collector() -> None:
    if sha256_path(SF051_REPORT) != SF051_REPORT_SHA256:
        raise RuntimeError("SF051 rejection source-lock mismatch")
    if json.loads(SF051_REPORT.read_text()).get("two_gate_milestone_passed"):
        raise RuntimeError("SF052 is unnecessary after a passing SF051")
    collector.TAG = TAG
    collector.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    collector.PREREGISTRATION = PREREGISTRATION
    collector.CHECKPOINT = CHECKPOINT
    collector.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    collector.TRAIN_REPORT = TRAIN_REPORT
    collector.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    collector.AGENTS = AGENTS
    collector.EPISODES = EPISODES
    collector.SEED = SEED
    collector.COLLECTION_STEP_LIMIT = COLLECTION_STEP_LIMIT
    collector.sf039.phase_dagger_config = exact_dagger_config
    collector.next_dagger_collection_passes = exact_collection_passes


def collect(output: Path = DEFAULT_OUTPUT, *, device_name: str = "cuda") -> dict:
    configure_collector()
    report = collector.collect(output=output, device_name=device_name)
    wrapper = Path(__file__).resolve()
    wrapper_key = str(wrapper.relative_to(ROOT))
    wrapper_sha = sha256_path(wrapper)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_sha256"][wrapper_key] = wrapper_sha
    metadata["source_sha256"][str(SF051_REPORT.relative_to(ROOT))] = (
        SF051_REPORT_SHA256
    )
    metadata["bounded_prefix_dataset"] = False
    metadata["measured_two_gate_exact_course"] = True
    metadata["action"]["plant_action_source"] = "sf050_phase_actor_mean"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    report["source_sha256"][wrapper_key] = wrapper_sha
    report["source_sha256"][str(SF051_REPORT.relative_to(ROOT))] = (
        SF051_REPORT_SHA256
    )
    report["metadata_sha256"] = sha256_path(metadata_path)
    report["measured_two_gate_exact_course"] = True
    report["sf051_report_sha256"] = SF051_REPORT_SHA256
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.output.resolve(), device_name=args.device),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
