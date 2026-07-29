#!/usr/bin/env python3
"""Collect SF045-driven public-phase DAgger prefixes on a fresh seed."""

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
from scripts.collect_vq2_public_phase_prefix import (
    PREFIX_STEPS,
    prefix_collection_passes,
)


TAG = "vq2_sf046_public_phase_prefix_dagger_512"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf046_public_phase_prefix_dagger_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf045_public_phase_full_fit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "760101030db10e1dfbf2c31fcadf8fbbf01555105748ecbf4e20abcc638c819b"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "0bcf6acb4c6d0334de95e10759335071d73b58143272fcabadeb52063d0aaf96"
)
SEED = 42046


def configure_collector() -> None:
    collector.TAG = TAG
    collector.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    collector.PREREGISTRATION = PREREGISTRATION
    collector.CHECKPOINT = CHECKPOINT
    collector.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    collector.TRAIN_REPORT = TRAIN_REPORT
    collector.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    collector.SEED = SEED
    collector.COLLECTION_STEP_LIMIT = PREFIX_STEPS
    collector.next_dagger_collection_passes = prefix_collection_passes


def collect(output: Path = DEFAULT_OUTPUT, *, device_name: str = "cuda") -> dict:
    configure_collector()
    report = collector.collect(output=output, device_name=device_name)
    wrapper = Path(__file__).resolve()
    wrapper_key = str(wrapper.relative_to(ROOT))
    wrapper_sha = sha256_path(wrapper)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_sha256"][wrapper_key] = wrapper_sha
    metadata["bounded_prefix_dataset"] = True
    metadata["bounded_prefix_steps"] = PREFIX_STEPS
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    terminal = np.load(output / "terminal.npy", mmap_mode="r")
    terminated = int((terminal.sum(axis=0) == 1).sum())
    report["source_sha256"][wrapper_key] = wrapper_sha
    report["metadata_sha256"] = sha256_path(metadata_path)
    report["bounded_prefix_dataset"] = True
    report["bounded_prefix_steps"] = PREFIX_STEPS
    report["terminated_agents"] = terminated
    report["surviving_agents"] = collector.AGENTS - terminated
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(json.dumps(collect(args.output.resolve(), device_name=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
