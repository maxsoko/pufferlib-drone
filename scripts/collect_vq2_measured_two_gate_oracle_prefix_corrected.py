#!/usr/bin/env python3
"""Collect the measured oracle prefix with SF016's admitted parity bound."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_measured_two_gate_oracle_prefix as collector
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_sf049_measured_two_gate_oracle_prefix_64"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf049_measured_two_gate_oracle_prefix_preregistration_2026-07-28.md"
)
SEED = 42049
PREFIX_STEPS = 1600


def collect(output: Path = DEFAULT_OUTPUT) -> dict:
    collector.TAG = TAG
    collector.PREREGISTRATION = PREREGISTRATION
    collector.SEED = SEED
    collector.PREFIX_STEPS = PREFIX_STEPS
    report = collector.collect(output=output)
    wrapper = Path(__file__).resolve()
    key = str(wrapper.relative_to(ROOT))
    digest = sha256_path(wrapper)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_sha256"][key] = digest
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    report["source_sha256"][key] = digest
    report["metadata_sha256"] = sha256_path(metadata_path)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(collect(args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

