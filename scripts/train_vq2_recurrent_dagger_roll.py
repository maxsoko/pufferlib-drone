#!/usr/bin/env python3
"""Continue SF019 with one narrower roll-weighted paired DAgger fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_vq2_recurrent_dagger_paired import (
    PairedTrainConfig,
    train,
)


TAG = "vq2_sf020_recurrent_dagger_roll_001"
PARENT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf019_recurrent_dagger_paired_001/policy_best.pt"
)
PARENT_SHA256 = (
    "a391516076f49549255bf323f94859f90a0d15a5f3406c436d3045682e681352"
)
DEFAULT_OUTPUT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG
)
PREREGISTRATION = (
    ROOT / "docs/vq2_sf020_recurrent_dagger_roll_preregistration_2026-07-28.md"
)
CONFIG = PairedTrainConfig(
    seed=42020,
    epochs=4,
    learning_rate=5e-5,
    action_weights=(1.0, 2.0, 4.0, 1.0),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    report = train(
        output=args.output.resolve(),
        device_name=args.device,
        config=CONFIG,
        parent_checkpoint=PARENT,
        parent_checkpoint_sha256=PARENT_SHA256,
        tag=TAG,
        preregistration=PREREGISTRATION,
        extra_source_paths=(Path(__file__).resolve(),),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
