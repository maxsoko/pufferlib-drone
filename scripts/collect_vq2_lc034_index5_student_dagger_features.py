#!/usr/bin/env python3
"""Collect phase-5 labels on states owned by the LC033 selected Puffer."""

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
import scripts.collect_vq2_lc012_index1_student_dagger_features as base


TAG = "vq2_lc034_index5_student_dagger_features_001"
SCHEMA = "vq2_lc034_index5_student_dagger_report_v1"
STATE_SCHEMA = "vq2_lc034_index5_student_dagger_state_v1"
AGENTS = EPISODES = 256
SEED = 431340
TARGET_PHASE = 5
STEP_LIMIT = 6000
MINIMUM_RECORDS = 30_000
MINIMUM_TARGET_PHASE_REACHED = 0.05
LC033 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc033_phase4_interpolation_bracket_001"
)
LC033_REPORT = LC033 / "report.json"
LC033_REPORT_SHA256 = (
    "57d358bd8acb0739853d95f2f4cd5872ca1db7d0e4415cfdcc19a83ef1941de2"
)
CHECKPOINT = LC033 / "a0p250/policy_selected.pt"
CHECKPOINT_SHA256 = (
    "5b344710e28edb9e2bc080efc4101b398c554cf509a6cdc436b0c1baded53678"
)
CHECKPOINT_REPORT = CHECKPOINT.parent / "report.json"
CHECKPOINT_REPORT_SHA256 = (
    "63839834e56c59d480a2c0ac13991515aa21d7f533b9793cd3cec16e4b1ab16b"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc034_index5_student_dagger_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc034_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    expected = {
        LC033_REPORT: LC033_REPORT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
        CHECKPOINT_REPORT: CHECKPOINT_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC034 bound input changed: {path}")
    aggregate = json.loads(LC033_REPORT.read_text())
    selected = aggregate.get("selected", {})
    child = json.loads(CHECKPOINT_REPORT.read_text())
    distribution = selected.get("maximum_raw_index_distribution", {})
    if (
        aggregate.get("schema")
        != "vq2_lc033_phase4_interpolation_bracket_report_v1"
        or not aggregate.get("diagnostic_valid")
        or not aggregate.get("numerically_admitted")
        or selected.get("alpha") != 0.25
        or selected.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or selected.get("mean_gates_passed") != 3.59375
        or selected.get("maximum_raw_index") != 6
        or selected.get("crash_rate") != 0.0625
        or int(distribution.get("4", 0)) != 5
        or int(distribution.get("5", 0)) != 6
        or int(distribution.get("6", 0)) != 5
        or child.get("schema") != "vq2_lc033_phase4_interpolation_screen_v1"
        or not child.get("numerically_admitted")
        or child.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or aggregate.get("safety", {}).get("flight_sim_packets_sent") != 0
        or aggregate.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC033 does not authorize phase-5 collection")


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.AGENTS, base.EPISODES, base.SEED = AGENTS, EPISODES, SEED
    base.STEP_LIMIT, base.MINIMUM_RECORDS = STEP_LIMIT, MINIMUM_RECORDS
    base.TARGET_PHASE = TARGET_PHASE
    base.MINIMUM_TARGET_PHASE_REACHED = MINIMUM_TARGET_PHASE_REACHED
    base.CHECKPOINT, base.CHECKPOINT_SHA256 = CHECKPOINT, CHECKPOINT_SHA256
    base.CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc033_phase4_interpolation_checkpoint_v1"
    )
    base.TRAIN_REPORT, base.TRAIN_REPORT_SHA256 = (
        CHECKPOINT_REPORT, CHECKPOINT_REPORT_SHA256,
    )
    base.LC011_REPORT, base.LC011_REPORT_SHA256 = (
        LC033_REPORT, LC033_REPORT_SHA256,
    )
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)
    base.verify_inputs = verify_inputs


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
            resume: bool = False) -> dict[str, Any]:
    configure()
    return base.collect(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_dataset_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
