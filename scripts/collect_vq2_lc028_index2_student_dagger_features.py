#!/usr/bin/env python3
"""Collect phase-2 labels on states owned by the LC027 selected Puffer."""

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


TAG = "vq2_lc028_index2_student_dagger_features_001"
SCHEMA = "vq2_lc028_index2_student_dagger_report_v1"
STATE_SCHEMA = "vq2_lc028_index2_student_dagger_state_v1"
AGENTS = EPISODES = 256
SEED = 431280
TARGET_PHASE = 2
STEP_LIMIT = 3500
MINIMUM_RECORDS = 100_000
LC027 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc027_phase2_interpolation_bracket_001"
)
LC027_REPORT = LC027 / "report.json"
LC027_REPORT_SHA256 = (
    "e161ca51447e4a324529f37b2d8175fd6d6667e77f67f3b61f805452100bbef2"
)
CHECKPOINT = LC027 / "a0p050/policy_selected.pt"
CHECKPOINT_SHA256 = (
    "4cdd636ab0a1d6741614c5f2f74a545a213492306e90d5d89dffff0430240bb9"
)
CHECKPOINT_REPORT = CHECKPOINT.parent / "report.json"
CHECKPOINT_REPORT_SHA256 = (
    "69845f35f66e1056f9228d6e0066272b3fe5f91a900c55af80713d5295e5da45"
)
CHECKPOINT_SCHEMA = "vq2_lc027_phase2_interpolation_checkpoint_v1"
PREREGISTRATION = (
    ROOT / "docs/vq2_lc028_index2_student_dagger_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc028_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    expected = {
        LC027_REPORT: LC027_REPORT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
        CHECKPOINT_REPORT: CHECKPOINT_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC028 bound input changed: {path}")
    bracket = json.loads(LC027_REPORT.read_text())
    screen = json.loads(CHECKPOINT_REPORT.read_text())
    selected = bracket.get("selected", {})
    if (
        bracket.get("schema") != "vq2_lc027_phase2_interpolation_bracket_report_v1"
        or not bracket.get("numerically_admitted")
        or selected.get("alpha") != 0.05
        or selected.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or selected.get("mean_gates_passed") != 3.5
        or selected.get("maximum_raw_index") != 6
        or selected.get("crash_rate") != 0.0625
        or screen.get("schema") != "vq2_lc027_phase2_interpolation_screen_v1"
        or screen.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or screen.get("safety", {}).get("flight_sim_packets_sent") != 0
        or screen.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC027 does not authorize phase-2 recollection")


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.AGENTS, base.EPISODES, base.SEED = AGENTS, EPISODES, SEED
    base.STEP_LIMIT, base.MINIMUM_RECORDS = STEP_LIMIT, MINIMUM_RECORDS
    base.TARGET_PHASE = TARGET_PHASE
    base.CHECKPOINT, base.CHECKPOINT_SHA256 = CHECKPOINT, CHECKPOINT_SHA256
    base.CHECKPOINT_SCHEMA_EXPECTED = CHECKPOINT_SCHEMA
    base.TRAIN_REPORT, base.TRAIN_REPORT_SHA256 = (
        CHECKPOINT_REPORT, CHECKPOINT_REPORT_SHA256,
    )
    base.LC011_REPORT, base.LC011_REPORT_SHA256 = (
        LC027_REPORT, LC027_REPORT_SHA256,
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
