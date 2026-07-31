#!/usr/bin/env python3
"""Collect oracle labels on LC021-owned public-index-2 states."""

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


TAG = "vq2_lc022_index2_student_dagger_features_001"
SCHEMA = "vq2_lc022_index2_student_dagger_report_v1"
STATE_SCHEMA = "vq2_lc022_index2_student_dagger_state_v1"
AGENTS = 256
EPISODES = 256
SEED = 431220
TARGET_PHASE = 2
STEP_LIMIT = 5000
MINIMUM_RECORDS = 100_000
LC021 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc021_cumulative_head_rollback_ladder_001"
)
LC021_REPORT = LC021 / "report.json"
LC021_REPORT_SHA256 = (
    "5e9f9a7d8cc1c53b1a47d35fe725a5f7188a28c5b6a35876737152a5ad723068"
)
CHECKPOINT = LC021 / "through_05/policy_selected.pt"
CHECKPOINT_SHA256 = (
    "a32cc7705145c47b63c5126128eaa1296a52bfb175b8bb7868b9839041275ca4"
)
CHECKPOINT_REPORT = CHECKPOINT.parent / "report.json"
CHECKPOINT_REPORT_SHA256 = (
    "bd45e3ec075c4574fbc136a00ddb2b57830a2b96ef781d11e46959b8d7ab53fa"
)
CHECKPOINT_SCHEMA = "vq2_lc021_cumulative_head_rollback_checkpoint_v1"
PREREGISTRATION = (
    ROOT / "docs/vq2_lc022_index2_student_dagger_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc022_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    expected = {
        LC021_REPORT: LC021_REPORT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
        CHECKPOINT_REPORT: CHECKPOINT_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC022 bound input changed: {path}")
    ladder = json.loads(LC021_REPORT.read_text())
    screen = json.loads(CHECKPOINT_REPORT.read_text())
    selected = ladder.get("selected", {})
    if (
        ladder.get("schema")
        != "vq2_lc021_cumulative_head_rollback_ladder_report_v1"
        or not ladder.get("numerically_admitted")
        or selected.get("restored_through_phase") != 5
        or selected.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or selected.get("mean_gates_passed") != 2.875
        or selected.get("maximum_raw_index") != 6
        or screen.get("schema") != "vq2_lc021_cumulative_head_rollback_screen_v1"
        or not screen.get("numerically_admitted")
        or screen.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or screen.get("safety", {}).get("flight_sim_packets_sent") != 0
        or screen.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC021 does not authorize phase-2 label collection")


def configure() -> None:
    base.TAG = TAG
    base.SCHEMA = SCHEMA
    base.STATE_SCHEMA = STATE_SCHEMA
    base.AGENTS = AGENTS
    base.EPISODES = EPISODES
    base.SEED = SEED
    base.STEP_LIMIT = STEP_LIMIT
    base.MINIMUM_RECORDS = MINIMUM_RECORDS
    base.TARGET_PHASE = TARGET_PHASE
    base.CHECKPOINT = CHECKPOINT
    base.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    base.CHECKPOINT_SCHEMA_EXPECTED = CHECKPOINT_SCHEMA
    base.TRAIN_REPORT = CHECKPOINT_REPORT
    base.TRAIN_REPORT_SHA256 = CHECKPOINT_REPORT_SHA256
    base.LC011_REPORT = LC021_REPORT
    base.LC011_REPORT_SHA256 = LC021_REPORT_SHA256
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
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
