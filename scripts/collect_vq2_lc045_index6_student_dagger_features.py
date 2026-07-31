#!/usr/bin/env python3
"""Collect rare phase-6 labels on states owned by the LC043 Puffer."""

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


TAG = "vq2_lc045_index6_student_dagger_features_001"
SCHEMA = "vq2_lc045_index6_student_dagger_report_v1"
STATE_SCHEMA = "vq2_lc045_index6_student_dagger_state_v1"
AGENTS = EPISODES = 1024
THREADS = 128
SEED = 431450
TARGET_PHASE = 6
STEP_LIMIT = 12_000
MINIMUM_RECORDS = 8_000
MINIMUM_TARGET_PHASE_REACHED = 0.001
LC043 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc043_phase5_interpolation_bracket_001"
)
LC043_REPORT = LC043 / "report.json"
LC043_REPORT_SHA256 = (
    "3dddf5bb2e3e6f3249a7215abe938a91625e3f9494c3f3c0984b22ddbbdaa115"
)
CHECKPOINT = LC043 / "a0p005/policy_selected.pt"
CHECKPOINT_SHA256 = (
    "d2f9c235cc0a966d8d96ddca7513026e6689cf14aeb8c0cff22891769b93ad52"
)
CHECKPOINT_REPORT = CHECKPOINT.parent / "report.json"
CHECKPOINT_REPORT_SHA256 = (
    "35f14abb43d7e6a9db7cac4370102131dfcc6c94bde68bc83ccd123e6774366d"
)
LC044_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc044_phase6_student_head_001/report.json"
)
LC044_REPORT_SHA256 = (
    "d7592fa112e0d3bc4d8cdef5281ac80f1391535bf9540001cc9608d22dd16f60"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc045_index6_student_dagger_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc045_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    expected = {
        LC043_REPORT: LC043_REPORT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
        CHECKPOINT_REPORT: CHECKPOINT_REPORT_SHA256,
        LC044_REPORT: LC044_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC045 bound input changed: {path}")
    aggregate = json.loads(LC043_REPORT.read_text())
    selected = aggregate.get("selected", {})
    child = json.loads(CHECKPOINT_REPORT.read_text())
    rejected = json.loads(LC044_REPORT.read_text())
    if (
        aggregate.get("schema")
        != "vq2_lc043_phase5_interpolation_bracket_report_v1"
        or not aggregate.get("diagnostic_valid")
        or not aggregate.get("numerically_admitted")
        or selected.get("alpha") != 0.005
        or selected.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or selected.get("mean_gates_passed") != 3.75
        or selected.get("maximum_raw_index") != 8
        or selected.get("crash_rate") != 0.03125
        or child.get("schema") != "vq2_lc043_phase5_interpolation_screen_v1"
        or not child.get("numerically_admitted")
        or rejected.get("schema") != "vq2_lc044_phase6_student_head_report_v1"
        or rejected.get("numerically_admitted")
        or rejected.get("selected_validation", {}).get("phases", {}).get(
            "6", {}
        ).get("improvement_factor") != 1.0173365731903161
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or aggregate.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC043/LC044 do not authorize phase-6 collection")


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.AGENTS, base.EPISODES, base.SEED = AGENTS, EPISODES, SEED
    base.THREADS = THREADS
    base.STEP_LIMIT, base.MINIMUM_RECORDS = STEP_LIMIT, MINIMUM_RECORDS
    base.TARGET_PHASE = TARGET_PHASE
    base.MINIMUM_TARGET_PHASE_REACHED = MINIMUM_TARGET_PHASE_REACHED
    base.CHECKPOINT, base.CHECKPOINT_SHA256 = CHECKPOINT, CHECKPOINT_SHA256
    base.CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc043_phase5_interpolation_checkpoint_v1"
    )
    base.TRAIN_REPORT, base.TRAIN_REPORT_SHA256 = (
        CHECKPOINT_REPORT, CHECKPOINT_REPORT_SHA256,
    )
    base.LC011_REPORT, base.LC011_REPORT_SHA256 = (
        LC043_REPORT, LC043_REPORT_SHA256,
    )
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), LC044_REPORT)
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
