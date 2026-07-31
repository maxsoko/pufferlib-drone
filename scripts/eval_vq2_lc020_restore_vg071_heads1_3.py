#!/usr/bin/env python3
"""Restore VG071 heads 1 through 3 in LC010S and screen 24 gates."""

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
import scripts.eval_vq2_lc015_restore_vg071_head1 as base


TAG = "vq2_lc020_restore_vg071_heads1_3_001"
SCHEMA = "vq2_lc020_restore_vg071_heads1_3_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc020_restore_vg071_heads1_3_checkpoint_v1"
SEED = 431200
LC019_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc019_restore_vg071_heads1_2_001/report.json"
)
LC019_REPORT_SHA256 = (
    "def94f520c4fe4dd1466799861e0eb33f51a46302dab6361e8804bacf57bd9dd"
)
LC019_CHECKPOINT = LC019_REPORT.parent / "policy_selected.pt"
LC019_CHECKPOINT_SHA256 = (
    "d75b954e9b6ec13c3628ae916a5d8b5174ea2f8f2edc4f482dc2795bd2997e2d"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc020_restore_vg071_heads1_3_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc020_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
ORIGINAL_VERIFY_INPUTS = base.verify_inputs


def verify_inputs() -> None:
    ORIGINAL_VERIFY_INPUTS()
    expected = {
        LC019_REPORT: LC019_REPORT_SHA256,
        LC019_CHECKPOINT: LC019_CHECKPOINT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC020 bound input changed: {path}")
    frontier = json.loads(LC019_REPORT.read_text())
    if (
        frontier.get("schema") != "vq2_lc019_restore_vg071_heads1_2_report_v1"
        or not frontier.get("numerically_admitted")
        or frontier.get("mean_gates_passed") != 2.6875
        or frontier.get("maximum_raw_index") != 5
        or frontier.get("crash_rate") != 0.1875
        or frontier.get("checkpoint_sha256") != LC019_CHECKPOINT_SHA256
        or frontier.get("safety", {}).get("flight_sim_packets_sent") != 0
        or frontier.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC019 does not authorize the head-3 rollback")


def configure() -> None:
    base.TAG = TAG
    base.SCHEMA = SCHEMA
    base.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    base.SEED = SEED
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.RESTORED_PHASES = (1, 2, 3)
    base.MIN_MEAN_GATES = 2.6875
    base.MIN_MAXIMUM_INDEX = 5
    base.MAX_CRASH_RATE = 0.50
    base.EXTRA_EVIDENCE_PATHS = (
        Path(__file__).resolve(), LC019_REPORT, LC019_CHECKPOINT,
    )
    base.verify_inputs = verify_inputs


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    configure()
    return base.run(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
