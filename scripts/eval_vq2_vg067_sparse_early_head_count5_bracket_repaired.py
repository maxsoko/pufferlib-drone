#!/usr/bin/env python3
"""Run the repaired-schema successor to the VG066 sparse-head bracket."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_vg066_sparse_early_head_count5_bracket as vg066


BASE_VERIFY_INPUTS = vg066.verify_inputs
TAG = "vq2_vg067_sparse_early_head_count5_bracket_repaired_001"
SCHEMA = "vq2_vg067_sparse_early_head_count5_bracket_report_v1"
STATE_SCHEMA = "vq2_vg067_sparse_early_head_count5_bracket_state_v1"
OFFSETS = (200,)
SEEDS = (429202,)
FAILURE = ROOT / "docs/vq2_vg066_component_schema_infrastructure_rejection_2026-07-31.json"
FAILURE_SHA256 = "6e37d72c89fc8ca287c6eb57dfa9320b690dfe34c16fe9227fc57a7aabcd44eb"
PREREGISTRATION = ROOT / "docs/vq2_vg067_sparse_early_head_count5_bracket_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg067_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_vg067_sparse_early_head_count5_bracket.py"
SCHEMA_REPAIR = ROOT / "docs/vq2_vg066_component_schema_infrastructure_repair_2026-07-31.json"
SUMMARIZER_TEST = ROOT / "tests/test_vq2_staged_count5_diagnostic.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    BASE_VERIFY_INPUTS()
    if sha256_path(FAILURE) != FAILURE_SHA256:
        raise RuntimeError("VG067 failure evidence changed")
    failure = json.loads(FAILURE.read_text())
    if (
        failure.get("schema")
        != "vq2_vg066_component_schema_infrastructure_rejection_v1"
        or not failure.get("terminally_rejected")
        or not failure.get("unchanged_retry_forbidden")
        or failure.get("candidate_components_completed") != 0
        or failure.get("live_authority")
    ):
        raise RuntimeError("VG066 does not authorize VG067")


def configure() -> None:
    vg066.TAG = TAG; vg066.SCHEMA = SCHEMA; vg066.STATE_SCHEMA = STATE_SCHEMA
    vg066.OFFSETS = OFFSETS; vg066.SEEDS = SEEDS
    vg066.PREREGISTRATION = PREREGISTRATION; vg066.RUNNER = RUNNER
    vg066.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    vg066.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FAILURE, SCHEMA_REPAIR, TEST, SUMMARIZER_TEST,
    )
    vg066.verify_inputs = verify_inputs


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False):
    configure()
    return vg066.run(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
