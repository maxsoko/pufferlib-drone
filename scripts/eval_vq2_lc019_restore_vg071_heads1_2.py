#!/usr/bin/env python3
"""Restore VG071 heads 1 and 2 in LC010S and screen the 24-gate course."""

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


TAG = "vq2_lc019_restore_vg071_heads1_2_001"
SCHEMA = "vq2_lc019_restore_vg071_heads1_2_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc019_restore_vg071_heads1_2_checkpoint_v1"
SEED = 431190
LC015_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc015_restore_vg071_head1_001/report.json"
)
LC015_REPORT_SHA256 = (
    "5c38e8436bff1b5e80ea4a8e32f9e756d078e9df52154c46606e2b5bdbbef9a5"
)
LC015_CHECKPOINT = LC015_REPORT.parent / "policy_selected.pt"
LC015_CHECKPOINT_SHA256 = (
    "666e4ccc03037524c7f56ad257fbd7231199b1ed72c93d737c5e1865e32d8b7c"
)
LC018_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc018_native_gate1_lowstd_ppo_001/report.json"
)
LC018_REPORT_SHA256 = (
    "a1dffa2283cf679540b5ecfd01854cc20c0290d81328c5941f6763936d198cba"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc019_restore_vg071_heads1_2_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc019_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
ORIGINAL_VERIFY_INPUTS = base.verify_inputs


def verify_inputs() -> None:
    ORIGINAL_VERIFY_INPUTS()
    expected = {
        LC015_REPORT: LC015_REPORT_SHA256,
        LC015_CHECKPOINT: LC015_CHECKPOINT_SHA256,
        LC018_REPORT: LC018_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC019 bound input changed: {path}")

    frontier = json.loads(LC015_REPORT.read_text())
    scratch = json.loads(LC018_REPORT.read_text())
    scratch_logs = [
        item for item in scratch.get("logs", [])
        if "gates_passed" in item.get("env", {})
    ]
    if (
        frontier.get("schema") != "vq2_lc015_restore_vg071_head1_report_v1"
        or not frontier.get("numerically_admitted")
        or frontier.get("mean_gates_passed") != 2.4375
        or frontier.get("maximum_raw_index") != 4
        or frontier.get("crash_rate") != 0.25
        or frontier.get("checkpoint_sha256") != LC015_CHECKPOINT_SHA256
        or frontier.get("safety", {}).get("flight_sim_packets_sent") != 0
        or scratch.get("schema") != "vq2_lc018_native_gate1_lowstd_ppo_report_v1"
        or not scratch.get("training_admitted")
        or scratch.get("agent_steps") != 10_010_624
        or float(scratch.get("agent_steps_per_second", 0.0)) < 150_000.0
        or not scratch_logs
        or any(float(item.get("env", {}).get("gates_passed", -1.0)) != 0.0
               for item in scratch_logs)
        or any(float(item.get("env", {}).get("ordered_gate0_sampled", -1.0)) != 0.0
               for item in scratch_logs)
        or scratch.get("safety", {}).get("flight_sim_packets_sent") != 0
        or scratch.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC015/LC018 do not authorize the head-2 rollback")


def configure() -> None:
    base.TAG = TAG
    base.SCHEMA = SCHEMA
    base.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    base.SEED = SEED
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.RESTORED_PHASES = (1, 2)
    base.MIN_MEAN_GATES = 2.4375
    base.MIN_MAXIMUM_INDEX = 4
    base.MAX_CRASH_RATE = 0.50
    base.EXTRA_EVIDENCE_PATHS = (
        Path(__file__).resolve(), LC015_REPORT, LC015_CHECKPOINT, LC018_REPORT,
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
