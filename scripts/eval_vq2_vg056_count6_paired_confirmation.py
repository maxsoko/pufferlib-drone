#!/usr/bin/env python3
"""Confirm VG055 against VG033 on larger independent six-gate offsets."""

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
import scripts.eval_vq2_vg055_count6_fraction_bracket as bracket


TAG = "vq2_vg056_vg055_count6_paired_confirmation_001"
SCHEMA = "vq2_vg056_count6_paired_confirmation_report_v1"
STATE_SCHEMA = "vq2_vg056_count6_paired_confirmation_state_v1"
ALPHAS = (0.0, 1.0)
OFFSETS = (64, 72, 80, 88)
SEEDS = (429179, 429180, 429181, 429182)
AGENTS = 64
EPISODES = 64
THREADS = 4
MAX_STEPS = 3072
UPDATE_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg055_vg033_to_vg053_count6_fraction_bracket_001/policy_selected.pt"
)
UPDATE_CHECKPOINT_SHA256 = "d4076edd16bc9b1f648a12c3003e1a1bfde33560b7eaa6cbf17817435d2bbc74"
UPDATE_REPORT = UPDATE_CHECKPOINT.parent / "report.json"
UPDATE_REPORT_SHA256 = "bb0269801f61f81c296941cfe12c673a0efa423a04e91f4bc3bac105ca75e62d"
UPDATE_ADMISSION = ROOT / "docs/vq2_vg055_count6_fraction_bracket_admission_2026-07-31.json"
UPDATE_ADMISSION_SHA256 = "ae892d09a468008a4130c39fd8ccf75bbc61de619823225baf412a10aa3293c1"
PREREGISTRATION = ROOT / "docs/vq2_vg056_count6_paired_confirmation_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg056_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
_bracket_source_identity = bracket.source_identity


def source_identity() -> dict[str, Any]:
    identity = _bracket_source_identity()
    identity["source_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = (
        sha256_path(Path(__file__).resolve())
    )
    return identity


def qualifies(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Require the larger screen to break the active Gate-4->5 bottleneck."""
    p, c = parent["gate_reach"], candidate["gate_reach"]
    downstream_parent = (
        parent["successes"], p["6"], p["5"], p["4"], p["3"],
        parent["mean_gates_passed"],
    )
    downstream_candidate = (
        candidate["successes"], c["6"], c["5"], c["4"], c["3"],
        candidate["mean_gates_passed"],
    )
    return bool(
        parent["all_hard_transport_pass"]
        and candidate["all_hard_transport_pass"]
        and c["1"] >= p["1"]
        and c["2"] >= p["2"]
        and candidate["crashes"] <= parent["crashes"]
        and c["5"] > p["5"]
        and downstream_candidate > downstream_parent
    )


def configure() -> None:
    bracket.TAG = TAG
    bracket.SCHEMA = SCHEMA
    bracket.STATE_SCHEMA = STATE_SCHEMA
    bracket.ALPHAS = ALPHAS
    bracket.OFFSETS = OFFSETS
    bracket.SEEDS = SEEDS
    bracket.AGENTS = AGENTS
    bracket.EPISODES = EPISODES
    bracket.THREADS = THREADS
    bracket.MAX_STEPS = MAX_STEPS
    bracket.UPDATE_CHECKPOINT = UPDATE_CHECKPOINT
    bracket.UPDATE_CHECKPOINT_SHA256 = UPDATE_CHECKPOINT_SHA256
    bracket.UPDATE_REPORT = UPDATE_REPORT
    bracket.UPDATE_REPORT_SHA256 = UPDATE_REPORT_SHA256
    bracket.UPDATE_ADMISSION = UPDATE_ADMISSION
    bracket.UPDATE_ADMISSION_SHA256 = UPDATE_ADMISSION_SHA256
    bracket.PREREGISTRATION = PREREGISTRATION
    bracket.RUNNER = RUNNER
    bracket.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    bracket.source_identity = source_identity
    bracket.qualifies = qualifies


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    configure()
    return bracket.run(
        output=output,
        device_name=device_name,
        resume=resume,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
