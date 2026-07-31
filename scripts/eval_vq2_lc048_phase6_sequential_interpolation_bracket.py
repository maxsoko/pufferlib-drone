#!/usr/bin/env python3
"""Run the LC046 phase-6 bracket through one CUDA context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc015_restore_vg071_head1 as screen
import scripts.eval_vq2_lc024_phase2_interpolation_bracket as bracket
import scripts.eval_vq2_lc047_phase6_parallel_interpolation_bracket as lc047


TAG = "vq2_lc048_phase6_sequential_interpolation_bracket_001"
SCHEMA = "vq2_lc048_phase6_sequential_interpolation_bracket_report_v1"
CHILD_SCHEMA = "vq2_lc048_phase6_interpolation_screen_v1"
CHECKPOINT_SCHEMA = "vq2_lc048_phase6_interpolation_checkpoint_v1"
ALPHAS = lc047.ALPHAS
SEED = lc047.SEED
TARGET_PHASE = lc047.TARGET_PHASE
REJECTION = ROOT / "docs/vq2_lc047_parallel_bracket_rejection_2026-07-31.json"
REJECTION_SHA256 = (
    "332e26c2d0877888622f7f75edcb1201097d4d301daeac3324931dbe03926b20"
)
PREREGISTRATION = (
    ROOT
    / "docs/vq2_lc048_phase6_sequential_interpolation_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc048_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CURRENT_ALPHA = 0.0


def verify_inputs() -> None:
    lc047.verify_inputs()
    if sha256_path(REJECTION) != REJECTION_SHA256:
        raise RuntimeError("LC048 bound LC047 rejection changed")
    rejected = json.loads(REJECTION.read_text())
    if (
        rejected.get("schema") != "vq2_lc047_parallel_bracket_rejection_v1"
        or not rejected.get("rejected")
        or not rejected.get("unchanged_retry_forbidden")
        or rejected.get("completed_child_reports") != 0
        or rejected.get("minimum_observed_wall_time_seconds", 0) < 660
        or rejected.get("flight_sim_packets_sent") != 0
        or rejected.get("submission_authorized")
    ):
        raise RuntimeError("LC047 does not authorize the sequential recovery")


def build_candidate(device: torch.device):
    lc047.CURRENT_ALPHA = CURRENT_ALPHA
    actor, payload, digest = lc047.build_candidate(device)
    payload["schema"] = CHECKPOINT_SCHEMA
    payload["tag"] = screen.TAG
    payload["surgery"]["execution_layout"] = "single_process_single_cuda_context"
    return actor, payload, digest


def configure_alpha(alpha: float) -> None:
    global CURRENT_ALPHA
    CURRENT_ALPHA = alpha
    slug = bracket.alpha_slug(alpha)
    screen.TAG = f"{TAG}_{slug}"
    screen.SCHEMA = CHILD_SCHEMA
    screen.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    screen.SEED = SEED
    screen.PREREGISTRATION = PREREGISTRATION
    screen.RUNNER = RUNNER
    screen.RESTORED_PHASES = (TARGET_PHASE,)
    screen.MIN_MEAN_GATES = 3.75
    screen.MIN_MAXIMUM_INDEX = 8
    screen.MAX_CRASH_RATE = 0.50
    screen.CONVERSION_OPERATION = (
        f"interpolate phase-{TARGET_PHASE} head toward fit at alpha {alpha:.4f}"
    )
    screen.EXTRA_EVIDENCE_PATHS = (
        Path(__file__).resolve(), REJECTION,
        lc047.LC043_REPORT, lc047.PARENT_CHECKPOINT, lc047.PARENT_REPORT,
        lc047.LC045_REPORT, lc047.FIT_CHECKPOINT, lc047.FIT_REPORT,
        ROOT / "scripts/eval_vq2_lc047_phase6_parallel_interpolation_bracket.py",
    )
    screen.verify_inputs = verify_inputs
    screen.build_candidate = build_candidate


def configure() -> None:
    bracket.TAG, bracket.SCHEMA, bracket.CHILD_SCHEMA = TAG, SCHEMA, CHILD_SCHEMA
    bracket.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    bracket.ALPHAS, bracket.SEED = ALPHAS, SEED
    bracket.TARGET_PHASE = TARGET_PHASE
    bracket.PREREGISTRATION, bracket.RUNNER = PREREGISTRATION, RUNNER
    bracket.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    bracket.HISTORICAL_MINIMUM_MEAN_GATES = 3.75
    bracket.HISTORICAL_MINIMUM_MAX_INDEX = 8
    bracket.verify_inputs = verify_inputs
    bracket.configure = configure_alpha
    bracket.build_candidate = build_candidate


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    configure()
    return bracket.run(
        output=output, device_name=device_name, resume=resume
    )


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
