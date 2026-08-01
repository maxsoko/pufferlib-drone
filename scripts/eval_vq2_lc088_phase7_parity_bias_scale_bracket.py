#!/usr/bin/env python3
"""Bracket mirrored phase-7 biases on the promoted LC087 Puffer policy."""

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
import scripts.eval_vq2_lc058_phase2_bias_milestone as base


TAG = "vq2_lc088_phase7_parity_bias_scale_bracket_001"
SCHEMA = "vq2_lc088_phase7_parity_bias_scale_bracket_report_v1"
GROUP_SIZE = 32
TARGET_PHASE = 7
TARGET_RAW_INDEX = 8
MAX_STEPS = 12_000
# LC087's successful phase-6 direction mirrored across the alternating course.
DIRECTION = (0.010, 0.050, 0.025, 0.0)
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
BIASES = tuple(
    (
        "baseline_lc087" if scale == 0.0 else f"mirrored_phase7_a{str(scale).replace('.', 'p')}",
        tuple(scale * value for value in DIRECTION),
    )
    for scale in SCALES
)
SEED = 431_880
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc087_phase6_half_bias_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "128bc970f169da81b4bd6b44dfbf73a9bf5ba0c66085d3e0cd13ca48f7f26b86"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "5023b42ef9b9bb2fbc4d016e4b5f5681e659da8c97217816769d3f2c9d21f6d5"
PREREGISTRATION = ROOT / "docs/vq2_lc088_phase7_parity_bias_scale_bracket_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc088_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc088_phase7_parity_bias_scale_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().clone() for name, value in parent_state.items()
    }
    state["indexed_phase_residual_output_bias"][TARGET_PHASE].add_(
        torch.tensor(BIASES[candidate_index][1], dtype=torch.float32)
    )
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    values = BIASES[candidate_index][1]
    return {
        "direction_scale": SCALES[candidate_index],
        "phase7_output_bias": list(values),
        "bias_l2": sum(value * value for value in values) ** 0.5,
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC088 bound input changed: {path}")
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    report = json.loads(PARENT_REPORT.read_text())
    candidate = report.get("items", [{}, {}])[1]
    if (
        payload.get("schema") != "vq2_lc087_phase6_half_bias_full_course_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or report.get("schema") != "vq2_lc087_phase6_half_bias_full_course_report_v1"
        or not report.get("diagnostic_valid")
        or not report.get("numerically_admitted")
        or report.get("promotion_target_pass_gain") != 2
        or report.get("mean_gate_gain") != 0.0390625
        or candidate.get("maximum_raw_index") != 9
        or candidate.get("promotion_target_passes") != 3
        or candidate.get("crash_rate") != report.get("items", [{}])[0].get("crash_rate")
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC087 does not authorize the LC088 offline parity screen")
    return payload


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.GROUP_SIZE = GROUP_SIZE
    base.BIAS_CANDIDATES = BIASES
    base.GROUPS = len(BIASES)
    base.TOTAL_AGENTS = GROUP_SIZE * base.GROUPS
    base.EPISODES = base.TOTAL_AGENTS
    base.SEED = SEED
    base.TARGET_PHASE = TARGET_PHASE
    base.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    base.MAX_STEPS = MAX_STEPS
    base.MINIMUM_PASS_GAIN = MINIMUM_PASS_GAIN
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT = PARENT_REPORT
    base.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.PARENT_CHILD_REPORT = PARENT_REPORT
    base.PARENT_CHILD_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)
    base.NEXT_AUTHORITY_SELECTED = (
        "Run one larger different-seed raw-index-8 confirmation of the selected mirrored phase-7 bias."
    )
    base.NEXT_AUTHORITY_NONE = (
        "Reject phase-7 parity transfer and collect phase-7 outcome-conditioned evidence from LC087."
    )
    base.VECTORIZED_CHECKPOINT_SURGERY = (
        "scaled mirrored phase-7 indexed Puffer output bias transferred from LC087 phase 6"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        base.verify_inputs,
        base.candidate_state_for_index,
        base.candidate_metadata_for_index,
    )
    base.verify_inputs = verify_inputs
    base.candidate_state_for_index = candidate_state_for_index
    base.candidate_metadata_for_index = candidate_metadata_for_index
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            base.verify_inputs,
            base.candidate_state_for_index,
            base.candidate_metadata_for_index,
        ) = originals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
