#!/usr/bin/env python3
"""Confirm the phase-6 success-action bias on 128 fresh paired seeds."""

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
import scripts.eval_vq2_lc058_phase2_bias_milestone as base


TAG = "vq2_lc083_phase6_success_bias_confirmation_001"
SCHEMA = "vq2_lc083_phase6_success_bias_confirmation_report_v1"
GROUP_SIZE = 128
TARGET_PHASE = 6
TARGET_RAW_INDEX = 7
MAX_STEPS = 12_000
SELECTED_BIAS = (0.010, -0.050, 0.025, 0.0)
BIASES = (
    ("baseline_lc073", (0.0, 0.0, 0.0, 0.0)),
    ("lc082_success_combo_strong", SELECTED_BIAS),
)
SEED = 431830
MINIMUM_PASS_GAIN = 2
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc073_phase2_constrained_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "5614a95cd2f9b428a99ddd43ec5e324c095344cae51b8d771aea3d5f05d35be8"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "ae68af50524b02425f9a970c7f87dccbcbb6b60a0debb9899fcf1baa167b5d75"
LC082_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc082_phase6_success_action_bias_001/report.json"
)
LC082_REPORT_SHA256 = "05762ed608bbb32bc868b54d198d74ff46bfb32388805be6dcfa516e22e882d4"
PREREGISTRATION = ROOT / "docs/vq2_lc083_phase6_success_bias_confirmation_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc083_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc083_phase6_success_bias_confirmation.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def candidate_state_for_index(parent_state, candidate_index):
    import torch
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
        "phase6_output_bias": list(values),
        "bias_l2": sum(value * value for value in values) ** 0.5,
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC082_REPORT: LC082_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC083 bound input changed: {path}")
    parent_report = json.loads(PARENT_REPORT.read_text())
    screen = json.loads(LC082_REPORT.read_text())
    selected = screen.get("causal_screen_selected", {})
    import torch
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        parent_report.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or screen.get("schema") != "vq2_lc082_phase6_success_action_bias_report_v1"
        or not screen.get("diagnostic_valid")
        or selected.get("name") != "success_combo_strong"
        or tuple(selected.get("phase6_output_bias", ())) != SELECTED_BIAS
        or selected.get("paired_target_gains_vs_baseline") != 1
        or selected.get("paired_target_losses_vs_baseline") != 0
        or selected.get("target_passes") != 2
        or screen.get("safety", {}).get("flight_sim_packets_sent") != 0
        or screen.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC073/LC082 do not authorize LC083")
    return parent


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
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), LC082_REPORT)
    base.NEXT_AUTHORITY_SELECTED = (
        "Run one paired 24-gate full-course screen of the confirmed success-action bias."
    )
    base.NEXT_AUTHORITY_NONE = "Reject the selected success-action bias and retain LC073."
    base.VECTORIZED_CHECKPOINT_SURGERY = (
        "exact confirmed phase-6 Puffer output bias measured from successful Puffer trajectories"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        base.verify_inputs, base.candidate_state_for_index,
        base.candidate_metadata_for_index,
    )
    base.verify_inputs = verify_inputs
    base.candidate_state_for_index = candidate_state_for_index
    base.candidate_metadata_for_index = candidate_metadata_for_index
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            base.verify_inputs, base.candidate_state_for_index,
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
