#!/usr/bin/env python3
"""Final larger-scale screen of the safe isolated LC111 phase-9 head."""

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
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc113_phase9_only_full_residual_screen as base


TAG = "vq2_lc114_phase9_only_large_scale_screen_001"
SCHEMA = "vq2_lc114_phase9_only_large_scale_screen_report_v1"
ALPHAS = (0.0, 0.30, 0.50, 1.0)
CANDIDATES = tuple(
    ("baseline_lc105" if alpha == 0.0 else f"lc111_phase9_scale_{alpha:g}", (0.0,) * 4)
    for alpha in ALPHAS
)
SEED = 432_140
LC113_REPORT = base.DEFAULT_OUTPUT / "report.json"
LC113_REPORT_SHA256 = "fda1c5194aa74794a19f2282e2fa89598540b3a5d5987ab8fd4a7c09b7cc49a0"
PREREGISTRATION = ROOT / "docs/vq2_lc114_phase9_only_large_scale_screen_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc114_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc114_phase9_only_large_scale_screen.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    parent = base.verify_inputs()
    if sha256_path(LC113_REPORT) != LC113_REPORT_SHA256:
        raise RuntimeError("LC114 bound LC113 report changed")
    rejected = json.loads(LC113_REPORT.read_text())
    items = rejected.get("items", [])
    if (
        rejected.get("schema") != "vq2_lc113_phase9_only_full_residual_screen_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or any(item.get("target_passes") != 0 for item in items)
        or any(
            item.get("maximum_raw_index_distribution", {}).get("9") != 1
            for item in items
        )
        or any(item.get("pre_target_terminals") != 128 for item in items)
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC113 does not authorize LC114")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.ALPHAS, base.CANDIDATES = ALPHAS, CANDIDATES
    base.SEED = SEED
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.configure()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), LC113_REPORT,
        base.base.ENDPOINT, base.base.ENDPOINT_REPORT,
        ROOT / "scripts/eval_vq2_lc113_phase9_only_full_residual_screen.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one independent pairwise-256 confirmation of the selected phase-9-only scale."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject the LC111 phase-9 endpoint family and retain LC105."
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    original_state = base.base.pairwise.candidate_state_for_index
    original_metadata = base.base.pairwise.candidate_metadata_for_index
    base.base.pairwise.candidate_state_for_index = base.candidate_state_for_index
    base.base.pairwise.candidate_metadata_for_index = base.candidate_metadata_for_index
    originals = (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
    )
    (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
    ) = (
        verify_inputs, base.base.pairwise.build_candidate_context,
        base.base.pairwise.initialize_actor_execution,
        base.base.pairwise.execute_actor_actions,
        base.candidate_state_for_index, base.candidate_metadata_for_index,
    )
    try:
        return milestone.run(
            output=output, device_name=device_name, resume=resume
        )
    finally:
        base.base.pairwise.candidate_state_for_index = original_state
        base.base.pairwise.candidate_metadata_for_index = original_metadata
        (
            milestone.verify_inputs, milestone.build_candidate_context,
            milestone.initialize_actor_execution, milestone.execute_actor_actions,
            milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
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
