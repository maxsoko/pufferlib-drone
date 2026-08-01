#!/usr/bin/env python3
"""Teacher-free repeated-source raw-12 screen of LC148 against LC143."""

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
import scripts.eval_vq2_lc110_phase8_success_action_bias as pairwise
import scripts.eval_vq2_lc149_phase11_state_dependent_milestone as lc149


TAG = "vq2_lc150_phase11_repeated_source_milestone_001"
SCHEMA = "vq2_lc150_phase11_repeated_source_milestone_report_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
TARGET_PHASE = 11
TARGET_RAW_INDEX = 12
MAX_STEPS = 17_000
SEED = 432_180
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
MINIMUM_PASS_GAIN = 1
BIASES = lc149.BIASES
PARENT_CHECKPOINT = lc149.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = lc149.PARENT_CHECKPOINT_SHA256
PARENT_REPORT = lc149.PARENT_REPORT
PARENT_REPORT_SHA256 = lc149.PARENT_REPORT_SHA256
CANDIDATE_CHECKPOINT = lc149.CANDIDATE_CHECKPOINT
CANDIDATE_CHECKPOINT_SHA256 = lc149.CANDIDATE_CHECKPOINT_SHA256
CANDIDATE_REPORT = lc149.CANDIDATE_REPORT
CANDIDATE_REPORT_SHA256 = lc149.CANDIDATE_REPORT_SHA256
LC149_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc149_phase11_state_dependent_milestone_001/report.json"
)
LC149_REPORT_SHA256 = "bda77d9a47d2c37c8c5f84c0e9a8d8766199c39b1224cca2effde40800d67c85"
PREREGISTRATION = ROOT / "docs/vq2_lc150_phase11_repeated_source_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc150_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc150_phase11_repeated_source_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    parent = lc149.verify_inputs()
    if sha256_path(LC149_REPORT) != LC149_REPORT_SHA256:
        raise RuntimeError("LC150 bound LC149 report changed")
    rejected = json.loads(LC149_REPORT.read_text())
    baseline, candidate = rejected.get("items", [{}, {}])
    if (
        rejected.get("schema") != "vq2_lc149_phase11_state_dependent_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or baseline.get("target_passes") != 0
        or candidate.get("target_passes") != 0
        or candidate.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC149 does not authorize LC150")
    return parent


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline, candidate = items
    if (
        baseline["target_passes"] == 0
        and candidate["target_passes"] == GROUP_SIZE
        and candidate["paired_target_gains_vs_baseline"] == GROUP_SIZE
        and candidate["paired_target_losses_vs_baseline"] == 0
        and candidate["transport_pass"]
        and candidate["pre_target_terminals"] <= baseline["pre_target_terminals"]
    ):
        return candidate
    return None


def configure() -> None:
    pairwise.TAG, pairwise.SCHEMA = TAG, SCHEMA
    pairwise.GROUP_SIZE, pairwise.PAIR_SIZE = GROUP_SIZE, PAIR_SIZE
    pairwise.BIASES = BIASES
    pairwise.SEED = SEED
    pairwise.TARGET_PHASE, pairwise.TARGET_RAW_INDEX = TARGET_PHASE, TARGET_RAW_INDEX
    pairwise.MAX_STEPS, pairwise.MINIMUM_PASS_GAIN = MAX_STEPS, MINIMUM_PASS_GAIN
    pairwise.PARENT_CHECKPOINT, pairwise.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    pairwise.PARENT_REPORT, pairwise.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    pairwise.PREREGISTRATION, pairwise.RUNNER, pairwise.TEST = (
        PREREGISTRATION, RUNNER, TEST,
    )
    pairwise.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    pairwise.configure()
    milestone.ENV_SEED_GROUP_SIZE = ENV_SEED_GROUP_SIZE
    milestone.ENV_SEED_INDEX_OFFSET = ENV_SEED_INDEX_OFFSET
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        LC149_REPORT, ROOT / "scripts/eval_vq2_lc149_phase11_state_dependent_milestone.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Use LC148 only as the source-trajectory parent for phase-12 offline search; independent admission remains failed and FlightSim remains forbidden."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC148 even on its source trajectory and retain LC143 for offline work; do not run FlightSim."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form Puffer actors over 256 identical seed-15 rows; LC148 changes only the four phase-11 residual parameter rows"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
        milestone.choose_candidate,
        pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
    )
    (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
        milestone.choose_candidate,
        pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
    ) = (
        verify_inputs, pairwise.build_candidate_context,
        pairwise.initialize_actor_execution, pairwise.execute_actor_actions,
        lc149.candidate_state_for_index, lc149.candidate_metadata_for_index,
        choose_candidate,
        lc149.candidate_state_for_index, lc149.candidate_metadata_for_index,
    )
    try:
        return milestone.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            milestone.verify_inputs, milestone.build_candidate_context,
            milestone.initialize_actor_execution, milestone.execute_actor_actions,
            milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
            milestone.choose_candidate,
            pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
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
