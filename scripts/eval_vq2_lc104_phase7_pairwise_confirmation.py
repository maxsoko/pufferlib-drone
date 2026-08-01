#!/usr/bin/env python3
"""Independent pairwise confirmation of the exact LC101 phase-7 endpoint."""

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
import scripts.eval_vq2_lc103_phase7_pairwise_batch_scale_screen as prior


TAG = "vq2_lc104_phase7_pairwise_confirmation_001"
SCHEMA = "vq2_lc104_phase7_pairwise_confirmation_report_v1"
GROUP_SIZE = 128
ALPHAS = (0.0, 1.0)
CANDIDATES = (
    ("baseline_lc094", (0.0,) * 4),
    ("lc101_exact", (0.0,) * 4),
)
SEED = 432_040
MINIMUM_PASS_GAIN = 1
LC103_REPORT = prior.DEFAULT_OUTPUT / "report.json"
LC103_REPORT_SHA256 = "3a85e0aefcf01a2ce6da4a70b7715ed1d92a34518fccc9433e124365ac66865c"
PREREGISTRATION = ROOT / "docs/vq2_lc104_phase7_pairwise_confirmation_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc104_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc104_phase7_pairwise_confirmation.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    parent = prior.verify_inputs()
    if sha256_path(LC103_REPORT) != LC103_REPORT_SHA256:
        raise RuntimeError("LC104 bound LC103 report changed")
    screen = json.loads(LC103_REPORT.read_text())
    selected = screen.get("causal_screen_selected", {})
    if (
        screen.get("schema") != "vq2_lc103_phase7_pairwise_batch_scale_screen_report_v1"
        or not screen.get("diagnostic_valid")
        or selected.get("endpoint_scale") != 1.0
        or selected.get("target_passes") != 2
        or selected.get("paired_target_gains_vs_baseline") != 1
        or selected.get("paired_target_losses_vs_baseline") != 0
        or selected.get("pre_target_terminals") != 126
        or screen.get("items", [{}])[0].get("target_passes") != 1
        or screen.get("safety", {}).get("flight_sim_packets_sent") != 0
        or screen.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC103 does not authorize LC104")
    return parent


def configure() -> None:
    prior.TAG, prior.SCHEMA = TAG, SCHEMA
    prior.GROUP_SIZE, prior.PAIR_SIZE = GROUP_SIZE, GROUP_SIZE * 2
    prior.ALPHAS, prior.CANDIDATES = ALPHAS, CANDIDATES
    prior.SEED, prior.MINIMUM_PASS_GAIN = SEED, MINIMUM_PASS_GAIN
    prior.PREREGISTRATION, prior.RUNNER, prior.TEST = PREREGISTRATION, RUNNER, TEST
    prior.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    prior.configure()
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), LC103_REPORT,
        prior.prior.FIT_CHECKPOINT, prior.prior.FIT_REPORT,
        ROOT / "scripts/eval_vq2_lc103_phase7_pairwise_batch_scale_screen.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Run one paired serialization-exact 24-gate full-course screen of the confirmed LC101 endpoint."
    )
    milestone.NEXT_AUTHORITY_NONE = "Reject LC101 and retain LC094."
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form Puffer actors, each executing the full 256-row pair"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
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
        verify_inputs, prior.build_candidate_context,
        prior.initialize_actor_execution, prior.execute_actor_actions,
        prior.prior.candidate_state_for_index,
        prior.prior.candidate_metadata_for_index,
    )
    try:
        return milestone.run(
            output=output, device_name=device_name, resume=resume
        )
    finally:
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
