#!/usr/bin/env python3
"""Large fresh-seed confirmation of the constrained phase-2 endpoint."""

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
import scripts.eval_vq2_lc071_phase2_constrained_endpoint_screen as base


TAG = "vq2_lc072_phase2_constrained_endpoint_confirmation_001"
SCHEMA = "vq2_lc072_phase2_constrained_endpoint_confirmation_report_v1"
GROUP_SIZE = 128
ALPHAS = (0.0, 1.0)
CANDIDATES = (
    ("baseline_lc062", (0.0, 0.0, 0.0, 0.0)),
    ("endpoint_a1p0", (0.0, 0.0, 0.0, 0.0)),
)
SEED = 431720
MINIMUM_PASS_GAIN = 4
LC071_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc071_phase2_constrained_endpoint_screen_001/report.json"
)
LC071_REPORT_SHA256 = "0db63287868c9517bf0d40484cd7167cb135c9c00934423945440cd283eb9745"
PREREGISTRATION = ROOT / "docs/vq2_lc072_phase2_constrained_endpoint_confirmation_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc072_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc072_phase2_constrained_endpoint_confirmation.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_BASE_VERIFY_INPUTS = base.verify_inputs


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.GROUP_SIZE = GROUP_SIZE
    base.ALPHAS = ALPHAS
    base.CANDIDATES = CANDIDATES
    base.SEED = SEED
    base.MINIMUM_PASS_GAIN = MINIMUM_PASS_GAIN
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT


def verify_inputs() -> dict[str, Any]:
    payload = _BASE_VERIFY_INPUTS()
    if sha256_path(LC071_REPORT) != LC071_REPORT_SHA256:
        raise RuntimeError("LC072 bound LC071 report changed")
    report = json.loads(LC071_REPORT.read_text())
    selected = report.get("causal_screen_selected", {})
    if (
        report.get("schema") != "vq2_lc071_phase2_constrained_endpoint_screen_report_v1"
        or not report.get("diagnostic_valid")
        or selected.get("endpoint_interpolation_alpha") != 1.0
        or selected.get("gate3_passes") != 10
        or selected.get("paired_gate3_gains_vs_baseline") != 2
        or selected.get("paired_gate3_losses_vs_baseline") != 0
        or report.get("items", [{}])[0].get("gate3_passes") != 8
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC071 does not authorize LC072")
    return payload


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    base.configure()
    generic = base.base
    originals = (
        generic.verify_inputs, generic.build_candidate_context,
        generic.apply_candidate_actions, generic.candidate_state_for_index,
        generic.candidate_metadata_for_index, generic.EXTRA_SOURCE_PATHS,
        generic.NEXT_AUTHORITY_SELECTED, generic.NEXT_AUTHORITY_NONE,
    )
    generic.verify_inputs = verify_inputs
    generic.build_candidate_context = base.build_candidate_context
    generic.apply_candidate_actions = base.residual.apply_candidate_actions
    generic.candidate_state_for_index = base.candidate_state_for_index
    generic.candidate_metadata_for_index = base.candidate_metadata_for_index
    generic.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), base.FIT_CHECKPOINT, base.FIT_REPORT,
        LC071_REPORT, ROOT / "scripts/eval_vq2_lc071_phase2_constrained_endpoint_screen.py",
    )
    generic.NEXT_AUTHORITY_SELECTED = (
        "Run one fresh-seed paired 24-gate full-course screen of LC062 and the confirmed constrained endpoint."
    )
    generic.NEXT_AUTHORITY_NONE = "Reject the constrained endpoint family and retain LC062."
    try:
        return generic.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            generic.verify_inputs, generic.build_candidate_context,
            generic.apply_candidate_actions, generic.candidate_state_for_index,
            generic.candidate_metadata_for_index, generic.EXTRA_SOURCE_PATHS,
            generic.NEXT_AUTHORITY_SELECTED, generic.NEXT_AUTHORITY_NONE,
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
