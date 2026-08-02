#!/usr/bin/env python3
"""Independent exact-context confirmation of LC216's all-24 completion."""

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
import scripts.eval_vq2_lc217_all24_action_sequence_milestone as base


BASE_CONFIGURE = base.configure
TAG = "vq2_lc218_all24_action_sequence_confirmation_001"
SCHEMA = "vq2_lc218_all24_action_sequence_confirmation_report_v1"
SEED = 432_218
LC217_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc217_all24_action_sequence_milestone_001/report.json"
LC217_REPORT_SHA256 = "547bb858c27db6c795467a6794e7477cbcc42693141cad8dca6234e00782caa9"
PREREGISTRATION = ROOT / "docs/vq2_lc218_all24_action_sequence_confirmation_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc218_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc218_all24_action_sequence_confirmation.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    parent = base.verify_inputs()
    if sha256_path(LC217_REPORT) != LC217_REPORT_SHA256:
        raise RuntimeError("LC218 bound LC217 report changed")
    promoted = json.loads(LC217_REPORT.read_text())
    selected = promoted.get("causal_screen_selected", {})
    baseline, candidate = promoted.get("items", [{}, {}])
    if (
        promoted.get("schema")
        != "vq2_lc217_all24_action_sequence_milestone_report_v1"
        or not promoted.get("diagnostic_valid")
        or selected.get("checkpoint_sha256") != base.CANDIDATE_CHECKPOINT_SHA256
        or selected.get("target_passes") != base.GROUP_SIZE
        or selected.get("maximum_raw_index_distribution", {}).get("24")
        != base.GROUP_SIZE
        or selected.get("paired_target_losses_vs_baseline") != 0
        or baseline.get("maximum_raw_index_distribution", {}).get("18")
        != base.GROUP_SIZE
        or candidate.get("pre_target_terminals") != 0
        or promoted.get("safety", {}).get("flight_sim_packets_sent") != 0
        or promoted.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC217 does not authorize independent confirmation")
    return parent


def configure() -> None:
    BASE_CONFIGURE()
    base.base.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), LC217_REPORT,
        base.CANDIDATE_CHECKPOINT, base.CANDIDATE_REPORT,
        ROOT / "scripts/eval_vq2_lc217_all24_action_sequence_milestone.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    base.base.milestone.NEXT_AUTHORITY_SELECTED = (
        "Freeze LC216 as independently confirmed exact all-24; complete runtime parity and zero-command shadow before bounded FlightSim authority."
    )
    base.base.milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC216 confirmation and retain LC213; do not run FlightSim."
    )


def configure_outer() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "SEED", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "verify_inputs", "configure",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG, SCHEMA, SEED, PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
        verify_inputs, configure,
    )
    for name, value in zip(names, values):
        setattr(base, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(base, name, value)


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_outer()
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("diagnostic_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
