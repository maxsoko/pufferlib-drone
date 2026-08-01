#!/usr/bin/env python3
"""Correct LC196 by screening phase-17 scales at the exact 256-row context."""

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
import scripts.eval_vq2_lc196_phase16_17_joint_interpolation_bracket as prior


BASE_CONFIGURE = prior.configure
TAG = "vq2_lc197_phase17_exact_context_bracket_001"
SCHEMA = "vq2_lc197_phase17_exact_context_bracket_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc197_phase17_exact_context_checkpoint_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
SCALE_PAIRS = ((0.0, 0.0), (0.0, 0.50), (0.0, 1.0))
CANDIDATES = (
    ("parent_lc189", (0.0,) * 4),
    ("phase17_scale_0.5", (0.0,) * 4),
    ("phase17_scale_1", (0.0,) * 4),
)
LC196_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc196_phase16_17_joint_interpolation_bracket_001/report.json"
LC196_REPORT_SHA256 = "32de9ea322b25d8e126be6bafce108c322d9bc3260e1b22775f0a787ca83c8cb"
PREREGISTRATION = ROOT / "docs/vq2_lc197_phase17_exact_context_bracket_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc197_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc197_phase17_exact_context_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        prior.PARENT_CHECKPOINT: prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT: prior.PARENT_REPORT_SHA256,
        prior.FIT_CHECKPOINT: prior.FIT_CHECKPOINT_SHA256,
        prior.FIT_REPORT: prior.FIT_REPORT_SHA256,
        LC196_REPORT: LC196_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC197 bound input changed: {path}")
    parent = prior.parent_payload()
    fitted = prior.fit_payload()
    parent_report = json.loads(prior.PARENT_REPORT.read_text())
    fit_report = json.loads(prior.FIT_REPORT.read_text())
    rejected = json.loads(LC196_REPORT.read_text())
    items = rejected.get("items", [])
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("training_admitted")
        or fitted.get("schema") != "vq2_lc194_phase16_17_full_residual_fit_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("parent_checkpoint_sha256") != prior.PARENT_CHECKPOINT_SHA256
        or not fit_report.get("numerically_admitted")
        or rejected.get("schema") != "vq2_lc196_phase16_17_joint_interpolation_bracket_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or rejected.get("group_size") != 32
        or len(items) != 13
        or any(item.get("maximum_raw_index_distribution", {}).get("6") != 32 for item in items)
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC194/LC196 do not authorize LC197")
    return parent


def configure() -> None:
    BASE_CONFIGURE()
    prior.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), prior.FIT_CHECKPOINT, prior.FIT_REPORT,
        LC196_REPORT, ROOT / "scripts/eval_vq2_lc196_phase16_17_joint_interpolation_bracket.py",
        ROOT / "scripts/train_vq2_lc194_phase16_17_full_residual_fit.py",
    )
    prior.milestone.NEXT_AUTHORITY_SELECTED = "Confirm the selected raw-18 source independently before extending to phase 18; no FlightSim authority."
    prior.milestone.NEXT_AUTHORITY_NONE = "Reject the LC194 phase-17 direction and retain LC189; do not run FlightSim."
    prior.milestone.VECTORIZED_CHECKPOINT_SURGERY = "three complete adapter Puffers, each executing the established 256-row recurrent context over paired 128-row groups"


def configure_wrapper() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "CHECKPOINT_SCHEMA", "GROUP_SIZE", "PAIR_SIZE",
        "SCALE_PAIRS", "CANDIDATES", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "verify_inputs", "configure",
    )
    originals = tuple(getattr(prior, name) for name in names)
    values = (
        TAG, SCHEMA, CHECKPOINT_SCHEMA, GROUP_SIZE, PAIR_SIZE,
        SCALE_PAIRS, CANDIDATES, PREREGISTRATION, RUNNER, TEST,
        DEFAULT_OUTPUT, verify_inputs, configure,
    )
    for name, value in zip(names, values):
        setattr(prior, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(prior, name, value)


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_wrapper()
    try:
        return prior.run(output=output, device_name=device_name, resume=resume)
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
