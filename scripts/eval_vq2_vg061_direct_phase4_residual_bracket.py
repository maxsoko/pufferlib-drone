#!/usr/bin/env python3
"""Bracket the direct VG060 phase-4 indexed head on fresh six-gate runs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseResidualActor
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_vg055_count6_fraction_bracket as bracket
import scripts.eval_vq2_vg059_phase4_indexed_residual_bracket as indexed


TAG = "vq2_vg061_vg060_direct_phase4_residual_bracket_001"
SCHEMA = "vq2_vg061_direct_phase4_residual_bracket_report_v1"
STATE_SCHEMA = "vq2_vg061_direct_phase4_residual_bracket_state_v1"
ALPHAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)
OFFSETS = (160, 168, 176, 184)
SEEDS = (429193, 429194, 429195, 429196)
AGENTS = 64
EPISODES = 64
THREADS = 4
MAX_STEPS = 3072
PARENT_CHECKPOINT = indexed.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = indexed.PARENT_CHECKPOINT_SHA256
PARENT_REPORT = indexed.PARENT_REPORT
PARENT_REPORT_SHA256 = indexed.PARENT_REPORT_SHA256
PARENT_ADMISSION = indexed.PARENT_ADMISSION
PARENT_ADMISSION_SHA256 = indexed.PARENT_ADMISSION_SHA256
UPDATE_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg060_direct_phase4_indexed_residual_001/policy_best.pt"
)
UPDATE_CHECKPOINT_SHA256 = "32aba1580844d20b51ddf7930892bf5367d79a44ec0d6d4d5e316fdf10246a26"
UPDATE_REPORT = UPDATE_CHECKPOINT.parent / "report.json"
UPDATE_REPORT_SHA256 = "68f5295328f8453824f3e3fa3dc6e0053afcf68ad0f5757fdeaebf95ce083f82"
UPDATE_ADMISSION = ROOT / "docs/vq2_vg060_direct_phase4_indexed_residual_admission_2026-07-31.json"
UPDATE_ADMISSION_SHA256 = "13f6c301f208c616f8c28e5287a07d2e8cf17934e242e9b2c0db5d888075f1c6"
REJECTION = ROOT / "docs/vq2_vg059_phase4_indexed_residual_bracket_rejection_2026-07-31.json"
REJECTION_SHA256 = "a181066247c73ab7222323a945a0c777c6ba72f3d25ab2992e0255f7755e858b"
PREREGISTRATION = ROOT / "docs/vq2_vg061_direct_phase4_residual_bracket_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg061_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_bracket_source_identity = bracket.source_identity


def verify_inputs() -> None:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        PARENT_ADMISSION: PARENT_ADMISSION_SHA256,
        UPDATE_CHECKPOINT: UPDATE_CHECKPOINT_SHA256,
        UPDATE_REPORT: UPDATE_REPORT_SHA256,
        UPDATE_ADMISSION: UPDATE_ADMISSION_SHA256,
        REJECTION: REJECTION_SHA256,
        bracket.GOAL: bracket.GOAL_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG061 source evidence changed: {path}")
    if not PREREGISTRATION.is_file() or not RUNNER.is_file():
        raise RuntimeError("VG061 source lock is incomplete")
    report = json.loads(UPDATE_REPORT.read_text())
    admission = json.loads(UPDATE_ADMISSION.read_text())
    rejection = json.loads(REJECTION.read_text())
    if (
        report.get("schema") != "vq2_vg060_direct_phase4_indexed_residual_report_v1"
        or not report.get("completed") or not report.get("numerically_admitted")
        or not report.get("base_parameters_exact")
        or report.get("checkpoint_sha256") != UPDATE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG060 report does not authorize VG061")
    if (
        admission.get("schema") != "vq2_vg060_direct_phase4_indexed_residual_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != UPDATE_CHECKPOINT_SHA256
        or admission.get("live_authority")
    ):
        raise RuntimeError("VG060 admission changed")
    if (
        rejection.get("schema") != "vq2_vg059_phase4_indexed_residual_bracket_rejection_v1"
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("live_authority")
    ):
        raise RuntimeError("VG059 rejection changed")


def load_endpoints() -> tuple[dict[str, Any], dict[str, Any]]:
    base = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    update = torch.load(UPDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    residual_name = "indexed_phase_action_residual"
    if (
        base.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or update.get("schema") != "vq2_vg060_direct_phase4_indexed_residual_checkpoint_v1"
        or update.get("model", {}).get("class") != "VQ2IndexedPhaseResidualActor"
        or not update.get("numerically_admitted")
        or residual_name not in update.get("model_state", {})
    ):
        raise RuntimeError("VG061 endpoint contract changed")
    for name, value in base["model_state"].items():
        if not torch.equal(value, update["model_state"][name]):
            raise RuntimeError(f"VG060 changed base parameter {name}")
    residual = update["model_state"][residual_name]
    if torch.count_nonzero(residual[:4]) or torch.count_nonzero(residual[5:]):
        raise RuntimeError("VG060 changed a non-target indexed head")
    parent = copy.deepcopy(update)
    parent["tag"] = f"{TAG}_zero_residual_parent"
    parent["model_state"][residual_name] = torch.zeros_like(residual)
    return parent, update


def source_identity() -> dict[str, Any]:
    identity = _bracket_source_identity()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/eval_vq2_vg059_phase4_indexed_residual_bracket.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    ):
        identity["source_sha256"][str(path.relative_to(ROOT))] = sha256_path(path)
    identity["indexed_residual"] = {
        "target_phase_index": 4,
        "phases_0_through_3_exact": True,
        "non_target_heads_zero": True,
    }
    return identity


def configure() -> None:
    bracket.TAG = TAG; bracket.SCHEMA = SCHEMA; bracket.STATE_SCHEMA = STATE_SCHEMA
    bracket.ALPHAS = ALPHAS; bracket.OFFSETS = OFFSETS; bracket.SEEDS = SEEDS
    bracket.AGENTS = AGENTS; bracket.EPISODES = EPISODES
    bracket.THREADS = THREADS; bracket.MAX_STEPS = MAX_STEPS
    bracket.PARENT_CHECKPOINT = PARENT_CHECKPOINT; bracket.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    bracket.PARENT_REPORT = PARENT_REPORT; bracket.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    bracket.PARENT_ADMISSION = PARENT_ADMISSION; bracket.PARENT_ADMISSION_SHA256 = PARENT_ADMISSION_SHA256
    bracket.UPDATE_CHECKPOINT = UPDATE_CHECKPOINT; bracket.UPDATE_CHECKPOINT_SHA256 = UPDATE_CHECKPOINT_SHA256
    bracket.UPDATE_REPORT = UPDATE_REPORT; bracket.UPDATE_REPORT_SHA256 = UPDATE_REPORT_SHA256
    bracket.UPDATE_ADMISSION = UPDATE_ADMISSION; bracket.UPDATE_ADMISSION_SHA256 = UPDATE_ADMISSION_SHA256
    bracket.REJECTION = REJECTION; bracket.REJECTION_SHA256 = REJECTION_SHA256
    bracket.PREREGISTRATION = PREREGISTRATION; bracket.RUNNER = RUNNER
    bracket.DEFAULT_OUTPUT = DEFAULT_OUTPUT; bracket.ACTOR_CLASS = VQ2IndexedPhaseResidualActor
    bracket.verify_inputs = verify_inputs; bracket.load_endpoints = load_endpoints
    bracket.source_identity = source_identity; bracket.qualifies = indexed.qualifies
    bracket.interpolation.interpolate_state = indexed.interpolate_state


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    configure()
    return bracket.run(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
