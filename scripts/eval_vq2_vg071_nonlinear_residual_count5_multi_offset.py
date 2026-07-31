#!/usr/bin/env python3
"""Paired multi-offset count-5 confirmation of the selected VG070 actor."""

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

from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseMLPResidualActor
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_vg055_count6_fraction_bracket as bracket
import scripts.eval_vq2_vg070_nonlinear_residual_count5_scale_bracket as vg070


TAG = "vq2_vg071_nonlinear_residual_count5_multi_offset_001"
SCHEMA = "vq2_vg071_nonlinear_residual_count5_multi_offset_report_v1"
STATE_SCHEMA = "vq2_vg071_nonlinear_residual_count5_multi_offset_state_v1"
ALPHAS = (0.0, 1.0)
OFFSETS = (216, 224, 232, 240)
SEEDS = (429205, 429206, 429207, 429208)
AGENTS = 32
EPISODES = 32
THREADS = 4
MAX_STEPS = 23040
NUM_GATES = 5
PARENT_CHECKPOINT = vg070.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = vg070.PARENT_CHECKPOINT_SHA256
PARENT_REPORT = vg070.PARENT_REPORT
PARENT_REPORT_SHA256 = vg070.PARENT_REPORT_SHA256
PARENT_ADMISSION = vg070.PARENT_ADMISSION
PARENT_ADMISSION_SHA256 = vg070.PARENT_ADMISSION_SHA256
UPDATE_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg070_nonlinear_residual_count5_scale_bracket_001/policy_selected.pt"
)
UPDATE_CHECKPOINT_SHA256 = "8fb8979ba9f2b7ca21bbcd9372b5b0f40335d072e61212a43dfe27e6a703607c"
UPDATE_REPORT = UPDATE_CHECKPOINT.parent / "report.json"
UPDATE_REPORT_SHA256 = "5a1582f22b523eeb3c4f0f80dc5c30c2c607bcb3c6cffa6033916be969bbabf8"
UPDATE_ADMISSION = ROOT / "docs/vq2_vg070_nonlinear_residual_count5_scale_bracket_admission_2026-07-31.json"
UPDATE_ADMISSION_SHA256 = "680134be22a7147e68b8f79b410312295505db02fbc4805812d5d4c487257e1e"
REJECTION = vg070.REJECTION
REJECTION_SHA256 = vg070.REJECTION_SHA256
PREREGISTRATION = ROOT / "docs/vq2_vg071_nonlinear_residual_count5_multi_offset_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg071_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_vg071_nonlinear_residual_count5_multi_offset.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
EXTRA_SOURCE_PATHS: tuple[Path, ...] = (Path(__file__).resolve(), TEST)


def verify_inputs() -> None:
    vg070.verify_inputs()
    expected = {
        UPDATE_CHECKPOINT: UPDATE_CHECKPOINT_SHA256,
        UPDATE_REPORT: UPDATE_REPORT_SHA256,
        UPDATE_ADMISSION: UPDATE_ADMISSION_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG071 evidence changed: {path}")
    report = json.loads(UPDATE_REPORT.read_text())
    admission = json.loads(UPDATE_ADMISSION.read_text())
    successor = admission.get("authorized_successor", {})
    if (
        report.get("schema") != vg070.SCHEMA
        or not report.get("numerically_admitted")
        or report.get("selected_alpha") != 0.05
        or report.get("checkpoint_sha256") != UPDATE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG070 selected endpoint changed")
    if (
        admission.get("schema")
        != "vq2_vg070_nonlinear_residual_count5_scale_bracket_admission_v1"
        or not admission.get("numerically_admitted")
        or successor.get("tag") != TAG
        or not successor.get("rollout_authority")
        or admission.get("live_authority")
    ):
        raise RuntimeError("VG070 admission does not authorize VG071")


def load_endpoints() -> tuple[dict[str, Any], dict[str, Any]]:
    base = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    selected = torch.load(UPDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        base.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or selected.get("schema")
        != "vq2_vg070_nonlinear_scale_synthetic_checkpoint_v1"
        or not selected.get("numerically_admitted")
        or selected.get("interpolation", {}).get("alpha") != 0.05
        or selected.get("model", {}).get("class")
        != "VQ2IndexedPhaseMLPResidualActor"
    ):
        raise RuntimeError("VG071 endpoint contract changed")
    update_state = {
        name: value.clone() for name, value in selected["model_state"].items()
    }
    parent_state = {name: value.clone() for name, value in update_state.items()}
    parent_state["indexed_phase_residual_output"].zero_()
    parent_state["indexed_phase_residual_output_bias"].zero_()
    common = {
        **base,
        "schema": "vq2_vg071_paired_synthetic_checkpoint_v1",
        "model": copy.deepcopy(selected["model"]),
        "selected_source_sha256": UPDATE_CHECKPOINT_SHA256,
        "synthetic_mapping": {
            "alpha_zero_outputs_exactly_zero": True,
            "alpha_one_is_vg070_selected_exact": True,
            "vg070_source_alpha": 0.05,
        },
    }
    return (
        {**copy.deepcopy(common), "tag": f"{TAG}_parent", "model_state": parent_state},
        {**copy.deepcopy(common), "tag": f"{TAG}_selected", "model_state": update_state},
    )


def qualifies(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return bool(
        parent["all_hard_transport_pass"] and candidate["all_hard_transport_pass"]
        and candidate["gate_reach"]["1"] >= parent["gate_reach"]["1"]
        and candidate["crashes"] <= parent["crashes"]
        and candidate["successes"] > parent["successes"]
    )


def selection_key(item: dict[str, Any]) -> tuple[float, ...]:
    summary, gates = item["aggregate"], item["aggregate"]["gate_reach"]
    return (
        float(summary["successes"]), -float(summary["crashes"]),
        float(gates["5"]), float(gates["4"]), float(gates["3"]),
        float(gates["2"]), float(summary["mean_gates_passed"]),
    )


def configure() -> None:
    bracket.TAG = TAG; bracket.SCHEMA = SCHEMA; bracket.STATE_SCHEMA = STATE_SCHEMA
    bracket.ALPHAS = ALPHAS; bracket.OFFSETS = OFFSETS; bracket.SEEDS = SEEDS
    bracket.AGENTS = AGENTS; bracket.EPISODES = EPISODES; bracket.THREADS = THREADS
    bracket.MAX_STEPS = MAX_STEPS; bracket.NUM_GATES = NUM_GATES
    bracket.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    bracket.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    bracket.PARENT_REPORT = PARENT_REPORT; bracket.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    bracket.PARENT_ADMISSION = PARENT_ADMISSION
    bracket.PARENT_ADMISSION_SHA256 = PARENT_ADMISSION_SHA256
    bracket.UPDATE_CHECKPOINT = UPDATE_CHECKPOINT
    bracket.UPDATE_CHECKPOINT_SHA256 = UPDATE_CHECKPOINT_SHA256
    bracket.UPDATE_REPORT = UPDATE_REPORT; bracket.UPDATE_REPORT_SHA256 = UPDATE_REPORT_SHA256
    bracket.UPDATE_ADMISSION = UPDATE_ADMISSION
    bracket.UPDATE_ADMISSION_SHA256 = UPDATE_ADMISSION_SHA256
    bracket.REJECTION = REJECTION; bracket.REJECTION_SHA256 = REJECTION_SHA256
    bracket.PREREGISTRATION = PREREGISTRATION; bracket.RUNNER = RUNNER
    bracket.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    bracket.ACTOR_CLASS = VQ2IndexedPhaseMLPResidualActor
    bracket.EXTRA_SOURCE_PATHS = EXTRA_SOURCE_PATHS
    bracket.verify_inputs = verify_inputs; bracket.load_endpoints = load_endpoints
    bracket.qualifies = qualifies; bracket.selection_key = selection_key


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False):
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
