#!/usr/bin/env python3
"""Paired 24-gate screen of the confirmed constrained phase-2 endpoint."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc061_phase2_bias_full_course as base
import scripts.eval_vq2_lc064_phase2_residual_direction as residual
import scripts.eval_vq2_lc071_phase2_constrained_endpoint_screen as endpoint_screen


TAG = "vq2_lc073_phase2_constrained_endpoint_full_course_001"
SCHEMA = "vq2_lc073_phase2_constrained_endpoint_full_course_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc073_phase2_constrained_endpoint_full_course_checkpoint_v1"
SEED = 431730
PARENT_CHECKPOINT = endpoint_screen.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = endpoint_screen.PARENT_CHECKPOINT_SHA256
PARENT_REPORT = endpoint_screen.PARENT_REPORT
PARENT_REPORT_SHA256 = endpoint_screen.PARENT_REPORT_SHA256
FIT_CHECKPOINT = endpoint_screen.FIT_CHECKPOINT
FIT_CHECKPOINT_SHA256 = endpoint_screen.FIT_CHECKPOINT_SHA256
FIT_REPORT = endpoint_screen.FIT_REPORT
FIT_REPORT_SHA256 = endpoint_screen.FIT_REPORT_SHA256
LC072_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc072_phase2_constrained_endpoint_confirmation_001/report.json"
)
LC072_REPORT_SHA256 = "8f4074df435c1f983f82014b83ef68055d5deff6848fdc43d955d98719255743"
PREREGISTRATION = ROOT / "docs/vq2_lc073_phase2_constrained_endpoint_full_course_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc073_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc073_phase2_constrained_endpoint_full_course.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


@functools.lru_cache(maxsize=1)
def endpoint_decoder() -> dict[str, torch.Tensor]:
    return torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)


def build_selected_candidate_state(
    parent_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    endpoint = endpoint_decoder()
    state["indexed_phase_residual_output"][base.TARGET_PHASE].copy_(
        endpoint["output_weight"]
    )
    state["indexed_phase_residual_output_bias"][base.TARGET_PHASE].copy_(
        endpoint["output_bias"]
    )
    return state


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, torch.Tensor]:
    states = [
        {name: value.detach().cpu().clone() for name, value in payload["model_state"].items()},
        build_selected_candidate_state(payload["model_state"]),
    ]
    context: dict[str, torch.Tensor] = {}
    for name in residual.RESIDUAL_NAMES:
        rows = torch.stack([state[name][base.TARGET_PHASE] for state in states])
        context[name] = rows.repeat_interleave(base.GROUP_SIZE, dim=0).to(device)
        context[f"parent_{name}"] = payload["model_state"][name][base.TARGET_PHASE].to(device)
    return context


def selected_candidate_metadata() -> dict[str, Any]:
    return {
        "decoder_endpoint_checkpoint_sha256": FIT_CHECKPOINT_SHA256,
        "endpoint_source_alpha": 0.075,
    }


def checkpoint_surgery_metadata(candidate_state_sha256: str) -> dict[str, Any]:
    return {
        "operation": "phase2_constrained_decoder_endpoint_replacement",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "target_phase": base.TARGET_PHASE,
        "decoder_endpoint_checkpoint_sha256": FIT_CHECKPOINT_SHA256,
        "endpoint_source_alpha": 0.075,
        "candidate_state_sha256": candidate_state_sha256,
    }


def configure() -> None:
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.SEED = SEED
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.LC060_REPORT = LC072_REPORT
    base.LC060_REPORT_SHA256 = LC072_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.CANDIDATE_NAME = "constrained_endpoint"
    base.NEXT_AUTHORITY_SELECTED = (
        "Retain LC073 as the offline whole-Puffer frontier and rediagnose the earliest remaining high-mass phase on the 24-gate proxy."
    )
    base.NEXT_AUTHORITY_NONE = "Reject the constrained endpoint and retain LC062."
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), PARENT_REPORT, LC072_REPORT,
        FIT_CHECKPOINT, FIT_REPORT,
        ROOT / "scripts/eval_vq2_lc071_phase2_constrained_endpoint_screen.py",
        ROOT / "scripts/eval_vq2_lc072_phase2_constrained_endpoint_confirmation.py",
        ROOT / "scripts/fit_vq2_lc070_phase2_constrained_endpoint.py",
    )


def verify_inputs() -> dict[str, Any]:
    parent = endpoint_screen.verify_inputs()
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
        LC072_REPORT: LC072_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC073 bound input changed: {path}")
    report = json.loads(LC072_REPORT.read_text())
    selected = report.get("causal_screen_selected", {})
    endpoint = endpoint_decoder()
    if (
        report.get("schema") != "vq2_lc072_phase2_constrained_endpoint_confirmation_report_v1"
        or not report.get("diagnostic_valid")
        or selected.get("endpoint_interpolation_alpha") != 1.0
        or selected.get("gate3_passes") != 31
        or selected.get("paired_gate3_gains_vs_baseline") != 10
        or selected.get("paired_gate3_losses_vs_baseline") != 3
        or report.get("items", [{}])[0].get("gate3_passes") != 24
        or endpoint.get("schema") != "vq2_lc070_phase2_constrained_endpoint_checkpoint_v1"
        or not endpoint.get("numerically_admitted")
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC070/LC072 do not authorize LC073")
    return parent


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        base.verify_inputs, base.build_candidate_context,
        base.apply_candidate_actions, base.build_selected_candidate_state,
        base.selected_candidate_metadata, base.checkpoint_surgery_metadata,
    )
    (
        base.verify_inputs, base.build_candidate_context,
        base.apply_candidate_actions, base.build_selected_candidate_state,
        base.selected_candidate_metadata, base.checkpoint_surgery_metadata,
    ) = (
        verify_inputs, build_candidate_context, residual.apply_candidate_actions,
        build_selected_candidate_state, selected_candidate_metadata,
        checkpoint_surgery_metadata,
    )
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            base.verify_inputs, base.build_candidate_context,
            base.apply_candidate_actions, base.build_selected_candidate_state,
            base.selected_candidate_metadata, base.checkpoint_surgery_metadata,
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
