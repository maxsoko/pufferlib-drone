#!/usr/bin/env python3
"""Exact-context output-scale bracket for the LC202 stacked adapter."""

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
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save
from scripts.eval_vq2_lc015_restore_vg071_head1 import state_sha256
from scripts.eval_vq2_lc203_stacked_adapter_raw18_milestone import (
    load_actor,
)
import scripts.eval_vq2_lc196_phase16_17_joint_interpolation_bracket as prior


BASE_CONFIGURE = prior.configure
TAG = "vq2_lc204_stacked_adapter_scale_bracket_001"
SCHEMA = "vq2_lc204_stacked_adapter_scale_bracket_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc204_stacked_adapter_scale_checkpoint_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
OUTPUT_SCALES = (0.0, 0.003, 0.01, 0.03, 0.10, 0.30, 0.60)
SCALE_PAIRS = tuple((scale, 0.0) for scale in OUTPUT_SCALES)
CANDIDATES = tuple(
    (
        "parent_lc189" if scale == 0.0 else f"lc202_output_scale_{scale:g}",
        (0.0,) * 4,
    )
    for scale in OUTPUT_SCALES
)
FIT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc202_phase16_17_stacked_adapter_001"
FIT_CHECKPOINT = FIT_DIR / "policy_selected.pt"
FIT_CHECKPOINT_SHA256 = "9574928925d778910729d5fa55a1bd86a05767e29ddff8564bb15ea375286a73"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "c048877bf078c636fa04b6ac36ef057711d092e3a47cc3e48159b093b9fada12"
LC203_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc203_stacked_adapter_raw18_milestone_001/report.json"
LC203_REPORT_SHA256 = "e5e6caa740dc46f20d19eb7f9969d26b16a8b88cf3b75b65194b6dddf7540b6a"
PREREGISTRATION = ROOT / "docs/vq2_lc204_stacked_adapter_scale_bracket_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc204_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc204_stacked_adapter_scale_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def fit_payload() -> dict[str, Any]:
    return torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    scale = OUTPUT_SCALES[candidate_index]
    if scale == 0.0:
        return {
            name: value.detach().cpu().clone()
            for name, value in parent_state.items()
        }
    state = {
        name: value.detach().cpu().clone()
        for name, value in fit_payload()["model_state"].items()
    }
    for name in (
        "continuation_adapter_output.weight",
        "continuation_adapter_output.bias",
    ):
        state[name].mul_(scale)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    scale = OUTPUT_SCALES[candidate_index]
    return {
        "adapter_output_scale": scale,
        "phase16_fit_scale": scale,
        "phase17_fit_scale": scale,
        "bias_l2": scale,
        "endpoint_phases": [16, 17] if scale else [],
    }


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline = items[0]
    eligible = [
        item
        for item in items[1:]
        if baseline["target_passes"] == 0
        and baseline.get("maximum_raw_index_distribution", {}).get("17")
        == GROUP_SIZE
        and item["target_passes"] == GROUP_SIZE
        and item["paired_target_gains_vs_baseline"] == GROUP_SIZE
        and item["paired_target_losses_vs_baseline"] == 0
        and item["transport_pass"]
        and item["pre_target_terminals"] <= baseline["pre_target_terminals"]
    ]
    return min(eligible, key=lambda item: item["adapter_output_scale"]) if eligible else None


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        prior.PARENT_CHECKPOINT: prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT: prior.PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
        LC203_REPORT: LC203_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC204 bound input changed: {path}")
    parent = prior.parent_payload()
    endpoint = fit_payload()
    fit_report = json.loads(FIT_REPORT.read_text())
    rejected = json.loads(LC203_REPORT.read_text())
    baseline, fitted = rejected.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or endpoint.get("schema") != "vq2_lc202_phase16_17_stacked_adapter_checkpoint_v1"
        or endpoint.get("numerically_admitted")
        or endpoint.get("parent_checkpoint_sha256") != prior.PARENT_CHECKPOINT_SHA256
        or fit_report.get("schema") != "vq2_lc202_phase16_17_stacked_adapter_report_v1"
        or fit_report.get("validation_improvement_factor", 0.0) < 180.0
        or not fit_report.get("frozen_lc189_state_exact")
        or rejected.get("schema")
        != "vq2_lc203_stacked_adapter_raw18_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or baseline.get("maximum_raw_index_distribution", {}).get("17") != 128
        or fitted.get("maximum_raw_index_distribution", {}).get("16") != 128
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC202/LC203 do not authorize LC204")
    zero_state = candidate_state_for_index(parent["model_state"], 0)
    if state_sha256(zero_state) != state_sha256(parent["model_state"]):
        raise RuntimeError("LC204 zero scale is not exact LC189")
    for name, value in parent["model_state"].items():
        if not torch.equal(endpoint["model_state"][name], value):
            raise RuntimeError(f"LC202 changed frozen tensor {name}")
    return parent


def configure() -> None:
    BASE_CONFIGURE()
    prior.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT, LC203_REPORT,
        ROOT / "scripts/eval_vq2_lc203_stacked_adapter_raw18_milestone.py",
        ROOT / "scripts/train_vq2_lc202_phase16_17_stacked_adapter.py",
    )
    prior.milestone.NEXT_AUTHORITY_SELECTED = (
        "Confirm the selected raw-18 stacked adapter independently; no FlightSim authority."
    )
    prior.milestone.NEXT_AUTHORITY_NONE = (
        "Reject the LC202 adapter direction and retain LC189; do not run FlightSim."
    )
    prior.milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "seven complete recurrent Puffers, each executing the established "
        "256-row context; only the new continuation output head is scaled"
    )


def save_selected_checkpoint(
    report: dict[str, Any], *, output: Path, parent: dict[str, Any]
) -> None:
    selected = report.get("causal_screen_selected")
    if selected is None:
        return
    state = candidate_state_for_index(
        parent["model_state"], selected["candidate_index"]
    )
    if state_sha256(state) != selected["candidate_state_sha256"]:
        raise RuntimeError("LC204 selected state changed after screen")
    endpoint = fit_payload()
    checkpoint = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": TAG,
        "model": endpoint["model"],
        "model_state": state,
        "parent_checkpoint_sha256": prior.PARENT_CHECKPOINT_SHA256,
        "fit_checkpoint_sha256": FIT_CHECKPOINT_SHA256,
        "adapter_output_scale": selected["adapter_output_scale"],
        "source_trajectory_admitted": True,
        "independent_screen_admitted": False,
        "deployment_candidate": False,
        "safety": report["safety"],
    }
    atomic_torch_save(output / "policy_source_selected.pt", checkpoint)


def configure_wrapper() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "CHECKPOINT_SCHEMA", "GROUP_SIZE", "PAIR_SIZE",
        "SCALE_PAIRS", "CANDIDATES", "FIT_CHECKPOINT", "FIT_CHECKPOINT_SHA256",
        "FIT_REPORT", "FIT_REPORT_SHA256", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "fit_payload", "candidate_state_for_index",
        "candidate_metadata_for_index", "choose_candidate", "verify_inputs",
        "configure", "save_selected_checkpoint", "load_actor",
    )
    originals = tuple(getattr(prior, name) for name in names)
    values = (
        TAG, SCHEMA, CHECKPOINT_SCHEMA, GROUP_SIZE, PAIR_SIZE, SCALE_PAIRS,
        CANDIDATES, FIT_CHECKPOINT, FIT_CHECKPOINT_SHA256, FIT_REPORT,
        FIT_REPORT_SHA256, PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
        fit_payload, candidate_state_for_index, candidate_metadata_for_index,
        choose_candidate, verify_inputs, configure, save_selected_checkpoint,
        load_actor,
    )
    for name, value in zip(names, values):
        setattr(prior, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(prior, name, value)


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
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
