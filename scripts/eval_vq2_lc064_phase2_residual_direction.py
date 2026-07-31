#!/usr/bin/env python3
"""Fast causal screen of state-dependent phase-2 residual directions."""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc058_phase2_bias_milestone as base


TAG = "vq2_lc064_phase2_residual_direction_001"
SCHEMA = "vq2_lc064_phase2_residual_direction_report_v1"
GROUP_SIZE = 32
TARGET_PHASE = 2
RESIDUAL_NAMES = (
    "indexed_phase_residual_input",
    "indexed_phase_residual_input_bias",
    "indexed_phase_residual_output",
    "indexed_phase_residual_output_bias",
)
ENCODER_NAMES = RESIDUAL_NAMES[:2]
DECODER_NAMES = RESIDUAL_NAMES[2:]
SPECS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("baseline_lc062", 0.0, ()),
    ("full_m0p0025", -0.0025, RESIDUAL_NAMES),
    ("full_m0p0050", -0.0050, RESIDUAL_NAMES),
    ("decoder_m0p0025", -0.0025, DECODER_NAMES),
    ("decoder_p0p0025", 0.0025, DECODER_NAMES),
    ("output_weight_m0p0025", -0.0025, (RESIDUAL_NAMES[2],)),
    ("encoder_m0p0025", -0.0025, ENCODER_NAMES),
    ("encoder_p0p0025", 0.0025, ENCODER_NAMES),
)
# The generic evaluator uses this tuple for candidate names only. Surgery and
# metadata are supplied by the exact state-dependent hooks below.
BIAS_CANDIDATES = tuple((name, (0.0, 0.0, 0.0, 0.0)) for name, _, _ in SPECS)
SEED = 431640
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc062_phase2_bias_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "06381161445f5f207a3b12f7c97b82c884f91a71fdb3e34e05189255cc58d04a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "5bf0fc1f95fec795dcf0c8212f7838a0f18a683baaece679228e87e667085af5"
REFERENCE_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc027_phase2_interpolation_bracket_001"
)
REFERENCE_CHECKPOINT = REFERENCE_DIR / "a0p050/policy_selected.pt"
REFERENCE_CHECKPOINT_SHA256 = "4cdd636ab0a1d6741614c5f2f74a545a213492306e90d5d89dffff0430240bb9"
REFERENCE_REPORT = REFERENCE_DIR / "report.json"
REFERENCE_REPORT_SHA256 = "e161ca51447e4a324529f37b2d8175fd6d6667e77f67f3b61f805452100bbef2"
FIT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc029_index2_student_dagger_head_001"
)
FIT_CHECKPOINT = FIT_DIR / "policy_best.pt"
FIT_CHECKPOINT_SHA256 = "eb72562c2655c3d5cb1300129fb85cd56f3cc1aa63a39766b3b27e81c38cfa61"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "3d9509e531452fb4c9a8103410643ece2521eab823ff66cd4848c0045c810219"
LC063_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc063_phase2_bias_multiaxis_001/report.json"
)
LC063_REPORT_SHA256 = "bd3eb8bad1107caab3f9f62348c8800eacf32c0b751c2d08e1a35260ca371bf6"
PREREGISTRATION = ROOT / "docs/vq2_lc064_phase2_residual_direction_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc064_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc064_phase2_residual_direction.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


@functools.lru_cache(maxsize=1)
def direction_rows() -> dict[str, torch.Tensor]:
    reference = torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=False)
    fitted = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    return {
        name: (
            fitted["model_state"][name][TARGET_PHASE]
            - reference["model_state"][name][TARGET_PHASE]
        ).detach().cpu()
        for name in RESIDUAL_NAMES
    }


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    _, alpha, components = SPECS[candidate_index]
    directions = direction_rows()
    for name in components:
        state[name][TARGET_PHASE].add_(directions[name], alpha=alpha)
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    _, alpha, components = SPECS[candidate_index]
    directions = direction_rows()
    squared = sum(float(directions[name].double().square().sum()) for name in components)
    delta_l2 = abs(alpha) * math.sqrt(squared)
    return {
        "alpha": alpha,
        "components": list(components),
        "parameter_delta_l2": delta_l2,
        "bias_l2": delta_l2,
    }


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, torch.Tensor]:
    states = [candidate_state_for_index(payload["model_state"], index) for index in range(len(SPECS))]
    context: dict[str, torch.Tensor] = {}
    for name in RESIDUAL_NAMES:
        rows = torch.stack([state[name][TARGET_PHASE] for state in states])
        context[name] = rows.repeat_interleave(GROUP_SIZE, dim=0).to(device)
        context[f"parent_{name}"] = payload["model_state"][name][TARGET_PHASE].to(device)
    return context


def apply_candidate_actions(
    actor_output: Any,
    next_recurrent: torch.Tensor,
    held_progress: torch.Tensor,
    context: dict[str, torch.Tensor],
) -> torch.Tensor:
    hidden = next_recurrent[0]
    feature = torch.tanh(
        torch.einsum("bh,brh->br", hidden, context[RESIDUAL_NAMES[0]])
        + context[RESIDUAL_NAMES[1]]
    )
    candidate_residual = (
        torch.einsum("br,bor->bo", feature, context[RESIDUAL_NAMES[2]])
        + context[RESIDUAL_NAMES[3]]
    )
    parent_feature = torch.tanh(
        hidden @ context[f"parent_{RESIDUAL_NAMES[0]}"].T
        + context[f"parent_{RESIDUAL_NAMES[1]}"]
    )
    parent_residual = (
        parent_feature @ context[f"parent_{RESIDUAL_NAMES[2]}"].T
        + context[f"parent_{RESIDUAL_NAMES[3]}"]
    )
    phase = torch.round(held_progress[:, 0] * base.OFFICIAL_PROGRESS_SCALE).to(torch.long)
    delta = candidate_residual - parent_residual
    return torch.tanh(
        actor_output.pre_tanh_mean + (phase == TARGET_PHASE)[:, None] * delta
    )


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.GROUP_SIZE = GROUP_SIZE
    base.BIAS_CANDIDATES = BIAS_CANDIDATES
    base.GROUPS = len(BIAS_CANDIDATES)
    base.TOTAL_AGENTS = GROUP_SIZE * base.GROUPS
    base.EPISODES = base.TOTAL_AGENTS
    base.SEED = SEED
    base.MINIMUM_PASS_GAIN = MINIMUM_PASS_GAIN
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT = PARENT_REPORT
    base.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.PARENT_CHILD_REPORT = PARENT_REPORT
    base.PARENT_CHILD_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), REFERENCE_CHECKPOINT, REFERENCE_REPORT,
        FIT_CHECKPOINT, FIT_REPORT, LC063_REPORT,
    )
    base.NEXT_AUTHORITY_SELECTED = (
        "Run one larger different-seed Gate-3 confirmation of the selected "
        "state-dependent whole-Puffer residual surgery."
    )
    base.NEXT_AUTHORITY_NONE = "Reject these residual directions and retain LC062."
    base.VECTORIZED_CHECKPOINT_SURGERY = (
        "exact per-group phase-2 indexed MLP parameter surgery evaluated from the recurrent hidden state"
    )


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        REFERENCE_CHECKPOINT: REFERENCE_CHECKPOINT_SHA256,
        REFERENCE_REPORT: REFERENCE_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
        LC063_REPORT: LC063_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC064 bound input changed: {path}")
    parent_report = json.loads(PARENT_REPORT.read_text())
    fit_report = json.loads(FIT_REPORT.read_text())
    rejected = json.loads(LC063_REPORT.read_text())
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    reference = torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=False)
    fitted = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        parent_report.get("schema") != "vq2_lc062_phase2_bias_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent.get("schema") != "vq2_lc062_phase2_bias_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or reference.get("schema") != "vq2_lc027_phase2_interpolation_checkpoint_v1"
        or fitted.get("schema") != "vq2_lc029_index2_student_dagger_head_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fit_report.get("selected_validation", {}).get("phases", {}).get("2", {}).get(
            "improvement_factor", 0.0
        ) < 2.32
        or rejected.get("schema") != "vq2_lc063_phase2_bias_multiaxis_report_v1"
        or rejected.get("causal_screen_selected") is not None
        or not rejected.get("diagnostic_valid")
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC062/LC063 and the phase-2 fit do not authorize LC064")
    return parent


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    configure()
    originals = (
        base.verify_inputs, base.build_candidate_context,
        base.apply_candidate_actions, base.candidate_state_for_index,
        base.candidate_metadata_for_index,
    )
    (
        base.verify_inputs, base.build_candidate_context,
        base.apply_candidate_actions, base.candidate_state_for_index,
        base.candidate_metadata_for_index,
    ) = (
        verify_inputs, build_candidate_context, apply_candidate_actions,
        candidate_state_for_index, candidate_metadata_for_index,
    )
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            base.verify_inputs, base.build_candidate_context,
            base.apply_candidate_actions, base.candidate_state_for_index,
            base.candidate_metadata_for_index,
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
