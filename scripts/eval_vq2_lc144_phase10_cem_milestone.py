#!/usr/bin/env python3
"""Teacher-free paired raw-11 screen of LC143 against LC141."""

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
import scripts.eval_vq2_lc058_phase2_bias_milestone as milestone
import scripts.eval_vq2_lc110_phase8_success_action_bias as pairwise


TAG = "vq2_lc144_phase10_cem_milestone_001"
SCHEMA = "vq2_lc144_phase10_cem_milestone_report_v1"
GROUP_SIZE = 128
TARGET_PHASE = 10
TARGET_RAW_INDEX = 11
MAX_STEPS = 15_000
SEED = 432_160
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc141_phase9_split_batch_cem_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "33d8b837fe6c94c522179191e0c869e7b24a4839b96b96bfed5f5d09b0b057da"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "ba24a298edb46fb2ef25bf643c6733c0fce49314e74a4991c933896a444c7306"
CANDIDATE_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc143_phase10_split_batch_cem_001"
)
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "c4fe8fc5e069ea7696f52a81a9dd55a702957b92ad73285d55c06c6783e46ea6"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "8ad28f6c61589e9430e2813a6b7c7b37538cdd885c12c2d0778d0dbf09800bd9"
SELECTED_DELTA = (
    -0.17414399981498718,
    0.12499915808439255,
    -0.0180757287889719,
    -0.030255507677793503,
)
BIASES = (
    ("parent_lc141", (0.0, 0.0, 0.0, 0.0)),
    ("lc143_phase10_cem", SELECTED_DELTA),
)
PREREGISTRATION = ROOT / "docs/vq2_lc144_phase10_cem_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc144_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc144_phase10_cem_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def candidate_payload() -> dict[str, Any]:
    return torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    if candidate_index == 0:
        source = parent_state
    elif candidate_index == 1:
        source = candidate_payload()["model_state"]
    else:
        raise ValueError("LC144 has only parent and candidate")
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    delta = BIASES[candidate_index][1]
    return {
        "phase10_output_bias": list(delta),
        "bias_l2": sum(value * value for value in delta) ** 0.5,
        "checkpoint_sha256": (
            PARENT_CHECKPOINT_SHA256 if candidate_index == 0
            else CANDIDATE_CHECKPOINT_SHA256
        ),
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC144 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    candidate = candidate_payload()
    training = json.loads(CANDIDATE_REPORT.read_text())
    generation = training.get("generations", [{}])[0]
    if (
        parent.get("schema") != "vq2_lc141_phase9_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc141_phase9_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or candidate.get("schema") != "vq2_lc143_phase10_split_batch_cem_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or not candidate.get("frozen_non_phase10_state_exact")
        or tuple(candidate.get("phase10_pre_tanh_output_bias_delta", ())) != SELECTED_DELTA
        or training.get("schema") != "vq2_lc143_phase10_split_batch_cem_report_v1"
        or not training.get("training_admitted")
        or training.get("candidate_selected_for_screen", {}).get("sha256")
        != CANDIDATE_CHECKPOINT_SHA256
        or generation.get("query_agents") != 512
        or generation.get("target_passes") != 170
        or not generation.get("transport_pass")
        or training.get("safety", {}).get("flight_sim_packets_sent") != 0
        or training.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC141/LC143 do not authorize LC144")
    return parent


def configure() -> None:
    pairwise.TAG, pairwise.SCHEMA = TAG, SCHEMA
    pairwise.GROUP_SIZE, pairwise.PAIR_SIZE = GROUP_SIZE, 2 * GROUP_SIZE
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
    milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        ROOT / "scripts/train_vq2_lc143_phase10_split_batch_cem.py",
        ROOT / "scripts/eval_vq2_lc110_phase8_success_action_bias.py",
    )
    milestone.NEXT_AUTHORITY_SELECTED = (
        "Advance the source-locked trajectory with phase-11 Puffer search; no FlightSim authority."
    )
    milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC143 and retain LC105 as the safe frontier; do not run FlightSim."
    )
    milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form Puffer actors, each executing a 256-row paired batch; LC143 differs only at phase 10"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
        pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
    )
    (
        milestone.verify_inputs, milestone.build_candidate_context,
        milestone.initialize_actor_execution, milestone.execute_actor_actions,
        milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
        pairwise.candidate_state_for_index, pairwise.candidate_metadata_for_index,
    ) = (
        verify_inputs, pairwise.build_candidate_context,
        pairwise.initialize_actor_execution, pairwise.execute_actor_actions,
        candidate_state_for_index, candidate_metadata_for_index,
        candidate_state_for_index, candidate_metadata_for_index,
    )
    try:
        return milestone.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            milestone.verify_inputs, milestone.build_candidate_context,
            milestone.initialize_actor_execution, milestone.execute_actor_actions,
            milestone.candidate_state_for_index, milestone.candidate_metadata_for_index,
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
