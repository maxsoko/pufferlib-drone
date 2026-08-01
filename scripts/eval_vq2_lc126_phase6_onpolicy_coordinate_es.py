#!/usr/bin/env python3
"""One vectorized closed-loop coordinate-ES generation around LC105 phase 6."""

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
import scripts.eval_vq2_lc058_phase2_bias_milestone as base


TAG = "vq2_lc126_phase6_onpolicy_coordinate_es_001"
SCHEMA = "vq2_lc126_phase6_onpolicy_coordinate_es_report_v1"
GROUP_SIZE = 32
TARGET_PHASE = 6
TARGET_RAW_INDEX = 10
MAX_STEPS = 12_000
BIASES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("baseline_lc105", (0.0, 0.0, 0.0, 0.0)),
    ("pitch_m0p0025", (-0.0025, 0.0, 0.0, 0.0)),
    ("pitch_p0p0025", (0.0025, 0.0, 0.0, 0.0)),
    ("roll_m0p0125", (0.0, -0.0125, 0.0, 0.0)),
    ("roll_p0p0125", (0.0, 0.0125, 0.0, 0.0)),
    ("thrust_m0p00625", (0.0, 0.0, -0.00625, 0.0)),
    ("thrust_p0p00625", (0.0, 0.0, 0.00625, 0.0)),
    ("joint_half_direction", (0.0025, -0.0125, 0.00625, 0.0)),
)
SEED = 432_050
MINIMUM_PASS_GAIN = 1
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc105_phase7_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "005e5e7929258fd282ab390fd230fda817700afaee790e78144200e67b3e10a4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c614e6929282399cac2f18465084f8337ed8770389a74e48b86ff1fcf4315e7d"
LC125_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc125_lc123_phase8_9_rescue_features_001"
)
LC125_REPORT = LC125_DIR / "report.json"
LC125_REPORT_SHA256 = "0021aa7d0a91825dd24ce1fe36ef75234170e5cf0a8cc2eb850a4854888fe3e4"
LC125_FEATURES = LC125_DIR / "features.bin"
LC125_FEATURES_SHA256 = "d505e3977e795aa81585bee69e1b9bad49d463f49c94fc9f2c7f07c2017444b9"
PREREGISTRATION = ROOT / "docs/vq2_lc126_phase6_onpolicy_coordinate_es_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc126_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc126_phase6_onpolicy_coordinate_es.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in parent_state.items()
    }
    state["indexed_phase_residual_output_bias"][TARGET_PHASE].add_(
        torch.tensor(BIASES[candidate_index][1], dtype=torch.float32)
    )
    return state


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    values = BIASES[candidate_index][1]
    return {
        "phase6_output_bias": list(values),
        "coordinate_es_generation": 1,
        "bias_l2": sum(value * value for value in values) ** 0.5,
    }


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC125_REPORT: LC125_REPORT_SHA256,
        LC125_FEATURES: LC125_FEATURES_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC126 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rescue = json.loads(LC125_REPORT.read_text())
    control, intervention = rescue.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or rescue.get("schema")
        != "vq2_lc125_lc123_phase8_9_rescue_features_report_v1"
        or not rescue.get("diagnostic_valid")
        or not rescue.get("training_oracle_rescued_phase8_9")
        or rescue.get("training_dataset_admitted")
        or rescue.get("failed_admission_predicates") != ["splittable_outcomes"]
        or rescue.get("feature_records") != 6_150
        or len(rescue.get("query_outcome_success_agents", [])) != 1
        or len(rescue.get("query_outcome_failure_agents", [])) != 2
        or control.get("target_passes") != 0
        or intervention.get("target_passes") != 1
        or intervention.get("paired_target_losses_vs_control") != 0
        or rescue.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rescue.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC125 do not authorize LC126")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA = TAG, SCHEMA
    base.GROUP_SIZE = GROUP_SIZE
    base.BIAS_CANDIDATES = BIASES
    base.GROUPS = len(BIASES)
    base.TOTAL_AGENTS = GROUP_SIZE * base.GROUPS
    base.EPISODES = base.TOTAL_AGENTS
    base.SEED = SEED
    base.TARGET_PHASE = TARGET_PHASE
    base.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    base.MAX_STEPS = MAX_STEPS
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
        Path(__file__).resolve(), LC125_REPORT, LC125_FEATURES,
    )
    base.NEXT_AUTHORITY_SELECTED = (
        "Run one 128-pair confirmation of the selected phase-6 on-policy bias."
    )
    base.NEXT_AUTHORITY_NONE = (
        "Use only the source-locked coordinate raw-index distributions to form one diagonal on-policy proposal; retain LC105."
    )
    base.VECTORIZED_CHECKPOINT_SURGERY = (
        "one complete 256-row LC105 Puffer actor plus exact per-group phase-6 output-bias coordinate perturbations"
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    originals = (
        base.verify_inputs, base.candidate_state_for_index,
        base.candidate_metadata_for_index,
    )
    base.verify_inputs = verify_inputs
    base.candidate_state_for_index = candidate_state_for_index
    base.candidate_metadata_for_index = candidate_metadata_for_index
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        (
            base.verify_inputs, base.candidate_state_for_index,
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
