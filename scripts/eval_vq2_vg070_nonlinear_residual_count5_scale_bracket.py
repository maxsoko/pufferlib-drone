#!/usr/bin/env python3
"""Teacher-free count-5 scale bracket for the admitted VG069 nonlinear heads."""

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


TAG = "vq2_vg070_nonlinear_residual_count5_scale_bracket_001"
SCHEMA = "vq2_vg070_nonlinear_residual_count5_scale_bracket_report_v1"
STATE_SCHEMA = "vq2_vg070_nonlinear_residual_count5_scale_bracket_state_v1"
ALPHAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
OFFSETS = (208,)
SEEDS = (429204,)
AGENTS = 64
EPISODES = 64
THREADS = 4
MAX_STEPS = 23040
NUM_GATES = 5
PARENT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg033_variable_gate_eight_source_refit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = "56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26"
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = "033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60"
PARENT_ADMISSION = ROOT / "docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
PARENT_ADMISSION_SHA256 = "dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44"
UPDATE_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg069_phase_independent_early_stop_001/policy_best.pt"
)
UPDATE_CHECKPOINT_SHA256 = "5e04e36cbc0b048b0863260b87f414de70745ef01bf45ca4d1cc3c7d8e1225f6"
UPDATE_REPORT = UPDATE_CHECKPOINT.parent / "report.json"
UPDATE_REPORT_SHA256 = "466d76224ff556f78d996e3c710aadeedb0104708414eaab5b6ae116dc93bc85"
UPDATE_ADMISSION = ROOT / "docs/vq2_vg069_phase_independent_early_stop_admission_2026-07-31.json"
UPDATE_ADMISSION_SHA256 = "cd17155e6c5090b6c81da1fda93b1339ef91bf43fdb0cf336e1a3e9300e74299"
REJECTION = ROOT / "docs/vq2_vg068_indexed_mlp_intervention_rejection_2026-07-31.json"
REJECTION_SHA256 = "1f8273e954f68f828306423b6e5ec567470c7abcbd65c0323d3f53af83327b18"
PREREGISTRATION = ROOT / "docs/vq2_vg070_nonlinear_residual_count5_scale_bracket_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg070_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_vg070_nonlinear_residual_count5_scale_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
EXTRA_SOURCE_PATHS: tuple[Path, ...] = (Path(__file__).resolve(), TEST)


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
            raise RuntimeError(f"VG070 evidence changed: {path}")
    update = json.loads(UPDATE_REPORT.read_text())
    admission = json.loads(UPDATE_ADMISSION.read_text())
    if (
        update.get("schema") != "vq2_vg069_phase_independent_early_stop_report_v1"
        or not update.get("numerically_admitted")
        or update.get("checkpoint_sha256") != UPDATE_CHECKPOINT_SHA256
        or update.get("vg068_history_replay_max_abs_error") != 0.0
    ):
        raise RuntimeError("VG069 is not the admitted nonlinear endpoint")
    successor = admission.get("authorized_successor", {})
    if (
        admission.get("schema")
        != "vq2_vg069_phase_independent_early_stop_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != UPDATE_CHECKPOINT_SHA256
        or successor.get("tag") != TAG
        or not successor.get("rollout_authority")
        or admission.get("live_authority")
    ):
        raise RuntimeError("VG069 admission does not authorize VG070")
    if not PREREGISTRATION.is_file() or not RUNNER.is_file():
        raise RuntimeError("VG070 source lock is incomplete")


def load_endpoints() -> tuple[dict[str, Any], dict[str, Any]]:
    base = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    fitted = torch.load(UPDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        base.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or fitted.get("schema")
        != "vq2_vg069_phase_independent_early_stop_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("model", {}).get("class")
        != "VQ2IndexedPhaseMLPResidualActor"
        or fitted.get("model", {}).get("residual_size") != 64
    ):
        raise RuntimeError("VG070 endpoint contract changed")
    actor = VQ2IndexedPhaseMLPResidualActor(
        hidden_size=256, residual_size=64, initial_std=0.15
    )
    actor.load_state_dict(fitted["model_state"])
    update_state = {name: value.clone() for name, value in actor.state_dict().items()}
    parent_state = {name: value.clone() for name, value in update_state.items()}
    parent_state["indexed_phase_residual_output"].zero_()
    parent_state["indexed_phase_residual_output_bias"].zero_()
    common = {
        **base,
        "schema": "vq2_vg070_nonlinear_scale_synthetic_checkpoint_v1",
        "model": {
            **base["model"], "class": "VQ2IndexedPhaseMLPResidualActor",
            "residual_size": 64,
        },
        "synthetic_mapping": {
            "source_checkpoint_sha256": UPDATE_CHECKPOINT_SHA256,
            "alpha_zero_outputs_exactly_zero": True,
            "alpha_scales_output_layers_and_biases_only": True,
        },
    }
    parent = {
        **copy.deepcopy(common), "tag": f"{TAG}_zero_parent",
        "model_state": parent_state,
    }
    update = {
        **copy.deepcopy(common), "tag": f"{TAG}_full_update",
        "model_state": update_state,
    }
    return parent, update


def qualifies(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    p, c = parent["gate_reach"], candidate["gate_reach"]
    return bool(
        parent["all_hard_transport_pass"] and candidate["all_hard_transport_pass"]
        and c["1"] >= p["1"]
        and candidate["crashes"] <= parent["crashes"]
        and candidate["successes"] > parent["successes"]
    )


def selection_key(item: dict[str, Any]) -> tuple[float, ...]:
    summary, gates = item["aggregate"], item["aggregate"]["gate_reach"]
    return (
        float(summary["successes"]), -float(summary["crashes"]),
        float(gates["5"]), float(gates["4"]), float(gates["3"]),
        float(gates["2"]), float(summary["mean_gates_passed"]),
        -float(item["alpha"]),
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
