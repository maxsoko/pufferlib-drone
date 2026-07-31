#!/usr/bin/env python3
"""Teacher-free count-5 scale bracket for sparse VG065 heads 1 through 3."""

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


TAG = "vq2_vg066_sparse_early_head_count5_bracket_001"
SCHEMA = "vq2_vg066_sparse_early_head_count5_bracket_report_v1"
STATE_SCHEMA = "vq2_vg066_sparse_early_head_count5_bracket_state_v1"
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
OFFSETS = (192,)
SEEDS = (429201,)
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
    / "vq2_vg065_group_stratified_intervention_ridge_001/policy_best.pt"
)
UPDATE_CHECKPOINT_SHA256 = "486256dd4c56687c2da123aef384bb213dccaa54eb48a37b437a1121608773d0"
UPDATE_REPORT = UPDATE_CHECKPOINT.parent / "report.json"
UPDATE_REPORT_SHA256 = "db129cc1275b58839d169eab1ac451bd454dc60ab115c35bcc89442d88211620"
REJECTION = ROOT / "docs/vq2_vg065_group_stratified_intervention_ridge_rejection_2026-07-31.json"
REJECTION_SHA256 = "777c2da4da877515399f9a0ace1a068ba2c81732f4750502cc0795983e5a12c8"
PREREGISTRATION = ROOT / "docs/vq2_vg066_sparse_early_head_count5_bracket_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg066_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
SPARSE_PHASES = (1, 2, 3)
EXTRA_SOURCE_PATHS: tuple[Path, ...] = (Path(__file__).resolve(),)


def verify_inputs() -> None:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        PARENT_ADMISSION: PARENT_ADMISSION_SHA256,
        UPDATE_CHECKPOINT: UPDATE_CHECKPOINT_SHA256,
        UPDATE_REPORT: UPDATE_REPORT_SHA256,
        REJECTION: REJECTION_SHA256,
        bracket.GOAL: bracket.GOAL_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG066 evidence changed: {path}")
    update = json.loads(UPDATE_REPORT.read_text())
    rejection = json.loads(REJECTION.read_text())
    if (
        update.get("schema") != "vq2_vg065_group_stratified_intervention_ridge_report_v1"
        or not update.get("completed") or update.get("numerically_admitted")
        or update.get("checkpoint_sha256") != UPDATE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG065 is not the source numerical rejection")
    if (
        rejection.get("schema")
        != "vq2_vg065_group_stratified_intervention_ridge_rejection_v1"
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("live_authority")
    ):
        raise RuntimeError("VG065 does not authorize VG066")
    if not PREREGISTRATION.is_file() or not RUNNER.is_file():
        raise RuntimeError("VG066 source lock is incomplete")


def load_endpoints() -> tuple[dict[str, Any], dict[str, Any]]:
    base = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    fitted = torch.load(UPDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        base.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or fitted.get("schema") != "vq2_vg065_group_stratified_intervention_ridge_checkpoint_v1"
        or fitted.get("numerically_admitted")
        or fitted.get("model", {}).get("class") != "VQ2IndexedPhaseResidualActor"
    ):
        raise RuntimeError("VG066 endpoint contract changed")
    actor = VQ2IndexedPhaseResidualActor(hidden_size=256, initial_std=0.15)
    actor.load_base_state(base["model_state"])
    parent_state = {name: value.clone() for name, value in actor.state_dict().items()}
    update_state = {name: value.clone() for name, value in parent_state.items()}
    source_residual = fitted["model_state"]["indexed_phase_action_residual"]
    for phase in SPARSE_PHASES:
        update_state["indexed_phase_action_residual"][phase] = source_residual[phase]
    common = {
        **base,
        "schema": "vq2_vg066_sparse_indexed_synthetic_checkpoint_v1",
        "model": {**base["model"], "class": "VQ2IndexedPhaseResidualActor"},
        "numerically_admitted": False,
        "sparse_mapping": {"source_checkpoint_sha256": UPDATE_CHECKPOINT_SHA256,
                           "nonzero_phases": list(SPARSE_PHASES),
                           "all_other_residual_heads_zero": True},
    }
    parent = {**copy.deepcopy(common), "tag": f"{TAG}_zero_parent", "model_state": parent_state}
    update = {**copy.deepcopy(common), "tag": f"{TAG}_sparse_update", "model_state": update_state}
    return parent, update


def qualifies(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    p, c = parent["gate_reach"], candidate["gate_reach"]
    return bool(
        parent["all_hard_transport_pass"] and candidate["all_hard_transport_pass"]
        and c["1"] >= p["1"] and candidate["crashes"] <= parent["crashes"]
        and (candidate["successes"], c["5"], c["4"], c["3"], c["2"], candidate["mean_gates_passed"])
        > (parent["successes"], p["5"], p["4"], p["3"], p["2"], parent["mean_gates_passed"])
    )


def selection_key(item: dict[str, Any]) -> tuple[float, ...]:
    summary, gates = item["aggregate"], item["aggregate"]["gate_reach"]
    return (
        float(summary["successes"]), float(gates["5"]), float(gates["4"]),
        float(gates["3"]), float(gates["2"]), float(summary["mean_gates_passed"]),
        -float(summary["crashes"]), -float(item["alpha"]),
    )


def configure() -> None:
    bracket.TAG = TAG; bracket.SCHEMA = SCHEMA; bracket.STATE_SCHEMA = STATE_SCHEMA
    bracket.ALPHAS = ALPHAS; bracket.OFFSETS = OFFSETS; bracket.SEEDS = SEEDS
    bracket.AGENTS = AGENTS; bracket.EPISODES = EPISODES; bracket.THREADS = THREADS
    bracket.MAX_STEPS = MAX_STEPS; bracket.NUM_GATES = NUM_GATES
    bracket.PARENT_CHECKPOINT = PARENT_CHECKPOINT; bracket.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    bracket.PARENT_REPORT = PARENT_REPORT; bracket.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    bracket.PARENT_ADMISSION = PARENT_ADMISSION; bracket.PARENT_ADMISSION_SHA256 = PARENT_ADMISSION_SHA256
    bracket.UPDATE_CHECKPOINT = UPDATE_CHECKPOINT; bracket.UPDATE_CHECKPOINT_SHA256 = UPDATE_CHECKPOINT_SHA256
    bracket.UPDATE_REPORT = UPDATE_REPORT; bracket.UPDATE_REPORT_SHA256 = UPDATE_REPORT_SHA256
    bracket.UPDATE_ADMISSION = REJECTION; bracket.UPDATE_ADMISSION_SHA256 = REJECTION_SHA256
    bracket.REJECTION = REJECTION; bracket.REJECTION_SHA256 = REJECTION_SHA256
    bracket.PREREGISTRATION = PREREGISTRATION; bracket.RUNNER = RUNNER
    bracket.DEFAULT_OUTPUT = DEFAULT_OUTPUT; bracket.ACTOR_CLASS = VQ2IndexedPhaseResidualActor
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
