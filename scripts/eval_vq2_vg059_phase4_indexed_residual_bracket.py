#!/usr/bin/env python3
"""Bracket a VG057-derived learned residual active only at public phase 4."""

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

from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseResidualActor
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_vg055_count6_fraction_bracket as bracket


TAG = "vq2_vg059_phase4_indexed_residual_bracket_001"
SCHEMA = "vq2_vg059_phase4_indexed_residual_bracket_report_v1"
STATE_SCHEMA = "vq2_vg059_phase4_indexed_residual_bracket_state_v1"
ALPHAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
OFFSETS = (128, 136, 144, 152)
SEEDS = (429188, 429189, 429190, 429191)
AGENTS = 64
EPISODES = 64
THREADS = 4
MAX_STEPS = 3072
TARGET_PHASE_INDEX = 4
SHARED_TO_INDEXED_SCALE = TARGET_PHASE_INDEX / 16.0
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
    / "vq2_vg057_warmed_phase_residual_001/policy_best.pt"
)
UPDATE_CHECKPOINT_SHA256 = "cf3825ed940945100f139c8b0f0a2012c8f2f9e33ff393fdf9f7a3053394d45b"
UPDATE_REPORT = UPDATE_CHECKPOINT.parent / "report.json"
UPDATE_REPORT_SHA256 = "843f44a91f1f40ded919f56504c5d82304351ae4203c4a30c541ad7c194c87b5"
UPDATE_ADMISSION = ROOT / "docs/vq2_vg057_warmed_phase_residual_admission_2026-07-31.json"
UPDATE_ADMISSION_SHA256 = "24edfd5c4cb8ff50fb002c9eb237a283f29197356b6583a2da6c75fa78c6e351"
REJECTION = ROOT / "docs/vq2_vg058_residual_scale_bracket_rejection_2026-07-31.json"
REJECTION_SHA256 = "3ffd88cfe456f52fdcd834592d3292e2c27fa6a52b8153d8a80a5c966bdf5d0a"
PREREGISTRATION = ROOT / "docs/vq2_vg059_phase4_indexed_residual_bracket_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg059_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_bracket_source_identity = bracket.source_identity
_bounded_interpolate_state = bracket.interpolation.interpolate_state


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
            raise RuntimeError(f"VG059 source evidence changed: {path}")
    if not PREREGISTRATION.is_file() or not RUNNER.is_file():
        raise RuntimeError("VG059 source lock is incomplete")
    rejection = json.loads(REJECTION.read_text())
    admission = json.loads(UPDATE_ADMISSION.read_text())
    if (
        rejection.get("schema") != "vq2_vg058_residual_scale_bracket_rejection_v1"
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("live_authority")
    ):
        raise RuntimeError("VG058 rejection does not authorize VG059")
    if (
        admission.get("schema") != "vq2_vg057_warmed_phase_residual_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != UPDATE_CHECKPOINT_SHA256
        or admission.get("live_authority")
    ):
        raise RuntimeError("VG057 admission changed")


def load_endpoints() -> tuple[dict[str, Any], dict[str, Any]]:
    base = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    fit = torch.load(UPDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    shared_name = "phase_action_residual.weight"
    if (
        base.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or fit.get("schema") != "vq2_vg057_warmed_phase_residual_checkpoint_v1"
        or fit.get("model", {}).get("class") != "VQ2PhaseResidualActor"
        or not fit.get("numerically_admitted")
        or shared_name not in fit.get("model_state", {})
    ):
        raise RuntimeError("VG059 endpoint contract changed")
    for name, value in base["model_state"].items():
        if not torch.equal(value, fit["model_state"][name]):
            raise RuntimeError(f"VG057 changed base parameter {name}")
    actor = VQ2IndexedPhaseResidualActor(hidden_size=256, initial_std=0.15)
    actor.load_base_state(base["model_state"])
    parent_state = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    update_state = {name: value.clone() for name, value in parent_state.items()}
    update_state["indexed_phase_action_residual"][TARGET_PHASE_INDEX] = (
        fit["model_state"][shared_name] * SHARED_TO_INDEXED_SCALE
    )
    common = {
        **base,
        "schema": "vq2_vg059_indexed_phase_residual_synthetic_checkpoint_v1",
        "model": {**base["model"], "class": "VQ2IndexedPhaseResidualActor"},
        "numerically_admitted": True,
        "indexed_residual_mapping": {
            "source_checkpoint_sha256": UPDATE_CHECKPOINT_SHA256,
            "source_parameter": shared_name,
            "target_parameter": "indexed_phase_action_residual[4]",
            "shared_to_indexed_scale": SHARED_TO_INDEXED_SCALE,
            "phases_0_through_3_exact": True,
        },
    }
    parent = {**common, "tag": f"{TAG}_zero_indexed_parent", "model_state": parent_state}
    update = {**common, "tag": f"{TAG}_phase4_indexed_update", "model_state": update_state}
    return parent, update


def interpolate_state(
    parent_state: dict[str, torch.Tensor],
    update_state: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 4.0:
        raise ValueError("VG059 residual scale must be in [0,4]")
    if alpha <= 1.0:
        return _bounded_interpolate_state(parent_state, update_state, alpha)
    result: dict[str, torch.Tensor] = {}
    for name, parent in parent_state.items():
        update = update_state[name]
        if parent.shape != update.shape or parent.dtype != update.dtype:
            raise RuntimeError(f"VG059 endpoint tensor mismatch: {name}")
        if parent.is_floating_point():
            result[name] = (
                parent.double() + alpha * (update.double() - parent.double())
            ).to(parent.dtype)
        elif torch.equal(parent, update):
            result[name] = parent.clone()
        else:
            raise RuntimeError(f"VG059 nonfloating endpoint differs: {name}")
    return result


def source_identity() -> dict[str, Any]:
    identity = _bracket_source_identity()
    for path in (Path(__file__).resolve(), ROOT / "pufferlib/vq2_recurrent_phase_residual.py"):
        identity["source_sha256"][str(path.relative_to(ROOT))] = sha256_path(path)
    identity["indexed_residual"] = {
        "target_phase_index": TARGET_PHASE_INDEX,
        "shared_to_indexed_scale": SHARED_TO_INDEXED_SCALE,
        "phases_0_through_3_exact": True,
    }
    return identity


def qualifies(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    p, c = parent["gate_reach"], candidate["gate_reach"]
    downstream_parent = (parent["successes"], p["6"], p["5"], parent["mean_gates_passed"])
    downstream_candidate = (candidate["successes"], c["6"], c["5"], candidate["mean_gates_passed"])
    return bool(
        parent["all_hard_transport_pass"] and candidate["all_hard_transport_pass"]
        and all(c[str(gate)] == p[str(gate)] for gate in range(1, 5))
        and candidate["crashes"] <= parent["crashes"]
        and c["5"] > p["5"] and downstream_candidate > downstream_parent
    )


def configure() -> None:
    bracket.TAG = TAG; bracket.SCHEMA = SCHEMA; bracket.STATE_SCHEMA = STATE_SCHEMA
    bracket.ALPHAS = ALPHAS; bracket.OFFSETS = OFFSETS; bracket.SEEDS = SEEDS
    bracket.AGENTS = AGENTS; bracket.EPISODES = EPISODES
    bracket.THREADS = THREADS; bracket.MAX_STEPS = MAX_STEPS
    bracket.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    bracket.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    bracket.PARENT_REPORT = PARENT_REPORT; bracket.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    bracket.PARENT_ADMISSION = PARENT_ADMISSION; bracket.PARENT_ADMISSION_SHA256 = PARENT_ADMISSION_SHA256
    bracket.UPDATE_CHECKPOINT = UPDATE_CHECKPOINT; bracket.UPDATE_CHECKPOINT_SHA256 = UPDATE_CHECKPOINT_SHA256
    bracket.UPDATE_REPORT = UPDATE_REPORT; bracket.UPDATE_REPORT_SHA256 = UPDATE_REPORT_SHA256
    bracket.UPDATE_ADMISSION = UPDATE_ADMISSION; bracket.UPDATE_ADMISSION_SHA256 = UPDATE_ADMISSION_SHA256
    bracket.REJECTION = REJECTION; bracket.REJECTION_SHA256 = REJECTION_SHA256
    bracket.PREREGISTRATION = PREREGISTRATION; bracket.RUNNER = RUNNER
    bracket.DEFAULT_OUTPUT = DEFAULT_OUTPUT; bracket.ACTOR_CLASS = VQ2IndexedPhaseResidualActor
    bracket.verify_inputs = verify_inputs; bracket.load_endpoints = load_endpoints
    bracket.source_identity = source_identity; bracket.qualifies = qualifies
    bracket.interpolation.interpolate_state = interpolate_state


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
