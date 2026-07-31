#!/usr/bin/env python3
"""Bracket the VG057 learned residual on fresh teacher-free six-gate runs."""

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

from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_vg055_count6_fraction_bracket as bracket


TAG = "vq2_vg058_vg057_residual_scale_bracket_001"
SCHEMA = "vq2_vg058_residual_scale_bracket_report_v1"
STATE_SCHEMA = "vq2_vg058_residual_scale_bracket_state_v1"
ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
OFFSETS = (96, 104, 112, 120)
SEEDS = (429184, 429185, 429186, 429187)
AGENTS = 32
EPISODES = 32
THREADS = 4
MAX_STEPS = 3072
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
REJECTION = ROOT / "docs/vq2_vg056_count6_paired_confirmation_rejection_2026-07-31.json"
REJECTION_SHA256 = "c97082e4f6b0d40c74ee841e9b0c42786194b3d004dc4bf2e49337282e6a87b1"
PREREGISTRATION = ROOT / "docs/vq2_vg058_residual_scale_bracket_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg058_vast.sh"
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
            raise RuntimeError(f"VG058 source evidence changed: {path}")
    if not PREREGISTRATION.is_file() or not RUNNER.is_file():
        raise RuntimeError("VG058 source lock is incomplete")
    report = json.loads(UPDATE_REPORT.read_text())
    admission = json.loads(UPDATE_ADMISSION.read_text())
    rejection = json.loads(REJECTION.read_text())
    if (
        report.get("schema") != "vq2_vg057_warmed_phase_residual_report_v1"
        or not report.get("completed") or not report.get("numerically_admitted")
        or not report.get("base_parameters_exact")
        or report.get("checkpoint_sha256") != UPDATE_CHECKPOINT_SHA256
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("VG057 report does not authorize VG058")
    if (
        admission.get("schema") != "vq2_vg057_warmed_phase_residual_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != UPDATE_CHECKPOINT_SHA256
        or admission.get("live_authority")
    ):
        raise RuntimeError("VG057 admission does not authorize VG058")
    if (
        rejection.get("schema") != "vq2_vg056_count6_paired_confirmation_rejection_v1"
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("live_authority")
    ):
        raise RuntimeError("VG056 rejection changed")


def load_endpoints() -> tuple[dict[str, Any], dict[str, Any]]:
    base = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    update = torch.load(UPDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        base.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or update.get("schema") != "vq2_vg057_warmed_phase_residual_checkpoint_v1"
        or update.get("model", {}).get("class") != "VQ2PhaseResidualActor"
        or not update.get("numerically_admitted")
    ):
        raise RuntimeError("VG058 endpoint actor contract changed")
    for name, value in base["model_state"].items():
        if name not in update["model_state"] or not torch.equal(
            value, update["model_state"][name]
        ):
            raise RuntimeError(f"VG057 changed frozen base parameter {name}")
    residual_name = "phase_action_residual.weight"
    if set(update["model_state"]) != {*base["model_state"], residual_name}:
        raise RuntimeError("VG057 residual state keys changed")
    parent = copy.deepcopy(update)
    parent["tag"] = f"{TAG}_zero_residual_parent"
    parent["model_state"][residual_name] = torch.zeros_like(
        parent["model_state"][residual_name]
    )
    return parent, update


def source_identity() -> dict[str, Any]:
    identity = _bracket_source_identity()
    identity["source_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = (
        sha256_path(Path(__file__).resolve())
    )
    return identity


def qualifies(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    p, c = parent["gate_reach"], candidate["gate_reach"]
    downstream_parent = (
        parent["successes"], p["6"], p["5"], p["4"], p["3"],
        parent["mean_gates_passed"],
    )
    downstream_candidate = (
        candidate["successes"], c["6"], c["5"], c["4"], c["3"],
        candidate["mean_gates_passed"],
    )
    return bool(
        parent["all_hard_transport_pass"]
        and candidate["all_hard_transport_pass"]
        and c["1"] >= p["1"] and c["2"] >= p["2"]
        and candidate["crashes"] <= parent["crashes"]
        and c["5"] > p["5"]
        and downstream_candidate > downstream_parent
    )


def configure() -> None:
    bracket.TAG = TAG; bracket.SCHEMA = SCHEMA; bracket.STATE_SCHEMA = STATE_SCHEMA
    bracket.ALPHAS = ALPHAS; bracket.OFFSETS = OFFSETS; bracket.SEEDS = SEEDS
    bracket.AGENTS = AGENTS; bracket.EPISODES = EPISODES
    bracket.THREADS = THREADS; bracket.MAX_STEPS = MAX_STEPS
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
    bracket.DEFAULT_OUTPUT = DEFAULT_OUTPUT; bracket.ACTOR_CLASS = VQ2PhaseResidualActor
    bracket.verify_inputs = verify_inputs; bracket.load_endpoints = load_endpoints
    bracket.source_identity = source_identity; bracket.qualifies = qualifies


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
