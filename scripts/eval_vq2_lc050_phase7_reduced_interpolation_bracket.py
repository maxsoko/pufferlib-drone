#!/usr/bin/env python3
"""Reduced paired bracket toward LC049's phase-7 proposal."""

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

from pufferlib.vq2_recurrent_phase_residual import (
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc015_restore_vg071_head1 as screen
import scripts.eval_vq2_lc024_phase2_interpolation_bracket as bracket


TAG = "vq2_lc050_phase7_reduced_interpolation_bracket_001"
SCHEMA = "vq2_lc050_phase7_reduced_interpolation_bracket_report_v1"
CHILD_SCHEMA = "vq2_lc050_phase7_interpolation_screen_v1"
CHECKPOINT_SCHEMA = "vq2_lc050_phase7_interpolation_checkpoint_v1"
ALPHAS = (0.0, 0.0025, 0.005, 0.01)
SEED = 431500
TARGET_PHASE = 7
LC048 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc048_phase6_sequential_interpolation_bracket_001"
)
LC048_REPORT = LC048 / "report.json"
LC048_REPORT_SHA256 = (
    "0f5849a41f7554b88565caff758ec9a60e25d3c359d3c9603b45646c1dceb669"
)
PARENT_CHECKPOINT = LC048 / "a0p003/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "5a2046a856aa29e8c8bff81789664ff68cd44d67e5ad6e6bf90ba02cf9dec084"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "b2fb203ca9a49e74371694da71f8bd72757a1d0aab3d930ce5c2c8afdd85b3de"
)
LC040_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc040_phase4plus_teacher_features_001/report.json"
)
LC040_REPORT_SHA256 = (
    "7b814fab385873ae23507e19ecd4323375584c59740fd25c8316bb19aa68e06a"
)
FIT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc049_phase7_teacher_suffix_head_001/policy_best.pt"
)
FIT_CHECKPOINT_SHA256 = (
    "439fc5e8a665918cd032530eb8fe20bd3e5b1ac758f2838bbb132aba345fd529"
)
FIT_REPORT = FIT_CHECKPOINT.parent / "report.json"
FIT_REPORT_SHA256 = (
    "3dd95d08c1f991973d0bd768faa9adf3dd21b810c3a47a5a293a4b76784f5df6"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc050_phase7_reduced_interpolation_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc050_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CURRENT_ALPHA = 0.0


def verify_inputs() -> None:
    expected = {
        LC048_REPORT: LC048_REPORT_SHA256,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC040_REPORT: LC040_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC050 bound input changed: {path}")
    aggregate = json.loads(LC048_REPORT.read_text())
    dataset = json.loads(LC040_REPORT.read_text())
    fitted = json.loads(FIT_REPORT.read_text())
    if (
        aggregate.get("schema")
        != "vq2_lc048_phase6_sequential_interpolation_bracket_report_v1"
        or not aggregate.get("numerically_admitted")
        or aggregate.get("selected", {}).get("alpha") != 0.0025
        or aggregate.get("selected", {}).get("checkpoint_sha256")
        != PARENT_CHECKPOINT_SHA256
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_phase_records", [])[TARGET_PHASE] != 81_216
        or dataset.get("teacher_plant_actions_executed") != 1_404_080
        or fitted.get("schema")
        != "vq2_lc049_phase7_teacher_suffix_head_report_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("checkpoint_sha256") != FIT_CHECKPOINT_SHA256
        or fitted.get("selected_validation", {}).get("phases", {}).get(
            "7", {}
        ).get("improvement_factor", 0.0) < 1.13
        or fitted.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fitted.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC040/LC048/LC049 do not authorize LC050")


def build_candidate(
    device: torch.device,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any], str]:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    fitted = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = parent.get("model", {})
    if (
        parent.get("schema") != "vq2_lc048_phase6_interpolation_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or fitted.get("schema")
        != "vq2_lc049_phase7_teacher_suffix_head_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or contract.get("class") != "VQ2UnboundedProgressMLPResidualActor"
    ):
        raise RuntimeError("LC050 checkpoint contract changed")
    actor = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(parent["model_state"])
    with torch.no_grad():
        for name in screen.RESIDUAL_NAMES:
            start = parent["model_state"][name][TARGET_PHASE].to(device)
            end = fitted["model_state"][name][TARGET_PHASE].to(device)
            getattr(actor, name)[TARGET_PHASE].copy_(
                torch.lerp(start, end, CURRENT_ALPHA)
            )
    state = {name: value.detach().cpu() for name, value in actor.state_dict().items()}
    for name, value in state.items():
        if name in screen.RESIDUAL_NAMES:
            keep = torch.arange(value.shape[0]) != TARGET_PHASE
            if not torch.equal(value[keep], parent["model_state"][name][keep]):
                raise RuntimeError(f"LC050 changed a non-target row: {name}")
        elif not torch.equal(value, parent["model_state"][name]):
            raise RuntimeError(f"LC050 changed frozen Puffer state: {name}")
    actor.eval()
    payload = {
        **parent,
        "schema": CHECKPOINT_SCHEMA, "tag": screen.TAG,
        "model_state": state, "best_epoch": fitted.get("best_epoch"),
        "optimizer_updates": 0, "numerically_admitted": False,
        "surgery": {
            "operation": "interpolate_phase_head_toward_teacher_suffix_fit",
            "phase": TARGET_PHASE, "alpha": CURRENT_ALPHA,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "fit_checkpoint_sha256": FIT_CHECKPOINT_SHA256,
        },
    }
    return actor, payload, screen.state_sha256(state)


def configure_alpha(alpha: float) -> None:
    global CURRENT_ALPHA
    CURRENT_ALPHA = alpha
    slug = bracket.alpha_slug(alpha)
    screen.TAG = f"{TAG}_{slug}"
    screen.SCHEMA = CHILD_SCHEMA
    screen.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    screen.SEED = SEED
    screen.PREREGISTRATION = PREREGISTRATION
    screen.RUNNER = RUNNER
    screen.RESTORED_PHASES = (TARGET_PHASE,)
    screen.MIN_MEAN_GATES = 3.8125
    screen.MIN_MAXIMUM_INDEX = 8
    screen.MAX_CRASH_RATE = 0.50
    screen.CONVERSION_OPERATION = (
        f"interpolate phase-{TARGET_PHASE} head toward fit at alpha {alpha:.4f}"
    )
    screen.EXTRA_EVIDENCE_PATHS = (
        Path(__file__).resolve(), LC048_REPORT, PARENT_CHECKPOINT,
        PARENT_REPORT, LC040_REPORT, FIT_CHECKPOINT, FIT_REPORT,
    )
    screen.verify_inputs = verify_inputs
    screen.build_candidate = build_candidate


def configure() -> None:
    bracket.TAG, bracket.SCHEMA, bracket.CHILD_SCHEMA = TAG, SCHEMA, CHILD_SCHEMA
    bracket.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    bracket.ALPHAS, bracket.SEED = ALPHAS, SEED
    bracket.TARGET_PHASE = TARGET_PHASE
    bracket.PREREGISTRATION, bracket.RUNNER = PREREGISTRATION, RUNNER
    bracket.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    bracket.HISTORICAL_MINIMUM_MEAN_GATES = 3.8125
    bracket.HISTORICAL_MINIMUM_MAX_INDEX = 8
    bracket.verify_inputs = verify_inputs
    bracket.configure = configure_alpha
    bracket.build_candidate = build_candidate


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    configure()
    return bracket.run(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
