#!/usr/bin/env python3
"""Paired closed-loop bracket toward the LC023 fitted phase-2 head."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase_residual import (
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc015_restore_vg071_head1 as base


TAG = "vq2_lc024_phase2_interpolation_bracket_001"
SCHEMA = "vq2_lc024_phase2_interpolation_bracket_report_v1"
CHILD_SCHEMA = "vq2_lc024_phase2_interpolation_screen_v1"
CHECKPOINT_SCHEMA = "vq2_lc024_phase2_interpolation_checkpoint_v1"
ALPHAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0)
TARGET_PHASE = 2
SEED = 431240
LC021 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc021_cumulative_head_rollback_ladder_001"
)
LC021_REPORT = LC021 / "report.json"
LC021_REPORT_SHA256 = (
    "5e9f9a7d8cc1c53b1a47d35fe725a5f7188a28c5b6a35876737152a5ad723068"
)
PARENT_CHECKPOINT = LC021 / "through_05/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "a32cc7705145c47b63c5126128eaa1296a52bfb175b8bb7868b9839041275ca4"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "bd45e3ec075c4574fbc136a00ddb2b57830a2b96ef781d11e46959b8d7ab53fa"
)
LC022_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc022_index2_student_dagger_features_001/report.json"
)
LC022_REPORT_SHA256 = (
    "833d4eed8e4dc68b48d1efd86d1551e6296d2ffc2225fdccef0e54edab5515b3"
)
FIT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc023_index2_student_dagger_head_001/policy_best.pt"
)
FIT_CHECKPOINT_SHA256 = (
    "fe05ac27e1715b3c9601beb1eff685c3a2de8c2da9bdd96b7cb70e52642deba6"
)
FIT_REPORT = FIT_CHECKPOINT.parent / "report.json"
FIT_REPORT_SHA256 = (
    "e701e7e7914a51813281841da86c20d40b7f51f591767dfab6a2afd28c7e9c8e"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc024_phase2_interpolation_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc024_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CURRENT_ALPHA = 0.0


def verify_inputs() -> None:
    expected = {
        LC021_REPORT: LC021_REPORT_SHA256,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC022_REPORT: LC022_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC024 bound input changed: {path}")
    ladder = json.loads(LC021_REPORT.read_text())
    dataset = json.loads(LC022_REPORT.read_text())
    fit = json.loads(FIT_REPORT.read_text())
    if (
        ladder.get("selected", {}).get("checkpoint_sha256")
        != PARENT_CHECKPOINT_SHA256
        or ladder.get("selected", {}).get("restored_through_phase") != 5
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_phase_records", [])[TARGET_PHASE] != 649_205
        or dataset.get("teacher_plant_actions_executed") != 0
        or fit.get("schema") != "vq2_lc023_index2_student_dagger_head_report_v1"
        or not fit.get("numerically_admitted")
        or fit.get("checkpoint_sha256") != FIT_CHECKPOINT_SHA256
        or fit.get("selected_validation", {}).get("phases", {}).get("2", {}).get(
            "improvement_factor", 0.0
        ) < 2.62
        or fit.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC021--LC023 do not authorize the phase-2 bracket")


def build_candidate(
    device: torch.device,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any], str]:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    fitted = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = parent.get("model", {})
    if (
        parent.get("schema")
        != "vq2_lc021_cumulative_head_rollback_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or fitted.get("schema")
        != "vq2_lc023_index2_student_dagger_head_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or contract.get("class") != "VQ2UnboundedProgressMLPResidualActor"
    ):
        raise RuntimeError("LC024 checkpoint contract changed")
    actor = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(parent["model_state"])
    with torch.no_grad():
        for name in base.RESIDUAL_NAMES:
            start = parent["model_state"][name][TARGET_PHASE].to(device)
            end = fitted["model_state"][name][TARGET_PHASE].to(device)
            getattr(actor, name)[TARGET_PHASE].copy_(
                torch.lerp(start, end, CURRENT_ALPHA)
            )
    state = {name: value.detach().cpu() for name, value in actor.state_dict().items()}
    for name, value in state.items():
        if name in base.RESIDUAL_NAMES:
            keep = torch.arange(value.shape[0]) != TARGET_PHASE
            if not torch.equal(value[keep], parent["model_state"][name][keep]):
                raise RuntimeError(f"LC024 changed a non-target row: {name}")
        elif not torch.equal(value, parent["model_state"][name]):
            raise RuntimeError(f"LC024 changed frozen Puffer state: {name}")
    actor.eval()
    payload = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": base.TAG,
        "model_state": state,
        "best_epoch": fitted.get("best_epoch"),
        "optimizer_updates": 0,
        "numerically_admitted": False,
        "surgery": {
            "operation": "interpolate_phase_head_toward_student_state_fit",
            "phase": TARGET_PHASE,
            "alpha": CURRENT_ALPHA,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "fit_checkpoint_sha256": FIT_CHECKPOINT_SHA256,
        },
    }
    return actor, payload, base.state_sha256(state)


def alpha_slug(alpha: float) -> str:
    return f"a{alpha:.3f}".replace(".", "p")


def configure(alpha: float) -> None:
    global CURRENT_ALPHA
    CURRENT_ALPHA = alpha
    slug = alpha_slug(alpha)
    base.TAG = f"{TAG}_{slug}"
    base.SCHEMA = CHILD_SCHEMA
    base.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    base.SEED = SEED
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
    base.RESTORED_PHASES = (TARGET_PHASE,)
    base.MIN_MEAN_GATES = 2.875
    base.MIN_MAXIMUM_INDEX = 6
    base.MAX_CRASH_RATE = 0.50
    base.CONVERSION_OPERATION = (
        f"interpolate phase-2 head toward LC023 at alpha {alpha:.3f}"
    )
    base.EXTRA_EVIDENCE_PATHS = (
        Path(__file__).resolve(), LC021_REPORT, PARENT_CHECKPOINT, PARENT_REPORT,
        LC022_REPORT, FIT_CHECKPOINT, FIT_REPORT,
    )
    base.verify_inputs = verify_inputs
    base.build_candidate = build_candidate


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    report_path = output / "report.json"
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {output}")
    if resume and report_path.is_file():
        return json.loads(report_path.read_text())
    started = time.perf_counter()
    items: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        configure(alpha)
        child_output = output / alpha_slug(alpha)
        child = base.run(
            output=child_output, device_name=device_name, resume=resume
        )
        identities.append(child["source_identity"])
        items.append({
            "alpha": alpha,
            "candidate_state_sha256": child["candidate_state_sha256"],
            "mean_gates_passed": child["mean_gates_passed"],
            "maximum_raw_index": child["maximum_raw_index"],
            "maximum_raw_index_distribution": child[
                "maximum_raw_index_distribution"
            ],
            "crash_rate": child["crash_rate"],
            "miss_rate": child["miss_rate"],
            "timeout_rate": child["timeout_rate"],
            "diagnostic_valid": child["diagnostic_valid"],
            "checkpoint": (
                str(Path(alpha_slug(alpha)) / child["checkpoint"])
                if child["checkpoint"] else None
            ),
            "checkpoint_sha256": child["checkpoint_sha256"],
            "child_report_sha256": sha256_path(child_output / "report.json"),
        })
    if any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("LC024 source identity changed within the bracket")
    baseline = items[0]
    admitted = [
        item for item in items[1:]
        if item["checkpoint"] is not None
        and item["mean_gates_passed"] > baseline["mean_gates_passed"]
        and item["maximum_raw_index"] >= baseline["maximum_raw_index"]
        and item["crash_rate"] <= baseline["crash_rate"]
    ]
    selected = max(
        admitted,
        key=lambda item: (
            item["mean_gates_passed"], item["maximum_raw_index"],
            -item["crash_rate"], -item["alpha"],
        ),
        default=None,
    )
    report = {
        "schema": SCHEMA,
        "tag": TAG,
        "completed": True,
        "diagnostic_valid": all(item["diagnostic_valid"] for item in items),
        "numerically_admitted": selected is not None,
        "paired_seed": SEED,
        "episodes_per_candidate": base.EPISODES,
        "num_gates": base.NUM_GATES,
        "baseline": baseline,
        "items": items,
        "selected": selected,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identities[0],
        "safety": {
            "teacher_plant_actions": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Retain the selected phase-2 interpolation for a larger offline screen."
            if selected else "Reject LC024 and retain the unchanged LC021 Puffer."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json_once(report_path, report)
    return report


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
