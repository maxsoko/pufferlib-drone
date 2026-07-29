#!/usr/bin/env python3
"""Evaluate the fixed whole-Puffer C009-to-C012 interpolation bracket."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_n294_visual_composite import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts import eval_vq2_measured_visual_suffix_handoff as handoff
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.interpolate_vq2_phase_checkpoints import interpolate_state_dict


TAG = "vq2_c014_c009_c012_interpolation_bracket"
DEFAULT_CANDIDATES = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / f"{TAG}_screens"
)
ALPHAS = (0.25, 0.5, 0.75)
GENERATOR_SHA256 = (
    "a4741142f75f03ccb87ccd524db42342377bd6719c6c791e002f2bf8f3ecafeb"
)


def candidate_name(alpha: float) -> str:
    return f"c014_alpha_{alpha:.2f}".replace(".", "p")


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema": "vq2_whole_puffer_interpolation_manifest_v1",
        "tag": TAG,
        "base_sha256": handoff.C009_CHECKPOINT_SHA256,
        "candidate_sha256": handoff.C012_CHECKPOINT_SHA256,
        "alphas": list(ALPHAS),
        "whole_checkpoint": True,
        "runtime_action_blend": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("C014 interpolation manifest contract changed")
    generator = ROOT / "scripts/interpolate_vq2_phase_checkpoints.py"
    if sha256_path(generator) != GENERATOR_SHA256:
        raise RuntimeError("C014 generator source-lock mismatch")
    base = torch.load(
        handoff.C009_CHECKPOINT, map_location="cpu", weights_only=False
    )
    candidate = torch.load(
        handoff.C012_CHECKPOINT, map_location="cpu", weights_only=False
    )
    for record, alpha in zip(manifest["candidates"], ALPHAS, strict=True):
        if float(record["alpha"]) != alpha:
            raise RuntimeError("C014 candidate order changed")
        path = root / record["path"]
        if sha256_path(path) != record["sha256"]:
            raise RuntimeError("C014 candidate hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_state = interpolate_state_dict(
            base["model_state"], candidate["model_state"], alpha
        )
        if any(
            not torch.equal(payload["model_state"][name], value)
            for name, value in expected_state.items()
        ):
            raise RuntimeError("C014 candidate is not an exact interpolation")
    return manifest


def load_bracket_suffix(
    device: torch.device, candidate: str
) -> tuple[VQ2PhaseResidualActor, dict[str, Any], tuple[Path, ...]]:
    manifest = verify_manifest(DEFAULT_CANDIDATES)
    names = [candidate_name(alpha) for alpha in ALPHAS]
    if candidate not in names:
        raise ValueError(f"unsupported C014 candidate: {candidate}")
    record = manifest["candidates"][names.index(candidate)]
    path = DEFAULT_CANDIDATES / record["path"]
    payload = torch.load(path, map_location=device, weights_only=False)
    contract = payload.get("model", {})
    expected_contract = {
        "class": "VQ2PhaseResidualActor",
        "legal_observation_size": PHASE_LEGAL_OBS_SIZE,
        "legacy_legal_observation_size": LEGAL_OBS_SIZE,
        "public_status_values": 1,
        "action_size": ACTION_SIZE,
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise RuntimeError("C014 actor ABI changed")
    actor = VQ2PhaseResidualActor(
        hidden_size=int(contract["hidden_size"]),
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(payload["model_state"])
    actor.eval()
    return actor, payload, (path, DEFAULT_CANDIDATES / "manifest.json")


def bracket_passes(report: dict[str, Any]) -> bool:
    metrics = report.get("native_metrics_diagnostic_only", {})
    return bool(
        report.get("diagnostic_valid")
        and report.get("reached_next_gate") == report.get("agents") == 128
        and report.get("failed_before_next_gate") == 0
        and metrics.get("env/crash", 0.0) == 0.0
        and metrics.get("env/timeout", 0.0) == 0.0
        and metrics.get("env/missed_gate", 0.0) == 0.0
        and metrics.get("env/out_of_order", 0.0) == 0.0
    )


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = verify_manifest(DEFAULT_CANDIDATES)
    manifest_path = DEFAULT_CANDIDATES / "manifest.json"
    original_loader = handoff.load_candidate_suffix
    reports: list[dict[str, Any]] = []
    try:
        handoff.load_candidate_suffix = load_bracket_suffix
        for index, (alpha, record) in enumerate(
            zip(ALPHAS, manifest["candidates"], strict=True)
        ):
            name = candidate_name(alpha)
            path = DEFAULT_CANDIDATES / record["path"]
            handoff.CANDIDATE_ARTIFACTS[name] = (
                path,
                record["sha256"],
                manifest_path,
                sha256_path(manifest_path),
            )
            report = handoff.run_handoff(
                output=output / name,
                agents=128,
                seed=43014 + index,
                device_name=device_name,
                camera_pitch_rad=0.0,
                alias_zoom=1.0,
                alias_hold_steps=0,
                suffix_candidate=name,
                tag=f"{TAG}_{name}_true_range_exact_128",
            )
            reports.append(report)
    finally:
        handoff.load_candidate_suffix = original_loader
    passed = [bracket_passes(report) for report in reports]
    aggregate = {
        "schema": "vq2_whole_puffer_interpolation_bracket_report_v1",
        "tag": TAG,
        "alphas": list(ALPHAS),
        "passed": passed,
        "promoted_alpha": next(
            (alpha for alpha, accepted in zip(ALPHAS, passed, strict=True) if accepted),
            None,
        ),
        "reports": [
            {
                "tag": report["tag"],
                "reached_next_gate": report["reached_next_gate"],
                "failed_before_next_gate": report["failed_before_next_gate"],
                "trace_sha256": report["trace"]["sha256"],
            }
            for report in reports
        ],
        "safety": {
            "flight_sim_packets_sent": 0,
            "teacher_actions_executed": 0,
            "student_updates": 0,
            "submission_actions": 0,
        },
    }
    (output / "bracket_report.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    run(output=args.output, device_name=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
