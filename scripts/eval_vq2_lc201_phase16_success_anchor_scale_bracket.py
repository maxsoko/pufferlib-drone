#!/usr/bin/env python3
"""Exact-context interpolation bracket from LC189 toward LC199."""

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
import scripts.eval_vq2_lc196_phase16_17_joint_interpolation_bracket as prior


BASE_CONFIGURE = prior.configure
TAG = "vq2_lc201_phase16_success_anchor_scale_bracket_001"
SCHEMA = "vq2_lc201_phase16_success_anchor_scale_bracket_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc201_phase16_success_anchor_scale_checkpoint_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
SCALE_PAIRS = ((0.0, 0.0), (0.01, 0.0), (0.03, 0.0), (0.10, 0.0), (0.30, 0.0))
CANDIDATES = tuple(
    ("parent_lc189" if scale16 == 0.0 else f"lc199_scale_{scale16:g}", (0.0,) * 4)
    for scale16, _ in SCALE_PAIRS
)
FIT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc199_phase16_success_anchor_fit_001"
FIT_CHECKPOINT = FIT_DIR / "policy_selected.pt"
FIT_CHECKPOINT_SHA256 = "ffceb4452072c87095728e3ffc4c6d68238383517a09b18dbbda9951cc25656c"
FIT_REPORT = FIT_DIR / "report.json"
FIT_REPORT_SHA256 = "f96e002f72e8a14fdea68c2becc55094ff3426bdd145571f5aca8b4ba54d5b0a"
LC200_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc200_phase16_success_anchor_milestone_001/report.json"
LC200_REPORT_SHA256 = "ebcb8caed767b22709b45b225f7675570b55254ee4610e5c180050761bd42af1"
PREREGISTRATION = ROOT / "docs/vq2_lc201_phase16_success_anchor_scale_bracket_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc201_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc201_phase16_success_anchor_scale_bracket.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        prior.PARENT_CHECKPOINT: prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT: prior.PARENT_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
        LC200_REPORT: LC200_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC201 bound input changed: {path}")
    parent = prior.parent_payload()
    fitted = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    fit_report = json.loads(FIT_REPORT.read_text())
    rejected = json.loads(LC200_REPORT.read_text())
    baseline, endpoint = rejected.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or fitted.get("schema") != "vq2_lc199_phase16_success_anchor_fit_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("parent_checkpoint_sha256") != prior.PARENT_CHECKPOINT_SHA256
        or fit_report.get("schema") != "vq2_lc199_phase16_success_anchor_fit_report_v1"
        or not fit_report.get("numerically_admitted")
        or fit_report.get("selected", {}).get("scale") != 0.3
        or rejected.get("schema") != "vq2_lc200_phase16_success_anchor_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or baseline.get("maximum_raw_index_distribution", {}).get("17") != 128
        or endpoint.get("maximum_raw_index_distribution", {}).get("16") != 128
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC199/LC200 do not authorize LC201")
    return parent


def configure() -> None:
    BASE_CONFIGURE()
    prior.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), FIT_CHECKPOINT, FIT_REPORT, LC200_REPORT,
        ROOT / "scripts/train_vq2_lc199_phase16_success_anchor_fit.py",
        ROOT / "scripts/eval_vq2_lc200_phase16_success_anchor_milestone.py",
    )
    prior.milestone.NEXT_AUTHORITY_SELECTED = "Confirm the selected raw-18 source independently before extending to phase 18; no FlightSim authority."
    prior.milestone.NEXT_AUTHORITY_NONE = "Reject the LC199 fitted direction and retain LC189; do not run FlightSim."
    prior.milestone.VECTORIZED_CHECKPOINT_SURGERY = "five complete adapter Puffers, each executing the established 256-row recurrent context over paired 128-row groups"


def configure_wrapper() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "CHECKPOINT_SCHEMA", "GROUP_SIZE", "PAIR_SIZE",
        "SCALE_PAIRS", "CANDIDATES", "FIT_CHECKPOINT", "FIT_CHECKPOINT_SHA256",
        "FIT_REPORT", "FIT_REPORT_SHA256", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "verify_inputs", "configure",
    )
    originals = tuple(getattr(prior, name) for name in names)
    values = (
        TAG, SCHEMA, CHECKPOINT_SCHEMA, GROUP_SIZE, PAIR_SIZE,
        SCALE_PAIRS, CANDIDATES, FIT_CHECKPOINT, FIT_CHECKPOINT_SHA256,
        FIT_REPORT, FIT_REPORT_SHA256, PREREGISTRATION, RUNNER, TEST,
        DEFAULT_OUTPUT, verify_inputs, configure,
    )
    for name, value in zip(names, values):
        setattr(prior, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(prior, name, value)


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_wrapper()
    try:
        return prior.run(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("diagnostic_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
