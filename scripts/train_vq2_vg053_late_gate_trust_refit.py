#!/usr/bin/env python3
"""Run one low-step tenth-source refit from VG033 using VG052 late gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_vq2_variable_gate_recurrent_bc import sha256_path, source_label
import scripts.train_vq2_variable_gate_nine_source_refit as vg040
import scripts.train_vq2_variable_gate_seven_source_refit as core


TAG = "vq2_vg053_late_gate_trust_refit_001"
REPORT_SCHEMA = "vq2_vg053_late_gate_trust_refit_report_v1"
STATE_SCHEMA = "vq2_vg053_late_gate_trust_refit_state_v1"
SEED = 429170
VG052_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg052_late_gate_local_dagger_vg033_visited_512"
)
VG052_REPORT_SHA256 = (
    "687141acf2ac95449772666ecda6a9deed2f59a77a23d666c17330c39070fa91"
)
VG052_METADATA_SHA256 = (
    "03b71103b956a660f32888ed8a9290c1a2251f214f736497e1d53b209ede1895"
)
VG052_ADMISSION = (
    ROOT / "docs/vq2_vg052_late_gate_local_dagger_admission_2026-07-31.json"
)
VG052_ADMISSION_SHA256 = (
    "95c3f6e6b9b838bf7fec40503136136098780eb5719b2971d579c72ecf63016e"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_vg053_late_gate_trust_refit_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg053_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
SOURCE_WEIGHTS = {
    "clean": 0.30,
    "dagger1": 0.03,
    "dagger2": 0.03,
    "dagger3": 0.06,
    "dagger4": 0.08,
    "dagger5": 0.08,
    "dagger6": 0.12,
    "dagger7": 0.15,
    "dagger8": 0.05,
    "dagger9": 0.10,
}


@dataclass(frozen=True)
class LateGateTrustConfig(vg040.NineSourceConfig):
    seed: int = SEED
    epochs: int = 1
    learning_rate: float = 5e-7
    clean_objective_weight: float = 0.30
    dagger1_objective_weight: float = 0.03
    dagger2_objective_weight: float = 0.03
    dagger3_objective_weight: float = 0.06
    dagger4_objective_weight: float = 0.08
    dagger5_objective_weight: float = 0.08
    dagger6_objective_weight: float = 0.12
    dagger7_objective_weight: float = 0.15
    dagger8_objective_weight: float = 0.05
    dagger9_objective_weight: float = 0.10
    dagger9_validation_agents: int = 64
    maximum_dagger9_validation_weighted_mse: float = 0.20


def source_paths() -> list[Path]:
    return [
        *vg040.source_paths(),
        Path(__file__).resolve(),
        PREREGISTRATION,
        RUNNER,
        VG052_DATASET / "report.json",
        VG052_DATASET / "metadata.json",
        VG052_ADMISSION,
    ]


def current_source_identity() -> tuple[str, dict[str, str]]:
    hashes = {source_label(path): sha256_path(path) for path in source_paths()}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, hashes


def verify_inputs() -> None:
    vg040.verify_inputs()
    expected = {
        VG052_DATASET / "report.json": VG052_REPORT_SHA256,
        VG052_DATASET / "metadata.json": VG052_METADATA_SHA256,
        VG052_ADMISSION: VG052_ADMISSION_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG053 source evidence hash mismatch: {path}")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG053 preregistration is missing")
    report = json.loads((VG052_DATASET / "report.json").read_text())
    admission = json.loads(VG052_ADMISSION.read_text())
    if (
        report.get("schema") != "vq2_vg052_late_gate_local_dagger_report_v1"
        or report.get("tag")
        != "vq2_vg052_late_gate_local_dagger_vg033_visited_512"
        or not report.get("admitted")
        or report.get("records") != 548255
        or report.get("phase_records", [0] * 6)[3:6]
        != [63023, 255090, 230142]
        or report.get("failed_admission_predicates")
        or report.get("safety", {}).get("teacher_actions_executed") != 0
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("VG052 report does not admit the late-gate corpus")
    if (
        admission.get("schema")
        != "vq2_vg052_late_gate_local_dagger_admission_v1"
        or not admission.get("admitted")
        or admission.get("artifact_sha256", {}).get("report")
        != VG052_REPORT_SHA256
        or admission.get("artifact_sha256", {}).get("metadata")
        != VG052_METADATA_SHA256
        or admission.get("live_authority")
    ):
        raise RuntimeError("VG052 admission does not authorize VG053")


def numerical_admission_predicate(
    *,
    best_epoch: int,
    best_score: float,
    baseline_validation: dict[str, Any],
    selected_validation: dict[str, Any],
    source_balance_audit: bool,
    config: LateGateTrustConfig,
) -> bool:
    caps = {
        "clean": config.maximum_clean_validation_weighted_mse,
        "dagger1": config.maximum_dagger1_validation_weighted_mse,
        "dagger2": config.maximum_dagger2_validation_weighted_mse,
        "dagger3": config.maximum_dagger3_validation_weighted_mse,
        "dagger4": config.maximum_dagger4_validation_weighted_mse,
        "dagger5": config.maximum_dagger5_validation_weighted_mse,
        "dagger6": config.maximum_dagger6_validation_weighted_mse,
        "dagger7": config.maximum_dagger7_validation_weighted_mse,
        "dagger8": config.maximum_dagger8_validation_weighted_mse,
        "dagger9": config.maximum_dagger9_validation_weighted_mse,
    }
    return bool(
        best_epoch == 1
        and np.isfinite(best_score)
        and best_score
        < float(baseline_validation["source_balanced_weighted_mse"])
        and selected_validation["dagger9"]["weighted_mse"]
        < baseline_validation["dagger9"]["weighted_mse"]
        and all(
            selected_validation[name]["weighted_mse"] <= cap
            for name, cap in caps.items()
        )
        and source_balance_audit
    )


def configure_core() -> None:
    vg040.configure_core()
    core.TAG = TAG
    core.REPORT_SCHEMA = REPORT_SCHEMA
    core.STATE_SCHEMA = STATE_SCHEMA
    core.SEED = SEED
    core.PREREGISTRATION = PREREGISTRATION
    core.RUNNER = RUNNER
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.SOURCE_WEIGHTS = dict(SOURCE_WEIGHTS)
    core.EXTRA_DATASET_SPECS = {
        **core.EXTRA_DATASET_SPECS,
        "dagger9": {
            "dataset": VG052_DATASET,
            "report_sha256": VG052_REPORT_SHA256,
            "metadata_sha256": VG052_METADATA_SHA256,
            "validation_agents_attribute": "dagger9_validation_agents",
        },
    }
    core.EXTRA_OBJECTIVE_WEIGHT_ATTRIBUTES = {
        **core.EXTRA_OBJECTIVE_WEIGHT_ATTRIBUTES,
        "dagger9": "dagger9_objective_weight",
    }
    core.ROW_OBJECTIVE_WEIGHTERS = {}
    core.DATASET_EVALUATORS = {}
    core.EXTRA_TRAINING_IDENTITY = {
        "trust_region_parent": "VG033",
        "epochs": 1,
        "learning_rate": 5e-7,
        "pre_vg039_anchor_mass": 0.85,
        "vg039_mass": 0.05,
        "vg052_late_gate_mass": 0.10,
        "causal_sequences_preserved": True,
        "runtime_abi_changed": False,
    }
    core.NUMERICAL_ADMISSION_PREDICATE = numerical_admission_predicate
    core.verify_inputs = verify_inputs
    core.source_paths = source_paths
    core.current_source_identity = current_source_identity


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: LateGateTrustConfig = LateGateTrustConfig(),
    resume: bool = False,
) -> dict[str, Any]:
    if config.epochs != 1 or config.learning_rate != 5e-7:
        raise RuntimeError("VG053 trust-region schedule changed")
    configure_core()
    if core.objective_weights(config) != SOURCE_WEIGHTS:
        raise RuntimeError("VG053 source weights changed")
    return core.train(
        output=output,
        device_name=device_name,
        config=config,
        resume=resume,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
