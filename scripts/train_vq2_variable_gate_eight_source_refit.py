#!/usr/bin/env python3
"""Refit VG028 while retaining all anchors and adding VG032 visited states."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase import (
    PHASE_LEGAL_OBS_SIZE,
    VQ2PhaseRecurrentActor,
)
from scripts.train_vq2_variable_gate_recurrent_bc import (
    sha256_path,
    source_label,
)
import scripts.train_vq2_variable_gate_seven_source_refit as core


TAG = "vq2_vg033_variable_gate_eight_source_refit_001"
REPORT_SCHEMA = "vq2_variable_gate_eight_source_refit_report_v1"
STATE_SCHEMA = "vq2_variable_gate_eight_source_refit_state_v1"
SEED = 429132

VG032_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg032_variable_gate_dagger_round7_vg028_visited_512"
)
VG032_REPORT_SHA256 = (
    "f725afd984ae5a02999bd01560b24d4728955cc8e4a3c1cad972591ed9556d41"
)
VG032_METADATA_SHA256 = (
    "b4d43848fe217ee490a57ff19f0eeff5752a58633c7181fde3472700b41a9701"
)
VG032_ADMISSION = (
    ROOT / "docs/vq2_vg032_variable_gate_dagger_round7_admission_2026-07-31.json"
)
VG032_ADMISSION_SHA256 = (
    "7887a69aba40a2c364336a516ef0ef3e45377118af149c4a27bc3376ee88e0be"
)

PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg028_variable_gate_seven_source_refit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "ec1206b5dbccb2ee679fa049534956b189b419da2b927f5e3a61eeaf5a359e65"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "b51b7a9b25bc9f213a8b93a3d60fca6a2226d2511f9475c0a6da7e12cd21909e"
)
PARENT_ADMISSION = (
    ROOT / "docs/vq2_vg028_seven_source_refit_admission_2026-07-31.json"
)
PARENT_ADMISSION_SHA256 = (
    "3282c98fab562a9593261538f452226190bc51b10e5e64495957cd348403755b"
)
GOAL_PROMPT = core.GOAL_PROMPT
GOAL_PROMPT_SHA256 = core.GOAL_PROMPT_SHA256
PREREGISTRATION = (
    ROOT / "docs/vq2_vg033_eight_source_refit_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg033_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
SOURCE_WEIGHTS = {
    "clean": 0.25,
    "dagger1": 0.03,
    "dagger2": 0.03,
    "dagger3": 0.06,
    "dagger4": 0.08,
    "dagger5": 0.08,
    "dagger6": 0.12,
    "dagger7": 0.35,
}

_ORIGINAL_VERIFY_INPUTS = core.verify_inputs
_ORIGINAL_SOURCE_PATHS = core.source_paths


@dataclass(frozen=True)
class EightSourceConfig(core.SevenSourceConfig):
    seed: int = SEED
    clean_objective_weight: float = 0.25
    dagger1_objective_weight: float = 0.03
    dagger2_objective_weight: float = 0.03
    dagger3_objective_weight: float = 0.06
    dagger4_objective_weight: float = 0.08
    dagger5_objective_weight: float = 0.08
    dagger6_objective_weight: float = 0.12
    dagger7_objective_weight: float = 0.35
    dagger7_validation_agents: int = 64
    maximum_dagger7_validation_weighted_mse: float = 0.12


def source_paths() -> list[Path]:
    return [
        *_ORIGINAL_SOURCE_PATHS(),
        Path(__file__).resolve(),
        VG032_DATASET / "report.json",
        VG032_DATASET / "metadata.json",
        VG032_ADMISSION,
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
    expected = {
        GOAL_PROMPT: GOAL_PROMPT_SHA256,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        PARENT_ADMISSION: PARENT_ADMISSION_SHA256,
        VG032_DATASET / "report.json": VG032_REPORT_SHA256,
        VG032_DATASET / "metadata.json": VG032_METADATA_SHA256,
        VG032_ADMISSION: VG032_ADMISSION_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG033 source evidence hash mismatch: {path}")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG033 preregistration is missing")

    parent = json.loads(PARENT_REPORT.read_text())
    parent_admission = json.loads(PARENT_ADMISSION.read_text())
    vg032 = json.loads((VG032_DATASET / "report.json").read_text())
    vg032_admission = json.loads(VG032_ADMISSION.read_text())
    if (
        parent.get("schema") != "vq2_variable_gate_seven_source_refit_report_v1"
        or parent.get("tag") != "vq2_vg028_variable_gate_seven_source_refit_001"
        or not parent.get("completed")
        or not parent.get("numerically_admitted")
        or parent.get("best_epoch") != 6
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG028 parent is not admitted for VG033")
    if (
        parent_admission.get("schema")
        != "vq2_vg028_seven_source_refit_admission_v1"
        or not parent_admission.get("numerically_admitted")
        or parent_admission.get("artifact_sha256", {}).get("checkpoint")
        != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG028 admission evidence does not bind VG033")
    if (
        vg032.get("schema")
        != "vq2_vg032_variable_gate_dagger_collection_report_v1"
        or vg032.get("tag")
        != "vq2_vg032_variable_gate_dagger_round7_vg028_visited_512"
        or not vg032.get("admitted")
        or vg032.get("records", 0) < 1_000_000
        or vg032.get("failed_admission_predicates")
        or not vg032.get("admission_predicates", {}).get(
            "minimum_phase_2_records"
        )
        or not vg032.get("admission_predicates", {}).get(
            "minimum_phase_3_records"
        )
    ):
        raise RuntimeError("VG032 report does not admit its dataset")
    if (
        vg032_admission.get("schema")
        != "vq2_vg032_variable_gate_dagger_admission_v1"
        or not vg032_admission.get("admitted")
        or vg032_admission.get("artifact_sha256", {}).get("report")
        != VG032_REPORT_SHA256
    ):
        raise RuntimeError("VG032 admission evidence does not bind VG033")


def _load_parent(model: VQ2PhaseRecurrentActor) -> dict[str, Any]:
    payload = torch.load(
        PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    contract = payload.get("model", {})
    if (
        payload.get("schema") != core.CHECKPOINT_SCHEMA
        or payload.get("tag") != "vq2_vg028_variable_gate_seven_source_refit_001"
        or contract.get("legal_observation_size") != PHASE_LEGAL_OBS_SIZE
        or contract.get("hidden_size") != model.hidden_size
        or payload.get("best_epoch") != 6
        or not payload.get("numerically_admitted")
    ):
        raise RuntimeError("VG028 parent actor contract changed")
    model.load_state_dict(payload["model_state"])
    return payload


def numerical_admission_predicate(
    *,
    best_epoch: int,
    best_score: float,
    baseline_validation: dict[str, Any],
    selected_validation: dict[str, Any],
    source_balance_audit: bool,
    config: EightSourceConfig,
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
    }
    return bool(
        best_epoch > 0
        and np.isfinite(best_score)
        and best_score
        < float(baseline_validation["source_balanced_weighted_mse"])
        and selected_validation["dagger7"]["weighted_mse"]
        < baseline_validation["dagger7"]["weighted_mse"]
        and all(
            selected_validation[name]["weighted_mse"] <= cap
            for name, cap in caps.items()
        )
        and source_balance_audit
    )


def configure_core() -> None:
    core.TAG = TAG
    core.REPORT_SCHEMA = REPORT_SCHEMA
    core.STATE_SCHEMA = STATE_SCHEMA
    core.SEED = SEED
    core.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    core.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    core.PARENT_REPORT = PARENT_REPORT
    core.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    core.PARENT_ADMISSION = PARENT_ADMISSION
    core.PARENT_ADMISSION_SHA256 = PARENT_ADMISSION_SHA256
    core.PREREGISTRATION = PREREGISTRATION
    core.RUNNER = RUNNER
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.SOURCE_WEIGHTS = dict(SOURCE_WEIGHTS)
    core.EXTRA_DATASET_SPECS = {
        "dagger7": {
            "dataset": VG032_DATASET,
            "report_sha256": VG032_REPORT_SHA256,
            "metadata_sha256": VG032_METADATA_SHA256,
            "validation_agents_attribute": "dagger7_validation_agents",
        }
    }
    core.EXTRA_OBJECTIVE_WEIGHT_ATTRIBUTES = {
        "dagger7": "dagger7_objective_weight"
    }
    core.NUMERICAL_ADMISSION_PREDICATE = numerical_admission_predicate
    core.verify_inputs = verify_inputs
    core._load_parent = _load_parent
    core.source_paths = source_paths
    core.current_source_identity = current_source_identity


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: EightSourceConfig = EightSourceConfig(),
    resume: bool = False,
) -> dict[str, Any]:
    _ORIGINAL_VERIFY_INPUTS()
    configure_core()
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
