#!/usr/bin/env python3
"""Test and collect a phase-16/17 oracle rescue from LC189's phase-17 failures."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc192_phase17_failure_state_dagger as prior


TAG = "vq2_lc193_phase16_17_rescue_dagger_001"
SCHEMA = "vq2_lc193_phase16_17_rescue_dagger_report_v1"
FEATURE_SCHEMA = "vq2_lc193_phase16_17_rescue_dagger_feature_v1"
GROUP_SIZE = 256
TOTAL_AGENTS = 512
MAX_STEPS = 33_000
TARGET_RAW_INDEX = 18
PHASE_MIN = 16
PHASE_MAX_EXCLUSIVE = 18
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc189_phase16_adapter_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "944be397e95d2bcc427e99d09e0b7885f1b0f1e7f2c2bacaf51a6e10398fc0b2"
LC192_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc192_phase17_failure_state_dagger_001/report.json"
LC192_REPORT_SHA256 = "b9ffe592be2f00e20b67d1df8ff012498751ed9ce4a989ed1bda07e8b774f144"
PREREGISTRATION = ROOT / "docs/vq2_lc193_phase16_17_rescue_dagger_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc193_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc193_phase16_17_rescue_dagger.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC192_REPORT: LC192_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC193 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC192_REPORT.read_text())
    control, intervention = rejected.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent.get("frozen_non_phase16_state_exact")
        or parent_report.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or rejected.get("schema") != "vq2_lc192_phase17_failure_state_dagger_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("training_dataset_admitted")
        or control.get("target_passes") != 0
        or intervention.get("target_passes") != 0
        or intervention.get("maximum_raw_index_distribution", {}).get("17") != GROUP_SIZE
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC192 do not authorize LC193")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC192_REPORT,
        ROOT / "scripts/collect_vq2_lc192_phase17_failure_state_dagger.py",
        ROOT / "scripts/collect_vq2_lc183_phase15_adapter_onpolicy_dagger.py",
        ROOT / "scripts/collect_vq2_lc125_lc123_phase8_9_rescue_features.py",
        ROOT / "pufferlib/vq2_oracle.py", ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py", ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "ocean/drone_race/drone_race.c", ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "src/vecenv.h", ROOT / "src/bindings.cu",
    )
    extension = Path(_C.__file__).resolve()
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {
            **{str(path.relative_to(ROOT)): sha256_path(path) for path in paths},
            "compiled_extension": sha256_path(extension),
        },
        "compiled_extension_path": str(extension),
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
    }


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        control, intervention = corrected["items"]
        control["name"] = "lc189_phase16_17_failure_control"
        intervention["name"] = "lc189_phase16_17_alignment_oracle"
        success = corrected["query_outcome_success_agents"]
        failure = corrected["query_outcome_failure_agents"]
        predicates = {
            "earlier_horizon_rescue": bool(
                control["target_passes"] == 0
                and intervention["target_passes"] == GROUP_SIZE
                and intervention["paired_target_gains_vs_control"] == GROUP_SIZE
                and intervention["paired_target_losses_vs_control"] == 0
            ),
            "all_rows_query_phase16_or17": corrected["query_agents"] == TOTAL_AGENTS,
            "split_success_failure_groups": bool(
                success == list(range(GROUP_SIZE, TOTAL_AGENTS))
                and failure == list(range(GROUP_SIZE))
            ),
            "failure_states_have_teacher_targets": bool(
                corrected.get("control_features_use_teacher_targets")
                and corrected["feature_records"] > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase16_17": bool(
                sum(corrected["feature_phase_records"][PHASE_MIN:PHASE_MAX_EXCLUSIVE])
                == corrected["feature_records"]
            ),
            "both_phases_represented": all(
                corrected["feature_phase_records"][phase] > 0
                for phase in range(PHASE_MIN, PHASE_MAX_EXCLUSIVE)
            ),
            "finite_in_envelope_features": corrected["teacher_action_envelope_violations"] == 0,
            "transport_and_pairing": bool(corrected["diagnostic_valid"] and corrected["initial_seed_groups_exact"]),
            "phase15_adapter_preserved_as_runtime_state": True,
            "training_only_authority": True,
        }
        corrected["feature_schema"] = FEATURE_SCHEMA
        corrected["feature_hidden_contract"] = "frozen 256-state base recurrent output; phase-15 adapter remains in trajectory only"
        corrected["training_dataset_admitted"] = all(predicates.values())
        corrected["admission_predicates"] = predicates
        corrected["failed_admission_predicates"] = [name for name, passed in predicates.items() if not passed]
        corrected["actor_execution"] = "two independent complete recurrent LC189 adapter Puffers over 256 rows each"
        corrected["next_authority"] = (
            "Fit source-locked phase-16/17 whole-Puffer residual rows while preserving the phase-15 adapter, then screen raw 18 and raw 17; no FlightSim authority."
            if corrected["training_dataset_admitted"] else
            "Reject LC193 and move the training-only rescue boundary earlier; do not fit or run FlightSim."
        )
    prior.prior.BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    names = (
        "TAG", "SCHEMA", "FEATURE_SCHEMA", "MAX_STEPS", "TARGET_RAW_INDEX",
        "PHASE_MIN", "PHASE_MAX_EXCLUSIVE", "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT", "PARENT_REPORT_SHA256",
        "PREREGISTRATION", "RUNNER", "TEST", "DEFAULT_OUTPUT",
        "verify_inputs", "source_identity", "corrected_writer",
    )
    originals = tuple(getattr(prior, name) for name in names)
    values = (
        TAG, SCHEMA, FEATURE_SCHEMA, MAX_STEPS, TARGET_RAW_INDEX,
        PHASE_MIN, PHASE_MAX_EXCLUSIVE, PARENT_CHECKPOINT,
        PARENT_CHECKPOINT_SHA256, PARENT_REPORT, PARENT_REPORT_SHA256,
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
        verify_inputs, source_identity, corrected_writer,
    )
    for name, value in zip(names, values):
        setattr(prior, name, value)
    return names, originals


def restore(snapshot: tuple[Any, ...]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(prior, name, value)


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure()
    try:
        return prior.collect(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("training_dataset_admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
