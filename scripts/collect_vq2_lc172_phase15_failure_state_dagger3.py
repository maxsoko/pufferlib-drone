#!/usr/bin/env python3
"""Collect a third phase-15 DAgger iteration from LC169's shifted failures."""

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
import scripts.collect_vq2_lc165_phase15_failure_state_dagger as prior


TAG = "vq2_lc172_phase15_failure_state_dagger3_001"
SCHEMA = "vq2_lc172_phase15_failure_state_dagger3_report_v1"
FEATURE_SCHEMA = "vq2_lc172_phase15_failure_state_dagger3_feature_v1"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc169_phase15_failure_state_dagger2_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "8ae1a6d9aebd01d584344f006724f5a81699a011a759356283ada1f271ece1b7"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "182e81da9cb41a7f8479f0a2cfa99baef9b408a63cdfd1024130e38349cccf9b"
LC171_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc171_phase15_dagger2_interpolation_bracket_001/report.json"
LC171_REPORT_SHA256 = "94aacd9da01fd38264ff020302c20cf56aaf5ffffb7cae9b6436e5a1a71a5909"
PREREGISTRATION = ROOT / "docs/vq2_lc172_phase15_failure_state_dagger3_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc172_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc172_phase15_failure_state_dagger3.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC171_REPORT: LC171_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC172 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC171_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc169_phase15_failure_state_dagger2_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc169_phase15_failure_state_dagger2_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or rejected.get("schema") != "vq2_lc171_phase15_dagger2_interpolation_bracket_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or any(item.get("target_passes") != 0 for item in rejected.get("items", []))
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC169/LC171 do not authorize LC172")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC171_REPORT,
        ROOT / "scripts/collect_vq2_lc165_phase15_failure_state_dagger.py",
        ROOT / "scripts/collect_vq2_lc168_phase15_failure_state_dagger2.py",
        ROOT / "scripts/collect_vq2_lc152_phase11_failure_state_dagger.py",
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
        control["name"] = "lc169_puffer_failure_control"
        intervention["name"] = "lc169_phase15_alignment_oracle"
        success = corrected["query_outcome_success_agents"]
        failure = corrected["query_outcome_failure_agents"]
        predicates = {
            "state_dependent_rescue": bool(
                control["target_passes"] == 0
                and intervention["target_passes"] == prior.GROUP_SIZE
                and intervention["paired_target_gains_vs_control"] == prior.GROUP_SIZE
                and intervention["paired_target_losses_vs_control"] == 0
            ),
            "all_rows_query_phase15": corrected["query_agents"] == prior.TOTAL_AGENTS,
            "split_success_failure_groups": bool(
                success == list(range(prior.GROUP_SIZE, prior.TOTAL_AGENTS))
                and failure == list(range(prior.GROUP_SIZE))
            ),
            "failure_states_have_teacher_targets": bool(
                corrected.get("control_features_use_teacher_targets")
                and corrected["feature_records"] > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase15": corrected["feature_phase_records"][prior.PHASE_MIN] == corrected["feature_records"],
            "finite_in_envelope_features": corrected["teacher_action_envelope_violations"] == 0,
            "transport_and_pairing": bool(corrected["diagnostic_valid"] and corrected["initial_seed_groups_exact"]),
            "training_only_authority": True,
        }
        corrected["feature_schema"] = FEATURE_SCHEMA
        corrected["training_dataset_admitted"] = all(predicates.values())
        corrected["admission_predicates"] = predicates
        corrected["failed_admission_predicates"] = [name for name, passed in predicates.items() if not passed]
        corrected["actor_execution"] = "two independent complete recurrent LC169 Puffer actors over 256 rows each"
        corrected["next_authority"] = (
            "Fit the third phase-15 whole-Puffer DAgger residual; no FlightSim authority."
            if corrected["training_dataset_admitted"] else
            "Reject LC172; do not fit or run FlightSim."
        )
    prior.BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        prior.TAG, prior.SCHEMA, prior.FEATURE_SCHEMA,
        prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.verify_inputs, prior.source_identity, prior.corrected_writer,
    )
    prior.TAG, prior.SCHEMA, prior.FEATURE_SCHEMA = TAG, SCHEMA, FEATURE_SCHEMA
    prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    prior.verify_inputs, prior.source_identity, prior.corrected_writer = verify_inputs, source_identity, corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        prior.TAG, prior.SCHEMA, prior.FEATURE_SCHEMA,
        prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.verify_inputs, prior.source_identity, prior.corrected_writer,
    ) = originals


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        return prior.collect(output=output, device_name=device_name, resume=resume)
    finally:
        restore(originals)


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
