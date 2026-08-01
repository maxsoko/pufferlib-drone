#!/usr/bin/env python3
"""Collect a second phase-15 DAgger iteration from LC166."""

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
import scripts.collect_vq2_lc165_phase15_failure_state_dagger as previous


TAG = "vq2_lc168_phase15_failure_state_dagger2_001"
SCHEMA = "vq2_lc168_phase15_failure_state_dagger2_report_v1"
FEATURE_SCHEMA = "vq2_lc168_phase15_failure_state_dagger2_feature_v1"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc166_phase15_failure_state_dagger_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "b9a6fc73bd7eaa81987f436d33c44325b6d0f5aef9f3394da9e99a0d6f29e518"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "825cae461cf4b095e45b441e072df13adaf5435c424f78304d909cc06c698048"
LC167_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc167_phase15_failure_state_dagger_milestone_001/report.json"
LC167_REPORT_SHA256 = "3ee78066586dee13716f3ab03f49e77d8a3066db4dcfa4a3c7be015a42c2045b"
PREREGISTRATION = ROOT / "docs/vq2_lc168_phase15_failure_state_dagger2_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc168_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc168_phase15_failure_state_dagger2.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC167_REPORT: LC167_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC168 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC167_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc166_phase15_failure_state_dagger_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc166_phase15_failure_state_dagger_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or rejected.get("schema") != "vq2_lc167_phase15_failure_state_dagger_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or any(item.get("target_passes") != 0 for item in rejected.get("items", []))
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC166/LC167 do not authorize LC168")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC167_REPORT,
        ROOT / "scripts/collect_vq2_lc165_phase15_failure_state_dagger.py",
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
        control["name"] = "lc166_puffer_failure_control"
        intervention["name"] = "lc166_phase15_alignment_oracle"
        success = corrected["query_outcome_success_agents"]
        failure = corrected["query_outcome_failure_agents"]
        predicates = {
            "state_dependent_rescue": bool(
                control["target_passes"] == 0
                and intervention["target_passes"] == previous.GROUP_SIZE
                and intervention["paired_target_gains_vs_control"] == previous.GROUP_SIZE
                and intervention["paired_target_losses_vs_control"] == 0
            ),
            "all_rows_query_phase15": corrected["query_agents"] == previous.TOTAL_AGENTS,
            "split_success_failure_groups": bool(
                success == list(range(previous.GROUP_SIZE, previous.TOTAL_AGENTS))
                and failure == list(range(previous.GROUP_SIZE))
            ),
            "failure_states_have_teacher_targets": bool(
                corrected.get("control_features_use_teacher_targets")
                and corrected["feature_records"] > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase15": bool(
                corrected["feature_phase_records"][previous.PHASE_MIN]
                == corrected["feature_records"]
            ),
            "finite_in_envelope_features": corrected["teacher_action_envelope_violations"] == 0,
            "transport_and_pairing": bool(
                corrected["diagnostic_valid"] and corrected["initial_seed_groups_exact"]
            ),
            "training_only_authority": True,
        }
        corrected["feature_schema"] = FEATURE_SCHEMA
        corrected["training_dataset_admitted"] = all(predicates.values())
        corrected["admission_predicates"] = predicates
        corrected["failed_admission_predicates"] = [name for name, passed in predicates.items() if not passed]
        corrected["actor_execution"] = (
            "two independent complete recurrent LC166 Puffer actors over 256 rows each"
        )
        corrected["next_authority"] = (
            "Fit the second phase-15 whole-Puffer DAgger residual; no FlightSim authority."
            if corrected["training_dataset_admitted"] else
            "Reject LC168; do not fit or run FlightSim."
        )
    previous.prior.BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        previous.TAG, previous.SCHEMA, previous.FEATURE_SCHEMA,
        previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256,
        previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256,
        previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT,
        previous.verify_inputs, previous.source_identity, previous.corrected_writer,
    )
    previous.TAG, previous.SCHEMA, previous.FEATURE_SCHEMA = TAG, SCHEMA, FEATURE_SCHEMA
    previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    previous.verify_inputs, previous.source_identity, previous.corrected_writer = (
        verify_inputs, source_identity, corrected_writer,
    )
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        previous.TAG, previous.SCHEMA, previous.FEATURE_SCHEMA,
        previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256,
        previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256,
        previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT,
        previous.verify_inputs, previous.source_identity, previous.corrected_writer,
    ) = originals


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        return previous.collect(output=output, device_name=device_name, resume=resume)
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
