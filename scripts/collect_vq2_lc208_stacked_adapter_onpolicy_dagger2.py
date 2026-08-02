#!/usr/bin/env python3
"""Second phase-16/17 DAgger collection on LC206-owned trajectories."""

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
import scripts.collect_vq2_lc205_stacked_adapter_onpolicy_dagger as base


TAG = "vq2_lc208_stacked_adapter_onpolicy_dagger2_001"
SCHEMA = "vq2_lc208_stacked_adapter_onpolicy_dagger2_report_v1"
FEATURE_SCHEMA = "vq2_lc208_stacked_adapter_onpolicy_dagger2_feature_v1"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc206_stacked_adapter_onpolicy_dagger_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "85bab7bf89ce1e140ebfc8cb714aedfb2427810f433ef7b1a524727ec47f51be"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "61032f4488c0e6b4f92858d2873a0810ea591006ac7025ac9db9baadfd50df0f"
LC207_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc207_stacked_adapter_onpolicy_milestone_001/report.json"
LC207_REPORT_SHA256 = "7903ddfa4f029e6caf3203a97a9397b7d8b5571c31065664101ff77705135386"
PREREGISTRATION = ROOT / "docs/vq2_lc208_stacked_adapter_onpolicy_dagger2_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc208_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc208_stacked_adapter_onpolicy_dagger2.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC207_REPORT: LC207_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC208 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC207_REPORT.read_text())
    baseline, candidate = rejected.get("items", [{}, {}])
    if (
        parent.get("schema")
        != "vq2_lc206_stacked_adapter_onpolicy_dagger_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent.get("model", {}).get("continuation_phase_min") != base.PHASE_MIN
        or parent.get("model", {}).get("continuation_phase_max_exclusive")
        != base.PHASE_MAX_EXCLUSIVE
        or parent_report.get("schema")
        != "vq2_lc206_stacked_adapter_onpolicy_dagger_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or not parent_report.get("frozen_lc189_state_exact")
        or rejected.get("schema")
        != "vq2_lc207_stacked_adapter_onpolicy_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or baseline.get("maximum_raw_index_distribution", {}).get("17") != 128
        or candidate.get("maximum_raw_index_distribution", {}).get("16") != 128
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC206/LC207 do not authorize LC208")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC207_REPORT,
        ROOT / "scripts/collect_vq2_lc205_stacked_adapter_onpolicy_dagger.py",
        ROOT / "scripts/eval_vq2_lc207_stacked_adapter_onpolicy_milestone.py",
        ROOT / "scripts/collect_vq2_lc125_lc123_phase8_9_rescue_features.py",
        ROOT / "pufferlib/vq2_oracle.py", ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
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
        control["name"] = "lc206_stacked_adapter_onpolicy_control"
        intervention["name"] = "lc206_phase16_17_alignment_oracle"
        success = corrected["query_outcome_success_agents"]
        failure = corrected["query_outcome_failure_agents"]
        predicates = {
            "state_dependent_rescue": bool(
                control["target_passes"] == 0
                and intervention["target_passes"] == base.GROUP_SIZE
                and intervention["paired_target_gains_vs_control"] == base.GROUP_SIZE
                and intervention["paired_target_losses_vs_control"] == 0
            ),
            "all_rows_query_phase16_or17": corrected["query_agents"] == base.TOTAL_AGENTS,
            "split_success_failure_groups": bool(
                success == list(range(base.GROUP_SIZE, base.TOTAL_AGENTS))
                and failure == list(range(base.GROUP_SIZE))
            ),
            "both_onpolicy_and_rescue_rows": bool(
                corrected.get("control_features_use_teacher_targets")
                and corrected["feature_records"]
                > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase16_17": bool(
                sum(corrected["feature_phase_records"][base.PHASE_MIN:base.PHASE_MAX_EXCLUSIVE])
                == corrected["feature_records"]
            ),
            "both_phases_represented": all(
                corrected["feature_phase_records"][phase] > 0
                for phase in range(base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE)
            ),
            "finite_in_envelope_features": corrected[
                "teacher_action_envelope_violations"
            ] == 0,
            "transport_and_pairing": bool(
                corrected["diagnostic_valid"]
                and corrected["initial_seed_groups_exact"]
            ),
            "features_exclude_continuation_state_and_output": True,
            "training_only_authority": True,
        }
        corrected["feature_schema"] = FEATURE_SCHEMA
        corrected["feature_hidden_contract"] = (
            "frozen LC189 base recurrent state only, 256 values"
        )
        corrected["feature_base_pre_tanh_contract"] = (
            "LC189 whole-Puffer output before LC206 continuation residual"
        )
        corrected["training_dataset_admitted"] = all(predicates.values())
        corrected["admission_predicates"] = predicates
        corrected["failed_admission_predicates"] = [
            name for name, passed in predicates.items() if not passed
        ]
        corrected["actor_execution"] = (
            "two independent complete recurrent LC206 stacked-adapter Puffers "
            "over 256 rows each"
        )
        corrected["next_authority"] = (
            "Continue only the phase-16/17 adapter on LC206-owned failure and "
            "rescue sequences, then screen raw 18; no FlightSim authority."
            if corrected["training_dataset_admitted"]
            else "Reject LC208; do not fit or run FlightSim."
        )
    base.BASE_WRITE_JSON_ONCE(path, corrected)


def configure_outer() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "FEATURE_SCHEMA", "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT", "PARENT_REPORT_SHA256",
        "PREREGISTRATION", "RUNNER", "TEST", "DEFAULT_OUTPUT",
        "verify_inputs", "source_identity", "corrected_writer",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG, SCHEMA, FEATURE_SCHEMA, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT, PARENT_REPORT_SHA256, PREREGISTRATION, RUNNER, TEST,
        DEFAULT_OUTPUT, verify_inputs, source_identity, corrected_writer,
    )
    for name, value in zip(names, values):
        setattr(base, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(base, name, value)


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_outer()
    try:
        return base.collect(output=output, device_name=device_name, resume=resume)
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
