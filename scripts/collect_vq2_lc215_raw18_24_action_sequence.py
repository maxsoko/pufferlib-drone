#!/usr/bin/env python3
"""Collect an exact training-only oracle continuation from LC213 to raw 24."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc183_phase15_adapter_onpolicy_dagger as common
import scripts.collect_vq2_lc193_phase16_17_rescue_dagger as prior


TAG = "vq2_lc215_raw18_24_action_sequence_001"
SCHEMA = "vq2_lc215_raw18_24_action_sequence_report_v1"
FEATURE_SCHEMA = "vq2_lc215_raw18_24_action_sequence_feature_v1"
GROUP_SIZE = 256
TOTAL_AGENTS = 512
MAX_STEPS = 45_000
TARGET_RAW_INDEX = 24
PHASE_MIN = 18
PHASE_MAX_EXCLUSIVE = 24
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc213_phase16_17_action_sequence_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "3ab6bebed9c1350601e4a664ef78caed65c755f84f92b2b0dbd99686cf99f156"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "e5d2f309ca259c68d3f11d61636bcd438423de43b05dadd6a8a35ccd99013b39"
LC214_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc214_phase16_17_action_sequence_milestone_001/report.json"
LC214_REPORT_SHA256 = "982fc16b9e7af6c7783c1196751e2a15f8c6d95dc621619056fa22e04c45a9f5"
PREREGISTRATION = ROOT / "docs/vq2_lc215_raw18_24_action_sequence_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc215_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc215_raw18_24_action_sequence.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC214_REPORT: LC214_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC215 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    milestone = json.loads(LC214_REPORT.read_text())
    selected = milestone.get("causal_screen_selected", {})
    if (
        parent.get("schema") != "vq2_lc213_phase16_17_action_sequence_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent.get("model", {}).get("sequence_length") != 2_329
        or parent_report.get("schema")
        != "vq2_lc213_phase16_17_action_sequence_report_v1"
        or not parent_report.get("numerically_admitted")
        or milestone.get("schema")
        != "vq2_lc214_phase16_17_action_sequence_milestone_report_v1"
        or not milestone.get("diagnostic_valid")
        or selected.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or selected.get("target_passes") != GROUP_SIZE // 2
        or selected.get("maximum_raw_index_distribution", {}).get("18")
        != GROUP_SIZE // 2
        or selected.get("paired_target_losses_vs_baseline") != 0
        or milestone.get("safety", {}).get("flight_sim_packets_sent") != 0
        or milestone.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC213/LC214 do not authorize LC215")
    return parent


def sequence_feature_components(
    actor: Any,
    result: Any,
    next_recurrent: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del actor
    if next_recurrent.shape != (1, TOTAL_AGENTS, 321):
        raise RuntimeError("LC215 action-sequence recurrent ABI changed")
    return (
        next_recurrent[0, :, :256].index_select(0, index),
        result.pre_tanh_mean.index_select(0, index),
    )


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC214_REPORT,
        ROOT / "scripts/collect_vq2_lc193_phase16_17_rescue_dagger.py",
        ROOT / "scripts/collect_vq2_lc183_phase15_adapter_onpolicy_dagger.py",
        ROOT / "scripts/eval_vq2_lc214_phase16_17_action_sequence_milestone.py",
        ROOT / "pufferlib/vq2_oracle.py", ROOT / "pufferlib/vq2_informed.py",
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
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": __import__("numpy").__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
    }


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        control, intervention = corrected["items"]
        control["name"] = "lc213_raw18_control"
        intervention["name"] = "lc213_raw18_24_alignment_oracle"
        phase_records = corrected["feature_phase_records"]
        predicates = {
            "all24_rescue": bool(
                control["target_passes"] == 0
                and intervention["target_passes"] == GROUP_SIZE
                and intervention["paired_target_gains_vs_control"] == GROUP_SIZE
                and intervention["paired_target_losses_vs_control"] == 0
            ),
            "all_rows_query_continuation": corrected["query_agents"] == TOTAL_AGENTS,
            "split_success_failure_groups": bool(
                corrected["query_outcome_success_agents"]
                == list(range(GROUP_SIZE, TOTAL_AGENTS))
                and corrected["query_outcome_failure_agents"]
                == list(range(GROUP_SIZE))
            ),
            "failure_states_have_teacher_targets": bool(
                corrected.get("control_features_use_teacher_targets")
                and corrected["feature_records"]
                > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase18_23": bool(
                sum(phase_records[PHASE_MIN:PHASE_MAX_EXCLUSIVE])
                == corrected["feature_records"]
            ),
            "every_later_phase_represented": all(
                phase_records[phase] > 0
                for phase in range(PHASE_MIN, PHASE_MAX_EXCLUSIVE)
            ),
            "finite_in_envelope_features": corrected[
                "teacher_action_envelope_violations"
            ] == 0,
            "transport_and_pairing": bool(
                corrected["diagnostic_valid"]
                and corrected["initial_seed_groups_exact"]
            ),
            "lc213_sequence_is_runtime_parent": True,
            "training_only_authority": True,
        }
        corrected["feature_schema"] = FEATURE_SCHEMA
        corrected["feature_hidden_contract"] = (
            "frozen base 256-state recurrent output; LC213 phase sequence remains in trajectory"
        )
        corrected["training_dataset_admitted"] = all(predicates.values())
        corrected["admission_predicates"] = predicates
        corrected["failed_admission_predicates"] = [
            name for name, passed in predicates.items() if not passed
        ]
        corrected["actor_execution"] = (
            "two independent complete recurrent LC213 action-sequence Puffers over 256 rows each"
        )
        corrected["next_authority"] = (
            "Concatenate the exact intervention sequence after LC213 and run one teacher-free all-24 screen; no FlightSim authority."
            if corrected["training_dataset_admitted"]
            else "Reject LC215 and retain LC213 at raw 18; do not run FlightSim."
        )
    common.BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[tuple[str, ...], tuple[Any, ...], Any]:
    names = (
        "TAG", "SCHEMA", "FEATURE_SCHEMA", "MAX_STEPS", "TARGET_RAW_INDEX",
        "PHASE_MIN", "PHASE_MAX_EXCLUSIVE", "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT", "PARENT_REPORT_SHA256",
        "LC192_REPORT", "LC192_REPORT_SHA256", "PREREGISTRATION", "RUNNER",
        "TEST", "DEFAULT_OUTPUT", "verify_inputs", "source_identity",
        "corrected_writer",
    )
    originals = tuple(getattr(prior, name) for name in names)
    values = (
        TAG, SCHEMA, FEATURE_SCHEMA, MAX_STEPS, TARGET_RAW_INDEX, PHASE_MIN,
        PHASE_MAX_EXCLUSIVE, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT, PARENT_REPORT_SHA256, LC214_REPORT, LC214_REPORT_SHA256,
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT, verify_inputs,
        source_identity, corrected_writer,
    )
    for name, value in zip(names, values):
        setattr(prior, name, value)
    feature_original = prior.prior.phase17_feature_components
    prior.prior.phase17_feature_components = sequence_feature_components
    return names, originals, feature_original


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...], Any]) -> None:
    names, originals, feature_original = snapshot
    for name, value in zip(names, originals):
        setattr(prior, name, value)
    prior.prior.phase17_feature_components = feature_original


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
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
