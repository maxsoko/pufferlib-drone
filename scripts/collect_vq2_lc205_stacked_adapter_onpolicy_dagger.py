#!/usr/bin/env python3
"""Collect phase-16/17 oracle labels on LC202 adapter-owned trajectories."""

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

from scripts.collect_vq2_lc129_phase8_9_expanded_rescue_corpus import SplitBatchActor
from scripts.eval_vq2_lc178_phase15_recurrent_adapter_milestone import load_actor
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc125_lc123_phase8_9_rescue_features as base


BASE_WRITE_JSON_ONCE = base.write_json_once
TAG = "vq2_lc205_stacked_adapter_onpolicy_dagger_001"
SCHEMA = "vq2_lc205_stacked_adapter_onpolicy_dagger_report_v1"
FEATURE_SCHEMA = "vq2_lc205_stacked_adapter_onpolicy_dagger_feature_v1"
GROUP_SIZE = 256
TOTAL_AGENTS = 512
MAX_STEPS = 33_000
TARGET_RAW_INDEX = 18
PHASE_MIN = 16
PHASE_MAX_EXCLUSIVE = 18
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc202_phase16_17_stacked_adapter_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "9574928925d778910729d5fa55a1bd86a05767e29ddff8564bb15ea375286a73"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c048877bf078c636fa04b6ac36ef057711d092e3a47cc3e48159b093b9fada12"
LC204_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc204_stacked_adapter_scale_bracket_001/report.json"
LC204_REPORT_SHA256 = "3c11a3b5ac0fdb0a5c85ecbcddc225f40bf5f088d44388bb124e70ede45aa52a"
PREREGISTRATION = ROOT / "docs/vq2_lc205_stacked_adapter_onpolicy_dagger_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc205_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc205_stacked_adapter_onpolicy_dagger.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC204_REPORT: LC204_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC205 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC204_REPORT.read_text())
    items = rejected.get("items", [])
    if (
        parent.get("schema") != "vq2_lc202_phase16_17_stacked_adapter_checkpoint_v1"
        or parent.get("numerically_admitted")
        or parent.get("model", {}).get("continuation_phase_min") != PHASE_MIN
        or parent.get("model", {}).get("continuation_phase_max_exclusive")
        != PHASE_MAX_EXCLUSIVE
        or parent_report.get("schema")
        != "vq2_lc202_phase16_17_stacked_adapter_report_v1"
        or not parent_report.get("frozen_lc189_state_exact")
        or parent_report.get("validation_improvement_factor", 0.0) < 180.0
        or rejected.get("schema")
        != "vq2_lc204_stacked_adapter_scale_bracket_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or len(items) != 7
        or items[0].get("maximum_raw_index_distribution", {}).get("17") != 128
        or items[-1].get("maximum_raw_index_distribution", {}).get("16") != 128
        or any(item.get("target_passes") != 0 for item in items)
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC202/LC204 do not authorize LC205")
    return parent


def split_actor_loader(payload: dict[str, Any], device: torch.device) -> SplitBatchActor:
    return SplitBatchActor((load_actor(payload, device), load_actor(payload, device)))


def adapter_feature_components(
    actor: SplitBatchActor,
    result: Any,
    next_recurrent: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exclude continuation state/output while labeling its own trajectory."""

    if next_recurrent.shape != (1, TOTAL_AGENTS, 384):
        raise RuntimeError("LC205 stacked-adapter recurrent ABI changed")
    residuals = []
    for group, child in enumerate(actor.actors):
        selected = slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)
        residuals.append(
            child.continuation_adapter_output(
                next_recurrent[0, selected, 320:384]
            )
        )
    continuation_residual = torch.cat(residuals, dim=0)
    return (
        next_recurrent[0, :, :256].index_select(0, index),
        (result.pre_tanh_mean - continuation_residual).index_select(0, index),
    )


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC204_REPORT,
        ROOT / "scripts/collect_vq2_lc125_lc123_phase8_9_rescue_features.py",
        ROOT / "scripts/collect_vq2_lc183_phase15_adapter_onpolicy_dagger.py",
        ROOT / "scripts/eval_vq2_lc204_stacked_adapter_scale_bracket.py",
        ROOT / "scripts/eval_vq2_lc178_phase15_recurrent_adapter_milestone.py",
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
        control["name"] = "lc202_stacked_adapter_onpolicy_control"
        intervention["name"] = "lc202_phase16_17_alignment_oracle"
        success = corrected["query_outcome_success_agents"]
        failure = corrected["query_outcome_failure_agents"]
        predicates = {
            "state_dependent_rescue": bool(
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
            "both_onpolicy_and_rescue_rows": bool(
                corrected.get("control_features_use_teacher_targets")
                and corrected["feature_records"]
                > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase16_17": bool(
                sum(corrected["feature_phase_records"][PHASE_MIN:PHASE_MAX_EXCLUSIVE])
                == corrected["feature_records"]
            ),
            "both_phases_represented": all(
                corrected["feature_phase_records"][phase] > 0
                for phase in range(PHASE_MIN, PHASE_MAX_EXCLUSIVE)
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
            "LC189 whole-Puffer output before LC202 continuation residual"
        )
        corrected["training_dataset_admitted"] = all(predicates.values())
        corrected["admission_predicates"] = predicates
        corrected["failed_admission_predicates"] = [
            name for name, passed in predicates.items() if not passed
        ]
        corrected["actor_execution"] = (
            "two independent complete recurrent LC202 stacked-adapter Puffers "
            "over 256 rows each"
        )
        corrected["next_authority"] = (
            "Refit only the phase-16/17 continuation adapter on LC202-owned "
            "failure and rescue sequences, then screen raw 18; no FlightSim authority."
            if corrected["training_dataset_admitted"]
            else "Reject LC205; do not fit or run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.FEATURE_SCHEMA,
        base.GROUP_SIZE, base.TOTAL_AGENTS, base.MAX_STEPS,
        base.TARGET_RAW_INDEX, base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE,
        base.ENV_SEED_GROUP_SIZE, base.ENV_SEED_INDEX_OFFSET,
        base.CAPTURE_CONTROL_FEATURES, base.CAPTURE_CONTROL_TEACHER_TARGETS,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor, base.feature_components,
    )
    base.TAG, base.SCHEMA, base.FEATURE_SCHEMA = TAG, SCHEMA, FEATURE_SCHEMA
    base.GROUP_SIZE, base.TOTAL_AGENTS, base.MAX_STEPS = GROUP_SIZE, TOTAL_AGENTS, MAX_STEPS
    base.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE = PHASE_MIN, PHASE_MAX_EXCLUSIVE
    base.ENV_SEED_GROUP_SIZE, base.ENV_SEED_INDEX_OFFSET = ENV_SEED_GROUP_SIZE, ENV_SEED_INDEX_OFFSET
    base.CAPTURE_CONTROL_FEATURES = True
    base.CAPTURE_CONTROL_TEACHER_TARGETS = True
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.write_json_once, base.milestone.load_actor = corrected_writer, split_actor_loader
    base.feature_components = adapter_feature_components
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.FEATURE_SCHEMA,
        base.GROUP_SIZE, base.TOTAL_AGENTS, base.MAX_STEPS,
        base.TARGET_RAW_INDEX, base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE,
        base.ENV_SEED_GROUP_SIZE, base.ENV_SEED_INDEX_OFFSET,
        base.CAPTURE_CONTROL_FEATURES, base.CAPTURE_CONTROL_TEACHER_TARGETS,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor, base.feature_components,
    ) = originals


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        base.collect(output=output, device_name=device_name, resume=resume)
        return json.loads((output / "report.json").read_text())
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
