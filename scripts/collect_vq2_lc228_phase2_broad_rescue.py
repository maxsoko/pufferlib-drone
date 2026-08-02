#!/usr/bin/env python3
"""Test and capture broad phase-2 oracle rescue of LC216 Gate-3 failures."""

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
from scripts.collect_vq2_lc129_phase8_9_expanded_rescue_corpus import SplitBatchActor
from scripts.eval_vq2_lc178_phase15_recurrent_adapter_milestone import load_actor
import scripts.collect_vq2_lc125_lc123_phase8_9_rescue_features as base


BASE_WRITE_JSON_ONCE = base.write_json_once
BASE_ACTOR_LOADER = base.milestone.load_actor

TAG = "vq2_lc228r_phase2_broad_rescue_001"
SCHEMA = "vq2_lc228r_phase2_broad_rescue_report_v1"
FEATURE_SCHEMA = "vq2_lc228r_phase2_broad_rescue_feature_v1"
GROUP_SIZE = 256
TOTAL_AGENTS = 512
MAX_STEPS = 12_000
TARGET_RAW_INDEX = 3
PHASE_MIN = 2
PHASE_MAX_EXCLUSIVE = 3
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 287
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc216_all24_action_sequence_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "672af0c4b014bb7d7399e2d709f268a487a83294b7b2a72ebfc79a48896a95b9"
)
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = (
    "b112413c778d984f369717643cdd8d69e0ec984e28c47e1b36b20aed18503617"
)
LC227_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc227_raw3_factor_isolation_control_001/report.json"
)
LC227_REPORT_SHA256 = (
    "d2994cb13d7bccc37e1b6ae13e075b67ff06a7d9921dd82f709b183a8359150e"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc228_phase2_broad_rescue_preregistration_2026-08-02.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc228_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc228_phase2_broad_rescue.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC227_REPORT: LC227_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC228 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    construction = json.loads(PARENT_REPORT.read_text())
    failure = json.loads(LC227_REPORT.read_text())
    candidate = failure.get("items", [{}, {}])[1]
    if (
        parent.get("schema") != "vq2_lc216_all24_action_sequence_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent.get("model", {}).get("class") != "VQ2PhaseActionSequenceActor"
        or parent.get("model", {}).get("sequence_length") != 9_592
        or construction.get("schema")
        != "vq2_lc216_all24_action_sequence_report_v1"
        or not construction.get("numerically_admitted")
        or construction.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or failure.get("schema") != "vq2_lc227_raw3_factor_isolation_report_v1"
        or not failure.get("diagnostic_valid")
        or candidate.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or candidate.get("target_passes") != 0
        or candidate.get("maximum_raw_index_distribution", {}).get("2") != 8
        or failure.get("safety", {}).get("flight_sim_packets_sent") != 0
        or failure.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC216/LC227 do not authorize LC228")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC227_REPORT,
        ROOT / "scripts/collect_vq2_lc125_lc123_phase8_9_rescue_features.py",
        ROOT / "scripts/collect_vq2_lc129_phase8_9_expanded_rescue_corpus.py",
        ROOT / "scripts/eval_vq2_lc178_phase15_recurrent_adapter_milestone.py",
        ROOT / "pufferlib/vq2_oracle.py", ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
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


def split_actor_loader(payload: dict[str, Any], device: torch.device) -> SplitBatchActor:
    return SplitBatchActor((load_actor(payload, device), load_actor(payload, device)))


def sequence_feature_components(
    actor: SplitBatchActor,
    result: Any,
    next_recurrent: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Store only the base 256-state Puffer, excluding adapter/counter state."""

    del actor
    if next_recurrent.shape != (1, TOTAL_AGENTS, 321):
        raise RuntimeError("LC228R action-sequence recurrent ABI changed")
    return (
        next_recurrent[0, :, :256].index_select(0, index),
        result.pre_tanh_mean.index_select(0, index),
    )


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        control, intervention = corrected["items"]
        control["name"] = "lc216_puffer_control"
        intervention["name"] = "lc216_phase2_alignment_oracle"
        success = np.asarray(
            corrected["query_outcome_success_agents"], dtype=np.int64
        )
        failure = np.asarray(
            corrected["query_outcome_failure_agents"], dtype=np.int64
        )
        intervention_success = int((success >= GROUP_SIZE).sum())
        control_failure = int((failure < GROUP_SIZE).sum())
        predicates = {
            "broad_state_dependent_rescue": bool(
                intervention["target_passes"] >= 250
                and intervention["target_passes"]
                >= control["target_passes"] + 128
                and intervention["paired_target_gains_vs_control"] >= 128
                and intervention["paired_target_losses_vs_control"] == 0
            ),
            "all_rows_query_phase2": corrected["query_agents"] == TOTAL_AGENTS,
            "dense_intervention_success": intervention_success >= 250,
            "dense_control_failure": control_failure >= 128,
            "both_groups_recorded": bool(
                corrected["feature_records"]
                > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase2": bool(
                corrected["feature_phase_records"][PHASE_MIN]
                == corrected["feature_records"]
            ),
            "finite_in_envelope_features": (
                corrected["teacher_action_envelope_violations"] == 0
            ),
            "transport_and_pairing": bool(
                corrected["diagnostic_valid"]
                and corrected["initial_seed_groups_exact"]
            ),
            "training_only_authority": True,
        }
        corrected["feature_schema"] = FEATURE_SCHEMA
        corrected["training_dataset_admitted"] = all(predicates.values())
        corrected["admission_predicates"] = predicates
        corrected["failed_admission_predicates"] = [
            name for name, passed in predicates.items() if not passed
        ]
        corrected["actor_execution"] = (
            "two independent complete recurrent LC216 Puffers over 256 rows each"
        )
        corrected["control_features_use_teacher_targets"] = True
        corrected["feature_hidden_contract"] = (
            "LC216 base recurrent state only, 256 values; adapter and sequence counter excluded"
        )
        corrected["next_authority"] = (
            "Fit one phase-2 whole-Puffer residual on the broad LC216 failure/rescue corpus; no FlightSim authority."
            if corrected["training_dataset_admitted"] else
            "Reject the phase-2 oracle target and do not fit or run FlightSim."
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
    base.GROUP_SIZE, base.TOTAL_AGENTS, base.MAX_STEPS = (
        GROUP_SIZE, TOTAL_AGENTS, MAX_STEPS,
    )
    base.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE = PHASE_MIN, PHASE_MAX_EXCLUSIVE
    base.ENV_SEED_GROUP_SIZE, base.ENV_SEED_INDEX_OFFSET = (
        ENV_SEED_GROUP_SIZE, ENV_SEED_INDEX_OFFSET,
    )
    base.CAPTURE_CONTROL_FEATURES = True
    base.CAPTURE_CONTROL_TEACHER_TARGETS = True
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.write_json_once, base.milestone.load_actor = corrected_writer, split_actor_loader
    base.feature_components = sequence_feature_components
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
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
