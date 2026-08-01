#!/usr/bin/env python3
"""Test and record state-dependent phase-11 rescue from exact LC143."""

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
import scripts.collect_vq2_lc125_lc123_phase8_9_rescue_features as base


BASE_WRITE_JSON_ONCE = base.write_json_once
BASE_ACTOR_LOADER = base.milestone.load_actor

TAG = "vq2_lc147_phase11_state_dependent_rescue_001"
SCHEMA = "vq2_lc147_phase11_state_dependent_rescue_report_v1"
FEATURE_SCHEMA = "vq2_lc147_phase11_state_dependent_rescue_feature_v1"
GROUP_SIZE = 256
TOTAL_AGENTS = 512
MAX_STEPS = 17_000
TARGET_RAW_INDEX = 12
PHASE_MIN = 11
PHASE_MAX_EXCLUSIVE = 12
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc143_phase10_split_batch_cem_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "c4fe8fc5e069ea7696f52a81a9dd55a702957b92ad73285d55c06c6783e46ea6"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "8ad28f6c61589e9430e2813a6b7c7b37538cdd885c12c2d0778d0dbf09800bd9"
LC144_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc144_phase10_cem_milestone_001/report.json"
)
LC144_REPORT_SHA256 = "e418dae3cdddfd34759cc75f6f2e239666ca3aae59a02d227f8e73fb9bfa6686"
LC146_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc146_phase11_wide_frontier_cem_001/report.json"
)
LC146_REPORT_SHA256 = "b615ac29ac3068f87fffc096526f3ce18c4f3ed112d01b1fd2534127906b787a"
PREREGISTRATION = ROOT / "docs/vq2_lc147_phase11_state_dependent_rescue_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc147_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc147_phase11_state_dependent_rescue.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC144_REPORT: LC144_REPORT_SHA256,
        LC146_REPORT: LC146_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC147 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    milestone = json.loads(LC144_REPORT.read_text())
    exhausted = json.loads(LC146_REPORT.read_text())
    selected = milestone.get("causal_screen_selected", {})
    generations = exhausted.get("generations", [])
    if (
        parent.get("schema") != "vq2_lc143_phase10_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc143_phase10_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or milestone.get("schema") != "vq2_lc144_phase10_cem_milestone_report_v1"
        or not milestone.get("diagnostic_valid")
        or selected.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or selected.get("target_passes") != 1
        or exhausted.get("schema") != "vq2_lc146_phase11_wide_frontier_cem_report_v1"
        or exhausted.get("training_admitted")
        or len(generations) != 2
        or any(item.get("target_passes") != 0 for item in generations)
        or any(not item.get("transport_pass") for item in generations)
        or exhausted.get("safety", {}).get("flight_sim_packets_sent") != 0
        or exhausted.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC143/LC144/LC146 do not authorize LC147")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC144_REPORT, LC146_REPORT,
        ROOT / "scripts/collect_vq2_lc125_lc123_phase8_9_rescue_features.py",
        ROOT / "scripts/collect_vq2_lc129_phase8_9_expanded_rescue_corpus.py",
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


def split_actor_loader(payload: dict[str, Any], device: torch.device) -> SplitBatchActor:
    return SplitBatchActor((BASE_ACTOR_LOADER(payload, device), BASE_ACTOR_LOADER(payload, device)))


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        control, intervention = corrected["items"]
        control["name"] = "lc143_puffer_control"
        intervention["name"] = "lc143_phase11_alignment_oracle"
        success = corrected["query_outcome_success_agents"]
        failure = corrected["query_outcome_failure_agents"]
        predicates = {
            "state_dependent_rescue": bool(
                control["target_passes"] == 0
                and intervention["target_passes"] >= 1
                and intervention["paired_target_gains_vs_control"] >= 1
                and intervention["paired_target_losses_vs_control"] == 0
            ),
            "all_rows_query_phase11": corrected["query_agents"] == TOTAL_AGENTS,
            "split_success_failure_groups": bool(
                len(success) >= 1 and len(failure) >= 1
                and min(success) >= GROUP_SIZE and max(failure) < GROUP_SIZE
            ),
            "both_groups_recorded": bool(
                corrected["feature_records"]
                > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase11": bool(
                corrected["feature_phase_records"][PHASE_MIN]
                == corrected["feature_records"]
            ),
            "finite_in_envelope_features": (
                corrected["teacher_action_envelope_violations"] == 0
            ),
            "transport_and_pairing": bool(
                corrected["diagnostic_valid"] and corrected["initial_seed_groups_exact"]
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
            "two independent complete recurrent LC143 Puffer actors over 256 rows each"
        )
        corrected["next_authority"] = (
            "Fit one state-dependent phase-11 Puffer residual and screen raw 12; no FlightSim authority."
            if corrected["training_dataset_admitted"] else
            "Reject the phase-11 alignment target; do not fit or run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.FEATURE_SCHEMA,
        base.GROUP_SIZE, base.TOTAL_AGENTS, base.MAX_STEPS,
        base.TARGET_RAW_INDEX, base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE,
        base.ENV_SEED_GROUP_SIZE, base.ENV_SEED_INDEX_OFFSET,
        base.CAPTURE_CONTROL_FEATURES,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor,
    )
    base.TAG, base.SCHEMA, base.FEATURE_SCHEMA = TAG, SCHEMA, FEATURE_SCHEMA
    base.GROUP_SIZE, base.TOTAL_AGENTS, base.MAX_STEPS = GROUP_SIZE, TOTAL_AGENTS, MAX_STEPS
    base.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE = PHASE_MIN, PHASE_MAX_EXCLUSIVE
    base.ENV_SEED_GROUP_SIZE, base.ENV_SEED_INDEX_OFFSET = (
        ENV_SEED_GROUP_SIZE, ENV_SEED_INDEX_OFFSET,
    )
    base.CAPTURE_CONTROL_FEATURES = True
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.write_json_once, base.milestone.load_actor = corrected_writer, split_actor_loader
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.FEATURE_SCHEMA,
        base.GROUP_SIZE, base.TOTAL_AGENTS, base.MAX_STEPS,
        base.TARGET_RAW_INDEX, base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE,
        base.ENV_SEED_GROUP_SIZE, base.ENV_SEED_INDEX_OFFSET,
        base.CAPTURE_CONTROL_FEATURES,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor,
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
