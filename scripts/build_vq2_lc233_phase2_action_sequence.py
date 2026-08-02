#!/usr/bin/env python3
"""Build LC233 by checkpointing LC230 seed-287 oracle phase-2 actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import FEATURE_DTYPE
from scripts.eval_vq2_lc015_restore_vg071_head1 import state_sha256
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc233_phase2_action_sequence_001"
SCHEMA = "vq2_lc233_phase2_action_sequence_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc233_phase2_action_sequence_checkpoint_v1"
NUMPY_SCHEMA = "vq2_lc233_phase2_action_sequence_numpy_checkpoint_v1"
SOURCE_AGENT = 8
SOURCE_PHASE = 2
EXPECTED_START_STEP = 1760
EXPECTED_END_STEP = 3192
EXPECTED_SEQUENCE_LENGTH = 1433
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
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc230_numpy_batch1_phase2_rescue_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = (
    "761bfea1700a86c8a9923036c440d1fbb7e5c535fe36c8f5d7d23d1b35e1273a"
)
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = (
    "514325b30725ff39cfe3e141560f8e117fbbd7156d69b482ff9baac044b9264f"
)
LC232_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc232_phase2_numpy_batch1_native_raw3_001/report.json"
)
LC232_REPORT_SHA256 = (
    "40d2aaa929be12e42b7f99a5e623a19fa391a3e7f5d3f7ba4e3710780d907c01"
)
CALLABLE = ROOT / "scripts/policy_callable_vq2_lc233_phase2_sequence.py"
PREREGISTRATION = (
    ROOT / "docs/vq2_lc233_phase2_action_sequence_preregistration_2026-08-02.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc233_vast.sh"
TEST = ROOT / "tests/test_build_vq2_lc233_phase2_action_sequence.py"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def source_sequence(records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = records[
        (records["agent_index"] == SOURCE_AGENT)
        & (records["phase_index"] == SOURCE_PHASE)
    ]
    order = np.argsort(selected["step"], kind="stable")
    selected = selected[order]
    steps = np.asarray(selected["step"], dtype=np.int64)
    actions = np.asarray(selected["teacher_action"], dtype=np.float32)
    if (
        len(selected) != EXPECTED_SEQUENCE_LENGTH
        or steps[0] != EXPECTED_START_STEP
        or steps[-1] != EXPECTED_END_STEP
        or not np.array_equal(
            steps, np.arange(EXPECTED_START_STEP, EXPECTED_END_STEP + 1)
        )
        or int(np.asarray(selected["terminal"], dtype=np.uint8).sum()) != 0
        or not np.isfinite(actions).all()
        or bool((np.abs(actions) > 1.0 + 1e-6).any())
    ):
        raise RuntimeError("LC233 source action sequence contract changed")
    return steps, actions


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        LC232_REPORT: LC232_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC233 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    construction = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC232_REPORT.read_text())
    intervention = dataset.get("items", [{}, {}])[1]
    if (
        parent.get("schema") != "vq2_lc216_all24_action_sequence_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or construction.get("schema")
        != "vq2_lc216_all24_action_sequence_report_v1"
        or not construction.get("numerically_admitted")
        or construction.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema")
        != "vq2_lc230_numpy_batch1_phase2_rescue_report_v1"
        or not dataset.get("diagnostic_valid")
        or intervention.get("target_passes") != 8
        or intervention.get("pre_target_terminals") != 0
        or SOURCE_AGENT not in dataset.get("query_outcome_success_agents", [])
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or rejected.get("schema")
        != "vq2_lc232_phase2_numpy_batch1_native_report_v1"
        or rejected.get("milestone_admitted")
        or rejected.get("outcome", {}).get("target_passes") != 0
        or rejected.get("outcome", {}).get("pre_target_terminals") != 8
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("LC216/LC230/LC232 do not authorize LC233")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST, CALLABLE,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        LC232_REPORT,
        ROOT / "scripts/collect_vq2_lc230_numpy_batch1_phase2_rescue.py",
        ROOT / "scripts/policy_callable_vq2_all24_sequence.py",
    )
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in paths
        },
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def build(*, output: Path = DEFAULT_OUTPUT, resume: bool = False) -> dict[str, Any]:
    parent = verify_inputs()
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC233 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    identity = source_identity()
    records = np.memmap(FEATURES, dtype=FEATURE_DTYPE, mode="r")
    steps, actions = source_sequence(records)
    state = {
        name: value.detach().cpu().clone()
        for name, value in parent["model_state"].items()
    }
    parent_state_hash = state_sha256(state)
    state["phase2_action_sequence"] = torch.from_numpy(actions.copy())
    candidate_state_hash = state_sha256(state)
    contract = {
        **parent["model"],
        "class": "VQ2Phase2AndLateActionSequenceActor",
        "phase2_sequence_phase": SOURCE_PHASE,
        "phase2_sequence_length": EXPECTED_SEQUENCE_LENGTH,
    }
    checkpoint = {
        **parent, "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "model": contract, "model_state": state,
        "numerically_admitted": True,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "phase2_action_sequence": {
            "source_agent": SOURCE_AGENT,
            "source_step_min": int(steps[0]), "source_step_max": int(steps[-1]),
            "sequence_length": len(actions), "sequence_sha256": array_sha256(actions),
            "parent_state_sha256": parent_state_hash,
            "candidate_state_sha256": candidate_state_hash,
        },
    }
    checkpoint_path = output / "policy_selected.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    checkpoint_sha256 = sha256_path(checkpoint_path)
    arrays = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, value in state.items()
    }
    arrays["__metadata_json__"] = np.asarray(json.dumps({
        "schema": NUMPY_SCHEMA,
        "source_checkpoint_sha256": checkpoint_sha256,
        "model": contract,
    }, sort_keys=True), dtype=np.str_)
    numpy_path = output / "policy_selected_numpy.npz"
    np.savez_compressed(numpy_path, **arrays)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": True,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "numpy_checkpoint": numpy_path.name,
        "numpy_checkpoint_sha256": sha256_path(numpy_path),
        "callable_sha256": sha256_path(CALLABLE),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_state_sha256": parent_state_hash,
        "candidate_state_sha256": candidate_state_hash,
        "source_agent": SOURCE_AGENT, "source_phase": SOURCE_PHASE,
        "source_step_min": int(steps[0]), "source_step_max": int(steps[-1]),
        "sequence_length": len(actions), "sequence_sha256": array_sha256(actions),
        "frozen_parent_state_exact": all(
            torch.equal(parent["model_state"][name], state[name])
            for name in parent["model_state"]
        ),
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "training_only_teacher_source": True,
            "runtime_privileged_values": 0, "runtime_teacher_actions": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Run teacher-free exact NumPy batch-1 Gate-3 admission; no FlightSim authority."
        ),
    }
    write_json_once(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = build(output=args.output.resolve(), resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("numerically_admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
