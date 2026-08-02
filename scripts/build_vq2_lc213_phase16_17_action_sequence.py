#!/usr/bin/env python3
"""Distill LC193's exact phase-16/17 rescue into a recurrent Puffer head."""

from __future__ import annotations

import argparse
import hashlib
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

from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseActionSequenceActor
from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import FEATURE_DTYPE
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc213_phase16_17_action_sequence_001"
SCHEMA = "vq2_lc213_phase16_17_action_sequence_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc213_phase16_17_action_sequence_checkpoint_v1"
PHASE_MIN = 16
PHASE_MAX_EXCLUSIVE = 18
PHASE16_STEPS = 1_392
PHASE17_STEPS = 937
SEQUENCE_LENGTH = PHASE16_STEPS + PHASE17_STEPS
ORACLE_AGENT_MIN = 256
ORACLE_AGENTS = 256
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc189_phase16_adapter_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "944be397e95d2bcc427e99d09e0b7885f1b0f1e7f2c2bacaf51a6e10398fc0b2"
DATASET_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc193_phase16_17_rescue_dagger_001"
DATASET_FEATURES = DATASET_DIR / "features.bin"
DATASET_FEATURES_SHA256 = "699bd082aec4e0d6471377949639ed48c69329f39e47f28634d1c3ec5c739a81"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "50b1a311a153dc0277fa2ba85d97e043cd31b022aa01f5b519acd0df83c10b64"
LC212_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc212_stacked_adapter_output_ppo_001/report.json"
LC212_REPORT_SHA256 = "38af0d12c21110f8c7cd2efd7dc28ff83d24e27f4503be06ceb27a52836fa807"
PREREGISTRATION = ROOT / "docs/vq2_lc213_phase16_17_action_sequence_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc213_vast.sh"
TEST = ROOT / "tests/test_build_vq2_lc213_phase16_17_action_sequence.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        DATASET_FEATURES: DATASET_FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        LC212_REPORT: LC212_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC213 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC212_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("training_admitted")
        or dataset.get("schema") != "vq2_lc193_phase16_17_rescue_dagger_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 715_520
        or dataset.get("feature_phase_records", [])[PHASE_MIN] != 454_656
        or dataset.get("feature_phase_records", [])[PHASE_MIN + 1] != 260_864
        or dataset.get("items", [{}, {}])[1].get("target_passes") != 256
        or rejected.get("schema") != "vq2_lc212_stacked_adapter_output_ppo_report_v1"
        or rejected.get("candidate_selected_for_screen") is not None
        or any(item.get("target_passes") for item in rejected.get("rollouts", []))
        or any(
            item.get("maximum_raw_index_distribution", {}).get("16") != 512
            for item in rejected.get("rollouts", [])
        )
        or any(
            report.get("safety", {}).get("flight_sim_packets_sent") != 0
            or report.get("safety", {}).get("submission_authorized")
            for report in (parent_report, dataset, rejected)
        )
    ):
        raise RuntimeError("LC189/LC193/LC212 do not authorize LC213")
    return parent


def extract_action_sequence() -> tuple[torch.Tensor, dict[str, Any]]:
    records = np.memmap(DATASET_FEATURES, dtype=FEATURE_DTYPE, mode="r")
    oracle = records[records["agent_index"] >= ORACLE_AGENT_MIN]
    if oracle.size != SEQUENCE_LENGTH * ORACLE_AGENTS:
        raise RuntimeError("LC193 oracle feature count changed")
    steps = oracle["step"].reshape(SEQUENCE_LENGTH, ORACLE_AGENTS)
    phases = oracle["phase_index"].reshape(SEQUENCE_LENGTH, ORACLE_AGENTS)
    actions = oracle["teacher_action"].reshape(
        SEQUENCE_LENGTH, ORACLE_AGENTS, 4
    )
    if not np.all(steps == steps[:, :1]):
        raise RuntimeError("LC193 oracle records are not step-aligned")
    if not np.array_equal(
        steps[:, 0], np.arange(steps[0, 0], steps[0, 0] + SEQUENCE_LENGTH)
    ):
        raise RuntimeError("LC193 oracle steps are not contiguous")
    if not np.all(phases == phases[:, :1]):
        raise RuntimeError("LC193 oracle phase labels differ across exact agents")
    expected_phases = np.concatenate((
        np.full(PHASE16_STEPS, PHASE_MIN, dtype=np.uint8),
        np.full(PHASE17_STEPS, PHASE_MIN + 1, dtype=np.uint8),
    ))
    if not np.array_equal(phases[:, 0], expected_phases):
        raise RuntimeError("LC193 oracle phase dwell contract changed")
    duplicate_max_error = float(np.max(np.abs(actions - actions[:, :1])))
    if duplicate_max_error != 0.0:
        raise RuntimeError("LC193 exact oracle actions differ between agents")
    sequence_np = np.asarray(actions[:, 0], dtype=np.float32)
    if not np.isfinite(sequence_np).all() or np.max(np.abs(sequence_np)) > 1.0:
        raise RuntimeError("LC193 oracle sequence violates the action envelope")
    sequence = torch.from_numpy(sequence_np.copy())
    return sequence, {
        "records": int(oracle.size),
        "sequence_length": SEQUENCE_LENGTH,
        "first_source_step": int(steps[0, 0]),
        "last_source_step": int(steps[-1, 0]),
        "phase16_steps": PHASE16_STEPS,
        "phase17_steps": PHASE17_STEPS,
        "exact_agent_duplicates": ORACLE_AGENTS,
        "duplicate_action_max_error": duplicate_max_error,
        "action_minimum": sequence_np.min(axis=0).tolist(),
        "action_maximum": sequence_np.max(axis=0).tolist(),
        "sequence_sha256": hashlib.sha256(sequence_np.tobytes()).hexdigest(),
    }


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, DATASET_FEATURES, DATASET_REPORT,
        LC212_REPORT, ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc178_phase15_recurrent_adapter_milestone.py",
        ROOT / "scripts/collect_vq2_lc193_phase16_17_rescue_dagger.py",
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
        raise RuntimeError("LC213 does not resume partial construction")
    output.mkdir(parents=True, exist_ok=True)
    sequence, sequence_report = extract_action_sequence()
    contract = parent["model"]
    actor = VQ2PhaseActionSequenceActor(
        target_phase=int(contract["adapter_target_phase"]),
        sequence_phase_min=PHASE_MIN,
        sequence_phase_max_exclusive=PHASE_MAX_EXCLUSIVE,
        sequence_length=SEQUENCE_LENGTH,
        hidden_size=int(contract["hidden_size"]),
        residual_size=int(contract["residual_size"]),
        adapter_size=int(contract["adapter_size"]),
        initial_std=float(contract["initial_std"]),
    )
    actor.load_phase_local_state(parent["model_state"], sequence)
    state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    frozen_exact = all(
        torch.equal(value, state[name]) for name, value in parent["model_state"].items()
    )
    checkpoint = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": TAG,
        "model": {
            **contract,
            "class": "VQ2PhaseActionSequenceActor",
            "sequence_phase_min": PHASE_MIN,
            "sequence_phase_max_exclusive": PHASE_MAX_EXCLUSIVE,
            "sequence_length": SEQUENCE_LENGTH,
            "sequence_source": "LC193 training-only exact oracle actions",
        },
        "model_state": state,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "numerically_admitted": bool(frozen_exact),
        "deployment_candidate": False,
    }
    checkpoint_path = output / "policy_selected.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": SCHEMA,
        "tag": TAG,
        "completed": True,
        "numerically_admitted": bool(frozen_exact),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "frozen_lc189_state_exact": frozen_exact,
        "sequence": sequence_report,
        "policy_contract": (
            "complete recurrent Puffer output; LC189 before phase 16, checkpointed "
            "action-sequence head during public phases 16 and 17"
        ),
        "source_identity": source_identity(),
        "safety": {
            "runtime_teacher_actions": 0,
            "offline_training_teacher_actions": int(sequence_report["records"]),
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Run one teacher-free exact-context LC189-versus-LC213 raw-18 screen; no FlightSim authority."
            if frozen_exact else "Reject LC213 and retain LC189."
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
