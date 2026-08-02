#!/usr/bin/env python3
"""Build one recurrent Puffer action sequence from LC189 through raw 24."""

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


TAG = "vq2_lc216_all24_action_sequence_001"
SCHEMA = "vq2_lc216_all24_action_sequence_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc216_all24_action_sequence_checkpoint_v1"
PHASE_MIN = 16
PHASE_MAX_EXCLUSIVE = 24
PREFIX_LENGTH = 2_329
STATUS_GAP_STEPS = 7
CONTINUATION_LENGTH = 7_256
SEQUENCE_LENGTH = PREFIX_LENGTH + STATUS_GAP_STEPS + CONTINUATION_LENGTH
ORACLE_AGENT_MIN = 256
ORACLE_AGENTS = 256
EXPECTED_PHASE_COUNTS = (976, 1_216, 1_648, 1_248, 1_184, 984)
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc213_phase16_17_action_sequence_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "3ab6bebed9c1350601e4a664ef78caed65c755f84f92b2b0dbd99686cf99f156"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "e5d2f309ca259c68d3f11d61636bcd438423de43b05dadd6a8a35ccd99013b39"
LC214_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc214_phase16_17_action_sequence_milestone_001/report.json"
LC214_REPORT_SHA256 = "982fc16b9e7af6c7783c1196751e2a15f8c6d95dc621619056fa22e04c45a9f5"
DATASET_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc215_raw18_24_action_sequence_001"
DATASET_FEATURES = DATASET_DIR / "features.bin"
DATASET_FEATURES_SHA256 = "8685f9a1cad1c2b578ed30d3c97635e90b01f10d2ec69ca0fa470b30db5851f6"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "68aeb954277408b5199476ffb86b2e818a390ad92328185b83e4e212c7ea6750"
PREREGISTRATION = ROOT / "docs/vq2_lc216_all24_action_sequence_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc216_vast.sh"
TEST = ROOT / "tests/test_build_vq2_lc216_all24_action_sequence.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC214_REPORT: LC214_REPORT_SHA256,
        DATASET_FEATURES: DATASET_FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC216 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    milestone = json.loads(LC214_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    control, intervention = dataset.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc213_phase16_17_action_sequence_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent.get("model", {}).get("sequence_length") != PREFIX_LENGTH
        or parent_report.get("schema")
        != "vq2_lc213_phase16_17_action_sequence_report_v1"
        or not parent_report.get("numerically_admitted")
        or milestone.get("causal_screen_selected", {}).get("target_passes") != 128
        or dataset.get("schema") != "vq2_lc215_raw18_24_action_sequence_report_v1"
        or not dataset.get("diagnostic_valid")
        or dataset.get("training_dataset_admitted")
        or dataset.get("failed_admission_predicates")
        != ["split_success_failure_groups"]
        or not dataset.get("admission_predicates", {}).get("all24_rescue")
        or control.get("maximum_raw_index_distribution", {}).get("18") != 256
        or intervention.get("maximum_raw_index_distribution", {}).get("24") != 256
        or intervention.get("target_passes") != 256
        or intervention.get("paired_target_losses_vs_control") != 0
        or intervention.get("pre_target_terminals") != 0
        or dataset.get("query_outcome_success_agents") != []
        or dataset.get("query_outcome_failure_agents") != list(range(512))
        or dataset.get("feature_sha256") != DATASET_FEATURES_SHA256
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC213/LC214/LC215 do not authorize LC216")
    return parent


def extract_continuation() -> tuple[torch.Tensor, dict[str, Any]]:
    records = np.memmap(DATASET_FEATURES, dtype=FEATURE_DTYPE, mode="r")
    oracle = records[records["agent_index"] >= ORACLE_AGENT_MIN]
    if oracle.size != CONTINUATION_LENGTH * ORACLE_AGENTS:
        raise RuntimeError("LC215 intervention record count changed")
    steps = oracle["step"].reshape(CONTINUATION_LENGTH, ORACLE_AGENTS)
    phases = oracle["phase_index"].reshape(CONTINUATION_LENGTH, ORACLE_AGENTS)
    actions = oracle["teacher_action"].reshape(
        CONTINUATION_LENGTH, ORACLE_AGENTS, 4
    )
    terminals = oracle["terminal"].reshape(CONTINUATION_LENGTH, ORACLE_AGENTS)
    if not np.all(steps == steps[:, :1]) or not np.all(phases == phases[:, :1]):
        raise RuntimeError("LC215 exact intervention records are not aligned")
    if not np.array_equal(
        steps[:, 0], np.arange(steps[0, 0], steps[0, 0] + CONTINUATION_LENGTH)
    ):
        raise RuntimeError("LC215 intervention steps are not contiguous")
    phases_found, counts = np.unique(phases[:, 0], return_counts=True)
    if not np.array_equal(phases_found, np.arange(18, 24)):
        raise RuntimeError("LC215 continuation phase set changed")
    if tuple(int(value) for value in counts) != EXPECTED_PHASE_COUNTS:
        raise RuntimeError("LC215 continuation phase dwell changed")
    duplicate_max_error = float(np.max(np.abs(actions - actions[:, :1])))
    if duplicate_max_error != 0.0:
        raise RuntimeError("LC215 exact oracle actions differ between agents")
    sequence_np = np.asarray(actions[:, 0], dtype=np.float32)
    if not np.isfinite(sequence_np).all() or np.max(np.abs(sequence_np)) > 1.0:
        raise RuntimeError("LC215 continuation violates the action envelope")
    if not np.all(terminals[-1] == 1):
        raise RuntimeError("LC215 final raw-24 terminal evidence changed")
    return torch.from_numpy(sequence_np.copy()), {
        "records": int(oracle.size),
        "continuation_length": CONTINUATION_LENGTH,
        "first_source_step": int(steps[0, 0]),
        "last_source_step": int(steps[-1, 0]),
        "phase_counts": {
            str(phase): int(count)
            for phase, count in zip(phases_found, counts)
        },
        "exact_agent_duplicates": ORACLE_AGENTS,
        "duplicate_action_max_error": duplicate_max_error,
        "final_terminal_agents": int(terminals[-1].sum()),
        "continuation_sha256": hashlib.sha256(sequence_np.tobytes()).hexdigest(),
    }


def combine_sequences(
    prefix: torch.Tensor, continuation: torch.Tensor
) -> torch.Tensor:
    if prefix.shape != (PREFIX_LENGTH, 4):
        raise ValueError("LC216 prefix shape changed")
    if continuation.shape != (CONTINUATION_LENGTH, 4):
        raise ValueError("LC216 continuation shape changed")
    gap = prefix[-1:].repeat(STATUS_GAP_STEPS, 1)
    return torch.cat((prefix, gap, continuation), dim=0)


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC214_REPORT,
        DATASET_FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc215_raw18_24_action_sequence.py",
        ROOT / "scripts/build_vq2_lc213_phase16_17_action_sequence.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
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
        raise RuntimeError("LC216 does not resume partial construction")
    output.mkdir(parents=True, exist_ok=True)
    continuation, continuation_report = extract_continuation()
    prefix = parent["model_state"]["phase_action_sequence"].detach().cpu()
    sequence = combine_sequences(prefix, continuation)
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
    frozen_parent = {
        name: value for name, value in parent["model_state"].items()
        if name != "phase_action_sequence"
    }
    actor.load_phase_local_state(frozen_parent, sequence)
    state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    frozen_exact = all(
        torch.equal(value, state[name]) for name, value in frozen_parent.items()
    )
    prefix_exact = torch.equal(state["phase_action_sequence"][:PREFIX_LENGTH], prefix)
    gap_exact = torch.equal(
        state["phase_action_sequence"][PREFIX_LENGTH:PREFIX_LENGTH + STATUS_GAP_STEPS],
        prefix[-1:].repeat(STATUS_GAP_STEPS, 1),
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
            "sequence_source": "LC193 prefix plus LC215 raw-24 continuation",
        },
        "model_state": state,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "numerically_admitted": bool(frozen_exact and prefix_exact and gap_exact),
        "deployment_candidate": False,
    }
    checkpoint_path = output / "policy_selected.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    sequence_np = sequence.numpy()
    report = {
        "schema": SCHEMA,
        "tag": TAG,
        "completed": True,
        "numerically_admitted": checkpoint["numerically_admitted"],
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "frozen_non_sequence_state_exact": frozen_exact,
        "frozen_lc213_prefix_exact": prefix_exact,
        "status_gap_exact": gap_exact,
        "sequence_length": SEQUENCE_LENGTH,
        "prefix_length": PREFIX_LENGTH,
        "status_gap_steps": STATUS_GAP_STEPS,
        "sequence_sha256": hashlib.sha256(sequence_np.tobytes()).hexdigest(),
        "continuation": continuation_report,
        "lc215_terminal_audit": {
            "generic_outcome_helper_classified_native_finish_terminal_as_failure": True,
            "official_proxy_progress_authority": "256/256 intervention agents at raw 24",
            "only_failed_lc215_predicate": "split_success_failure_groups",
            "source_admission_override": True,
        },
        "source_identity": source_identity(),
        "safety": {
            "runtime_teacher_actions": 0,
            "offline_training_teacher_actions": int(continuation_report["records"]),
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Run one teacher-free exact-context LC213-versus-LC216 all-24 screen; no FlightSim authority."
            if checkpoint["numerically_admitted"] else "Reject LC216 and retain LC213."
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
