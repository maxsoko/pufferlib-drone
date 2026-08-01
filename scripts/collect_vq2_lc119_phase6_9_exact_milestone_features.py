#!/usr/bin/env python3
"""Capture LC117's exact phase-6--9 milestone intervention features."""

from __future__ import annotations

import argparse
import ctypes
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

from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc115_phase9_teacher_rescue as harness
import scripts.eval_vq2_lc116_phase8_9_teacher_rescue as window


TAG = "vq2_lc119_phase6_9_exact_milestone_features_001"
SCHEMA = "vq2_lc119_phase6_9_exact_milestone_features_report_v1"
FEATURE_SCHEMA = "vq2_lc119_exact_milestone_hidden_feature_v1"
PHASE_MIN = 6
PHASE_MAX_EXCLUSIVE = 10
EXPECTED_RECORDS = 60_082
EXPECTED_QUERY_AGENTS = 19
EXPECTED_SUCCESS_AGENTS = 4
EXPECTED_FAILURE_AGENTS = 15
LC117_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc117_teacher_rescue_horizon_ladder_001/report.json"
)
LC117_REPORT_SHA256 = "e2897fc5f79532ccf85bc8397186dc8652980a1e242fb9e6dfe892e8e4be08e5"
LC118_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc118_phase6_9_rescue_features_001/report.json"
)
LC118_REPORT_SHA256 = "24e71bb7ebb3401b058b58e3ff892f9b77092e0ac94ac7c7dc9fbeddf7ce01c0"
LC118_FEATURES = LC118_REPORT.parent / "features.bin"
LC118_FEATURES_SHA256 = "46a70062606dfed5ec32d3deb550aa28d41ac469bdbe18667f916864ff6c6a13"
PREREGISTRATION = ROOT / "docs/vq2_lc119_phase6_9_exact_milestone_features_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc119_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc119_phase6_9_exact_milestone_features.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
FEATURE_DTYPE = np.dtype([
    ("hidden", "<f2", (256,)),
    ("base_pre_tanh", "<f4", (4,)),
    ("teacher_action", "<f4", (4,)),
    ("phase_index", "u1"),
    ("agent_index", "<u2"),
    ("step", "<u2"),
    ("terminal", "u1"),
])


def verify_inputs() -> None:
    for path, digest in {
        LC117_REPORT: LC117_REPORT_SHA256,
        LC118_REPORT: LC118_REPORT_SHA256,
        LC118_FEATURES: LC118_FEATURES_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC119 bound input changed: {path}")
    window.verify_inputs()
    ladder = json.loads(LC117_REPORT.read_text())
    rejected = json.loads(LC118_REPORT.read_text())
    if (
        ladder.get("training_oracle_rescue_horizon") != PHASE_MIN
        or ladder.get("horizons_executed") != [7, 6]
        or ladder.get("items", [{}, {}])[-1].get("offline_teacher_plant_actions")
        != EXPECTED_RECORDS
        or rejected.get("schema") != "vq2_lc118_phase6_9_rescue_features_report_v1"
        or rejected.get("training_dataset_admitted")
        or rejected.get("feature_records") != 120_224
        or rejected.get("query_agents") != 38
        or rejected.get("query_outcome_success_agents") != 8
        or rejected.get("query_outcome_failure_agents") != 30
        or rejected.get("failed_admission_predicates")
        != ["exact_feature_and_plant_count", "only_rescue_window_recorded"]
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC117/LC118 do not authorize LC119")


def outcome_agents(records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    agents = np.unique(records["agent_index"])
    success: list[int] = []
    failure: list[int] = []
    for agent in agents:
        selected = records[records["agent_index"] == agent]
        last = selected[np.argmax(selected["step"])]
        (failure if int(last["terminal"]) else success).append(int(agent))
    return (
        np.asarray(success, dtype=np.int32),
        np.asarray(failure, dtype=np.int32),
    )


class FeatureRecorder:
    def __init__(self) -> None:
        self.result: Any = None
        self.candidate: Any = None
        self.step = 0
        self.pending: np.ndarray | None = None
        self.pending_agents: np.ndarray | None = None
        self.chunks: list[np.ndarray] = []

    def select(
        self,
        student: np.ndarray,
        teacher: np.ndarray,
        active: np.ndarray,
        phase_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.result is None or self.candidate is None or self.pending is not None:
            raise RuntimeError("LC119 actor/plant instrumentation order changed")
        teacher_mask = np.zeros(harness.TOTAL_AGENTS, dtype=bool)
        selected_group = harness.group_slice(1)
        teacher_mask[selected_group] = (
            active[selected_group]
            & (phase_index[selected_group] >= PHASE_MIN)
            & (phase_index[selected_group] < PHASE_MAX_EXCLUSIVE)
        )
        selected = np.flatnonzero(teacher_mask)
        records = np.empty(selected.size, dtype=FEATURE_DTYPE)
        if selected.size:
            index = torch.from_numpy(selected).to(self.result.pre_tanh_mean.device)
            records["hidden"] = self.candidate[0].index_select(
                0, index
            ).detach().cpu().numpy().astype(np.float16)
            records["base_pre_tanh"] = self.result.pre_tanh_mean.index_select(
                0, index
            ).detach().cpu().numpy()
            records["teacher_action"] = teacher[selected]
            records["phase_index"] = phase_index[selected].astype(np.uint8)
            records["agent_index"] = selected.astype(np.uint16)
            records["step"] = np.uint16(self.step)
            records["terminal"] = 0
        self.pending = records
        self.pending_agents = selected
        plant = student.copy()
        plant[teacher_mask] = teacher[teacher_mask]
        return plant, teacher_mask

    def finish_step(self, terminals: np.ndarray) -> None:
        if self.pending is None or self.pending_agents is None:
            raise RuntimeError("LC119 plant step occurred without a pending feature row")
        if self.pending.size:
            self.pending["terminal"] = terminals[self.pending_agents].astype(np.uint8)
            self.chunks.append(self.pending)
        self.pending = None
        self.pending_agents = None
        self.step += 1

    def records(self) -> np.ndarray:
        if self.pending is not None:
            raise RuntimeError("LC119 ended with an unexecuted feature row")
        if not self.chunks:
            return np.empty(0, dtype=FEATURE_DTYPE)
        return np.concatenate(self.chunks)


class InstrumentedActor:
    def __init__(self, actor: Any, recorder: FeatureRecorder) -> None:
        self.actor = actor
        self.recorder = recorder

    def initial_state(self, *args: Any, **kwargs: Any) -> Any:
        return self.actor.initial_state(*args, **kwargs)

    def forward_step(self, *args: Any, **kwargs: Any) -> Any:
        result, candidate = self.actor.forward_step(*args, **kwargs)
        self.recorder.result = result
        self.recorder.candidate = candidate
        return result, candidate


class InstrumentedVector:
    def __init__(self, vector: Any, recorder: FeatureRecorder) -> None:
        self.vector = vector
        self.recorder = recorder
        raw = (ctypes.c_float * harness.TOTAL_AGENTS).from_address(
            vector.terminals_ptr
        )
        self.terminals = np.ctypeslib.as_array(raw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.vector, name)

    def cpu_step(self, action_ptr: int) -> None:
        self.vector.cpu_step(action_ptr)
        self.recorder.finish_step(self.terminals > 0.5)


def source_identity(harness_report: Path) -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        LC117_REPORT, LC118_REPORT, LC118_FEATURES, harness_report,
        ROOT / "scripts/eval_vq2_lc115_phase9_teacher_rescue.py",
        ROOT / "scripts/eval_vq2_lc116_phase8_9_teacher_rescue.py",
        ROOT / "pufferlib/vq2_oracle.py",
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
        },
    }


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    from pufferlib import _C

    verify_inputs()
    report_path = output / "report.json"
    feature_path = output / "features.bin"
    harness_output = output / "harness"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC119 does not resume a partial instrumented rollout")
    output.mkdir(parents=True, exist_ok=True)

    recorder = FeatureRecorder()
    original_loader = harness.milestone.load_actor
    original_create_vec = _C.create_vec
    original_window = (
        window.TAG, window.SCHEMA, window.TEACHER_PHASE_MIN,
        window.PREREGISTRATION, window.RUNNER, window.TEST, window.DEFAULT_OUTPUT,
        window.select_plant_actions,
    )

    def load_actor(payload: dict[str, Any], device: torch.device) -> Any:
        return InstrumentedActor(original_loader(payload, device), recorder)

    def create_vec(config: dict[str, Any], gpu: int = 0) -> Any:
        return InstrumentedVector(original_create_vec(config, gpu=gpu), recorder)

    harness.milestone.load_actor = load_actor
    _C.create_vec = create_vec
    window.TAG = f"{TAG}_harness"
    window.SCHEMA = "vq2_lc119_exact_milestone_harness_v1"
    window.TEACHER_PHASE_MIN = PHASE_MIN
    window.PREREGISTRATION, window.RUNNER, window.TEST = (
        PREREGISTRATION, RUNNER, TEST,
    )
    window.DEFAULT_OUTPUT = harness_output
    window.select_plant_actions = recorder.select
    started = time.perf_counter()
    try:
        harness_report = window.run(
            output=harness_output, device_name=device_name, resume=False
        )
    finally:
        harness.milestone.load_actor = original_loader
        _C.create_vec = original_create_vec
        (
            window.TAG, window.SCHEMA, window.TEACHER_PHASE_MIN,
            window.PREREGISTRATION, window.RUNNER, window.TEST,
            window.DEFAULT_OUTPUT, window.select_plant_actions,
        ) = original_window

    records = recorder.records()
    feature_tmp = output / ".features.bin.tmp"
    records.tofile(feature_tmp)
    with feature_tmp.open("rb") as stream:
        os.fsync(stream.fileno())
    feature_tmp.replace(feature_path)
    feature_sha = sha256_path(feature_path)
    success, failure = outcome_agents(records)
    phases = np.bincount(
        records["phase_index"].astype(np.int64), minlength=33
    )
    control, intervention = harness_report["items"]
    finite = bool(
        np.isfinite(records["hidden"]).all()
        and np.isfinite(records["base_pre_tanh"]).all()
        and np.isfinite(records["teacher_action"]).all()
    )
    envelope_violations = int(
        (np.abs(records["teacher_action"]) > 1.0 + 1e-6).any(axis=1).sum()
    )
    predicates = {
        "exact_harness_rescue": bool(
            harness_report.get("diagnostic_valid")
            and harness_report.get("training_oracle_rescued_phase8_9")
            and control.get("target_passes") == 0
            and intervention.get("target_passes") == EXPECTED_SUCCESS_AGENTS
            and intervention.get("paired_target_gains_vs_control")
            == EXPECTED_SUCCESS_AGENTS
            and intervention.get("paired_target_losses_vs_control") == 0
        ),
        "exact_feature_count": len(records) == EXPECTED_RECORDS,
        "exact_query_and_outcome_counts": (
            len(np.unique(records["agent_index"])) == EXPECTED_QUERY_AGENTS
            and len(success) == EXPECTED_SUCCESS_AGENTS
            and len(failure) == EXPECTED_FAILURE_AGENTS
        ),
        "only_candidate_group": bool(
            len(records) and records["agent_index"].min() >= harness.GROUP_SIZE
        ),
        "only_phase6_9": bool(
            phases[PHASE_MIN:PHASE_MAX_EXCLUSIVE].sum() == EXPECTED_RECORDS
            and phases[:PHASE_MIN].sum() == 0
            and phases[PHASE_MAX_EXCLUSIVE:].sum() == 0
        ),
        "finite_in_envelope_features": finite and envelope_violations == 0,
        "training_only_authority": (
            harness_report.get("safety", {}).get("flight_sim_packets_sent") == 0
            and not harness_report.get("safety", {}).get("submission_authorized")
        ),
    }
    admitted = all(predicates.values())
    harness_report_path = harness_output / "report.json"
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "training_dataset_admitted": admitted,
        "admission_predicates": predicates,
        "failed_admission_predicates": [
            name for name, passed in predicates.items() if not passed
        ],
        "feature_schema": FEATURE_SCHEMA,
        "feature_dtype": FEATURE_DTYPE.descr,
        "feature_itemsize": FEATURE_DTYPE.itemsize,
        "feature_records": len(records), "feature_sha256": feature_sha,
        "feature_phase_records": phases.tolist(),
        "query_agents": len(np.unique(records["agent_index"])),
        "query_outcome_success_agents": success.tolist(),
        "query_outcome_failure_agents": failure.tolist(),
        "teacher_action_envelope_violations": envelope_violations,
        "harness_report_sha256": sha256_path(harness_report_path),
        "harness_items": harness_report["items"],
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": source_identity(harness_report_path),
        "safety": {
            "runtime_teacher_actions": 0,
            "offline_training_teacher_actions": len(records),
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Run one source-locked whole-Puffer phase-6--9 distillation; no FlightSim authority."
            if admitted else
            "Reject LC119 and diagnose the failed exact milestone predicates."
        ),
    }
    write_json_once(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_dataset_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
