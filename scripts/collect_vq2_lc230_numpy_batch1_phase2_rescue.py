#!/usr/bin/env python3
"""Collect paired exact-batch-1 LC216 control/oracle Gate-3 features."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
from scripts.policy_callable_vq2_all24_sequence import NumpyLC216Policy
import scripts.collect_vq2_lc125_lc123_phase8_9_rescue_features as base


BASE_WRITE_JSON_ONCE = base.write_json_once
TAG = "vq2_lc230_numpy_batch1_phase2_rescue_001"
SCHEMA = "vq2_lc230_numpy_batch1_phase2_rescue_report_v1"
FEATURE_SCHEMA = "vq2_lc230_numpy_batch1_phase2_rescue_feature_v1"
GROUP_SIZE = 8
TOTAL_AGENTS = 16
THREADS = 8
SEED = 432_224
MAX_STEPS = 10_000
TARGET_RAW_INDEX = 3
PHASE_MIN = 2
PHASE_MAX_EXCLUSIVE = 3
ENV_SEED_GROUP_SIZE = GROUP_SIZE
ENV_SEED_INDEX_OFFSET = 287
NUMPY_CHECKPOINT = ROOT / "checkpoints/vq2_lc216_all24_action_sequence_numpy.npz"
NUMPY_CHECKPOINT_SHA256 = (
    "248d3574d3354862891b27942d239fb0d123dc8832e8c1e04db5fe12c0f3da1f"
)
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
LC229_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc229_numpy_batch1_native_raw3_001/report.json"
)
LC229_REPORT_SHA256 = (
    "5b04510999480f2bae03d332377aaf30fb76e2d24902165be3950ab08b50dddd"
)
CALLABLE = ROOT / "scripts/policy_callable_vq2_all24_sequence.py"
CALLABLE_SHA256 = (
    "ce64fc1422fbe29a40c239ab921c023e73cac008fa9511ee0dbc7c9e6aa73773"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc230_numpy_batch1_phase2_rescue_preregistration_2026-08-02.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc230_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc230_numpy_batch1_phase2_rescue.py"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


class NumpyBatch1Actor:
    """Torch-shaped adapter that preserves independent NumPy batch-1 math."""

    def __init__(self, *, agents: int, device: torch.device) -> None:
        self.policies = [NumpyLC216Policy(NUMPY_CHECKPOINT) for _ in range(agents)]
        self.device = device
        self.agents = agents

    def initial_state(
        self, batch_size: int, *, device: torch.device | None = None,
    ) -> torch.Tensor:
        if batch_size != self.agents:
            raise ValueError("LC230 actor batch size changed")
        for policy in self.policies:
            policy.reset()
        return torch.zeros(
            (1, batch_size, 256), dtype=torch.float32,
            device=self.device if device is None else device,
        )

    def forward_step(
        self, observation: torch.Tensor, recurrent: torch.Tensor,
    ) -> tuple[Any, torch.Tensor]:
        del recurrent
        if observation.shape != (self.agents, 4119):
            raise ValueError("LC230 actor observation ABI changed")
        rows = observation.detach().cpu().numpy().astype(np.float32, copy=False)
        mean = np.empty((self.agents, 4), dtype=np.float32)
        hidden = np.empty((self.agents, 256), dtype=np.float32)
        for index, policy in enumerate(self.policies):
            mean[index] = policy(rows[index])
            hidden[index] = policy.base_hidden
        clipped = np.clip(mean, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = np.arctanh(clipped).astype(np.float32)
        mean_tensor = torch.from_numpy(mean).to(self.device)
        result = SimpleNamespace(
            mean=mean_tensor,
            pre_tanh_mean=torch.from_numpy(pre_tanh).to(self.device),
        )
        next_recurrent = torch.from_numpy(hidden[None]).to(self.device)
        return result, next_recurrent


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        NUMPY_CHECKPOINT: NUMPY_CHECKPOINT_SHA256,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC229_REPORT: LC229_REPORT_SHA256,
        CALLABLE: CALLABLE_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC230 bound input changed: {path}")
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent = json.loads(PARENT_REPORT.read_text())
    failed = json.loads(LC229_REPORT.read_text())
    if (
        payload.get("schema") != "vq2_lc216_all24_action_sequence_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or parent.get("schema") != "vq2_lc216_all24_action_sequence_report_v1"
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or failed.get("schema") != "vq2_lc229_numpy_batch1_native_report_v1"
        or not failed.get("diagnostic_valid")
        or failed.get("milestone_admitted")
        or failed.get("outcome", {}).get("target_passes") != 0
        or failed.get("outcome", {}).get("pre_target_terminals") != GROUP_SIZE
        or failed.get("outcome", {}).get("maximum_raw_index_distribution", {}).get("2")
        != GROUP_SIZE
        or failed.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("LC216/LC229 do not authorize LC230")
    return payload


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST, CALLABLE,
        NUMPY_CHECKPOINT, PARENT_CHECKPOINT, PARENT_REPORT, LC229_REPORT,
        ROOT / "pufferlib/vq2_informed.py", ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "scripts/collect_vq2_lc125_lc123_phase8_9_rescue_features.py",
        ROOT / "scripts/eval_vq2_lc007_converted_vg071_long_course.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h", ROOT / "src/vecenv.h",
        ROOT / "src/bindings.cu", ROOT / "src/bindings_cpu.cpp",
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
            "torch": torch.__version__, "numpy": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }


def load_numpy_actor(payload: dict[str, Any], device: torch.device) -> NumpyBatch1Actor:
    del payload
    return NumpyBatch1Actor(agents=TOTAL_AGENTS, device=device)


def numpy_feature_components(
    actor: NumpyBatch1Actor, result: Any, next_recurrent: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del actor
    return (
        next_recurrent[0].index_select(0, index),
        result.pre_tanh_mean.index_select(0, index),
    )


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        control, intervention = corrected["items"]
        control["name"] = "lc216_exact_numpy_batch1_control"
        intervention["name"] = "lc216_exact_numpy_batch1_phase2_oracle"
        predicates = {
            "exact_control_failure": bool(
                control["target_passes"] == 0
                and control["pre_target_terminals"] == GROUP_SIZE
                and control["maximum_raw_index_distribution"].get("2") == GROUP_SIZE
            ),
            "paired_oracle_rescue": bool(
                intervention["target_passes"] == GROUP_SIZE
                and intervention["paired_target_gains_vs_control"] == GROUP_SIZE
                and intervention["paired_target_losses_vs_control"] == 0
                and intervention["pre_target_terminals"] == 0
            ),
            "both_control_and_rescue_features": bool(
                corrected["feature_records"]
                > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase2": bool(
                corrected["feature_phase_records"][PHASE_MIN]
                == corrected["feature_records"]
            ),
            "finite_in_envelope_targets": (
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
            "sixteen independent exact NumpyLC216Policy batch-1 actors; paired "
            "control and training-only phase-2 oracle plant groups"
        )
        corrected["control_features_use_teacher_targets"] = True
        corrected["feature_hidden_contract"] = "LC216 NumPy base recurrent state, 256 values"
        corrected["next_authority"] = (
            "Fit one phase-2 whole-Puffer residual and screen it first through exact NumPy batch-1 native closure."
            if corrected["training_dataset_admitted"] else
            "Reject the phase-2 oracle target and keep FlightSim frozen."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    names = (
        "TAG", "SCHEMA", "FEATURE_SCHEMA", "GROUP_SIZE", "TOTAL_AGENTS",
        "THREADS", "SEED", "MAX_STEPS", "TARGET_RAW_INDEX", "PHASE_MIN",
        "PHASE_MAX_EXCLUSIVE", "ENV_SEED_GROUP_SIZE", "ENV_SEED_INDEX_OFFSET",
        "CAPTURE_CONTROL_FEATURES", "CAPTURE_CONTROL_TEACHER_TARGETS",
        "PARENT_CHECKPOINT", "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT",
        "PARENT_REPORT_SHA256", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "verify_inputs", "source_identity", "write_json_once",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG, SCHEMA, FEATURE_SCHEMA, GROUP_SIZE, TOTAL_AGENTS, THREADS, SEED,
        MAX_STEPS, TARGET_RAW_INDEX, PHASE_MIN, PHASE_MAX_EXCLUSIVE,
        ENV_SEED_GROUP_SIZE, ENV_SEED_INDEX_OFFSET, True, True,
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256, PARENT_REPORT,
        PARENT_REPORT_SHA256, PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
        verify_inputs, source_identity, corrected_writer,
    )
    for name, value in zip(names, values):
        setattr(base, name, value)
    actor_original = base.milestone.load_actor
    feature_original = base.feature_components
    base.milestone.load_actor = load_numpy_actor
    base.feature_components = numpy_feature_components
    return names, originals, actor_original, feature_original


def restore(snapshot: tuple[Any, ...]) -> None:
    names, originals, actor_original, feature_original = snapshot
    for name, value in zip(names, originals):
        setattr(base, name, value)
    base.milestone.load_actor = actor_original
    base.feature_components = feature_original


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure()
    try:
        base.collect(output=output, device_name=device_name, resume=resume)
        return json.loads((output / "report.json").read_text())
    finally:
        restore(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("diagnostic_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
