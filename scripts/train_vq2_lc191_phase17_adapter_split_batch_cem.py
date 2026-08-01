#!/usr/bin/env python3
"""Search LC189's phase-17 residual while preserving its recurrent adapter."""

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
import scripts.train_vq2_lc189_phase16_adapter_split_batch_cem as prior


TAG = "vq2_lc191_phase17_adapter_split_batch_cem_001"
SCHEMA = "vq2_lc191_phase17_adapter_split_batch_cem_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc191_phase17_adapter_split_batch_cem_checkpoint_v1"
TARGET_PHASE = 17
TARGET_RAW_INDEX = 18
MAX_STEPS = 33_000
GENERATIONS = 2
FROZEN_STATE_FIELD = "frozen_non_phase17_state_exact"
DELTA_FIELD = "phase17_pre_tanh_output_bias_delta"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc189_phase16_adapter_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "944be397e95d2bcc427e99d09e0b7885f1b0f1e7f2c2bacaf51a6e10398fc0b2"
LC190_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc190_phase16_adapter_cem_milestone_001/report.json"
LC190_REPORT_SHA256 = "4f08d7a87a0a7c71191a81be0b81e267cfeedf562c462d1b0e93682617ef9310"
PREREGISTRATION = ROOT / "docs/vq2_lc191_phase17_adapter_split_batch_cem_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc191_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc191_phase17_adapter_split_batch_cem.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC190_REPORT: LC190_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC191 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    milestone = json.loads(LC190_REPORT.read_text())
    selected = milestone.get("causal_screen_selected", {})
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent.get("frozen_non_phase16_state_exact")
        or parent_report.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or parent_report.get("candidate_selected_for_screen", {}).get("sha256") != PARENT_CHECKPOINT_SHA256
        or milestone.get("schema") != "vq2_lc190_phase16_adapter_cem_milestone_report_v1"
        or not milestone.get("diagnostic_valid")
        or selected.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or selected.get("target_passes") != 128
        or selected.get("paired_target_gains_vs_baseline") != 128
        or selected.get("paired_target_losses_vs_baseline") != 0
        or milestone.get("safety", {}).get("flight_sim_packets_sent") != 0
        or milestone.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC190 do not authorize LC191")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC190_REPORT,
        ROOT / "scripts/train_vq2_lc189_phase16_adapter_split_batch_cem.py",
        ROOT / "scripts/eval_vq2_lc190_phase16_adapter_cem_milestone.py",
        ROOT / "pufferlib/vq2_informed.py", ROOT / "pufferlib/vq2_public_phase.py",
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
        corrected["actor_execution"] = "two independent complete recurrent LC189 adapter Puffers over 256 rows each"
        corrected["optimization"] = "512 complete recurrent LC189 candidates; one constant legal phase-17 residual per trajectory"
        corrected["next_authority"] = (
            "Run one deterministic LC189-versus-LC191 raw-18 screen; no FlightSim authority."
            if corrected.get("training_admitted") else
            "Reject phase-17 constant bias and start adapter-owned DAgger; do not run FlightSim."
        )
    prior.BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA,
        prior.TARGET_PHASE, prior.TARGET_RAW_INDEX, prior.MAX_STEPS, prior.GENERATIONS,
        prior.FROZEN_STATE_FIELD, prior.DELTA_FIELD,
        prior.PARENT_DIR, prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.verify_inputs, prior.source_identity, prior.corrected_writer,
    )
    prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    prior.TARGET_PHASE, prior.TARGET_RAW_INDEX, prior.MAX_STEPS = TARGET_PHASE, TARGET_RAW_INDEX, MAX_STEPS
    prior.GENERATIONS = GENERATIONS
    prior.FROZEN_STATE_FIELD, prior.DELTA_FIELD = FROZEN_STATE_FIELD, DELTA_FIELD
    prior.PARENT_DIR, prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256 = PARENT_DIR, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    prior.verify_inputs, prior.source_identity, prior.corrected_writer = verify_inputs, source_identity, corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA,
        prior.TARGET_PHASE, prior.TARGET_RAW_INDEX, prior.MAX_STEPS, prior.GENERATIONS,
        prior.FROZEN_STATE_FIELD, prior.DELTA_FIELD,
        prior.PARENT_DIR, prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.verify_inputs, prior.source_identity, prior.corrected_writer,
    ) = originals


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        return prior.train(output=output, device_name=device_name, resume=resume)
    finally:
        restore(originals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("training_admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
