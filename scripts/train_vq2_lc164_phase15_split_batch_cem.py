#!/usr/bin/env python3
"""Search LC162's phase-15 Puffer residual on 512 exact seed-15 copies."""

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
import scripts.train_vq2_lc162_phase14_split_batch_cem as previous


TAG = "vq2_lc164_phase15_split_batch_cem_001"
SCHEMA = "vq2_lc164_phase15_split_batch_cem_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc164_phase15_split_batch_cem_checkpoint_v1"
TARGET_PHASE = 15
TARGET_RAW_INDEX = 16
MAX_STEPS = 27_000
GENERATIONS = 2
FROZEN_STATE_FIELD = "frozen_non_phase15_state_exact"
DELTA_FIELD = "phase15_pre_tanh_output_bias_delta"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc162_phase14_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "4a45d1814e2f76b776ed2a278bd9a026dab5c1caf346792bc26032eea874770f"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "a85ae9c0da3c52110891d03cf67eafeebf76a3123b62a210d3740e541d32fabe"
LC163_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc163_phase14_cem_milestone_001/report.json"
LC163_REPORT_SHA256 = "e944f3fbce50f42bf7a3fb41a917ee287858e2564c0444d582e82313d6b26ac0"
PREREGISTRATION = ROOT / "docs/vq2_lc164_phase15_split_batch_cem_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc164_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc164_phase15_split_batch_cem.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC163_REPORT: LC163_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC164 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    milestone = json.loads(LC163_REPORT.read_text())
    selected = milestone.get("causal_screen_selected", {})
    if (
        parent.get("schema") != "vq2_lc162_phase14_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent.get("frozen_non_phase14_state_exact")
        or parent_report.get("schema") != "vq2_lc162_phase14_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or parent_report.get("candidate_selected_for_screen", {}).get("sha256")
        != PARENT_CHECKPOINT_SHA256
        or milestone.get("schema") != "vq2_lc163_phase14_cem_milestone_report_v1"
        or not milestone.get("diagnostic_valid")
        or selected.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or selected.get("target_passes") != 128
        or selected.get("paired_target_gains_vs_baseline") != 128
        or selected.get("paired_target_losses_vs_baseline") != 0
        or milestone.get("safety", {}).get("flight_sim_packets_sent") != 0
        or milestone.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC162/LC163 do not authorize LC164")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC163_REPORT,
        ROOT / "scripts/train_vq2_lc140_phase9_repeated_seed_cem.py",
        ROOT / "scripts/train_vq2_lc143_phase10_split_batch_cem.py",
        ROOT / "scripts/train_vq2_lc162_phase14_split_batch_cem.py",
        ROOT / "scripts/eval_vq2_lc163_phase14_cem_milestone.py",
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
        corrected["actor_execution"] = (
            "two independent complete recurrent LC162 Puffer actors over 256 rows each"
        )
        corrected["actor_batches"] = 2
        corrected["actor_batch_size"] = previous.previous.prior.ACTOR_BATCH_SIZE
        corrected["optimization"] = (
            "512 complete recurrent LC162 Puffer candidates; one constant legal phase-15 pre-tanh residual per trajectory"
        )
        corrected["next_authority"] = (
            "Run one deterministic LC162-versus-LC164 raw-16 screen; no FlightSim authority."
            if corrected.get("training_admitted") else
            "Reject the phase-15 constant-bias family and collect a state-dependent DAgger rescue; do not run FlightSim."
        )
    previous.previous.prior.BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, Any]:
    outer = (
        previous.TAG, previous.SCHEMA, previous.CHECKPOINT_SCHEMA,
        previous.TARGET_PHASE, previous.TARGET_RAW_INDEX, previous.MAX_STEPS,
        previous.FROZEN_STATE_FIELD, previous.DELTA_FIELD,
        previous.PARENT_DIR, previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256,
        previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256,
        previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT,
        previous.verify_inputs, previous.source_identity, previous.corrected_writer,
    )
    previous.TAG, previous.SCHEMA, previous.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    previous.TARGET_PHASE, previous.TARGET_RAW_INDEX, previous.MAX_STEPS = (
        TARGET_PHASE, TARGET_RAW_INDEX, MAX_STEPS,
    )
    previous.FROZEN_STATE_FIELD, previous.DELTA_FIELD = FROZEN_STATE_FIELD, DELTA_FIELD
    previous.PARENT_DIR, previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256 = (
        PARENT_DIR, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    previous.verify_inputs, previous.source_identity, previous.corrected_writer = (
        verify_inputs, source_identity, corrected_writer,
    )
    return outer, previous.configure()


def restore(originals: tuple[Any, Any]) -> None:
    outer, inner = originals
    previous.restore(inner)
    (
        previous.TAG, previous.SCHEMA, previous.CHECKPOINT_SCHEMA,
        previous.TARGET_PHASE, previous.TARGET_RAW_INDEX, previous.MAX_STEPS,
        previous.FROZEN_STATE_FIELD, previous.DELTA_FIELD,
        previous.PARENT_DIR, previous.PARENT_CHECKPOINT, previous.PARENT_CHECKPOINT_SHA256,
        previous.PARENT_REPORT, previous.PARENT_REPORT_SHA256,
        previous.PREREGISTRATION, previous.RUNNER, previous.TEST, previous.DEFAULT_OUTPUT,
        previous.verify_inputs, previous.source_identity, previous.corrected_writer,
    ) = outer


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    engine = previous.previous.prior.base
    generation_original = engine.GENERATIONS
    try:
        engine.GENERATIONS = GENERATIONS
        engine.train(output=output, device_name=device_name, resume=resume)
        return json.loads((output / "report.json").read_text())
    finally:
        engine.GENERATIONS = generation_original
        restore(originals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
