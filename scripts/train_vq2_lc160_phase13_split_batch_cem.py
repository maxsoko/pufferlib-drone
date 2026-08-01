#!/usr/bin/env python3
"""Search LC158's phase-13 Puffer residual on 512 exact seed-15 copies."""

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
import scripts.train_vq2_lc143_phase10_split_batch_cem as prior


TAG = "vq2_lc160_phase13_split_batch_cem_001"
SCHEMA = "vq2_lc160_phase13_split_batch_cem_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc160_phase13_split_batch_cem_checkpoint_v1"
TARGET_PHASE = 13
TARGET_RAW_INDEX = 14
MAX_STEPS = 21_000
GENERATIONS = 2
FROZEN_STATE_FIELD = "frozen_non_phase13_state_exact"
DELTA_FIELD = "phase13_pre_tanh_output_bias_delta"
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc158_phase12_split_batch_cem_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "fe8dd2ec4d88dc1777d212b4f4917ff3be1ae894ae35c41624d353c51289dcad"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "3312802858cc9eb77c84354a23140e2193830cb0d50bca01918fab419a0fd329"
LC159_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc159_phase12_cem_milestone_001/report.json"
)
LC159_REPORT_SHA256 = "c39cb4c2fbba64342bb7393d7135c416ed236c617f21775fbca7c968fd6052c7"
PREREGISTRATION = ROOT / "docs/vq2_lc160_phase13_split_batch_cem_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc160_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc160_phase13_split_batch_cem.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC159_REPORT: LC159_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC160 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    milestone = json.loads(LC159_REPORT.read_text())
    selected = milestone.get("causal_screen_selected", {})
    if (
        parent.get("schema") != "vq2_lc158_phase12_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent.get("frozen_non_phase12_state_exact")
        or parent_report.get("schema") != "vq2_lc158_phase12_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or parent_report.get("candidate_selected_for_screen", {}).get("sha256")
        != PARENT_CHECKPOINT_SHA256
        or milestone.get("schema") != "vq2_lc159_phase12_cem_milestone_report_v1"
        or not milestone.get("diagnostic_valid")
        or selected.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or selected.get("target_passes") != 128
        or selected.get("paired_target_gains_vs_baseline") != 128
        or selected.get("paired_target_losses_vs_baseline") != 0
        or milestone.get("safety", {}).get("flight_sim_packets_sent") != 0
        or milestone.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC158/LC159 do not authorize LC160")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC159_REPORT,
        ROOT / "scripts/train_vq2_lc140_phase9_repeated_seed_cem.py",
        ROOT / "scripts/train_vq2_lc143_phase10_split_batch_cem.py",
        ROOT / "scripts/train_vq2_lc158_phase12_split_batch_cem.py",
        ROOT / "scripts/eval_vq2_lc159_phase12_cem_milestone.py",
        ROOT / "scripts/collect_vq2_lc129_phase8_9_expanded_rescue_corpus.py",
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
            "two independent complete recurrent LC158 Puffer actors over 256 rows each"
        )
        corrected["actor_batches"] = 2
        corrected["actor_batch_size"] = prior.ACTOR_BATCH_SIZE
        corrected["optimization"] = (
            "512 complete recurrent LC158 Puffer candidates; one constant legal phase-13 pre-tanh residual per trajectory"
        )
        corrected["next_authority"] = (
            "Run one deterministic LC158-versus-LC160 raw-14 screen; no FlightSim authority."
            if corrected.get("training_admitted") else
            "Reject the phase-13 constant-bias family and collect a state-dependent DAgger rescue; do not run FlightSim."
        )
    prior.BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    outer = (
        prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA,
        prior.TARGET_PHASE, prior.TARGET_RAW_INDEX, prior.MAX_STEPS,
        prior.FROZEN_STATE_FIELD, prior.DELTA_FIELD,
        prior.PARENT_DIR, prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.verify_inputs, prior.source_identity, prior.corrected_writer,
    )
    prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    prior.TARGET_PHASE, prior.TARGET_RAW_INDEX, prior.MAX_STEPS = (
        TARGET_PHASE, TARGET_RAW_INDEX, MAX_STEPS,
    )
    prior.FROZEN_STATE_FIELD, prior.DELTA_FIELD = FROZEN_STATE_FIELD, DELTA_FIELD
    prior.PARENT_DIR, prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256 = (
        PARENT_DIR, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    prior.verify_inputs, prior.source_identity, prior.corrected_writer = (
        verify_inputs, source_identity, corrected_writer,
    )
    inner = prior.configure()
    return outer, inner


def restore(originals: tuple[tuple[Any, ...], tuple[Any, ...]]) -> None:
    outer, inner = originals
    prior.restore(inner)
    (
        prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA,
        prior.TARGET_PHASE, prior.TARGET_RAW_INDEX, prior.MAX_STEPS,
        prior.FROZEN_STATE_FIELD, prior.DELTA_FIELD,
        prior.PARENT_DIR, prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.verify_inputs, prior.source_identity, prior.corrected_writer,
    ) = outer


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    generation_original = prior.base.GENERATIONS
    try:
        prior.base.GENERATIONS = GENERATIONS
        prior.base.train(output=output, device_name=device_name, resume=resume)
        return json.loads((output / "report.json").read_text())
    finally:
        prior.base.GENERATIONS = generation_original
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
