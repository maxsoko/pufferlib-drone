#!/usr/bin/env python3
"""Search LC143's phase-11 Puffer residual on 512 exact seed-15 copies."""

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
import scripts.train_vq2_lc140_phase9_repeated_seed_cem as base


BASE_WRITE_JSON_ONCE = base.write_json_once
BASE_ACTOR_LOADER = base.milestone.load_actor

TAG = "vq2_lc145_phase11_split_batch_cem_001"
SCHEMA = "vq2_lc145_phase11_split_batch_cem_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc145_phase11_split_batch_cem_checkpoint_v1"
TARGET_PHASE = 11
TARGET_RAW_INDEX = 12
MAX_STEPS = 17_000
ACTOR_BATCH_SIZE = 256
FROZEN_STATE_FIELD = "frozen_non_phase11_state_exact"
DELTA_FIELD = "phase11_pre_tanh_output_bias_delta"
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc143_phase10_split_batch_cem_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "c4fe8fc5e069ea7696f52a81a9dd55a702957b92ad73285d55c06c6783e46ea6"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "8ad28f6c61589e9430e2813a6b7c7b37538cdd885c12c2d0778d0dbf09800bd9"
LC141_CHECKPOINT_SHA256 = "33d8b837fe6c94c522179191e0c869e7b24a4839b96b96bfed5f5d09b0b057da"
LC144_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc144_phase10_cem_milestone_001/report.json"
)
LC144_REPORT_SHA256 = "e418dae3cdddfd34759cc75f6f2e239666ca3aae59a02d227f8e73fb9bfa6686"
PREREGISTRATION = ROOT / "docs/vq2_lc145_phase11_split_batch_cem_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc145_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc145_phase11_split_batch_cem.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC144_REPORT: LC144_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC145 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    milestone = json.loads(LC144_REPORT.read_text())
    selected = milestone.get("causal_screen_selected", {})
    if (
        parent.get("schema") != "vq2_lc143_phase10_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent.get("frozen_non_phase10_state_exact")
        or parent.get("parent_checkpoint_sha256") != LC141_CHECKPOINT_SHA256
        or parent_report.get("schema") != "vq2_lc143_phase10_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or parent_report.get("candidate_selected_for_screen", {}).get("sha256")
        != PARENT_CHECKPOINT_SHA256
        or milestone.get("schema") != "vq2_lc144_phase10_cem_milestone_report_v1"
        or not milestone.get("diagnostic_valid")
        or selected.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or selected.get("target_passes") != 1
        or selected.get("paired_target_gains_vs_baseline") != 1
        or selected.get("paired_target_losses_vs_baseline") != 0
        or milestone.get("safety", {}).get("flight_sim_packets_sent") != 0
        or milestone.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC143/LC144 do not authorize LC145")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC144_REPORT,
        ROOT / "scripts/train_vq2_lc140_phase9_repeated_seed_cem.py",
        ROOT / "scripts/train_vq2_lc143_phase10_split_batch_cem.py",
        ROOT / "scripts/eval_vq2_lc144_phase10_cem_milestone.py",
        ROOT / "scripts/collect_vq2_lc129_phase8_9_expanded_rescue_corpus.py",
        ROOT / "pufferlib/vq2_informed.py",
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
        corrected["actor_execution"] = (
            "two independent complete recurrent LC143 Puffer actors over 256 rows each"
        )
        corrected["actor_batches"] = 2
        corrected["actor_batch_size"] = ACTOR_BATCH_SIZE
        corrected["optimization"] = (
            "512 complete recurrent LC143 Puffer candidates; one constant legal phase-11 pre-tanh residual per trajectory"
        )
        corrected["next_authority"] = (
            "Run one deterministic LC143-versus-LC145 raw-12 screen; no FlightSim authority."
            if corrected.get("training_admitted") else
            "Reject LC145 and retain LC105 as the safe frontier; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.MAX_STEPS,
        base.FROZEN_STATE_FIELD, base.DELTA_FIELD,
        base.PARENT_DIR, base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.MAX_STEPS = TARGET_PHASE, TARGET_RAW_INDEX, MAX_STEPS
    base.FROZEN_STATE_FIELD, base.DELTA_FIELD = FROZEN_STATE_FIELD, DELTA_FIELD
    base.PARENT_DIR, base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_DIR, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
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
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.MAX_STEPS,
        base.FROZEN_STATE_FIELD, base.DELTA_FIELD,
        base.PARENT_DIR, base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor,
    ) = originals


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        base.train(output=output, device_name=device_name, resume=resume)
        return json.loads((output / "report.json").read_text())
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
    return 0 if report["training_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
