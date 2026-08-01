#!/usr/bin/env python3
"""Correct LC140 to two source-matched 256-row Puffer actor batches."""

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
BASE_VERIFY_INPUTS = base.verify_inputs

TAG = "vq2_lc141_phase9_split_batch_cem_001"
SCHEMA = "vq2_lc141_phase9_split_batch_cem_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc141_phase9_split_batch_cem_checkpoint_v1"
ACTOR_BATCH_SIZE = 256
LC140_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc140_phase9_repeated_seed_cem_001/report.json"
)
LC140_REPORT_SHA256 = "c88e6198b2191d4fa39b4b18eb246ff64b0b1eb7cad74080ba86c060ef7851da"
PREREGISTRATION = ROOT / "docs/vq2_lc141_phase9_split_batch_cem_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc141_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc141_phase9_split_batch_cem.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    parent = BASE_VERIFY_INPUTS()
    if sha256_path(LC140_REPORT) != LC140_REPORT_SHA256:
        raise RuntimeError("LC141 bound LC140 report changed")
    rejected = json.loads(LC140_REPORT.read_text())
    generation = rejected.get("generations", [{}])[0]
    if (
        rejected.get("schema") != "vq2_lc140_phase9_repeated_seed_cem_report_v1"
        or rejected.get("training_admitted")
        or rejected.get("candidate_selected_for_screen") is not None
        or len(rejected.get("generations", [])) != 1
        or generation.get("initial_states_exact") is not True
        or generation.get("query_agents") != 0
        or generation.get("maximum_raw_index_distribution", {}).get("6") != 512
        or generation.get("transport_pass")
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC140 does not authorize its split-batch correction")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        base.PARENT_CHECKPOINT, base.PARENT_REPORT, base.LC139_REPORT,
        LC140_REPORT,
        ROOT / "scripts/train_vq2_lc140_phase9_repeated_seed_cem.py",
        ROOT / "scripts/collect_vq2_lc129_phase8_9_expanded_rescue_corpus.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
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
    return SplitBatchActor((
        BASE_ACTOR_LOADER(payload, device),
        BASE_ACTOR_LOADER(payload, device),
    ))


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        corrected["actor_execution"] = (
            "two independent complete recurrent LC105 Puffer actors over 256 rows each"
        )
        corrected["actor_batches"] = 2
        corrected["actor_batch_size"] = ACTOR_BATCH_SIZE
        corrected["corrected_lc140_execution_mismatch"] = (
            "LC140 used one 512-row actor and reached raw 6; LC141 restores the source-measured 256-row actor shape"
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.write_json_once = corrected_writer
    base.milestone.load_actor = split_actor_loader
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
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
