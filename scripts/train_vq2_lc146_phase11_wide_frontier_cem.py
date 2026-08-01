#!/usr/bin/env python3
"""Widen phase-11 CEM around LC145's measured constant-bias frontier."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_lc140_phase9_repeated_seed_cem as base
import scripts.train_vq2_lc145_phase11_split_batch_cem as prior


BASE_WRITE_JSON_ONCE = base.write_json_once

TAG = "vq2_lc146_phase11_wide_frontier_cem_001"
SCHEMA = "vq2_lc146_phase11_wide_frontier_cem_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc146_phase11_wide_frontier_cem_checkpoint_v1"
GENERATIONS = 2
INITIAL_MEAN = np.array(
    (0.1574796438217163, 0.16625714302062988, -0.048462092876434326,
     0.02503732033073902), dtype=np.float32,
)
INITIAL_STD = np.array((0.15, 0.15, 0.10, 0.025), dtype=np.float32)
MAXIMUM_ABS_DELTA = np.array((0.60, 0.60, 0.50, 0.10), dtype=np.float32)
LC145_REPORT = prior.DEFAULT_OUTPUT / "report.json"
LC145_REPORT_SHA256 = "4aee89937e58c4c37f7a92909fe50a4ae00549f988f846a35075a1a75796dc42"
PREREGISTRATION = ROOT / "docs/vq2_lc146_phase11_wide_frontier_cem_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc146_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc146_phase11_wide_frontier_cem.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    parent = prior.verify_inputs()
    if sha256_path(LC145_REPORT) != LC145_REPORT_SHA256:
        raise RuntimeError("LC146 bound LC145 report changed")
    rejected = json.loads(LC145_REPORT.read_text())
    generations = rejected.get("generations", [])
    if (
        rejected.get("schema") != "vq2_lc145_phase11_split_batch_cem_report_v1"
        or rejected.get("training_admitted")
        or rejected.get("candidate_selected_for_screen") is not None
        or len(generations) != 4
        or any(item.get("query_agents") != 512 for item in generations)
        or any(item.get("target_passes") != 0 for item in generations)
        or any(not item.get("transport_pass") for item in generations)
        or generations[-1].get("best_delta") != INITIAL_MEAN.tolist()
        or generations[-1].get("phase_return_max", 0.0)
        <= generations[0].get("phase_return_max", 0.0)
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC145 does not authorize the wide frontier bracket")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        prior.PARENT_CHECKPOINT, prior.PARENT_REPORT, prior.LC144_REPORT,
        LC145_REPORT,
        ROOT / "scripts/train_vq2_lc140_phase9_repeated_seed_cem.py",
        ROOT / "scripts/train_vq2_lc145_phase11_split_batch_cem.py",
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
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
    }


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        corrected["actor_execution"] = (
            "two independent complete recurrent LC143 Puffer actors over 256 rows each"
        )
        corrected["search_family"] = "wide constant phase-11 output-bias frontier"
        corrected["next_authority"] = (
            "Run one deterministic LC143-versus-LC146 raw-12 screen; no FlightSim authority."
            if corrected.get("training_admitted") else
            "Reject the constant phase-11 bias family; use a state-dependent Puffer residual and do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = prior.configure()
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.GENERATIONS = GENERATIONS
    base.INITIAL_MEAN = INITIAL_MEAN
    base.INITIAL_STD = INITIAL_STD
    base.MAXIMUM_ABS_DELTA = MAXIMUM_ABS_DELTA
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.write_json_once = corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    # Restore the values captured before prior.configure(), including its base
    # execution hooks.  Search-only constants are reset explicitly.
    prior.restore(originals)
    base.GENERATIONS = 4
    base.INITIAL_MEAN = np.zeros(4, dtype=np.float32)
    base.INITIAL_STD = np.array((0.08, 0.08, 0.08, 0.01), dtype=np.float32)
    base.MAXIMUM_ABS_DELTA = np.array((0.35, 0.35, 0.35, 0.05), dtype=np.float32)


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
