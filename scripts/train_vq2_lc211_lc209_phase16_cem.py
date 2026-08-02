#!/usr/bin/env python3
"""CEM-search a phase-16 Puffer residual around the LC209 adapter."""

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
import scripts.train_vq2_lc189_phase16_adapter_split_batch_cem as base


TAG = "vq2_lc211_lc209_phase16_cem_001"
SCHEMA = "vq2_lc211_lc209_phase16_cem_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc211_lc209_phase16_cem_checkpoint_v1"
TARGET_PHASE = 16
TARGET_RAW_INDEX = 18
MAX_STEPS = 33_000
GENERATIONS = 3
FROZEN_STATE_FIELD = "frozen_non_phase16_state_exact"
DELTA_FIELD = "phase16_pre_tanh_output_bias_delta"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc209_stacked_adapter_onpolicy_dagger2_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "50654f1f1e16e919fd74c35e5841f36986d759d36ff101eaf70ddc5843fcdea4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "ff8801c14a3e4620a056723bf9ca4566d61b059a4c8e7262d5a1accc666fd41b"
LC210_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc210_stacked_adapter_onpolicy2_milestone_001/report.json"
LC210_REPORT_SHA256 = "3096f9bac8406bb8432118616fc2cc24b160015b0851713d8036a78829c4b270"
PREREGISTRATION = ROOT / "docs/vq2_lc211_lc209_phase16_cem_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc211_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc211_lc209_phase16_cem.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC210_REPORT: LC210_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC211 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC210_REPORT.read_text())
    baseline, candidate = rejected.get("items", [{}, {}])
    if (
        parent.get("schema")
        != "vq2_lc209_stacked_adapter_onpolicy_dagger2_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent.get("model", {}).get("continuation_phase_min") != 16
        or parent.get("model", {}).get("continuation_phase_max_exclusive") != 18
        or parent_report.get("schema")
        != "vq2_lc209_stacked_adapter_onpolicy_dagger2_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or not parent_report.get("frozen_lc189_state_exact")
        or rejected.get("schema")
        != "vq2_lc210_stacked_adapter_onpolicy2_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or baseline.get("maximum_raw_index_distribution", {}).get("17") != 128
        or candidate.get("maximum_raw_index_distribution", {}).get("16") != 128
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC209/LC210 do not authorize LC211")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC210_REPORT,
        ROOT / "scripts/train_vq2_lc189_phase16_adapter_split_batch_cem.py",
        ROOT / "scripts/train_vq2_lc140_phase9_repeated_seed_cem.py",
        ROOT / "scripts/eval_vq2_lc210_stacked_adapter_onpolicy2_milestone.py",
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
            "two independent complete recurrent LC209 stacked-adapter Puffers "
            "over 256 rows each"
        )
        corrected["actor_batches"] = 2
        corrected["actor_batch_size"] = 256
        corrected["optimization"] = (
            "three generations of 512 complete recurrent LC209 Puffer candidates; "
            "one constant legal phase-16 pre-tanh residual per trajectory"
        )
        corrected["next_authority"] = (
            "Run one exact-context LC189-versus-LC211 raw-18 screen; no FlightSim authority."
            if corrected.get("training_admitted")
            else "Reject the LC209 phase-16 CEM family and retain LC189."
        )
    base.BASE_WRITE_JSON_ONCE(path, corrected)


def configure_outer() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "CHECKPOINT_SCHEMA", "TARGET_PHASE",
        "TARGET_RAW_INDEX", "MAX_STEPS", "GENERATIONS", "FROZEN_STATE_FIELD",
        "DELTA_FIELD", "PARENT_DIR", "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT", "PARENT_REPORT_SHA256",
        "PREREGISTRATION", "RUNNER", "TEST", "DEFAULT_OUTPUT",
        "verify_inputs", "source_identity", "corrected_writer",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG, SCHEMA, CHECKPOINT_SCHEMA, TARGET_PHASE, TARGET_RAW_INDEX,
        MAX_STEPS, GENERATIONS, FROZEN_STATE_FIELD, DELTA_FIELD, PARENT_DIR,
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256, PARENT_REPORT,
        PARENT_REPORT_SHA256, PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
        verify_inputs, source_identity, corrected_writer,
    )
    for name, value in zip(names, values):
        setattr(base, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(base, name, value)


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_outer()
    try:
        return base.train(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


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
