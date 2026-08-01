#!/usr/bin/env python3
"""Search LC187's phase-16 residual while preserving its recurrent adapter."""

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

from scripts.collect_vq2_lc129_phase8_9_expanded_rescue_corpus import SplitBatchActor
from scripts.eval_vq2_lc178_phase15_recurrent_adapter_milestone import load_actor
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_lc140_phase9_repeated_seed_cem as base


BASE_WRITE_JSON_ONCE = base.write_json_once
TAG = "vq2_lc189_phase16_adapter_split_batch_cem_001"
SCHEMA = "vq2_lc189_phase16_adapter_split_batch_cem_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
TARGET_PHASE = 16
TARGET_RAW_INDEX = 17
MAX_STEPS = 30_000
GENERATIONS = 2
FROZEN_STATE_FIELD = "frozen_non_phase16_state_exact"
DELTA_FIELD = "phase16_pre_tanh_output_bias_delta"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "a51d2a065edcc1154d6262a87146183ab786a9111150274c476762724ac6e196"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "4269220c21f3fafc1b7ef72275e44a5f2e2757f901734bfd28b34d52ac80fa55"
LC188_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc188_phase15_adapter_onpolicy_dagger2_milestone_001/report.json"
LC188_REPORT_SHA256 = "7fceb50ffcaea3532cc894c93521efc62022c8d4fd26ab6e496166dcbacf9c8e"
PREREGISTRATION = ROOT / "docs/vq2_lc189_phase16_adapter_split_batch_cem_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc189_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc189_phase16_adapter_split_batch_cem.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC188_REPORT: LC188_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC189 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    milestone = json.loads(LC188_REPORT.read_text())
    selected = milestone.get("causal_screen_selected", {})
    if (
        parent.get("schema") != "vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent.get("model", {}).get("adapter_target_phase") != 15
        or parent_report.get("schema") != "vq2_lc187_phase15_adapter_onpolicy_dagger2_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or milestone.get("schema") != "vq2_lc188_phase15_adapter_onpolicy_dagger2_milestone_report_v1"
        or not milestone.get("diagnostic_valid")
        or selected.get("candidate_state_sha256") != parent_report.get("candidate_state_sha256")
        or selected.get("target_passes") != 128
        or selected.get("paired_target_gains_vs_baseline") != 128
        or selected.get("paired_target_losses_vs_baseline") != 0
        or milestone.get("safety", {}).get("flight_sim_packets_sent") != 0
        or milestone.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC187/LC188 do not authorize LC189")
    return parent


def split_adapter_loader(payload: dict[str, Any], device: torch.device) -> SplitBatchActor:
    return SplitBatchActor((load_actor(payload, device), load_actor(payload, device)))


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC188_REPORT,
        ROOT / "scripts/train_vq2_lc140_phase9_repeated_seed_cem.py",
        ROOT / "scripts/eval_vq2_lc188_phase15_adapter_onpolicy_dagger2_milestone.py",
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
        corrected["actor_execution"] = "two independent complete recurrent LC187 adapter Puffers over 256 rows each"
        corrected["actor_batches"] = 2
        corrected["actor_batch_size"] = 256
        corrected["optimization"] = "512 complete recurrent LC187 adapter Puffer candidates; one constant legal phase-16 pre-tanh residual per trajectory"
        corrected["next_authority"] = (
            "Run one deterministic LC187-versus-LC189 raw-17 screen; no FlightSim authority."
            if corrected.get("training_admitted") else
            "Reject the phase-16 constant-bias family and start adapter-owned DAgger; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.MAX_STEPS, base.GENERATIONS,
        base.FROZEN_STATE_FIELD, base.DELTA_FIELD,
        base.PARENT_DIR, base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.MAX_STEPS = TARGET_PHASE, TARGET_RAW_INDEX, MAX_STEPS
    base.GENERATIONS = GENERATIONS
    base.FROZEN_STATE_FIELD, base.DELTA_FIELD = FROZEN_STATE_FIELD, DELTA_FIELD
    base.PARENT_DIR, base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = PARENT_DIR, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    base.verify_inputs, base.source_identity, base.write_json_once = verify_inputs, source_identity, corrected_writer
    base.milestone.load_actor = split_adapter_loader
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA,
        base.TARGET_PHASE, base.TARGET_RAW_INDEX, base.MAX_STEPS, base.GENERATIONS,
        base.FROZEN_STATE_FIELD, base.DELTA_FIELD,
        base.PARENT_DIR, base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor,
    ) = originals


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        return base.train(output=output, device_name=device_name, resume=resume)
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
