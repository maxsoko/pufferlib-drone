#!/usr/bin/env python3
"""Expand LC125 across unseen seeds while preserving 256-row actor numerics."""

from __future__ import annotations

import argparse
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

from pufferlib.vq2_recurrent import RecurrentActorOutput
from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import outcome_agents
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.collect_vq2_lc125_lc123_phase8_9_rescue_features as base


TAG = "vq2_lc129_phase8_9_expanded_rescue_corpus_001"
SCHEMA = "vq2_lc129_phase8_9_expanded_rescue_corpus_report_v1"
CHILD_SCHEMA = "vq2_lc129_expanded_rescue_child_v1"
GROUP_SIZE = 256
TOTAL_AGENTS = 512
ACTOR_BATCH_SIZE = 256
NEW_RELATIVE_SEED_MIN = 128
ORIGINAL_DIR = base.DEFAULT_OUTPUT
ORIGINAL_REPORT = ORIGINAL_DIR / "report.json"
ORIGINAL_REPORT_SHA256 = "0021aa7d0a91825dd24ce1fe36ef75234170e5cf0a8cc2eb850a4854888fe3e4"
ORIGINAL_FEATURES = ORIGINAL_DIR / "features.bin"
ORIGINAL_FEATURES_SHA256 = "d505e3977e795aa81585bee69e1b9bad49d463f49c94fc9f2c7f07c2017444b9"
LC128_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc128_phase6_onpolicy_ppo_milestone_001/report.json"
)
LC128_REPORT_SHA256 = "e572752909c5893fa41c29c4182ba88c5742a17d77c1b006c49e3111f2c0eb8f"
PREREGISTRATION = ROOT / "docs/vq2_lc129_phase8_9_expanded_rescue_corpus_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc129_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc129_phase8_9_expanded_rescue_corpus.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


class SplitBatchActor:
    """Execute each paired environment group in its own 256-row actor."""

    def __init__(self, actors: tuple[Any, Any]) -> None:
        self.actors = actors

    def initial_state(
        self, batch_size: int, *, device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if batch_size != TOTAL_AGENTS:
            raise ValueError("LC129 split actor requires exactly 512 rows")
        states = [
            actor.initial_state(ACTOR_BATCH_SIZE, device=device, dtype=dtype)
            for actor in self.actors
        ]
        return torch.cat(states, dim=1)

    def forward_step(
        self, observation: torch.Tensor, state: torch.Tensor
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.shape[0] != TOTAL_AGENTS or state.shape[1] != TOTAL_AGENTS:
            raise ValueError("LC129 split actor input/state batch changed")
        outputs: list[RecurrentActorOutput] = []
        next_states: list[torch.Tensor] = []
        for group, actor in enumerate(self.actors):
            selected = slice(group * ACTOR_BATCH_SIZE, (group + 1) * ACTOR_BATCH_SIZE)
            output, next_state = actor.forward_step(
                observation[selected], state[:, selected]
            )
            outputs.append(output)
            next_states.append(next_state)
        return (
            RecurrentActorOutput(
                mean=torch.cat([output.mean for output in outputs], dim=0),
                pre_tanh_mean=torch.cat(
                    [output.pre_tanh_mean for output in outputs], dim=0
                ),
                log_std=torch.cat([output.log_std for output in outputs], dim=0),
            ),
            torch.cat(next_states, dim=1),
        )


def verify_inputs() -> dict[str, Any]:
    payload = base.verify_inputs()
    for path, digest in {
        ORIGINAL_REPORT: ORIGINAL_REPORT_SHA256,
        ORIGINAL_FEATURES: ORIGINAL_FEATURES_SHA256,
        LC128_REPORT: LC128_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC129 bound input changed: {path}")
    original = json.loads(ORIGINAL_REPORT.read_text())
    rejected = json.loads(LC128_REPORT.read_text())
    control, candidate = rejected.get("items", [{}, {}])
    if (
        original.get("schema")
        != "vq2_lc125_lc123_phase8_9_rescue_features_report_v1"
        or not original.get("diagnostic_valid")
        or not original.get("training_oracle_rescued_phase8_9")
        or original.get("training_dataset_admitted")
        or original.get("failed_admission_predicates") != ["splittable_outcomes"]
        or original.get("feature_records") != 6_150
        or len(original.get("query_outcome_success_agents", [])) != 1
        or len(original.get("query_outcome_failure_agents", [])) != 2
        or rejected.get("schema")
        != "vq2_lc128_phase6_onpolicy_ppo_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or control.get("maximum_raw_index_distribution", {}).get("9") != 1
        or candidate.get("maximum_raw_index_distribution", {}).get("9") != 0
        or candidate.get("maximum_raw_index_distribution", {}).get("8") != 0
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC125/LC128 do not authorize LC129")
    return payload


def expanded_source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        base.PARENT_CHECKPOINT, base.PARENT_REPORT,
        ORIGINAL_REPORT, ORIGINAL_FEATURES, LC128_REPORT,
        ROOT / "scripts/collect_vq2_lc125_lc123_phase8_9_rescue_features.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_oracle.py",
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


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    report_path = output / "report.json"
    feature_path = output / "features.bin"
    child_output = output / "expanded_child"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC129 does not resume a partial expanded collection")
    output.mkdir(parents=True, exist_ok=True)

    original_loader = base.milestone.load_actor
    original_writer = base.write_json_once
    original_values = (
        base.TAG, base.SCHEMA, base.GROUP_SIZE, base.TOTAL_AGENTS,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.source_identity,
    )

    def split_loader(payload: dict[str, Any], device: torch.device) -> SplitBatchActor:
        return SplitBatchActor((
            original_loader(payload, device),
            original_loader(payload, device),
        ))

    def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
        corrected = dict(payload)
        if path == child_output / "report.json":
            corrected["actor_execution"] = (
                "two complete saved-form LC123 Puffer actors; one independent 256-row execution per paired group"
            )
            corrected["actor_batch_size"] = ACTOR_BATCH_SIZE
            corrected["actor_batches"] = 2
        original_writer(path, corrected)

    base.TAG, base.SCHEMA = f"{TAG}_expanded_child", CHILD_SCHEMA
    base.GROUP_SIZE, base.TOTAL_AGENTS = GROUP_SIZE, TOTAL_AGENTS
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = child_output
    base.source_identity = expanded_source_identity
    base.milestone.load_actor = split_loader
    base.write_json_once = corrected_writer
    started = time.perf_counter()
    try:
        base.collect(
            output=child_output, device_name=device_name, resume=False
        )
    finally:
        base.milestone.load_actor = original_loader
        base.write_json_once = original_writer
        (
            base.TAG, base.SCHEMA, base.GROUP_SIZE, base.TOTAL_AGENTS,
            base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
            base.source_identity,
        ) = original_values

    child_report_path = child_output / "report.json"
    child_feature_path = child_output / "features.bin"
    child = json.loads(child_report_path.read_text())
    original_records = np.array(
        np.memmap(ORIGINAL_FEATURES, dtype=base.FEATURE_DTYPE, mode="r"), copy=True
    )
    child_records = np.memmap(
        child_feature_path, dtype=base.FEATURE_DTYPE, mode="r"
    )
    new_agent_min = GROUP_SIZE + NEW_RELATIVE_SEED_MIN
    new_records = np.array(
        child_records[child_records["agent_index"] >= new_agent_min], copy=True
    )
    combined = np.concatenate((original_records, new_records))
    feature_tmp = output / ".features.bin.tmp"
    combined.tofile(feature_tmp)
    with feature_tmp.open("rb") as stream:
        os.fsync(stream.fileno())
    feature_tmp.replace(feature_path)
    feature_sha = sha256_path(feature_path)
    success_agents, failure_agents = outcome_agents(combined)
    new_success_agents, new_failure_agents = outcome_agents(new_records)
    phase_counts = np.bincount(
        combined["phase_index"].astype(np.int64), minlength=33
    )
    finite = bool(
        np.isfinite(combined["hidden"]).all()
        and np.isfinite(combined["base_pre_tanh"]).all()
        and np.isfinite(combined["teacher_action"]).all()
    )
    envelope_violations = int(
        (np.abs(combined["teacher_action"]) > 1.0 + 1e-6).any(axis=1).sum()
    )
    control, intervention = child.get("items", [{}, {}])
    predicates = {
        "expanded_diagnostic_valid": bool(
            child.get("diagnostic_valid")
            and child.get("training_oracle_rescued_phase8_9")
            and intervention.get("target_passes", 0) >= control.get("target_passes", 0) + 1
            and intervention.get("paired_target_losses_vs_control") == 0
        ),
        "new_seed_records_present": len(new_records) > 0,
        "new_seed_success_present": len(new_success_agents) >= 1,
        "combined_splittable_outcomes": (
            len(success_agents) >= 2 and len(failure_agents) >= 2
        ),
        "only_unique_seed_ranges": bool(
            len(original_records)
            and original_records["agent_index"].max() < new_agent_min
            and (not len(new_records) or new_records["agent_index"].min() >= new_agent_min)
        ),
        "only_phase8_9": bool(
            phase_counts[base.PHASE_MIN:base.PHASE_MAX_EXCLUSIVE].sum() == len(combined)
            and phase_counts[:base.PHASE_MIN].sum() == 0
            and phase_counts[base.PHASE_MAX_EXCLUSIVE:].sum() == 0
        ),
        "finite_in_envelope_features": finite and envelope_violations == 0,
        "training_only_authority": (
            child.get("safety", {}).get("flight_sim_packets_sent") == 0
            and not child.get("safety", {}).get("submission_authorized")
        ),
    }
    admitted = all(predicates.values())
    identity = expanded_source_identity()
    identity["child_report_sha256"] = sha256_path(child_report_path)
    identity["child_features_sha256"] = sha256_path(child_feature_path)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "training_dataset_admitted": admitted,
        "admission_predicates": predicates,
        "failed_admission_predicates": [
            name for name, passed in predicates.items() if not passed
        ],
        "feature_schema": "vq2_lc129_combined_phase8_9_rescue_feature_v1",
        "feature_dtype": base.FEATURE_DTYPE.descr,
        "feature_itemsize": base.FEATURE_DTYPE.itemsize,
        "feature_records": len(combined), "feature_sha256": feature_sha,
        "original_feature_records": len(original_records),
        "new_feature_records": len(new_records),
        "feature_phase_records": phase_counts.tolist(),
        "query_agents": int(np.unique(combined["agent_index"]).size),
        "query_outcome_success_agents": success_agents.tolist(),
        "query_outcome_failure_agents": failure_agents.tolist(),
        "new_query_outcome_success_agents": new_success_agents.tolist(),
        "new_query_outcome_failure_agents": new_failure_agents.tolist(),
        "expanded_child_items": child.get("items"),
        "expanded_child_report_sha256": sha256_path(child_report_path),
        "expanded_child_features_sha256": sha256_path(child_feature_path),
        "actor_execution": (
            "two independent complete 256-row LC123 Puffer actors in one 512-environment native vector"
        ),
        "actor_batch_size": ACTOR_BATCH_SIZE,
        "relative_seed_ranges": {
            "original": [0, 127], "new": [128, 255],
        },
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0,
            "offline_training_teacher_actions": len(combined),
            "student_updates": 0, "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Fit one source-locked whole-Puffer phase-8/9 endpoint on the combined LC129 corpus; no FlightSim authority."
            if admitted else
            "Reject LC129 and retain LC105; do not fit an unsplittable corpus."
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
