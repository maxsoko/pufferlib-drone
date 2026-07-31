#!/usr/bin/env python3
"""Collect oracle labels on long-horizon states visited by VG025."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_variable_gate_dagger as collector
import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP


TAG = "vq2_vg027_variable_gate_dagger_round6_vg025_visited_512"
SEED = 429119
AGENTS = 512
EPISODES = 512
COLLECTION_STEP_LIMIT = 4096
MINIMUM_RECORDS = 1_000_000
MINIMUM_GATE1_RATE = 0.95
MINIMUM_GATE2_RATE = 0.20
MINIMUM_POST_GATE2_RECORDS = 1

CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg025_variable_gate_six_source_refit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "85671c31a7c7a25cf49414cd51259357a2efd64b5d2b441fea27c51b93bf77aa"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "b9d22d2867b8518a0f8b1b8a0fbbad9d0977f3a57e189b83d1362be98f38dcb6"
)
VG025_ADMISSION = (
    ROOT / "docs/vq2_vg025_six_source_refit_admission_2026-07-30.json"
)
VG025_ADMISSION_SHA256 = (
    "633a34d5b6470b271f3e376ab0d4c35ef179ad471c23cb3af8b840f090468ac8"
)
VG026_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg026_variable_gate_recurrent_teacher_free_256/report.json"
)
VG026_REPORT_SHA256 = (
    "324634e247ee26e7e3eedc4c5602cf02aa6799a94ebb4518bc03935eee01ced9"
)
VG026_REJECTION = (
    ROOT / "docs/vq2_vg026_variable_gate_teacher_free_rejection_2026-07-30.json"
)
VG026_REJECTION_SHA256 = (
    "866e240c97f678a093888083a36f6e9b3d5bb0e3ce1b1d0554f599aed0665229"
)
SF016_REPORT = collector.SF016_REPORT
SF016_REPORT_SHA256 = collector.SF016_REPORT_SHA256
PREREGISTRATION = (
    ROOT / "docs/vq2_vg027_variable_gate_dagger_round6_preregistration_2026-07-30.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg027_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG

_BASE_DAGGER_COLLECTION_PREDICATES = collector.dagger_collection_predicates


def vg027_collection_predicates(
    metrics: dict[str, float],
    **kwargs: Any,
) -> dict[str, bool]:
    """Retain generic predicates and require actual phase-2 label coverage."""

    predicates = _BASE_DAGGER_COLLECTION_PREDICATES(metrics, **kwargs)
    phase_records = kwargs["phase_records"]
    predicates["post_gate_2_records_present"] = bool(
        isinstance(phase_records, np.ndarray)
        and phase_records.shape == (ENGINE_GATE_CAP + 1,)
        and phase_records[2] >= MINIMUM_POST_GATE2_RECORDS
    )
    return predicates


def verify_inputs() -> None:
    """Bind VG027 to the admitted parent, rejected screen, and oracle."""

    expected = {
        collector.GOAL_PROMPT: collector.GOAL_PROMPT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        VG025_ADMISSION: VG025_ADMISSION_SHA256,
        VG026_REPORT: VG026_REPORT_SHA256,
        VG026_REJECTION: VG026_REJECTION_SHA256,
        SF016_REPORT: SF016_REPORT_SHA256,
        ROOT / "pufferlib/vq2_oracle.py": collector.ORACLE_QUERY_SHA256,
    }
    for path, digest in expected.items():
        if collector.sha256_path(path) != digest:
            raise RuntimeError(f"VG027 source evidence hash mismatch: {path}")

    train = json.loads(TRAIN_REPORT.read_text())
    admission = json.loads(VG025_ADMISSION.read_text())
    screen = json.loads(VG026_REPORT.read_text())
    rejection = json.loads(VG026_REJECTION.read_text())
    oracle = json.loads(SF016_REPORT.read_text())
    if (
        train.get("schema") != "vq2_variable_gate_six_source_refit_report_v1"
        or train.get("tag") != "vq2_vg025_variable_gate_six_source_refit_001"
        or not train.get("completed")
        or not train.get("numerically_admitted")
        or train.get("best_epoch") != 5
        or train.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG025 training result is not VG027-admissible")
    if (
        admission.get("schema") != "vq2_vg025_six_source_refit_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != CHECKPOINT_SHA256
        or admission.get("artifact_sha256", {}).get("report")
        != TRAIN_REPORT_SHA256
    ):
        raise RuntimeError("VG025 admission evidence does not bind VG027")
    if (
        screen.get("schema")
        != "vq2_variable_gate_recurrent_teacher_free_admission_v1"
        or screen.get("tag")
        != "vq2_vg026_variable_gate_recurrent_teacher_free_256"
        or not screen.get("completed")
        or screen.get("admitted")
        or screen.get("total_episodes") != 256
        or screen.get("successes") != 0
        or screen.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG026 report is not the source-locked rejection")
    if (
        rejection.get("schema") != "vq2_vg026_teacher_free_rejection_v1"
        or rejection.get("admitted")
        or not rejection.get("completed")
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("candidate_sha256") != CHECKPOINT_SHA256
        or rejection.get("episodes") != 256
        or rejection.get("successes") != 0
        or rejection.get("gate1_reach") != 256
        or rejection.get("gate2_reach") != 97
        or rejection.get("gate3_reach") != 2
        or rejection.get("gate4_reach") != 0
        or rejection.get("crashes") != 168
        or rejection.get("crashes_low") != 72
        or rejection.get("crashes_xy") != 96
        or rejection.get("misses") != 88
    ):
        raise RuntimeError("VG026 rejection evidence does not authorize VG027")
    if not oracle.get("parity_passed"):
        raise RuntimeError("SF016 did not admit the VG027 oracle query")


def configure_collector() -> None:
    """Bind the generic collector to the one preregistered VG027 run."""

    evaluator.CHECKPOINT = CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256

    collector.TAG = TAG
    collector.SCHEMA = "vq2_variable_gate_dagger_time_major_v1"
    collector.STATE_SCHEMA = "vq2_vg027_variable_gate_dagger_collection_state_v1"
    collector.REPORT_SCHEMA = "vq2_vg027_variable_gate_dagger_collection_report_v1"
    collector.REJECTION_SCHEMA = (
        "vq2_vg027_variable_gate_dagger_collection_rejection_v1"
    )
    collector.COLLECTION_LABEL = "VG027"
    collector.AGENTS = AGENTS
    collector.EPISODES = EPISODES
    collector.SEED = SEED
    collector.COLLECTION_STEP_LIMIT = COLLECTION_STEP_LIMIT
    collector.MINIMUM_RECORDS = MINIMUM_RECORDS
    collector.MINIMUM_GATE1_RATE = MINIMUM_GATE1_RATE
    collector.MINIMUM_GATE2_RATE = MINIMUM_GATE2_RATE
    collector.MAXIMUM_CRASH_RATE = 1.0
    collector.CRASH_RATE_IS_ADMISSION = False
    collector.GATE2_REACH_IS_ADMISSION = True
    collector.PREREGISTRATION = PREREGISTRATION
    collector.RUNNER = RUNNER
    collector.EVIDENCE_PATHS = (
        CHECKPOINT,
        TRAIN_REPORT,
        VG025_ADMISSION,
        VG026_REPORT,
        VG026_REJECTION,
        SF016_REPORT,
    )
    collector.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)
    collector.EVIDENCE_SHA256 = {
        "vg025_checkpoint_sha256": CHECKPOINT_SHA256,
        "vg025_report_sha256": TRAIN_REPORT_SHA256,
        "vg025_admission_sha256": VG025_ADMISSION_SHA256,
        "vg026_report_sha256": VG026_REPORT_SHA256,
        "vg026_rejection_sha256": VG026_REJECTION_SHA256,
        "sf016_report_sha256": SF016_REPORT_SHA256,
    }
    collector.QUERY_ACTION_SOURCE = (
        "sf016_admitted_alignment_oracle_query_at_vg025_visited_state"
    )
    collector.PLANT_ACTION_SOURCE = "vg025_recurrent_actor_deterministic_mean"
    collector.dagger_collection_predicates = vg027_collection_predicates
    collector.verify_inputs = verify_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    verify_inputs()
    configure_collector()
    report = collector.collect(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
