#!/usr/bin/env python3
"""Collect oracle labels on early transition states visited by VG014."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_variable_gate_dagger as collector
import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator


TAG = "vq2_vg016_variable_gate_dagger_round3_vg014_visited_512"
SEED = 429079
AGENTS = 512
EPISODES = 512
COLLECTION_STEP_LIMIT = 1024
MINIMUM_RECORDS = 350_000
MINIMUM_GATE1_RATE = 0.90

CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg014_variable_gate_three_source_refit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "60d69a1f80a1551b190cd4b517dfa029bc3dbe48728a853a47b658df0866d79e"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "067d9c198c14959dbb7221d34f9fec0f065a3cee41043e3ce0ff1f9a7bbf0a16"
)
VG014_ADMISSION = (
    ROOT / "docs/vq2_vg014_three_source_refit_admission_2026-07-30.json"
)
VG014_ADMISSION_SHA256 = (
    "073fc31e143e2d8e61e40cead685698c47e012b7153b32622bd4dc202fbcfa72"
)
VG015_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg015_variable_gate_recurrent_teacher_free_256/report.json"
)
VG015_REPORT_SHA256 = (
    "bb655666484125a46ce23612c2bc4cb4a18295991c10654b94a7e9c0be782cda"
)
VG015_REJECTION = (
    ROOT / "docs/vq2_vg015_variable_gate_teacher_free_rejection_2026-07-30.json"
)
VG015_REJECTION_SHA256 = (
    "568f948e861cef0634d7e1464c8d7869d1767eb77a1829b380a529036c1cbde6"
)
SF016_REPORT = collector.SF016_REPORT
SF016_REPORT_SHA256 = collector.SF016_REPORT_SHA256
PREREGISTRATION = (
    ROOT
    / "docs/vq2_vg016_variable_gate_dagger_round3_preregistration_2026-07-30.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg016_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    """Bind VG016 to the admitted parent, rejected screen, and oracle."""

    expected = {
        collector.GOAL_PROMPT: collector.GOAL_PROMPT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        VG014_ADMISSION: VG014_ADMISSION_SHA256,
        VG015_REPORT: VG015_REPORT_SHA256,
        VG015_REJECTION: VG015_REJECTION_SHA256,
        SF016_REPORT: SF016_REPORT_SHA256,
        ROOT / "pufferlib/vq2_oracle.py": collector.ORACLE_QUERY_SHA256,
    }
    for path, digest in expected.items():
        if collector.sha256_path(path) != digest:
            raise RuntimeError(f"VG016 source evidence hash mismatch: {path}")

    train = json.loads(TRAIN_REPORT.read_text())
    admission = json.loads(VG014_ADMISSION.read_text())
    screen = json.loads(VG015_REPORT.read_text())
    rejection = json.loads(VG015_REJECTION.read_text())
    oracle = json.loads(SF016_REPORT.read_text())
    if (
        train.get("schema")
        != "vq2_variable_gate_three_source_refit_report_v1"
        or train.get("tag")
        != "vq2_vg014_variable_gate_three_source_refit_001"
        or not train.get("completed")
        or not train.get("numerically_admitted")
        or train.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG014 training result is not VG016-admissible")
    if (
        admission.get("schema")
        != "vq2_vg014_three_source_refit_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != CHECKPOINT_SHA256
        or admission.get("artifact_sha256", {}).get("report")
        != TRAIN_REPORT_SHA256
    ):
        raise RuntimeError("VG014 admission evidence does not bind VG016")
    if (
        screen.get("schema")
        != "vq2_variable_gate_recurrent_teacher_free_admission_v1"
        or screen.get("tag")
        != "vq2_vg015_variable_gate_recurrent_teacher_free_256"
        or not screen.get("completed")
        or screen.get("admitted")
        or screen.get("total_episodes") != 256
        or screen.get("successes") != 0
        or screen.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG015 report is not the source-locked rejection")
    if (
        rejection.get("schema") != "vq2_vg015_teacher_free_rejection_v1"
        or rejection.get("admitted")
        or not rejection.get("completed")
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("candidate_sha256") != CHECKPOINT_SHA256
        or rejection.get("episodes") != 256
        or rejection.get("successes") != 0
        or rejection.get("gate1_reach") != 246
        or rejection.get("gate2_reach") != 2
        or rejection.get("gate3_reach") != 0
        or rejection.get("crashes") != 137
    ):
        raise RuntimeError("VG015 rejection evidence does not authorize VG016")
    if not oracle.get("parity_passed"):
        raise RuntimeError("SF016 did not admit the VG016 oracle query")


def configure_collector() -> None:
    """Bind the generic collector to the one preregistered VG016 run."""

    evaluator.CHECKPOINT = CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256

    collector.TAG = TAG
    collector.SCHEMA = "vq2_variable_gate_dagger_time_major_v1"
    collector.STATE_SCHEMA = "vq2_vg016_variable_gate_dagger_collection_state_v1"
    collector.REPORT_SCHEMA = "vq2_vg016_variable_gate_dagger_collection_report_v1"
    collector.REJECTION_SCHEMA = (
        "vq2_vg016_variable_gate_dagger_collection_rejection_v1"
    )
    collector.COLLECTION_LABEL = "VG016"
    collector.AGENTS = AGENTS
    collector.EPISODES = EPISODES
    collector.SEED = SEED
    collector.COLLECTION_STEP_LIMIT = COLLECTION_STEP_LIMIT
    collector.MINIMUM_RECORDS = MINIMUM_RECORDS
    collector.MINIMUM_GATE1_RATE = MINIMUM_GATE1_RATE
    collector.MINIMUM_GATE2_RATE = 0.0
    collector.MAXIMUM_CRASH_RATE = 0.05
    collector.CRASH_RATE_IS_ADMISSION = False
    collector.GATE2_REACH_IS_ADMISSION = False
    collector.PREREGISTRATION = PREREGISTRATION
    collector.RUNNER = RUNNER
    collector.EVIDENCE_PATHS = (
        CHECKPOINT,
        TRAIN_REPORT,
        VG014_ADMISSION,
        VG015_REPORT,
        VG015_REJECTION,
        SF016_REPORT,
    )
    collector.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)
    collector.EVIDENCE_SHA256 = {
        "vg014_checkpoint_sha256": CHECKPOINT_SHA256,
        "vg014_report_sha256": TRAIN_REPORT_SHA256,
        "vg014_admission_sha256": VG014_ADMISSION_SHA256,
        "vg015_report_sha256": VG015_REPORT_SHA256,
        "vg015_rejection_sha256": VG015_REJECTION_SHA256,
        "sf016_report_sha256": SF016_REPORT_SHA256,
    }
    collector.QUERY_ACTION_SOURCE = (
        "sf016_admitted_alignment_oracle_query_at_vg014_visited_state"
    )
    collector.PLANT_ACTION_SOURCE = "vg014_recurrent_actor_deterministic_mean"
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
