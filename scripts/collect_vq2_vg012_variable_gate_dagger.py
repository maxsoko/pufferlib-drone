#!/usr/bin/env python3
"""Collect oracle labels on states visited by the rejected VG010 actor."""

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


TAG = "vq2_vg012_variable_gate_dagger_round2_vg010_visited_512"
SEED = 429064
AGENTS = 512
EPISODES = 512
COLLECTION_STEP_LIMIT = 2048
MINIMUM_RECORDS = 150_000
MINIMUM_GATE1_RATE = 0.60

CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg010_variable_gate_source_balanced_refit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "a093475371a94bc284310bbe64609fe515a35e50be0e68a7fdfb781b55f03822"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "dfe943b593f99c4ccfa53b04290a30cd3a58ea9fefd11d9489786e064b8734a5"
)
VG010_ADMISSION = (
    ROOT / "docs/vq2_vg010_variable_gate_dagger_refit_admission_2026-07-30.json"
)
VG010_ADMISSION_SHA256 = (
    "c0fbfc72c0c609b137d7240aa8347baa6f59926e2c26e137144d363e9e820850"
)
VG011_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg011_variable_gate_recurrent_teacher_free_256/report.json"
)
VG011_REPORT_SHA256 = (
    "3e470b1896881eaab74aa1391fdb51c3639d6ed1149e0678b3d4e35cb0e7cd3d"
)
VG011_REJECTION = (
    ROOT / "docs/vq2_vg011_variable_gate_teacher_free_rejection_2026-07-30.json"
)
VG011_REJECTION_SHA256 = (
    "53d5ed65b680b83c278993c2517553043de1444460e8780ad2ddb8f0a61271f5"
)
SF016_REPORT = collector.SF016_REPORT
SF016_REPORT_SHA256 = collector.SF016_REPORT_SHA256
PREREGISTRATION = (
    ROOT
    / "docs/vq2_vg012_variable_gate_dagger_round2_preregistration_2026-07-30.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg012_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    """Bind VG012 to the admitted parent, rejected screen, and oracle."""

    expected = {
        collector.GOAL_PROMPT: collector.GOAL_PROMPT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        VG010_ADMISSION: VG010_ADMISSION_SHA256,
        VG011_REPORT: VG011_REPORT_SHA256,
        VG011_REJECTION: VG011_REJECTION_SHA256,
        SF016_REPORT: SF016_REPORT_SHA256,
        ROOT / "pufferlib/vq2_oracle.py": collector.ORACLE_QUERY_SHA256,
    }
    for path, digest in expected.items():
        if collector.sha256_path(path) != digest:
            raise RuntimeError(f"VG012 source evidence hash mismatch: {path}")

    train = json.loads(TRAIN_REPORT.read_text())
    admission = json.loads(VG010_ADMISSION.read_text())
    screen = json.loads(VG011_REPORT.read_text())
    rejection = json.loads(VG011_REJECTION.read_text())
    oracle = json.loads(SF016_REPORT.read_text())
    if (
        train.get("schema")
        != "vq2_variable_gate_source_balanced_refit_report_v1"
        or train.get("tag")
        != "vq2_vg010_variable_gate_source_balanced_refit_001"
        or not train.get("completed")
        or not train.get("numerically_admitted")
        or train.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG010 training result is not VG012-admissible")
    if (
        admission.get("schema")
        != "vq2_vg010_source_balanced_refit_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != CHECKPOINT_SHA256
        or admission.get("artifact_sha256", {}).get("report")
        != TRAIN_REPORT_SHA256
    ):
        raise RuntimeError("VG010 admission evidence does not bind VG012")
    if (
        screen.get("schema")
        != "vq2_variable_gate_recurrent_teacher_free_admission_v1"
        or screen.get("tag")
        != "vq2_vg011_variable_gate_recurrent_teacher_free_256"
        or not screen.get("completed")
        or screen.get("admitted")
        or screen.get("total_episodes") != 256
        or screen.get("successes") != 0
        or screen.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG011 report is not the source-locked rejection")
    if (
        rejection.get("schema") != "vq2_vg011_teacher_free_rejection_v1"
        or rejection.get("admitted")
        or not rejection.get("completed")
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("candidate_sha256") != CHECKPOINT_SHA256
        or rejection.get("episodes") != 256
        or rejection.get("successes") != 0
        or rejection.get("gate1_reach") != 176
        or rejection.get("gate2_reach") != 0
        or rejection.get("crashes") != 193
    ):
        raise RuntimeError("VG011 rejection evidence does not authorize VG012")
    if not oracle.get("parity_passed"):
        raise RuntimeError("SF016 did not admit the VG012 oracle query")


def configure_collector() -> None:
    """Bind the generic collector to the one preregistered VG012 run."""

    evaluator.CHECKPOINT = CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256

    collector.TAG = TAG
    collector.SCHEMA = "vq2_variable_gate_dagger_time_major_v1"
    collector.STATE_SCHEMA = "vq2_vg012_variable_gate_dagger_collection_state_v1"
    collector.REPORT_SCHEMA = "vq2_vg012_variable_gate_dagger_collection_report_v1"
    collector.REJECTION_SCHEMA = (
        "vq2_vg012_variable_gate_dagger_collection_rejection_v1"
    )
    collector.COLLECTION_LABEL = "VG012"
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
        VG010_ADMISSION,
        VG011_REPORT,
        VG011_REJECTION,
        SF016_REPORT,
    )
    collector.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)
    collector.EVIDENCE_SHA256 = {
        "vg010_checkpoint_sha256": CHECKPOINT_SHA256,
        "vg010_report_sha256": TRAIN_REPORT_SHA256,
        "vg010_admission_sha256": VG010_ADMISSION_SHA256,
        "vg011_report_sha256": VG011_REPORT_SHA256,
        "vg011_rejection_sha256": VG011_REJECTION_SHA256,
        "sf016_report_sha256": SF016_REPORT_SHA256,
    }
    collector.QUERY_ACTION_SOURCE = (
        "sf016_admitted_alignment_oracle_query_at_vg010_visited_state"
    )
    collector.PLANT_ACTION_SOURCE = "vg010_recurrent_actor_deterministic_mean"
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
