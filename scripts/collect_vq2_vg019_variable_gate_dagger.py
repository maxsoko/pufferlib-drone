#!/usr/bin/env python3
"""Collect oracle labels on early transition states visited by VG017."""

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


TAG = "vq2_vg019_variable_gate_dagger_round4_vg017_visited_512"
SEED = 429093
AGENTS = 512
EPISODES = 512
COLLECTION_STEP_LIMIT = 1024
MINIMUM_RECORDS = 300_000
MINIMUM_GATE1_RATE = 0.95

CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg017_variable_gate_four_source_refit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "6769476f309beb4bb502c64a83c4f992b47da2b1a8d8b6913398b4b19b3ab4a8"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "476c730649c0f127feb4975fe2ad44f66be64b0ea8dde4d48c5ada7bb0d4c86b"
)
VG017_ADMISSION = ROOT / "docs/vq2_vg017_four_source_refit_admission_2026-07-30.json"
VG017_ADMISSION_SHA256 = (
    "df664cbac28ee676cbd5e056cb1f06745bf3e622710b3faa085eca90425741b6"
)
VG018_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg018_variable_gate_recurrent_teacher_free_256/report.json"
)
VG018_REPORT_SHA256 = (
    "40be8bf0041e9b183c5ad057ae6b4e5a20eb82ef21a972a8f0390e9d3c65ca09"
)
VG018_REJECTION = (
    ROOT / "docs/vq2_vg018_variable_gate_teacher_free_rejection_2026-07-30.json"
)
VG018_REJECTION_SHA256 = (
    "b0d7ddb33b887cdfeb840710915ebb07ff9ec8455d8e1890c1be722db6e8af36"
)
SF016_REPORT = collector.SF016_REPORT
SF016_REPORT_SHA256 = collector.SF016_REPORT_SHA256
PREREGISTRATION = (
    ROOT
    / "docs/vq2_vg019_variable_gate_dagger_round4_preregistration_2026-07-30.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg019_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    """Bind VG019 to the admitted parent, rejected screen, and oracle."""

    expected = {
        collector.GOAL_PROMPT: collector.GOAL_PROMPT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        VG017_ADMISSION: VG017_ADMISSION_SHA256,
        VG018_REPORT: VG018_REPORT_SHA256,
        VG018_REJECTION: VG018_REJECTION_SHA256,
        SF016_REPORT: SF016_REPORT_SHA256,
        ROOT / "pufferlib/vq2_oracle.py": collector.ORACLE_QUERY_SHA256,
    }
    for path, digest in expected.items():
        if collector.sha256_path(path) != digest:
            raise RuntimeError(f"VG019 source evidence hash mismatch: {path}")

    train = json.loads(TRAIN_REPORT.read_text())
    admission = json.loads(VG017_ADMISSION.read_text())
    screen = json.loads(VG018_REPORT.read_text())
    rejection = json.loads(VG018_REJECTION.read_text())
    oracle = json.loads(SF016_REPORT.read_text())
    if (
        train.get("schema")
        != "vq2_variable_gate_four_source_refit_report_v1"
        or train.get("tag")
        != "vq2_vg017_variable_gate_four_source_refit_001"
        or not train.get("completed")
        or not train.get("numerically_admitted")
        or train.get("best_epoch") != 11
        or train.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG017 training result is not VG019-admissible")
    if (
        admission.get("schema")
        != "vq2_vg017_four_source_refit_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != CHECKPOINT_SHA256
        or admission.get("artifact_sha256", {}).get("report")
        != TRAIN_REPORT_SHA256
    ):
        raise RuntimeError("VG017 admission evidence does not bind VG019")
    if (
        screen.get("schema")
        != "vq2_variable_gate_recurrent_teacher_free_admission_v1"
        or screen.get("tag")
        != "vq2_vg018_variable_gate_recurrent_teacher_free_256"
        or not screen.get("completed")
        or screen.get("admitted")
        or screen.get("total_episodes") != 256
        or screen.get("successes") != 0
        or screen.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG018 report is not the source-locked rejection")
    if (
        rejection.get("schema") != "vq2_vg018_teacher_free_rejection_v1"
        or rejection.get("admitted")
        or not rejection.get("completed")
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("candidate_sha256") != CHECKPOINT_SHA256
        or rejection.get("episodes") != 256
        or rejection.get("successes") != 0
        or rejection.get("gate1_reach") != 254
        or rejection.get("gate2_reach") != 4
        or rejection.get("gate3_reach") != 1
        or rejection.get("crashes") != 168
        or rejection.get("misses") != 88
    ):
        raise RuntimeError("VG018 rejection evidence does not authorize VG019")
    if not oracle.get("parity_passed"):
        raise RuntimeError("SF016 did not admit the VG019 oracle query")


def configure_collector() -> None:
    """Bind the generic collector to the one preregistered VG019 run."""

    evaluator.CHECKPOINT = CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256

    collector.TAG = TAG
    collector.SCHEMA = "vq2_variable_gate_dagger_time_major_v1"
    collector.STATE_SCHEMA = "vq2_vg019_variable_gate_dagger_collection_state_v1"
    collector.REPORT_SCHEMA = "vq2_vg019_variable_gate_dagger_collection_report_v1"
    collector.REJECTION_SCHEMA = (
        "vq2_vg019_variable_gate_dagger_collection_rejection_v1"
    )
    collector.COLLECTION_LABEL = "VG019"
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
        VG017_ADMISSION,
        VG018_REPORT,
        VG018_REJECTION,
        SF016_REPORT,
    )
    collector.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)
    collector.EVIDENCE_SHA256 = {
        "vg017_checkpoint_sha256": CHECKPOINT_SHA256,
        "vg017_report_sha256": TRAIN_REPORT_SHA256,
        "vg017_admission_sha256": VG017_ADMISSION_SHA256,
        "vg018_report_sha256": VG018_REPORT_SHA256,
        "vg018_rejection_sha256": VG018_REJECTION_SHA256,
        "sf016_report_sha256": SF016_REPORT_SHA256,
    }
    collector.QUERY_ACTION_SOURCE = (
        "sf016_admitted_alignment_oracle_query_at_vg017_visited_state"
    )
    collector.PLANT_ACTION_SOURCE = "vg017_recurrent_actor_deterministic_mean"
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
