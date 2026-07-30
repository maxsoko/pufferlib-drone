#!/usr/bin/env python3
"""Collect oracle labels on transition/recovery states visited by VG022."""

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


TAG = "vq2_vg024_variable_gate_dagger_round5_vg022_visited_512"
SEED = 429109
AGENTS = 512
EPISODES = 512
COLLECTION_STEP_LIMIT = 1024
MINIMUM_RECORDS = 300_000
MINIMUM_GATE1_RATE = 0.95

CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg022_five_source_device_decode_continuation_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "bd3f93d46e2b04af4a1ae843ac8a920bd75e2954079dbba3c6a3da541bb93915"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "23d281824813e14c3d77c4446a36f4af4d29769c01bdd0a46202caae017c95b3"
)
VG022_ADMISSION = (
    ROOT / "docs/vq2_vg022_five_source_device_decode_admission_2026-07-30.json"
)
VG022_ADMISSION_SHA256 = (
    "ac93a00d46139a97610731acac37f6b652be69c7c2d8aa30d865c1a97c593994"
)
VG023_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg023_variable_gate_recurrent_teacher_free_256/report.json"
)
VG023_REPORT_SHA256 = (
    "2a3023fdedf080e6272e74b88e61d5b5e23f6395cfc3b2c1cd40d049c5504665"
)
VG023_REJECTION = (
    ROOT / "docs/vq2_vg023_variable_gate_teacher_free_rejection_2026-07-30.json"
)
VG023_REJECTION_SHA256 = (
    "88f403f028eece863e6971e12d6d0fae675cf3e4aec6b85f7b5cb75274329d34"
)
SF016_REPORT = collector.SF016_REPORT
SF016_REPORT_SHA256 = collector.SF016_REPORT_SHA256
PREREGISTRATION = (
    ROOT / "docs/vq2_vg024_variable_gate_dagger_round5_preregistration_2026-07-30.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg024_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    """Bind VG024 to the admitted parent, rejected screen, and oracle."""

    expected = {
        collector.GOAL_PROMPT: collector.GOAL_PROMPT_SHA256,
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        VG022_ADMISSION: VG022_ADMISSION_SHA256,
        VG023_REPORT: VG023_REPORT_SHA256,
        VG023_REJECTION: VG023_REJECTION_SHA256,
        SF016_REPORT: SF016_REPORT_SHA256,
        ROOT / "pufferlib/vq2_oracle.py": collector.ORACLE_QUERY_SHA256,
    }
    for path, digest in expected.items():
        if collector.sha256_path(path) != digest:
            raise RuntimeError(f"VG024 source evidence hash mismatch: {path}")

    train = json.loads(TRAIN_REPORT.read_text())
    admission = json.loads(VG022_ADMISSION.read_text())
    screen = json.loads(VG023_REPORT.read_text())
    rejection = json.loads(VG023_REJECTION.read_text())
    oracle = json.loads(SF016_REPORT.read_text())
    if (
        train.get("schema")
        != "vq2_variable_gate_five_source_device_decode_report_v1"
        or train.get("tag")
        != "vq2_vg022_five_source_device_decode_continuation_001"
        or not train.get("completed")
        or not train.get("numerically_admitted")
        or train.get("best_epoch") != 9
        or train.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG022 training result is not VG024-admissible")
    if (
        admission.get("schema")
        != "vq2_vg022_five_source_device_decode_admission_v1"
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != CHECKPOINT_SHA256
        or admission.get("artifact_sha256", {}).get("report")
        != TRAIN_REPORT_SHA256
    ):
        raise RuntimeError("VG022 admission evidence does not bind VG024")
    if (
        screen.get("schema")
        != "vq2_variable_gate_recurrent_teacher_free_admission_v1"
        or screen.get("tag")
        != "vq2_vg023_variable_gate_recurrent_teacher_free_256"
        or not screen.get("completed")
        or screen.get("admitted")
        or screen.get("total_episodes") != 256
        or screen.get("successes") != 0
        or screen.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG023 report is not the source-locked rejection")
    if (
        rejection.get("schema") != "vq2_vg023_teacher_free_rejection_v1"
        or rejection.get("admitted")
        or not rejection.get("completed")
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("candidate_sha256") != CHECKPOINT_SHA256
        or rejection.get("episodes") != 256
        or rejection.get("successes") != 0
        or rejection.get("gate1_reach") != 255
        or rejection.get("gate2_reach") != 26
        or rejection.get("gate3_reach") != 0
        or rejection.get("crashes") != 193
        or rejection.get("crashes_xy") != 101
        or rejection.get("misses") != 63
    ):
        raise RuntimeError("VG023 rejection evidence does not authorize VG024")
    if not oracle.get("parity_passed"):
        raise RuntimeError("SF016 did not admit the VG024 oracle query")


def configure_collector() -> None:
    """Bind the generic collector to the one preregistered VG024 run."""

    evaluator.CHECKPOINT = CHECKPOINT
    evaluator.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.TRAIN_REPORT = TRAIN_REPORT
    evaluator.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256

    collector.TAG = TAG
    collector.SCHEMA = "vq2_variable_gate_dagger_time_major_v1"
    collector.STATE_SCHEMA = "vq2_vg024_variable_gate_dagger_collection_state_v1"
    collector.REPORT_SCHEMA = "vq2_vg024_variable_gate_dagger_collection_report_v1"
    collector.REJECTION_SCHEMA = (
        "vq2_vg024_variable_gate_dagger_collection_rejection_v1"
    )
    collector.COLLECTION_LABEL = "VG024"
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
        VG022_ADMISSION,
        VG023_REPORT,
        VG023_REJECTION,
        SF016_REPORT,
    )
    collector.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)
    collector.EVIDENCE_SHA256 = {
        "vg022_checkpoint_sha256": CHECKPOINT_SHA256,
        "vg022_report_sha256": TRAIN_REPORT_SHA256,
        "vg022_admission_sha256": VG022_ADMISSION_SHA256,
        "vg023_report_sha256": VG023_REPORT_SHA256,
        "vg023_rejection_sha256": VG023_REJECTION_SHA256,
        "sf016_report_sha256": SF016_REPORT_SHA256,
    }
    collector.QUERY_ACTION_SOURCE = (
        "sf016_admitted_alignment_oracle_query_at_vg022_visited_state"
    )
    collector.PLANT_ACTION_SOURCE = "vg022_recurrent_actor_deterministic_mean"
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
