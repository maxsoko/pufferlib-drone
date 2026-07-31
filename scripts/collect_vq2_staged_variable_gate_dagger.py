#!/usr/bin/env python3
"""Run a manifest-bound higher-phase variable-gate DAgger collection."""

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


MANIFEST_SCHEMA = "vq2_staged_variable_gate_dagger_manifest_v1"
_BASE_COLLECTION_PREDICATES = collector.dagger_collection_predicates


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported staged DAgger manifest")
    for name in (
        "tag",
        "collection_label",
        "state_schema",
        "report_schema",
        "rejection_schema",
        "train_report_schema",
        "train_report_tag",
        "candidate_admission_schema",
        "screen_evidence_schema",
        "screen_tag",
        "query_action_source",
        "plant_action_source",
    ):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise RuntimeError(f"staged DAgger {name} is missing")
    for name in (
        "checkpoint",
        "train_report",
        "candidate_admission",
        "screen_evidence",
        "oracle_report",
        "preregistration",
        "runner",
        "goal_prompt",
    ):
        if not isinstance(manifest.get(name), str):
            raise RuntimeError(f"staged DAgger {name} path is missing")
        _root_path(manifest[name])
    for name in (
        "checkpoint_sha256",
        "train_report_sha256",
        "candidate_admission_sha256",
        "screen_evidence_sha256",
        "oracle_report_sha256",
        "goal_prompt_sha256",
    ):
        value = manifest.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"staged DAgger {name} is not a SHA-256")

    agents = int(manifest.get("agents", 0))
    episodes = int(manifest.get("episodes", 0))
    if agents != 512 or episodes != agents:
        raise RuntimeError("staged DAgger requires exactly 512 one-shot agents")
    if int(manifest.get("seed", 0)) <= 429130:
        raise RuntimeError("staged DAgger seed is not fresh")
    if int(manifest.get("collection_step_limit", 0)) not in range(4096, 8193):
        raise RuntimeError("staged DAgger step limit must be in [4096,8192]")
    if int(manifest.get("minimum_records", 0)) < 1_000_000:
        raise RuntimeError("staged DAgger requires at least one million records")
    rates = (
        float(manifest.get("minimum_gate1_rate", -1.0)),
        float(manifest.get("minimum_gate2_rate", -1.0)),
        float(manifest.get("minimum_gate3_rate", -1.0)),
    )
    if not (1.0 >= rates[0] >= rates[1] >= rates[2] > 0.0):
        raise RuntimeError("staged DAgger gate-reach rates are inconsistent")
    if int(manifest.get("minimum_phase2_records", 0)) <= 0:
        raise RuntimeError("staged DAgger requires phase-2 records")
    if int(manifest.get("minimum_phase3_records", 0)) <= 0:
        raise RuntimeError("staged DAgger requires phase-3 records")
    return manifest


def verify_bound_inputs(manifest: dict[str, Any]) -> None:
    expected = {
        _root_path(manifest["checkpoint"]): manifest["checkpoint_sha256"],
        _root_path(manifest["train_report"]): manifest["train_report_sha256"],
        _root_path(manifest["candidate_admission"]): (
            manifest["candidate_admission_sha256"]
        ),
        _root_path(manifest["screen_evidence"]): (
            manifest["screen_evidence_sha256"]
        ),
        _root_path(manifest["oracle_report"]): manifest["oracle_report_sha256"],
        _root_path(manifest["goal_prompt"]): manifest["goal_prompt_sha256"],
        ROOT / "pufferlib/vq2_oracle.py": collector.ORACLE_QUERY_SHA256,
    }
    for path, digest in expected.items():
        if collector.sha256_path(path) != digest:
            raise RuntimeError(f"staged DAgger evidence hash mismatch: {path}")

    train = json.loads(_root_path(manifest["train_report"]).read_text())
    admission = json.loads(
        _root_path(manifest["candidate_admission"]).read_text()
    )
    screen = json.loads(_root_path(manifest["screen_evidence"]).read_text())
    oracle = json.loads(_root_path(manifest["oracle_report"]).read_text())
    if (
        train.get("schema") != manifest["train_report_schema"]
        or train.get("tag") != manifest["train_report_tag"]
        or not train.get("completed")
        or not train.get("numerically_admitted")
        or train.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
        or train.get("minimum_transition_window_exposure", 0.0) < 4.0
        or not train.get("equal_source_weight_audit")
    ):
        raise RuntimeError("staged DAgger actor training report is not admitted")
    if (
        admission.get("schema") != manifest["candidate_admission_schema"]
        or not admission.get("completed")
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != manifest["checkpoint_sha256"]
        or admission.get("artifact_sha256", {}).get("report")
        != manifest["train_report_sha256"]
        or admission.get("safety", {}).get("flight_sim_packets_sent") != 0
        or admission.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("staged DAgger actor admission is not legal")
    if (
        screen.get("schema") != manifest["screen_evidence_schema"]
        or screen.get("screen_tag") != manifest["screen_tag"]
        or not screen.get("completed")
        or screen.get("admitted")
        or not screen.get("unchanged_retry_forbidden")
        or screen.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
        or screen.get("episodes") != 256
        or not screen.get("hard_transport_pass")
        or screen.get("gate_reach", {}).get("3", 0) <= 0
        or screen.get("safety", {}).get("flight_sim_packets_sent") != 0
        or screen.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("staged DAgger screen evidence is not an eligible frontier")
    if not oracle.get("parity_passed"):
        raise RuntimeError("staged DAgger oracle query is not admitted")


def configure_collector(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    checkpoint = _root_path(manifest["checkpoint"])
    train_report = _root_path(manifest["train_report"])
    candidate_admission = _root_path(manifest["candidate_admission"])
    screen_evidence = _root_path(manifest["screen_evidence"])
    oracle_report = _root_path(manifest["oracle_report"])
    goal_prompt = _root_path(manifest["goal_prompt"])

    evaluator.CHECKPOINT = checkpoint
    evaluator.CHECKPOINT_SHA256 = manifest["checkpoint_sha256"]
    evaluator.TRAIN_REPORT = train_report
    evaluator.TRAIN_REPORT_SHA256 = manifest["train_report_sha256"]

    minimum_gate3_rate = float(manifest["minimum_gate3_rate"])
    minimum_phase2_records = int(manifest["minimum_phase2_records"])
    minimum_phase3_records = int(manifest["minimum_phase3_records"])

    def higher_phase_predicates(
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, bool]:
        predicates = _BASE_COLLECTION_PREDICATES(metrics, **kwargs)
        phase_records = kwargs["phase_records"]
        phase_shape = (
            isinstance(phase_records, np.ndarray)
            and phase_records.shape == (ENGINE_GATE_CAP + 1,)
        )
        predicates["gate_3_reach_rate"] = bool(
            metrics.get("env/ordered_gate2_sampled", 0.0)
            >= minimum_gate3_rate
        )
        predicates["minimum_phase_2_records"] = bool(
            phase_shape and phase_records[2] >= minimum_phase2_records
        )
        predicates["minimum_phase_3_records"] = bool(
            phase_shape and phase_records[3] >= minimum_phase3_records
        )
        return predicates

    def bound_verify_inputs() -> None:
        verify_bound_inputs(manifest)

    collector.TAG = manifest["tag"]
    collector.SCHEMA = "vq2_variable_gate_dagger_time_major_v1"
    collector.STATE_SCHEMA = manifest["state_schema"]
    collector.REPORT_SCHEMA = manifest["report_schema"]
    collector.REJECTION_SCHEMA = manifest["rejection_schema"]
    collector.COLLECTION_LABEL = manifest["collection_label"]
    collector.AGENTS = int(manifest["agents"])
    collector.EPISODES = int(manifest["episodes"])
    collector.SEED = int(manifest["seed"])
    collector.COLLECTION_STEP_LIMIT = int(manifest["collection_step_limit"])
    collector.MINIMUM_RECORDS = int(manifest["minimum_records"])
    collector.MINIMUM_GATE1_RATE = float(manifest["minimum_gate1_rate"])
    collector.MINIMUM_GATE2_RATE = float(manifest["minimum_gate2_rate"])
    collector.MAXIMUM_CRASH_RATE = 1.0
    collector.CRASH_RATE_IS_ADMISSION = False
    collector.GATE2_REACH_IS_ADMISSION = True
    collector.PREREGISTRATION = _root_path(manifest["preregistration"])
    collector.RUNNER = _root_path(manifest["runner"])
    collector.GOAL_PROMPT = goal_prompt
    collector.GOAL_PROMPT_SHA256 = manifest["goal_prompt_sha256"]
    collector.EVIDENCE_PATHS = (
        checkpoint,
        train_report,
        candidate_admission,
        screen_evidence,
        oracle_report,
    )
    collector.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        manifest_path.resolve(),
    )
    collector.EVIDENCE_SHA256 = {
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "train_report_sha256": manifest["train_report_sha256"],
        "candidate_admission_sha256": manifest[
            "candidate_admission_sha256"
        ],
        "screen_evidence_sha256": manifest["screen_evidence_sha256"],
        "oracle_report_sha256": manifest["oracle_report_sha256"],
    }
    collector.QUERY_ACTION_SOURCE = manifest["query_action_source"]
    collector.PLANT_ACTION_SOURCE = manifest["plant_action_source"]
    collector.dagger_collection_predicates = higher_phase_predicates
    collector.verify_inputs = bound_verify_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    verify_bound_inputs(manifest)
    configure_collector(manifest_path, manifest)
    report = collector.collect(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
