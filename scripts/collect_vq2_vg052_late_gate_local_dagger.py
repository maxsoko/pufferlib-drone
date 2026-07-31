#!/usr/bin/env python3
"""Collect VG033-visited Gate-4/5 local-start states with SF016 labels."""

from __future__ import annotations

import argparse
import json
import math
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
from scripts.eval_vq2_variable_gate_oracle import (
    load_variable_config,
    variable_environment,
)


MANIFEST_SCHEMA = "vq2_vg052_late_gate_local_dagger_manifest_v1"
EXPECTED_EPISODES = 512
EXPECTED_AGENTS = 512
EXPECTED_NUM_GATES = 6
EXPECTED_GATE_MIN = 3
EXPECTED_GATE_MAX_EXCLUSIVE = 5
EXPECTED_OFFSET_MIN = 2.0
EXPECTED_OFFSET_MAX = 5.0
MINIMUM_RECORDS = 250_000
MINIMUM_PHASE3_RECORDS = 50_000
MINIMUM_PHASE4_RECORDS = 50_000


def root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported VG052 manifest")
    required_strings = (
        "tag",
        "state_schema",
        "report_schema",
        "rejection_schema",
        "checkpoint",
        "train_report",
        "candidate_admission",
        "six_gate_baseline",
        "six_gate_report",
        "oracle_report",
        "goal_prompt",
        "preregistration",
        "runner",
        "query_action_source",
        "plant_action_source",
    )
    for name in required_strings:
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise RuntimeError(f"VG052 {name} is missing")
    for name in required_strings[4:13]:
        root_path(manifest[name])
    for name in (
        "checkpoint_sha256",
        "train_report_sha256",
        "candidate_admission_sha256",
        "six_gate_baseline_sha256",
        "six_gate_report_sha256",
        "oracle_report_sha256",
        "goal_prompt_sha256",
    ):
        value = manifest.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"VG052 {name} is not a SHA-256")
    exact = {
        "agents": EXPECTED_AGENTS,
        "episodes": EXPECTED_EPISODES,
        "num_gates": EXPECTED_NUM_GATES,
        "gate_local_start_gate_min": EXPECTED_GATE_MIN,
        "gate_local_start_gate_max_exclusive": EXPECTED_GATE_MAX_EXCLUSIVE,
        "minimum_records": MINIMUM_RECORDS,
        "minimum_phase3_records": MINIMUM_PHASE3_RECORDS,
        "minimum_phase4_records": MINIMUM_PHASE4_RECORDS,
    }
    for name, expected in exact.items():
        if int(manifest.get(name, -1)) != expected:
            raise RuntimeError(f"VG052 {name} changed")
    if float(manifest.get("gate_local_start_offset_min", -1.0)) != EXPECTED_OFFSET_MIN:
        raise RuntimeError("VG052 local-start minimum offset changed")
    if float(manifest.get("gate_local_start_offset_max", -1.0)) != EXPECTED_OFFSET_MAX:
        raise RuntimeError("VG052 local-start maximum offset changed")
    step_limit = int(manifest.get("collection_step_limit", 0))
    if step_limit not in range(1024, 3073):
        raise RuntimeError("VG052 step limit must remain in [1024,3072]")
    if int(manifest.get("seed", 0)) <= 429168:
        raise RuntimeError("VG052 seed label is not fresh")
    return manifest


def verify_bound_inputs(manifest: dict[str, Any]) -> None:
    expected = {
        root_path(manifest["checkpoint"]): manifest["checkpoint_sha256"],
        root_path(manifest["train_report"]): manifest["train_report_sha256"],
        root_path(manifest["candidate_admission"]): (
            manifest["candidate_admission_sha256"]
        ),
        root_path(manifest["six_gate_baseline"]): (
            manifest["six_gate_baseline_sha256"]
        ),
        root_path(manifest["six_gate_report"]): manifest["six_gate_report_sha256"],
        root_path(manifest["oracle_report"]): manifest["oracle_report_sha256"],
        root_path(manifest["goal_prompt"]): manifest["goal_prompt_sha256"],
        ROOT / "pufferlib/vq2_oracle.py": collector.ORACLE_QUERY_SHA256,
    }
    for path, digest in expected.items():
        if collector.sha256_path(path) != digest:
            raise RuntimeError(f"VG052 evidence hash mismatch: {path}")

    train = json.loads(root_path(manifest["train_report"]).read_text())
    admission = json.loads(root_path(manifest["candidate_admission"]).read_text())
    baseline = json.loads(root_path(manifest["six_gate_baseline"]).read_text())
    six_gate = json.loads(root_path(manifest["six_gate_report"]).read_text())
    oracle = json.loads(root_path(manifest["oracle_report"]).read_text())
    if (
        train.get("schema") != "vq2_variable_gate_eight_source_refit_report_v1"
        or train.get("tag") != "vq2_vg033_variable_gate_eight_source_refit_001"
        or not train.get("completed")
        or not train.get("numerically_admitted")
        or train.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
    ):
        raise RuntimeError("VG052 parent training report is not admitted")
    if (
        admission.get("schema") != "vq2_vg033_eight_source_refit_admission_v1"
        or not admission.get("completed")
        or not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != manifest["checkpoint_sha256"]
        or admission.get("safety", {}).get("flight_sim_packets_sent") != 0
        or admission.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("VG052 parent admission changed")
    if (
        baseline.get("schema")
        != "vq2_vg051_count6_multi_offset_baseline_evidence_v1"
        or not baseline.get("completed")
        or baseline.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
        or baseline.get("gate_reach", {}).get("5") != 0
        or baseline.get("conclusion", {}).get("live_authority")
    ):
        raise RuntimeError("VG052 six-gate baseline does not establish the bottleneck")
    if (
        six_gate.get("schema") != "vq2_count6_multi_offset_baseline_v1"
        or not six_gate.get("completed")
        or six_gate.get("aggregate", {}).get("gate_reach", {}).get("5") != 0
        or six_gate.get("safety", {}).get("flight_sim_packets_sent") != 0
        or six_gate.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("VG052 source six-gate report changed")
    if not oracle.get("parity_passed"):
        raise RuntimeError("VG052 SF016 oracle query is not admitted")


def late_gate_environment(manifest: dict[str, Any]) -> dict[str, int | float]:
    values = variable_environment(EXPECTED_NUM_GATES)
    step_limit = int(manifest["collection_step_limit"])
    values.update({
        "evaluation_episode_limit": 1,
        "evaluation_episode_offset": 0,
        "max_steps": step_limit,
        "time_limit_seconds": step_limit / 64.0,
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
        "observable_gate_index_denominator": float(ENGINE_GATE_CAP),
        "gate_local_start_curriculum": 1,
        "gate_local_start_probability": 1.0,
        "gate_local_start_offset_min": EXPECTED_OFFSET_MIN,
        "gate_local_start_offset_max": EXPECTED_OFFSET_MAX,
        "gate_local_start_gate_min": EXPECTED_GATE_MIN,
        "gate_local_start_gate_max_exclusive": EXPECTED_GATE_MAX_EXCLUSIVE,
        "mixed_start_curriculum": 0,
        "segment_start_probability": 0.0,
        "use_custom_start": 0,
    })
    return values


def configure_collector(manifest_path: Path, manifest: dict[str, Any]) -> None:
    checkpoint = root_path(manifest["checkpoint"])
    train_report = root_path(manifest["train_report"])
    evidence_paths = (
        checkpoint,
        train_report,
        root_path(manifest["candidate_admission"]),
        root_path(manifest["six_gate_baseline"]),
        root_path(manifest["six_gate_report"]),
        root_path(manifest["oracle_report"]),
    )

    evaluator.CHECKPOINT = checkpoint
    evaluator.CHECKPOINT_SHA256 = manifest["checkpoint_sha256"]
    evaluator.TRAIN_REPORT = train_report
    evaluator.TRAIN_REPORT_SHA256 = manifest["train_report_sha256"]

    def dagger_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
        config, overrides = load_variable_config(
            pufferl_module,
            agents=EXPECTED_AGENTS,
            episodes=EXPECTED_EPISODES,
            seed=int(manifest["seed"]),
            num_gates=EXPECTED_NUM_GATES,
        )
        config["env"].update(late_gate_environment(manifest))
        return config, overrides

    def late_gate_predicates(
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, bool]:
        lengths = kwargs["lengths"]
        terminal_count = kwargs["terminal_count"]
        terminal_is_last = kwargs["terminal_is_last"]
        phase_records = kwargs["phase_records"]
        hard_zero = (
            "out_of_order",
            "action_envelope_violation",
            "wire_rate_envelope_violation",
            "thrust_envelope_violation",
        )
        phase_shape = (
            isinstance(phase_records, np.ndarray)
            and phase_records.shape == (ENGINE_GATE_CAP + 1,)
        )
        predicates = {
            "native_episode_count": metrics.get("env/n") == EXPECTED_EPISODES,
            "fixed_six_gate_course": (
                abs(metrics.get("env/gate_count6_episode", -1.0) - 1.0)
                <= 1e-12
            ),
            "all_resets_gate_local": (
                abs(metrics.get("env/gate_local_start_rate", -1.0) - 1.0)
                <= 1e-12
                and abs(metrics.get("env/gate_local_reset_count", -1.0) - 1.0)
                <= 1e-12
            ),
            "zero_hard_safety_envelope_metrics": not any(
                metrics.get(f"env/{name}", math.inf) != 0.0
                for name in hard_zero
            ),
            "episode_length_shape": lengths.shape == (EXPECTED_AGENTS,),
            "episode_lengths_positive": bool(np.all(lengths > 0)),
            "episode_lengths_bounded": bool(
                np.all(lengths <= int(manifest["collection_step_limit"]))
            ),
            "one_terminal_per_episode": bool(np.all(terminal_count == 1)),
            "terminal_is_last": bool(np.all(terminal_is_last)),
            "label_count_matches_lengths": kwargs["labels"] == int(lengths.sum()),
            "minimum_record_count": kwargs["labels"] >= MINIMUM_RECORDS,
            "executed_action_parity": kwargs["executed_action_max_error"] <= 1e-7,
            "query_actions_finite": kwargs.get("query_nonfinite", 1) == 0,
            "query_actions_in_envelope": (
                kwargs.get("query_action_envelope_violations", 1) == 0
            ),
            "phase_changes_only_on_public_ticks": (
                kwargs["phase_changes_off_tick"] == 0
            ),
            "phase_never_decreases": kwargs["phase_decreases"] == 0,
            # The collector initializes its held public phase at zero. Each
            # deliberately late-started agent therefore contributes exactly
            # one diagnostic jump on step zero; subsequent updates remain
            # ordered one-gate increments.
            "initial_late_phase_jump_exact": (
                kwargs["phase_skips"] == EXPECTED_AGENTS
            ),
            "phase_encoding_exact": kwargs["raw_phase_encoding_max_error"] <= 1e-6,
            "phase_record_shape": phase_shape,
            "minimum_phase_3_records": bool(
                phase_shape and phase_records[3] >= MINIMUM_PHASE3_RECORDS
            ),
            "minimum_phase_4_records": bool(
                phase_shape and phase_records[4] >= MINIMUM_PHASE4_RECORDS
            ),
            "no_early_phase_records": bool(
                phase_shape and int(phase_records[:3].sum()) == 0
            ),
        }
        return {name: bool(passed) for name, passed in predicates.items()}

    def bound_verify_inputs() -> None:
        verify_bound_inputs(manifest)

    collector.TAG = manifest["tag"]
    collector.SCHEMA = "vq2_late_gate_local_dagger_time_major_v1"
    collector.STATE_SCHEMA = manifest["state_schema"]
    collector.REPORT_SCHEMA = manifest["report_schema"]
    collector.REJECTION_SCHEMA = manifest["rejection_schema"]
    collector.COLLECTION_LABEL = "VG052"
    collector.AGENTS = EXPECTED_AGENTS
    collector.EPISODES = EXPECTED_EPISODES
    collector.SEED = int(manifest["seed"])
    collector.COLLECTION_STEP_LIMIT = int(manifest["collection_step_limit"])
    collector.MINIMUM_RECORDS = MINIMUM_RECORDS
    collector.MINIMUM_GATE1_RATE = 0.0
    collector.MINIMUM_GATE2_RATE = 0.0
    collector.MAXIMUM_CRASH_RATE = 1.0
    collector.CRASH_RATE_IS_ADMISSION = False
    collector.GATE2_REACH_IS_ADMISSION = False
    collector.DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / manifest["tag"]
    collector.PREREGISTRATION = root_path(manifest["preregistration"])
    collector.RUNNER = root_path(manifest["runner"])
    collector.GOAL_PROMPT = root_path(manifest["goal_prompt"])
    collector.GOAL_PROMPT_SHA256 = manifest["goal_prompt_sha256"]
    collector.EVIDENCE_PATHS = evidence_paths
    collector.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), manifest_path.resolve())
    collector.EVIDENCE_SHA256 = {
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "train_report_sha256": manifest["train_report_sha256"],
        "candidate_admission_sha256": manifest["candidate_admission_sha256"],
        "six_gate_baseline_sha256": manifest["six_gate_baseline_sha256"],
        "six_gate_report_sha256": manifest["six_gate_report_sha256"],
        "oracle_report_sha256": manifest["oracle_report_sha256"],
    }
    collector.QUERY_ACTION_SOURCE = manifest["query_action_source"]
    collector.PLANT_ACTION_SOURCE = manifest["plant_action_source"]
    collector.dagger_config = dagger_config
    collector.dagger_collection_predicates = late_gate_predicates
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
