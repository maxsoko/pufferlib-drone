#!/usr/bin/env python3
"""Separate Gate-1 misses from the N144/N145 roll-floor adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import policy_callable_six_gate_composite as composite
    from policy_callable_checkpoint import CheckpointPolicy
    from verify_gate1_promoted_adapter_traces import _convert_observation
except ModuleNotFoundError:
    from scripts import policy_callable_six_gate_composite as composite
    from scripts.policy_callable_checkpoint import CheckpointPolicy
    from scripts.verify_gate1_promoted_adapter_traces import _convert_observation


POSITIVE_MODES = frozenset({"adaptive", "early", "close"})
SEVERE_DIRECT_RIGHT_M = composite.GATE1_SEVERE_DIRECT_RIGHT_M
ACTION_ATOL = 1e-4


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direct_right_m(observation: np.ndarray) -> float:
    return composite._inverse_tanh_norm(float(observation[12]), 5.0)


def severe_direct_positive_intercept(
    observation: np.ndarray,
    actions: list[float],
) -> tuple[list[float], str, float | None, float]:
    """Apply a positive floor only to an already-severe direct left miss."""

    mode, roll_command = composite._promoted_gate3_roll_command(observation)
    direct_right_m = _direct_right_m(observation)
    if (
        mode in POSITIVE_MODES
        and roll_command is not None
        and direct_right_m < SEVERE_DIRECT_RIGHT_M
    ):
        actions[1] = max(actions[1], roll_command)
    return actions, mode, roll_command, direct_right_m


def _severe_direct_eligible(
    mode: str,
    roll_command: float | None,
    direct_right_m: float,
) -> bool:
    return (
        mode in POSITIVE_MODES
        and roll_command is not None
        and direct_right_m < SEVERE_DIRECT_RIGHT_M
    )


def _n144_positive_intercept(
    observation: np.ndarray,
    actions: list[float],
) -> list[float]:
    """Reproduce the rejected N144 law independently of the deployed helper."""

    mode, roll_command = composite._promoted_gate3_roll_command(observation)
    if mode in POSITIVE_MODES:
        assert roll_command is not None
        actions[1] = max(actions[1], roll_command)
    return actions


def _source(path: Path, report: dict) -> dict:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "acceptance_passed": bool(report.get("acceptance_passed")),
        "official_active_gate_index": report.get("official_active_gate_index"),
    }


def _project_recorded(paths: Iterable[Path]) -> dict:
    changes: list[dict] = []
    sources: list[dict] = []
    modes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    samples = 0
    eligible_samples: list[dict] = []
    non_roll_max_error = 0.0
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        sources.append(_source(path, report))
        for sample_index, sample in enumerate(
            (report.get("policy_trace") or {}).get("samples") or []
        ):
            gate = int(sample.get("official_active_gate_index", -1))
            if gate != 0:
                continue
            recorded = np.asarray(sample.get("normalized_action", []), dtype=float)
            if recorded.shape != (4,):
                raise ValueError(f"{path}: sample {sample_index}: expected four actions")
            observation, trace_format = _convert_observation(
                sample.get("observation", []), gate
            )
            proposed, mode, roll_floor, direct_right_m = (
                severe_direct_positive_intercept(observation, recorded.tolist())
            )
            proposed_array = np.asarray(proposed, dtype=float)
            error = np.abs(proposed_array - recorded)
            modes[mode] += 1
            formats[trace_format] += 1
            samples += 1
            non_roll_max_error = max(
                non_roll_max_error, float(np.max(error[[0, 2, 3]]))
            )
            if _severe_direct_eligible(mode, roll_floor, direct_right_m):
                eligible_samples.append(
                    {
                        "path": str(path),
                        "sample": sample_index,
                        "elapsed_s": sample.get("elapsed_s"),
                        "mode": mode,
                        "direct_right_m": direct_right_m,
                        "recorded_roll": float(recorded[1]),
                        "roll_floor": roll_floor,
                    }
                )
            if float(np.max(error)) > 0.0:
                changes.append(
                    {
                        "path": str(path),
                        "sample": sample_index,
                        "elapsed_s": sample.get("elapsed_s"),
                        "mode": mode,
                        "direct_right_m": direct_right_m,
                        "recorded_roll": float(recorded[1]),
                        "proposed_roll": float(proposed_array[1]),
                        "roll_floor": roll_floor,
                    }
                )
    return {
        "reports": len(sources),
        "gate1_samples": samples,
        "mode_counts": dict(sorted(modes.items())),
        "trace_format_counts": dict(sorted(formats.items())),
        "proposal_changes": len(changes),
        "proposal_eligible_samples": len(eligible_samples),
        "proposal_non_roll_max_error": non_roll_max_error,
        "eligible_samples": eligible_samples,
        "changed_samples": changes,
        "sources": sources,
    }


def _replay_n144_failure(path: Path, checkpoint: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    trace = (report.get("policy_trace") or {}).get("samples") or []
    model = CheckpointPolicy.load(
        str(checkpoint),
        input_dim=32,
        native_bf16=False,
        layout_precision_bytes=4,
    )
    model.reset_state()
    samples: list[dict] = []
    base_change_count = 0
    n144_rounded_replay_mismatches = 0
    proposal_change_count = 0
    n144_non_roll_max_error = 0.0
    proposal_non_roll_max_error = 0.0
    boundary_distances: list[float] = []
    for sample_index, sample in enumerate(trace):
        if int(sample.get("official_active_gate_index", -1)) != 0:
            continue
        observation = np.asarray(sample.get("observation", []), dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action", []), dtype=float)
        if observation.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"{path}: sample {sample_index}: invalid trace shape")
        base = np.asarray(model.infer(observation), dtype=float)
        n144_rounded = np.asarray(
            _n144_positive_intercept(observation, base.tolist()),
            dtype=float,
        )
        proposed_values, mode, roll_floor, direct_right_m = (
            severe_direct_positive_intercept(observation, base.tolist())
        )
        proposed = np.asarray(proposed_values, dtype=float)
        recorded_base_error = np.abs(recorded - base)
        n144_recorded_error = np.abs(n144_rounded - recorded)
        proposal_base_error = np.abs(proposed - base)
        recorded_changed = float(np.max(recorded_base_error)) > ACTION_ATOL
        n144_mismatch = float(np.max(n144_recorded_error)) > ACTION_ATOL
        proposal_changed = float(np.max(proposal_base_error)) > 0.0
        base_change_count += int(recorded_changed)
        n144_rounded_replay_mismatches += int(n144_mismatch)
        proposal_change_count += int(proposal_changed)
        n144_non_roll_max_error = max(
            n144_non_roll_max_error,
            float(np.max(n144_recorded_error[[0, 2, 3]])),
        )
        proposal_non_roll_max_error = max(
            proposal_non_roll_max_error,
            float(np.max(proposal_base_error[[0, 2, 3]])),
        )
        if recorded_changed or n144_mismatch or proposal_changed:
            threshold_distance_m = abs(direct_right_m - (-0.3))
            if n144_mismatch:
                boundary_distances.append(threshold_distance_m)
            samples.append(
                {
                    "sample": sample_index,
                    "elapsed_s": sample.get("elapsed_s"),
                    "mode_from_rounded_trace": mode,
                    "direct_right_m": direct_right_m,
                    "old_threshold_distance_m": threshold_distance_m,
                    "base_roll": float(base[1]),
                    "recorded_n144_roll": float(recorded[1]),
                    "rounded_trace_n144_roll": float(n144_rounded[1]),
                    "proposed_roll": float(proposed[1]),
                    "roll_floor": roll_floor,
                    "recorded_changed_from_base": recorded_changed,
                    "n144_rounded_replay_mismatch": n144_mismatch,
                    "proposal_changed_from_base": proposal_changed,
                }
            )
    return {
        "source": _source(path, report),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "inference_ticks": int((report.get("policy_trace") or {}).get("inference_ticks", -1)),
        "trace_samples": len(trace),
        "recorded_n144_changes_from_base": base_change_count,
        "n144_rounded_replay_mismatches": n144_rounded_replay_mismatches,
        "minimum_old_threshold_distance_m": (
            min(boundary_distances) if boundary_distances else None
        ),
        "proposal_changes_from_base": proposal_change_count,
        "n144_non_roll_max_error": n144_non_roll_max_error,
        "proposal_non_roll_max_error": proposal_non_roll_max_error,
        "diagnostic_samples": samples,
    }


def _replay_n145_failure(path: Path, checkpoint: Path) -> dict:
    """Replay a deployed N145 failure against its recurrent base and law."""

    report = json.loads(path.read_text(encoding="utf-8"))
    trace = (report.get("policy_trace") or {}).get("samples") or []
    inference_ticks = int(
        (report.get("policy_trace") or {}).get("inference_ticks", -1)
    )
    sample_hz = float((report.get("policy_trace") or {}).get("sample_hz", 0.0))
    trace_complete = inference_ticks == len(trace)
    if not trace_complete:
        return {
            "source": _source(path, report),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "inference_ticks": inference_ticks,
            "trace_samples": len(trace),
            "sample_hz": sample_hz,
            "trace_complete": False,
            "base_replay_valid": False,
            "base_replay_reason": (
                "sampled recurrent trace omits hidden-state advances"
            ),
        }
    model = CheckpointPolicy.load(
        str(checkpoint),
        input_dim=32,
        native_bf16=False,
        layout_precision_bytes=4,
    )
    model.reset_state()
    diagnostic_samples: list[dict] = []
    recorded_changes_from_base = 0
    proposal_changes_from_base = 0
    proposal_recorded_mismatches = 0
    recorded_base_max_action_error = 0.0
    proposal_recorded_max_action_error = 0.0
    proposal_recorded_non_roll_max_error = 0.0
    first_recorded_change_from_base: dict | None = None
    for sample_index, sample in enumerate(trace):
        if int(sample.get("official_active_gate_index", -1)) != 0:
            continue
        observation = np.asarray(sample.get("observation", []), dtype=np.float32)
        recorded = np.asarray(sample.get("normalized_action", []), dtype=float)
        if observation.shape != (32,) or recorded.shape != (4,):
            raise ValueError(f"{path}: sample {sample_index}: invalid trace shape")
        base = np.asarray(model.infer(observation), dtype=float)
        proposed_values, mode, roll_floor, direct_right_m = (
            severe_direct_positive_intercept(observation, base.tolist())
        )
        proposed = np.asarray(proposed_values, dtype=float)
        recorded_base_error = np.abs(recorded - base)
        proposal_base_error = np.abs(proposed - base)
        proposal_recorded_error = np.abs(proposed - recorded)
        recorded_changed = float(np.max(recorded_base_error)) > ACTION_ATOL
        proposal_changed = float(np.max(proposal_base_error)) > 0.0
        proposal_mismatch = float(np.max(proposal_recorded_error)) > ACTION_ATOL
        recorded_changes_from_base += int(recorded_changed)
        proposal_changes_from_base += int(proposal_changed)
        proposal_recorded_mismatches += int(proposal_mismatch)
        recorded_base_max_action_error = max(
            recorded_base_max_action_error,
            float(np.max(recorded_base_error)),
        )
        proposal_recorded_max_action_error = max(
            proposal_recorded_max_action_error,
            float(np.max(proposal_recorded_error)),
        )
        proposal_recorded_non_roll_max_error = max(
            proposal_recorded_non_roll_max_error,
            float(np.max(proposal_recorded_error[[0, 2, 3]])),
        )
        if recorded_changed or proposal_changed or proposal_mismatch:
            diagnostic = {
                "sample": sample_index,
                "elapsed_s": sample.get("elapsed_s"),
                "mode_from_rounded_trace": mode,
                "direct_right_m": direct_right_m,
                "base_roll": float(base[1]),
                "recorded_n145_roll": float(recorded[1]),
                "proposed_n145_roll": float(proposed[1]),
                "roll_floor": roll_floor,
                "recorded_changed_from_base": recorded_changed,
                "proposal_changed_from_base": proposal_changed,
                "proposal_recorded_mismatch": proposal_mismatch,
                "recorded_base_max_action_error": float(
                    np.max(recorded_base_error)
                ),
                "proposal_recorded_max_action_error": float(
                    np.max(proposal_recorded_error)
                ),
            }
            diagnostic_samples.append(diagnostic)
            if recorded_changed and first_recorded_change_from_base is None:
                first_recorded_change_from_base = diagnostic
    return {
        "source": _source(path, report),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "inference_ticks": int(
            (report.get("policy_trace") or {}).get("inference_ticks", -1)
        ),
        "trace_samples": len(trace),
        "sample_hz": sample_hz,
        "trace_complete": True,
        "base_replay_valid": True,
        "recorded_n145_changes_from_base": recorded_changes_from_base,
        "proposal_changes_from_base": proposal_changes_from_base,
        "proposal_recorded_mismatches": proposal_recorded_mismatches,
        "recorded_base_max_action_error": recorded_base_max_action_error,
        "proposal_recorded_max_action_error": proposal_recorded_max_action_error,
        "proposal_recorded_non_roll_max_error": (
            proposal_recorded_non_roll_max_error
        ),
        "first_recorded_change_from_base": first_recorded_change_from_base,
        "diagnostic_samples": diagnostic_samples,
    }


def verify(
    accepted_paths: Iterable[Path],
    candidate005_path: Path,
    candidate006_path: Path,
    checkpoint: Path,
    candidate007_path: Path | None = None,
) -> dict:
    accepted = _project_recorded(accepted_paths)
    candidate005 = _project_recorded([candidate005_path])
    candidate006 = _replay_n144_failure(candidate006_path, checkpoint)
    candidate007 = (
        {
            "projection": _project_recorded([candidate007_path]),
            "base_replay": _replay_n145_failure(candidate007_path, checkpoint),
        }
        if candidate007_path is not None
        else None
    )
    blockers: list[str] = []
    accepted_progress_violations = [
        source
        for source in accepted["sources"]
        if int(source.get("official_active_gate_index") or 0) < 1
    ]
    if accepted_progress_violations:
        blockers.append("accepted_source_without_gate1_pass")
    if accepted["proposal_changes"] != 0:
        blockers.append("proposal_changes_accepted_trace")
    if candidate005["proposal_changes"] != 6:
        blockers.append("proposal_does_not_preserve_candidate005_target")
    if candidate005["proposal_non_roll_max_error"] != 0.0:
        blockers.append("proposal_changes_candidate005_non_roll")
    source005 = candidate005["sources"][0]
    if source005["acceptance_passed"] or int(
        source005.get("official_active_gate_index") or 0
    ) != 0:
        blockers.append("candidate005_contract_invalid")
    source006 = candidate006["source"]
    if source006["acceptance_passed"] or int(
        source006.get("official_active_gate_index") or 0
    ) != 0:
        blockers.append("candidate006_contract_invalid")
    if candidate006["inference_ticks"] != candidate006["trace_samples"]:
        blockers.append("candidate006_trace_not_complete")
    if candidate006["recorded_n144_changes_from_base"] <= 0:
        blockers.append("candidate006_n144_change_not_exercised")
    if candidate006["n144_rounded_replay_mismatches"] <= 0:
        blockers.append("old_threshold_boundary_not_exercised")
    boundary_distance = candidate006["minimum_old_threshold_distance_m"]
    if boundary_distance is None or boundary_distance > 1e-5:
        blockers.append("old_threshold_boundary_not_close")
    if candidate006["proposal_changes_from_base"] != 0:
        blockers.append("proposal_changes_candidate006_base")
    if candidate006["n144_non_roll_max_error"] > ACTION_ATOL:
        blockers.append("candidate006_non_roll_replay_error")
    if candidate006["proposal_non_roll_max_error"] != 0.0:
        blockers.append("proposal_changes_candidate006_non_roll")
    if candidate007 is not None:
        source007 = candidate007["projection"]["sources"][0]
        if source007["acceptance_passed"] or int(
            source007.get("official_active_gate_index") or 0
        ) != 0:
            blockers.append("candidate007_contract_invalid")
        if candidate007["projection"]["proposal_changes"] != 0:
            blockers.append("candidate007_deployed_law_not_idempotent")
        if candidate007["projection"]["proposal_non_roll_max_error"] != 0.0:
            blockers.append("candidate007_projection_changes_non_roll")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "contract": {
            "official_gate": 0,
            "positive_modes": sorted(POSITIVE_MODES),
            "severe_direct_right_m_exclusive_max": SEVERE_DIRECT_RIGHT_M,
            "expected_accepted_changes": 0,
            "expected_candidate005_changes": 6,
            "expected_candidate006_changes": 0,
            "candidate007_expected_projection_changes": 0,
            "action_atol": ACTION_ATOL,
        },
        "accepted": accepted,
        "candidate005": candidate005,
        "candidate006": candidate006,
        "candidate007": candidate007,
        "accepted_progress_violations": accepted_progress_violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate005", type=Path, required=True)
    parser.add_argument("--candidate006", type=Path, required=True)
    parser.add_argument("--candidate007", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--json-path", type=Path)
    args = parser.parse_args()
    result = verify(
        args.accepted,
        args.candidate005,
        args.candidate006,
        args.checkpoint,
        args.candidate007,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
