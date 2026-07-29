#!/usr/bin/env python3
"""Verify a passive Windows live-policy shadow trace and Linux replay parity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

try:
    from policy_callable_checkpoint import CheckpointPolicy
except ModuleNotFoundError:
    from scripts.policy_callable_checkpoint import CheckpointPolicy


ROOT = Path(__file__).resolve().parents[1]
LOCAL_MANIFEST_PATHS = {
    "runner": ROOT / "scripts" / "drone_sitl_competition_smoke.py",
    "policy_callable": ROOT / "scripts" / "policy_callable_checkpoint.py",
    "policy_contract": ROOT / "scripts" / "drone_policy_contract.py",
    "gate_detector": ROOT / "scripts" / "drone_gate_detector.py",
    "camera_receiver": ROOT / "scripts" / "drone_camera_receiver.py",
    "sitl_adapter": ROOT / "scripts" / "drone_sitl_adapter.py",
    "state_estimator": ROOT / "scripts" / "drone_state_estimator.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_shadow_report(
    report: dict[str, Any],
    checkpoint: Path,
    *,
    action_atol: float = 1e-4,
    model=None,
    policy_callable_path: Path | None = None,
    additional_checkpoints: dict[str, Path] | None = None,
    hybrid_prefix_confidence: bool = False,
    min_inference_hz: float = 0.0,
) -> dict[str, Any]:
    blockers: list[str] = []
    control = report.get("control_inputs", {})
    sitl = report.get("sitl", {})
    trace = report.get("policy_trace", {})
    samples = trace.get("samples", [])
    reset = control.get("official_reset_start", {})

    if report.get("control_mode") != "policy-attitude":
        blockers.append("control_mode_not_policy_attitude")
    if control.get("policy_shadow_only") is not True:
        blockers.append("policy_shadow_only_not_true")
    if control.get("arm_on_start") is not False:
        blockers.append("shadow_arm_on_start_not_false")
    if int(control.get("arm_commands_sent", -1)) != 0:
        blockers.append("shadow_arm_commands_nonzero")
    if int(control.get("disarm_commands_sent", -1)) != 0:
        blockers.append("shadow_disarm_commands_nonzero")
    if reset.get("reset_sent") is not False:
        blockers.append("shadow_reset_was_sent")
    if int(sitl.get("commands_sent", -1)) != 0:
        blockers.append("shadow_control_commands_nonzero")
    if sitl.get("command_kind") != "shadow_policy_attitude_no_setpoint":
        blockers.append("shadow_command_kind_mismatch")
    if not report.get("acceptance_passed", False):
        blockers.append("windows_shadow_acceptance_failed")
    if int(sitl.get("telemetry", {}).get("messages_seen", 0)) <= 0:
        blockers.append("no_live_telemetry")
    if int(sitl.get("telemetry", {}).get("race_statuses", 0)) <= 0:
        blockers.append("no_live_race_status")
    if int(report.get("vision", {}).get("frames_seen", 0)) <= 0:
        blockers.append("no_live_camera_frames")
    if int(report.get("vision", {}).get("detector_detections", 0)) <= 0:
        blockers.append("no_live_gate_detections")
    if not control.get("state_estimator", {}).get("calibration"):
        blockers.append("missing_passive_imu_calibration")
    cadence = control.get("policy_shadow_cadence_probe", {})
    if int(cadence.get("mavlink_setpoints_sent", -1)) != 0:
        blockers.append("shadow_cadence_probe_sent_setpoints")
    if int(cadence.get("commands_counted", 0)) <= 0:
        blockers.append("shadow_cadence_probe_missing")
    cadence_hz = float(cadence.get("effective_command_hz", 0.0))
    if not 50.0 <= cadence_hz < 100.0:
        blockers.append(f"shadow_cadence_out_of_bounds:{cadence_hz:.3f}")
    if int(cadence.get("command_rate_violations", -1)) != 0:
        blockers.append("shadow_cadence_rate_violation")

    inference_ticks = int(trace.get("inference_ticks", -1))
    if inference_ticks <= 0:
        blockers.append("no_policy_inference_ticks")
    if inference_ticks != len(samples):
        blockers.append(
            f"incomplete_policy_trace:{len(samples)}!={inference_ticks}"
        )
    inference_duration_s = float(sitl.get("duration_s", 0.0))
    inference_hz = (
        inference_ticks / inference_duration_s
        if inference_duration_s > 0.0 and inference_ticks >= 0
        else 0.0
    )
    if inference_hz < min_inference_hz:
        blockers.append(
            f"policy_inference_rate_too_low:{inference_hz:.3f}<"
            f"{min_inference_hz:.3f}"
        )

    phase_counts = [0] * 6
    visible_samples = 0
    for sample_index, sample in enumerate(samples):
        observation = sample.get("observation", [])
        gate_index = sample.get("official_active_gate_index")
        if len(observation) != 32:
            blockers.append(f"observation_size_{sample_index}:{len(observation)}")
            continue
        if not all(math.isfinite(float(value)) for value in observation):
            blockers.append(f"nonfinite_observation_{sample_index}")
            continue
        if any(abs(float(value)) > 1.000001 for value in observation):
            blockers.append(f"out_of_bounds_observation_{sample_index}")
        if not isinstance(gate_index, int) or not 0 <= gate_index < 6:
            blockers.append(f"invalid_gate_index_{sample_index}:{gate_index}")
            continue
        phase_counts[gate_index] += 1
        if float(observation[10]) > 0.5:
            visible_samples += 1
        if abs(float(observation[23]) - gate_index / 6.0) > 1e-6:
            blockers.append(f"progress_abi_mismatch_{sample_index}")
        expected_flags = [1.0 if index == gate_index else 0.0 for index in range(6)]
        if any(
            abs(float(observation[24 + index]) - expected) > 1e-6
            for index, expected in enumerate(expected_flags)
        ):
            blockers.append(f"phase_abi_mismatch_{sample_index}")
        if hybrid_prefix_confidence:
            if not 0.0 <= float(observation[30]) <= 1.0:
                blockers.append(f"prefix_confidence_abi_mismatch_{sample_index}")
            if not 0.0 <= float(observation[31]) <= 1.0:
                blockers.append(f"prefix_elapsed_abi_mismatch_{sample_index}")
            if gate_index >= 3 and (
                abs(float(observation[30])) > 1e-7
                or abs(float(observation[31])) > 1e-7
            ):
                blockers.append(f"tail_reserved_abi_mismatch_{sample_index}")
        elif (
            abs(float(observation[30])) > 1e-7
            or abs(float(observation[31])) > 1e-7
        ):
            blockers.append(f"reserved_abi_mismatch_{sample_index}")
    if visible_samples <= 0:
        blockers.append("no_gate_visible_policy_observations")

    manifest = trace.get("deployment_manifest", {})
    manifest_matches: dict[str, bool] = {}
    local_manifest_paths = dict(LOCAL_MANIFEST_PATHS)
    if policy_callable_path is not None:
        local_manifest_paths["policy_callable"] = policy_callable_path
    for name, local_path in local_manifest_paths.items():
        windows_hash = manifest.get(name, {}).get("sha256")
        local_hash = sha256_file(local_path)
        matched = windows_hash == local_hash
        manifest_matches[name] = matched
        if not matched:
            blockers.append(f"deployment_hash_mismatch:{name}")
    checkpoint_hash = sha256_file(checkpoint)
    manifest_checkpoint_hash = manifest.get("checkpoint", {}).get("sha256")
    manifest_matches["checkpoint"] = manifest_checkpoint_hash == checkpoint_hash
    if manifest_checkpoint_hash != checkpoint_hash:
        blockers.append("deployment_hash_mismatch:checkpoint")
    for name, path in (additional_checkpoints or {}).items():
        local_hash = sha256_file(path)
        windows_hash = manifest.get(name, {}).get("sha256")
        matched = windows_hash == local_hash
        manifest_matches[name] = matched
        if not matched:
            blockers.append(f"deployment_hash_mismatch:{name}")

    if model is None:
        model = CheckpointPolicy.load(
            str(checkpoint),
            input_dim=32,
            native_bf16=False,
            layout_precision_bytes=4,
        )
    reset_model = getattr(model, "reset_state", None)
    if callable(reset_model):
        reset_model()
    max_action_error = 0.0
    max_action_error_sample = None
    replayed_samples = 0
    for sample_index, sample in enumerate(samples):
        observation = sample.get("observation", [])
        expected = sample.get("normalized_action", [])
        if len(observation) != 32 or len(expected) != 4:
            continue
        actual = model.infer(observation)
        error = max(abs(float(a) - float(b)) for a, b in zip(actual, expected))
        if error > max_action_error:
            max_action_error = error
            max_action_error_sample = sample_index
        replayed_samples += 1
    if replayed_samples != len(samples):
        blockers.append(f"replay_sample_count:{replayed_samples}!={len(samples)}")
    if max_action_error > action_atol:
        blockers.append(
            f"action_replay_error:{max_action_error:.9g}>{action_atol:.9g}"
        )

    return {
        "passed": not blockers,
        "blockers": blockers,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "action_atol": action_atol,
        "max_action_error": max_action_error,
        "max_action_error_sample": max_action_error_sample,
        "inference_ticks": inference_ticks,
        "inference_duration_s": inference_duration_s,
        "inference_hz": inference_hz,
        "min_inference_hz": min_inference_hz,
        "trace_samples": len(samples),
        "replayed_samples": replayed_samples,
        "visible_samples": visible_samples,
        "phase_counts": phase_counts,
        "shadow_cadence_probe": cadence,
        "deployment_manifest_matches": manifest_matches,
    }


class _CallableModel:
    def __init__(self, module) -> None:
        self.module = module

    def reset_state(self) -> None:
        self.module.reset()

    def infer(self, observation):
        return self.module.infer(observation)


def _load_callable_model(path: Path) -> _CallableModel:
    resolved = path.resolve()
    spec = importlib.util.spec_from_file_location(
        f"_shadow_policy_callable_{resolved.stem}", resolved
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load policy callable: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "infer", None)):
        raise ValueError(f"policy callable has no infer function: {resolved}")
    if not callable(getattr(module, "reset", None)):
        raise ValueError(f"policy callable has no reset function: {resolved}")
    return _CallableModel(module)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gate2-checkpoint", type=Path)
    parser.add_argument("--gate5-checkpoint", type=Path)
    parser.add_argument("--gate6-checkpoint", type=Path)
    parser.add_argument("--gate4-checkpoint", type=Path)
    parser.add_argument(
        "--hybrid-prefix-confidence", action="store_true"
    )
    parser.add_argument("--policy-callable", type=Path)
    parser.add_argument("--json-path", type=Path)
    parser.add_argument("--action-atol", type=float, default=1e-4)
    parser.add_argument("--min-inference-hz", type=float, default=0.0)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    composite_requested = bool(
        args.gate4_checkpoint or args.gate5_checkpoint or args.gate6_checkpoint
    )
    vq2_composite_requested = args.gate2_checkpoint is not None
    if composite_requested and vq2_composite_requested:
        parser.error("VQ2 Gate-2 and six-gate composite verification are exclusive")
    if composite_requested and not (
        args.gate5_checkpoint and args.gate6_checkpoint and args.policy_callable
    ):
        parser.error(
            "composite verification requires --gate5-checkpoint, "
            "--gate6-checkpoint, and --policy-callable"
        )
    model = None
    additional_checkpoints = None
    if vq2_composite_requested:
        if args.policy_callable is None:
            parser.error("VQ2 Gate-2 composite verification requires --policy-callable")
        os.environ["PUFFER_POLICY_CHECKPOINT_PATH"] = str(args.checkpoint.resolve())
        os.environ["PUFFER_POLICY_PREFIX_CHECKPOINT_PATH"] = str(
            args.checkpoint.resolve()
        )
        os.environ["PUFFER_POLICY_GATE2_CHECKPOINT_PATH"] = str(
            args.gate2_checkpoint.resolve()
        )
        os.environ["PUFFER_POLICY_INPUT_DIM"] = "32"
        os.environ["PUFFER_POLICY_LAYOUT_PRECISION_BYTES"] = "4"
        os.environ["PUFFER_POLICY_NATIVE_BF16"] = "0"
        os.environ["PUFFER_POLICY_RACE_PHASE_DENOMINATOR"] = "6"
        model = _load_callable_model(args.policy_callable)
        additional_checkpoints = {
            "gate2_checkpoint": args.gate2_checkpoint,
        }
    elif composite_requested:
        os.environ["PUFFER_POLICY_CHECKPOINT_PATH"] = str(args.checkpoint.resolve())
        gate4_checkpoint = args.gate4_checkpoint or args.checkpoint
        if args.hybrid_prefix_confidence and args.gate4_checkpoint is None:
            parser.error(
                "hybrid verification requires --gate4-checkpoint"
            )
        if args.hybrid_prefix_confidence:
            os.environ["PUFFER_POLICY_PREFIX_CHECKPOINT_PATH"] = str(
                args.checkpoint.resolve()
            )
        os.environ["PUFFER_POLICY_GATE4_CHECKPOINT_PATH"] = str(
            gate4_checkpoint.resolve()
        )
        os.environ["PUFFER_POLICY_GATE5_CHECKPOINT_PATH"] = str(
            args.gate5_checkpoint.resolve()
        )
        os.environ["PUFFER_POLICY_GATE6_CHECKPOINT_PATH"] = str(
            args.gate6_checkpoint.resolve()
        )
        os.environ["PUFFER_POLICY_INPUT_DIM"] = "32"
        os.environ["PUFFER_POLICY_LAYOUT_PRECISION_BYTES"] = "4"
        os.environ["PUFFER_POLICY_NATIVE_BF16"] = "0"
        model = _load_callable_model(args.policy_callable)
        additional_checkpoints = {
            "gate4_checkpoint": gate4_checkpoint,
            "gate5_checkpoint": args.gate5_checkpoint,
            "gate6_checkpoint": args.gate6_checkpoint,
        }
    result = verify_shadow_report(
        report,
        args.checkpoint,
        action_atol=args.action_atol,
        model=model,
        policy_callable_path=args.policy_callable,
        additional_checkpoints=additional_checkpoints,
        hybrid_prefix_confidence=args.hybrid_prefix_confidence,
        min_inference_hz=args.min_inference_hz,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
