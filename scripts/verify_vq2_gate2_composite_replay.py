#!/usr/bin/env python3
"""Source-lock the two-Puffer VQ2 composite against the official N295 trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    import policy_callable_vq2_gate2_composite as composite
    from policy_callable_checkpoint import CheckpointPolicy
except ModuleNotFoundError:  # Imported as scripts.verify_vq2_gate2_composite_replay.
    from scripts import policy_callable_vq2_gate2_composite as composite
    from scripts.policy_callable_checkpoint import CheckpointPolicy


OBSERVATIONS = 32
ACTIONS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_prefix(report_path: Path) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    samples = report.get("policy_trace", {}).get("samples", [])
    if not samples:
        raise ValueError("official report contains no policy_trace.samples")
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        observation = np.asarray(sample.get("observation"), dtype=np.float32)
        action = np.asarray(sample.get("normalized_action"), dtype=np.float32)
        if observation.shape != (OBSERVATIONS,):
            raise ValueError(f"prefix sample {index} has invalid observation width")
        if action.shape != (ACTIONS,):
            raise ValueError(f"prefix sample {index} has invalid action width")
        if int(sample.get("official_active_gate_index", -1)) != 0:
            raise ValueError("official prefix extends beyond Gate 1")
        observations.append(observation)
        actions.append(action)
    return np.stack(observations), np.stack(actions)


def load_first_gate_observation(report_path: Path, gate_index: int) -> np.ndarray:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for sample_index, sample in enumerate(
        report.get("policy_trace", {}).get("samples", [])
    ):
        if int(sample.get("official_active_gate_index", -1)) != gate_index:
            continue
        observation = np.asarray(sample.get("observation"), dtype=np.float32)
        if observation.shape != (OBSERVATIONS,):
            raise ValueError(
                f"transition sample {sample_index} has invalid observation width")
        return observation
    raise ValueError(f"transition report has no official gate index {gate_index}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix_checkpoint", type=Path)
    parser.add_argument("gate2_checkpoint", type=Path)
    parser.add_argument("official_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--transition-report", type=Path)
    parser.add_argument("--runtime-duration-seconds", type=float, default=0.0)
    parser.add_argument("--gate2-time-limit-seconds", type=float, default=0.0)
    parser.add_argument("--world-frame-gate-features", action="store_true")
    parser.add_argument("--linear-gate-features", action="store_true")
    parser.add_argument("--predict-gate-kinematics", action="store_true")
    parser.add_argument(
        "--initial-gate-rate-world", type=float, nargs=3,
        default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--initial-gate-position-world-scale", type=float, nargs=3,
        default=(1.0, 1.0, 1.0), metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--initial-gate-position-world", type=float, nargs=3,
        metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--association-jump-threshold-m", type=float, default=0.0)
    parser.add_argument("--predictor-dt-seconds", type=float, default=1.0 / 60.0)
    args = parser.parse_args()

    os.environ["PUFFER_POLICY_PREFIX_CHECKPOINT_PATH"] = str(
        args.prefix_checkpoint.resolve())
    os.environ["PUFFER_POLICY_GATE2_CHECKPOINT_PATH"] = str(
        args.gate2_checkpoint.resolve())
    os.environ["PUFFER_POLICY_INPUT_DIM"] = "32"
    os.environ["PUFFER_POLICY_LAYOUT_PRECISION_BYTES"] = "4"
    os.environ["PUFFER_POLICY_NATIVE_BF16"] = "0"
    os.environ["PUFFER_POLICY_RACE_PHASE_DENOMINATOR"] = "6"
    os.environ["PUFFER_POLICY_GATE2_FIXED_PREFIX_REPORT_PATH"] = str(
        args.official_report.resolve())
    os.environ["PUFFER_POLICY_RUNTIME_DURATION_SECONDS"] = str(
        args.runtime_duration_seconds)
    os.environ["PUFFER_POLICY_GATE2_TIME_LIMIT_SECONDS"] = str(
        args.gate2_time_limit_seconds)
    os.environ["PUFFER_POLICY_GATE2_WORLD_FRAME_FEATURES"] = str(
        int(args.world_frame_gate_features))
    os.environ["PUFFER_POLICY_GATE2_LINEAR_GATE_FEATURES"] = str(
        int(args.linear_gate_features))
    os.environ["PUFFER_POLICY_GATE2_PREDICT_GATE_KINEMATICS"] = str(
        int(args.predict_gate_kinematics))
    os.environ["PUFFER_POLICY_GATE2_INITIAL_FORWARD_GATE_RATE_M_S"] = str(
        args.initial_gate_rate_world[0])
    os.environ["PUFFER_POLICY_GATE2_INITIAL_GATE_RATE_Y_M_S"] = str(
        args.initial_gate_rate_world[1])
    os.environ["PUFFER_POLICY_GATE2_INITIAL_GATE_RATE_Z_M_S"] = str(
        args.initial_gate_rate_world[2])
    os.environ["PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_SCALE_X"] = str(
        args.initial_gate_position_world_scale[0])
    os.environ["PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_SCALE_Y"] = str(
        args.initial_gate_position_world_scale[1])
    os.environ["PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_SCALE_Z"] = str(
        args.initial_gate_position_world_scale[2])
    if args.initial_gate_position_world is not None:
        for axis, value in zip(
            ("X", "Y", "Z"), args.initial_gate_position_world
        ):
            os.environ[
                f"PUFFER_POLICY_GATE2_INITIAL_GATE_POSITION_WORLD_{axis}"
            ] = str(value)
    os.environ["PUFFER_POLICY_GATE2_ASSOCIATION_JUMP_THRESHOLD_M"] = str(
        args.association_jump_threshold_m)
    os.environ["PUFFER_POLICY_GATE2_FORWARD_PREDICTOR_DT_SECONDS"] = str(
        args.predictor_dt_seconds)

    observations, recorded_actions = load_policy_prefix(args.official_report)
    composite.reset()
    emitted_actions = np.asarray(
        [composite.infer(row) for row in observations], dtype=np.float32)
    prefix_max_action_error = float(np.max(np.abs(
        emitted_actions - recorded_actions)))

    if args.transition_report is None:
        transition = observations[-1].copy()
        transition[19:23] = recorded_actions[-1]
        transition[23] = np.float32(1.0 / 6.0)
        transition[24:30] = 0.0
        transition[25] = 1.0
    else:
        transition = load_first_gate_observation(args.transition_report, 1)
    composite_action = np.asarray(composite.infer(transition), dtype=np.float32)

    gate2 = CheckpointPolicy.load(
        str(args.gate2_checkpoint), input_dim=32, num_layers=3,
        layout_precision_bytes=4)
    gate2.reset_state()
    for row in observations:
        gate2.infer(row)
    manual_state = composite._CompositeState(
        prefix=gate2,
        gate2=gate2,
        runtime_duration_s=args.runtime_duration_seconds,
        gate2_time_limit_s=args.gate2_time_limit_seconds,
        world_frame_gate_features=args.world_frame_gate_features,
        linear_gate_features=args.linear_gate_features,
        predict_gate_kinematics=args.predict_gate_kinematics,
        initial_gate_rate_world=tuple(args.initial_gate_rate_world),
        initial_gate_position_world_scale=tuple(
            args.initial_gate_position_world_scale),
        initial_gate_position_world=(
            None
            if args.initial_gate_position_world is None
            else tuple(args.initial_gate_position_world)
        ),
        gate_rate_world=tuple(args.initial_gate_rate_world),
        forward_predictor_dt_s=args.predictor_dt_seconds,
    )
    standalone_observation = transition.tolist()
    if args.predict_gate_kinematics:
        standalone_observation = composite._apply_gate_kinematic_predictor(
            manual_state, standalone_observation)
    standalone_observation = composite._gate2_observation(
        standalone_observation,
        runtime_duration_s=args.runtime_duration_seconds,
        gate2_time_limit_s=args.gate2_time_limit_seconds,
        world_frame_gate_features=args.world_frame_gate_features,
        linear_gate_features=args.linear_gate_features,
    )
    standalone_action = np.asarray(
        gate2.infer(standalone_observation), dtype=np.float32)
    gate2_selector_max_action_error = float(np.max(np.abs(
        composite_action - standalone_action)))

    callable_path = Path(composite.__file__).resolve()
    passed = (
        prefix_max_action_error <= 1e-6
        and gate2_selector_max_action_error == 0.0
        and np.isfinite(composite_action).all()
    )
    report = {
        "passed": bool(passed),
        "runtime_controller": "two_recurrent_puffer_checkpoints",
        "action_semantics": "whole_vector_selected_by_quantized_official_progress",
        "official_report": str(args.official_report),
        "official_report_sha256": sha256_file(args.official_report),
        "transition_report": (
            None if args.transition_report is None else str(args.transition_report)),
        "transition_report_sha256": (
            None
            if args.transition_report is None
            else sha256_file(args.transition_report)
        ),
        "prefix_checkpoint": str(args.prefix_checkpoint),
        "prefix_checkpoint_sha256": sha256_file(args.prefix_checkpoint),
        "gate2_checkpoint": str(args.gate2_checkpoint),
        "gate2_checkpoint_sha256": sha256_file(args.gate2_checkpoint),
        "callable": str(callable_path),
        "callable_sha256": sha256_file(callable_path),
        "prefix_samples": int(len(observations)),
        "prefix_max_action_error": prefix_max_action_error,
        "gate2_selector_max_action_error": gate2_selector_max_action_error,
        "gate2_transition_action": composite_action.tolist(),
        "runtime_duration_seconds": args.runtime_duration_seconds,
        "gate2_time_limit_seconds": args.gate2_time_limit_seconds,
        "world_frame_gate_features": args.world_frame_gate_features,
        "linear_gate_features": args.linear_gate_features,
        "predict_gate_kinematics": args.predict_gate_kinematics,
        "initial_gate_rate_world": list(args.initial_gate_rate_world),
        "initial_gate_position_world_scale": list(
            args.initial_gate_position_world_scale),
        "initial_gate_position_world": (
            None
            if args.initial_gate_position_world is None
            else list(args.initial_gate_position_world)
        ),
        "association_jump_threshold_m": args.association_jump_threshold_m,
        "predictor_dt_seconds": args.predictor_dt_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
