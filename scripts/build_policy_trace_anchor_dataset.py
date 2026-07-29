#!/usr/bin/env python3
"""Convert a sparse official policy trace into recurrent policy anchors.

The official report samples policy observations below command rate.  This tool
linearly fills those intervals at the requested rate, advances the proven base
MinGRU with its own preceding yaw action, and labels every row with either the
base decoder action or a bounded observable-feedback target.  The resulting
ordinary BC dataset constrains a new gate head only in feature directions
actually observed in the official simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy
from train_full_policy_bc import ACTIONS, OBSERVATIONS, RECORD_WIDTH


BASE_OBSERVATIONS = 23


def _inverse_tanh_norm(value: float, scale: float) -> float:
    return float(np.arctanh(np.clip(value, -0.999999, 0.999999)) * scale)


def observable_feedback_action(
    observation: np.ndarray,
    *,
    pitch_scale: float = 1.0,
    roll_scale: float = 1.0,
    thrust_scale: float = 1.0,
    thrust_offset: float = 0.0,
) -> np.ndarray:
    """Reproduce the native teacher using only the deployed observation.

    Gate-motion observations are relative rates, so vehicle velocity has the
    opposite sign for the forward and right axes.  The down-axis convention is
    already aligned with vehicle z velocity in the policy contract.
    """
    if observation.shape not in ((BASE_OBSERVATIONS,), (OBSERVATIONS,)):
        raise ValueError(f"observation has shape {observation.shape}")

    gate_forward_rate = _inverse_tanh_norm(float(observation[0]), 5.0)
    gate_right_rate = _inverse_tanh_norm(float(observation[1]), 3.0)
    gate_down_rate = _inverse_tanh_norm(float(observation[2]), 3.0)
    gate_right = _inverse_tanh_norm(float(observation[12]), 5.0)
    gate_down = _inverse_tanh_norm(float(observation[13]), 5.0)

    forward_velocity = -gate_forward_rate
    right_velocity = -gate_right_rate
    pitch = 0.24 * np.clip(
        -(1.5 - forward_velocity) * 0.35, -0.60, 0.60
    )
    roll = -0.40 * np.clip(
        gate_right * 0.08 - right_velocity * 0.10, -0.50, 0.50
    )
    thrust = np.clip(
        -gate_down * 0.08 - gate_down_rate * 0.25, -0.80, 0.60
    )
    return np.asarray(
        [
            np.clip(pitch * pitch_scale, -1.0, 1.0),
            np.clip(roll * roll_scale, -1.0, 1.0),
            np.clip(thrust * thrust_scale + thrust_offset, -1.0, 1.0),
            0.0,
        ],
        dtype=np.float32,
    )


def expand_gate_phase_onehot_observation(
    observation: np.ndarray,
    *,
    official_gate_index: int,
    denominator: int = 6,
) -> np.ndarray:
    """Expand the legacy 23-value live trace into the six-gate 32-value ABI."""
    if observation.shape not in ((BASE_OBSERVATIONS,), (OBSERVATIONS,)):
        raise ValueError(f"trace observation has shape {observation.shape}")
    if denominator != 6:
        raise ValueError("gate-phase one-hot expansion requires denominator 6")
    if not 0 <= official_gate_index < denominator:
        raise ValueError(
            f"official_gate_index must be in [0, {denominator - 1}]"
        )
    expanded = np.zeros(OBSERVATIONS, dtype=np.float32)
    expanded[:BASE_OBSERVATIONS] = observation[:BASE_OBSERVATIONS]
    expanded[23] = np.float32(official_gate_index / denominator)
    expanded[24 + official_gate_index] = np.float32(1.0)
    return expanded


def interpolate_recorded_action(
    current: np.ndarray,
    following: np.ndarray,
    *,
    fraction: float,
    same_phase: bool,
) -> np.ndarray:
    """Interpolate sparse teacher actions without smearing a gate transition."""
    if current.shape != (ACTIONS,) or following.shape != (ACTIONS,):
        raise ValueError("recorded action must contain four values")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if not same_phase:
        return current.copy()
    return current + np.float32(fraction) * (following - current)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--command-hz", type=float, default=60.0)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument(
        "--label-mode",
        choices=("source", "observable-feedback", "recorded"),
        default="source",
        help=(
            "Use source-policy actions, recorded live-teacher actions, or bounded "
            "feedback labels computed only from the recorded policy observation."
        ),
    )
    parser.add_argument("--checkpoint-input-dim", type=int, default=32)
    parser.add_argument(
        "--checkpoint-layout-precision-bytes", type=int, choices=(2, 4), default=4
    )
    parser.add_argument(
        "--expand-gate-phase-onehot-denominator",
        type=int,
        default=0,
        help=(
            "Expand a legacy 23-value live trace into the 32-value ABI using "
            "official gate progress and one-hot phase. Only denominator 6 is legal."
        ),
    )
    parser.add_argument("--feedback-pitch-scale", type=float, default=1.0)
    parser.add_argument("--feedback-roll-scale", type=float, default=1.0)
    parser.add_argument("--feedback-thrust-scale", type=float, default=1.0)
    parser.add_argument("--feedback-thrust-offset", type=float, default=0.0)
    parser.add_argument(
        "--override-race-phase-denominator",
        type=int,
        default=0,
        help="Replace observation[22] from official_active_gate_index when positive.",
    )
    args = parser.parse_args()
    if args.command_hz <= 0.0 or args.copies < 1:
        parser.error("command-hz and copies must be positive")

    report = json.loads(args.trace.read_text(encoding="utf-8"))
    samples = sorted(
        report.get("policy_trace", {}).get("samples", []),
        key=lambda sample: float(sample["elapsed_s"]),
    )
    if not samples:
        raise ValueError("report contains no policy trace samples")

    source = (
        CheckpointPolicy.load(
            str(args.checkpoint),
            input_dim=args.checkpoint_input_dim,
            layout_precision_bytes=args.checkpoint_layout_precision_bytes,
        )
        if args.label_mode == "source"
        else None
    )
    episodes: list[np.ndarray] = []
    phase_counts: dict[int, int] = {}
    for _copy in range(args.copies):
        if source is not None:
            source.reset_state()
        last_yaw = 0.0
        rows: list[np.ndarray] = []
        for index, sample in enumerate(samples):
            current_gate = int(sample.get("official_active_gate_index") or 0)
            current_raw = np.asarray(sample["observation"], dtype=np.float32)
            current = (
                expand_gate_phase_onehot_observation(
                    current_raw,
                    official_gate_index=current_gate,
                    denominator=args.expand_gate_phase_onehot_denominator,
                )
                if args.expand_gate_phase_onehot_denominator > 0
                else current_raw
            )
            if current.shape != (OBSERVATIONS,):
                raise ValueError(
                    "trace observation must already contain 32 values or use "
                    "--expand-gate-phase-onehot-denominator 6"
                )
            current_recorded_action = np.asarray(
                sample.get("normalized_action", []), dtype=np.float32
            )
            if index + 1 < len(samples):
                following_sample = samples[index + 1]
                following_gate = int(
                    following_sample.get("official_active_gate_index") or 0
                )
                following_raw = np.asarray(
                    samples[index + 1]["observation"], dtype=np.float32
                )
                following = (
                    expand_gate_phase_onehot_observation(
                        following_raw,
                        official_gate_index=following_gate,
                        denominator=args.expand_gate_phase_onehot_denominator,
                    )
                    if args.expand_gate_phase_onehot_denominator > 0
                    else following_raw
                )
                following_recorded_action = np.asarray(
                    following_sample.get("normalized_action", []), dtype=np.float32
                )
                interval = max(
                    0.0,
                    float(samples[index + 1]["elapsed_s"])
                    - float(sample["elapsed_s"]),
                )
                steps = max(1, int(round(interval * args.command_hz)))
            else:
                following = current
                following_gate = current_gate
                following_recorded_action = current_recorded_action
                steps = 1
            for step in range(steps):
                fraction = step / max(steps, 1)
                observation = current + np.float32(fraction) * (following - current)
                if args.expand_gate_phase_onehot_denominator > 0:
                    # Gate transitions are discrete official events, never an
                    # interpolated fractional identity.
                    observation[23:32] = current[23:32]
                elif args.override_race_phase_denominator > 0:
                    observation[22] = np.float32(
                        min(max(current_gate, 0), args.override_race_phase_denominator)
                        / args.override_race_phase_denominator
                    )
                recurrent_observation = observation.copy()
                recurrent_observation[22] = np.float32(last_yaw)
                if args.label_mode == "source":
                    if source is None:
                        raise RuntimeError("source checkpoint was not loaded")
                    action = np.asarray(
                        source.infer(recurrent_observation), dtype=np.float32
                    )
                elif args.label_mode == "recorded":
                    action = interpolate_recorded_action(
                        current_recorded_action,
                        following_recorded_action,
                        fraction=fraction,
                        same_phase=current_gate == following_gate,
                    )
                else:
                    action = observable_feedback_action(
                        observation,
                        pitch_scale=args.feedback_pitch_scale,
                        roll_scale=args.feedback_roll_scale,
                        thrust_scale=args.feedback_thrust_scale,
                        thrust_offset=args.feedback_thrust_offset,
                    )
                last_yaw = float(action[3])
                row = np.zeros(RECORD_WIDTH, dtype=np.float32)
                row[:OBSERVATIONS] = observation
                row[OBSERVATIONS:OBSERVATIONS + ACTIONS] = action
                row[-1] = 1.0 if not rows else 0.0
                rows.append(row)
                phase_counts[current_gate] = phase_counts.get(current_gate, 0) + 1
        episodes.append(np.stack(rows))

    output = np.concatenate(episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(args.output)
    print(
        f"wrote={args.output} copies={args.copies} records={len(output)} "
        f"label_mode={args.label_mode} phase_counts={phase_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
