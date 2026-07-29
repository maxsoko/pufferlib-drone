#!/usr/bin/env python3
"""Evaluate the visual suffix from a warmed, measured N399 Gate-1 handoff.

This is an offline-only fixture. It replays the source-locked gain-10 visual
prefix to warm both recurrent Puffers, then starts the native visual plant from
the retained N399 Gate-2 transition state. No classical action reaches the
plant and no FlightSim packet is sent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.torch_pufferl import _cpu_tensor
from pufferlib.vq2_informed import ACTION_HISTORY, ENV_OBS_SIZE, LEGAL_OBS_SIZE, MASK_SIZE
from pufferlib.vq2_n294_visual_composite import (
    ACTION_SIZE,
    LegacyN294ObservationAdapter,
    select_complete_actions,
    visual_suffix_observation,
)
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.collect_vq2_public_phase_dagger import (
    PHASE_PRIVILEGED_INDEX,
    update_held_phase,
)
from scripts.eval_vq2_n294_visual_suffix_composite import (
    BACKEND_ENV_NAME,
    DT_SECONDS,
    N294_SHA256,
    SECOND_GATE_PHASE,
    TIME_LIMIT_SECONDS,
    composite_config,
    flatten_log,
    load_n294,
    load_suffix,
)
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state


TAG = "vq2_c003_measured_visual_handoff_exact128"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
PREFIX_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c002r_gain10p0_exact1/report.json"
)
PREFIX_REPORT_SHA256 = "deb0301428e7c781d5e981e298ed87d66f0aa7c65c5d12e99ec5cd9301055f6b"
PREFIX_TRACE = PREFIX_REPORT.parent / "trace_agent0.npz"
PREFIX_TRACE_SHA256 = "f33ab23c6977e6e3b94937bc2b4f41cc9e1adf34651e5f8b5373675f3d4557af"
N404_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_n404_n380_n399_contract_trace/report.json"
)
N404_REPORT_SHA256 = "364050aace586f3c154b712e3bc6df43073edb645aa57f5b57af6547b8d50312"
N404_TRACE = N404_REPORT.parent / "trace.npz"
N404_TRACE_SHA256 = "6f28c959c44efe6c0c7b15607a0d92f93c0c99896bdd297db6d44d4af828fd88"

MEASURED_START_ELAPSED_SECONDS = 3.281
MEASURED_START_VELOCITY = np.asarray([4.677, 0.006, 0.173], dtype=np.float32)
MEASURED_START_QUATERNION = np.asarray(
    [0.998520, 0.052695, -0.013450, 0.000931], dtype=np.float32
)
MEASURED_START_BODY_RATES = np.asarray([0.1226, -0.0006, 0.0], dtype=np.float32)
MEASURED_GATE2 = np.asarray([14.74, 8.70, 1.095], dtype=np.float32)
INERT_PHASE_SCALE_GATES = (
    (60.0, 8.70, 1.095),
    (90.0, 8.70, 1.095),
    (120.0, 8.70, 1.095),
    (150.0, 8.70, 1.095),
)
MEASURED_ALIAS_VECTOR = np.asarray([8.488131, 4.977876, -0.364377], dtype=np.float32)
MEASURED_ALIAS_ZOOM = float(
    np.linalg.norm(MEASURED_GATE2) / np.linalg.norm(MEASURED_ALIAS_VECTOR)
)
FIRST_GATE_PHASE = np.float32(1.0 / 6.0)
PHASE_EPSILON = np.float32(1e-6)

C007_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c007_measured_visual_suffix_dagger_fit_001/policy_best.pt"
)
C007_CHECKPOINT_SHA256 = (
    "393f5de6f9dd97ae81d855b23d77c96e97926507e7c2b89611714df3d60a24c5"
)
C007_REPORT = C007_CHECKPOINT.parent / "report.json"
C007_REPORT_SHA256 = (
    "bfc99b3d5177c5e83d9d46653480c26c66e05eebbc8d1d1bbc960bacd77c6e49"
)
C009_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c009_measured_visual_suffix_warmed_dagger_fit_001/policy_best.pt"
)
C009_CHECKPOINT_SHA256 = (
    "f037e56ec07ae134a87b3cf86bc9e1f3312d2310918cdf4f6fb84a9e96b259f2"
)
C009_REPORT = C009_CHECKPOINT.parent / "report.json"
C009_REPORT_SHA256 = (
    "296adbfdb8c6e2c8360e872a47f604b54ac7265f7628944c4d8c3b7b426baf34"
)
C012_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c012_c011_aggregate_warmed_dagger_fit_001/policy_best.pt"
)
C012_CHECKPOINT_SHA256 = (
    "2362ed2250ce41743d9153c64d7ccd3760171a468733c1b62cf34ebb2694f9f0"
)
C012_REPORT = C012_CHECKPOINT.parent / "report.json"
C012_REPORT_SHA256 = (
    "d7d6a66ea9f686a53d3516a11d8651b215805835327be8a79899ea455457d8f4"
)
CANDIDATE_ARTIFACTS = {
    "c007": (
        C007_CHECKPOINT,
        C007_CHECKPOINT_SHA256,
        C007_REPORT,
        C007_REPORT_SHA256,
    ),
    "c009": (
        C009_CHECKPOINT,
        C009_CHECKPOINT_SHA256,
        C009_REPORT,
        C009_REPORT_SHA256,
    ),
    "c012": (
        C012_CHECKPOINT,
        C012_CHECKPOINT_SHA256,
        C012_REPORT,
        C012_REPORT_SHA256,
    ),
}


def centered_mask_zoom(observations: np.ndarray, zoom: float) -> np.ndarray:
    """Zoom each legal 64x64 soft mask around its intensity centroid."""

    values = np.asarray(observations, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < MASK_SIZE:
        raise ValueError("observations must contain a batched 64x64 mask")
    if not np.isfinite(zoom) or zoom <= 0.0:
        raise ValueError("mask zoom must be finite and positive")
    result = values.copy()
    if abs(zoom - 1.0) <= 1e-8:
        return result
    masks = np.clip(values[:, :MASK_SIZE], 0.0, 1.0).reshape(-1, 64, 64)
    rows = np.arange(64, dtype=np.float32)[None, :, None]
    cols = np.arange(64, dtype=np.float32)[None, None, :]
    mass = masks.sum(axis=(1, 2))
    safe_mass = np.maximum(mass, np.float32(1e-8))
    center_row = (masks * rows).sum(axis=(1, 2)) / safe_mass
    center_col = (masks * cols).sum(axis=(1, 2)) / safe_mass
    empty = mass <= np.float32(1e-8)
    center_row[empty] = np.float32(31.5)
    center_col[empty] = np.float32(31.5)
    source_row = center_row[:, None, None] + (
        rows - center_row[:, None, None]
    ) / np.float32(zoom)
    source_col = center_col[:, None, None] + (
        cols - center_col[:, None, None]
    ) / np.float32(zoom)
    row0 = np.floor(source_row).astype(np.int32)
    col0 = np.floor(source_col).astype(np.int32)
    row1 = row0 + 1
    col1 = col0 + 1
    valid = (row0 >= 0) & (col0 >= 0) & (row1 < 64) & (col1 < 64)
    row0c = np.clip(row0, 0, 63)
    row1c = np.clip(row1, 0, 63)
    col0c = np.clip(col0, 0, 63)
    col1c = np.clip(col1, 0, 63)
    batch = np.arange(masks.shape[0])[:, None, None]
    wr = source_row - row0
    wc = source_col - col0
    sampled = (
        masks[batch, row0c, col0c] * (1.0 - wr) * (1.0 - wc)
        + masks[batch, row1c, col0c] * wr * (1.0 - wc)
        + masks[batch, row0c, col1c] * (1.0 - wr) * wc
        + masks[batch, row1c, col1c] * wr * wc
    )
    sampled *= valid
    result[:, :MASK_SIZE] = np.clip(sampled, 0.0, 1.0).reshape(-1, MASK_SIZE)
    return result


def load_warm_prefix() -> dict[str, np.ndarray]:
    if sha256_path(PREFIX_REPORT) != PREFIX_REPORT_SHA256:
        raise RuntimeError("C002R prefix report source-lock mismatch")
    if sha256_path(PREFIX_TRACE) != PREFIX_TRACE_SHA256:
        raise RuntimeError("C002R prefix trace source-lock mismatch")
    report = json.loads(PREFIX_REPORT.read_text())
    if not report.get("diagnostic_valid") or not report.get("harness_admitted"):
        raise RuntimeError("C002R prefix did not pass its legal harness checks")
    with np.load(PREFIX_TRACE) as payload:
        trace = {name: payload[name].copy() for name in payload.files}
    required = {
        "legacy_observation",
        "visual_observation",
        "n294_action",
        "suffix_action",
        "selected_action",
        "held_phase",
        "suffix_selected",
    }
    if not required.issubset(trace):
        raise RuntimeError("C002R trace ABI changed")
    prefix = np.asarray(trace["held_phase"] < FIRST_GATE_PHASE - PHASE_EPSILON)
    count = int(prefix.sum())
    if (
        prefix.ndim != 1
        or count <= 0
        or not np.all(prefix[:count])
        or prefix[count:].any()
    ):
        raise RuntimeError("C002R prefix is not a contiguous index-0 sequence")
    result = {name: values[:count] for name, values in trace.items()}
    if result["legacy_observation"].shape != (count, 32):
        raise RuntimeError("legacy prefix width changed")
    if result["visual_observation"].shape[1] != 4119:
        raise RuntimeError("visual prefix width changed")
    if np.asarray(result["suffix_selected"], dtype=bool).any():
        raise RuntimeError("suffix owned an index-0 prefix action")
    if np.max(np.abs(result["selected_action"] - result["n294_action"])) > 5e-5:
        raise RuntimeError("C002R prefix did not select whole N294 actions")
    return result


def load_measured_previous_action() -> np.ndarray:
    if sha256_path(N404_REPORT) != N404_REPORT_SHA256:
        raise RuntimeError("N404 report source-lock mismatch")
    if sha256_path(N404_TRACE) != N404_TRACE_SHA256:
        raise RuntimeError("N404 trace source-lock mismatch")
    report = json.loads(N404_REPORT.read_text())
    if float(report["metadata"]["start_elapsed_time"]) != MEASURED_START_ELAPSED_SECONDS:
        raise RuntimeError("N404 transition clock changed")
    with np.load(N404_TRACE) as payload:
        first = np.asarray(payload["observations"][0], dtype=np.float32)
    if first.shape != (32,):
        raise RuntimeError("N404 first observation ABI changed")
    return first[19:23].copy()


def load_candidate_suffix(
    device: torch.device, candidate: str
) -> tuple[VQ2PhaseResidualActor, dict[str, Any], tuple[Path, ...]]:
    """Load one source-locked visual suffix admitted for this fixture."""

    if candidate == "sf066":
        actor, payload = load_suffix(device)
        return actor, payload, ()
    if candidate not in CANDIDATE_ARTIFACTS:
        raise ValueError(f"unsupported suffix candidate: {candidate}")
    checkpoint, checkpoint_sha256, report_path, report_sha256 = (
        CANDIDATE_ARTIFACTS[candidate]
    )
    frozen = {
        checkpoint: checkpoint_sha256,
        report_path: report_sha256,
    }
    for path, expected in frozen.items():
        actual = sha256_path(path)
        if actual != expected:
            raise RuntimeError(f"source-lock mismatch for {path}: {actual}")
    report = json.loads(report_path.read_text())
    if not report.get("combined_numerical_admission"):
        raise RuntimeError(f"{candidate.upper()} did not pass combined numerical admission")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "vq2_public_phase_recurrent_checkpoint_v1":
        raise RuntimeError(f"unsupported {candidate.upper()} checkpoint schema")
    contract = payload.get("model", {})
    expected_contract = {
        "class": "VQ2PhaseResidualActor",
        "legal_observation_size": PHASE_LEGAL_OBS_SIZE,
        "legacy_legal_observation_size": LEGAL_OBS_SIZE,
        "public_status_values": 1,
        "action_size": ACTION_SIZE,
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise RuntimeError(f"{candidate.upper()} actor ABI changed")
    if payload.get("safety", {}).get(
        "stored_training_only_privileged_values_per_actor_record"
    ) != 0:
        raise RuntimeError(f"{candidate.upper()} actor records contain privileged values")
    actor = VQ2PhaseResidualActor(
        hidden_size=int(contract["hidden_size"]),
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(payload["model_state"])
    actor.eval()
    return actor, payload, tuple(frozen)


def measured_suffix_config(
    pufferl_module: Any,
    *,
    agents: int,
    seed: int,
    camera_pitch_rad: float,
) -> tuple[dict[str, Any], list[str]]:
    config, overrides = composite_config(
        pufferl_module,
        agents=agents,
        seed=seed,
        camera_pitch_rad=camera_pitch_rad,
    )
    environment = config["env"]
    environment.update(
        {
            # The privileged phase authority is current_gate / num_gates.
            # Keep six solely as the checkpoint's feature scale; the screen
            # still stops on the next ordered transition and never treats six
            # as an official VQ2 finish count.
            "num_gates": 6,
            "use_custom_start": 1,
            "start_gate_index": 1,
            "start_elapsed_time": MEASURED_START_ELAPSED_SECONDS,
            "start_x": 0.0,
            "start_y": 0.0,
            "start_z": 0.0,
            "start_vx": float(MEASURED_START_VELOCITY[0]),
            "start_vy": float(MEASURED_START_VELOCITY[1]),
            "start_vz": float(MEASURED_START_VELOCITY[2]),
            "start_qw": float(MEASURED_START_QUATERNION[0]),
            "start_qx": float(MEASURED_START_QUATERNION[1]),
            "start_qy": float(MEASURED_START_QUATERNION[2]),
            "start_qz": float(MEASURED_START_QUATERNION[3]),
            "start_wx": float(MEASURED_START_BODY_RATES[0]),
            "start_wy": float(MEASURED_START_BODY_RATES[1]),
            "start_wz": float(MEASURED_START_BODY_RATES[2]),
            "gate0_x": 0.0,
            "gate0_y": 0.0,
            "gate0_z": 0.0,
            "gate1_x": float(MEASURED_GATE2[0]),
            "gate1_y": float(MEASURED_GATE2[1]),
            "gate1_z": float(MEASURED_GATE2[2]),
            "observable_gate_progress": 1,
            "observable_gate_index_denominator": 6.0,
            "observable_gate_phase_onehot": 1,
            "evaluation_episode_limit": 1,
            "evaluation_episode_offset": 0,
        }
    )
    for gate_index, (x, y, z) in enumerate(INERT_PHASE_SCALE_GATES, start=2):
        environment[f"gate{gate_index}_x"] = x
        environment[f"gate{gate_index}_y"] = y
        environment[f"gate{gate_index}_z"] = z
    return config, overrides


def _repeat_state(state: torch.Tensor, agents: int) -> torch.Tensor:
    if state.ndim != 3 or state.shape[1] != 1:
        raise ValueError("warm recurrent state must have one batch row")
    return state.repeat(1, agents, 1)


def warm_suffix_stepwise(
    suffix: torch.nn.Module,
    observations: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    """Replay a prefix with the exact deployment-time one-step GRU call."""

    values = np.asarray(observations, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4119:
        raise ValueError("visual warm prefix must have shape [steps,4119]")
    state = suffix.initial_state(1, device=device)
    actions: list[np.ndarray] = []
    with torch.no_grad():
        for row in values:
            output, state = suffix.forward_step(
                torch.from_numpy(row).to(device)[None], state
            )
            actions.append(output.mean[0].cpu().numpy())
    return np.asarray(actions, dtype=np.float32), state


def run_handoff(
    *,
    output: Path = DEFAULT_OUTPUT,
    agents: int = 128,
    seed: int = 43003,
    device_name: str = "cuda",
    camera_pitch_rad: float = 0.0,
    alias_zoom: float = 1.0,
    alias_hold_steps: int = 0,
    suffix_candidate: str = "sf066",
    tag: str = TAG,
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if agents <= 0 or alias_hold_steps < 0:
        raise ValueError("agents must be positive and alias hold nonnegative")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled backend is not drone_race_vision")

    prefix = load_warm_prefix()
    measured_previous_action = load_measured_previous_action()
    device = torch.device(device_name)
    suffix, suffix_payload, suffix_source_paths = load_candidate_suffix(
        device, suffix_candidate
    )
    n294 = load_n294(device)
    config, overrides = measured_suffix_config(
        pufferl,
        agents=agents,
        seed=seed,
        camera_pitch_rad=camera_pitch_rad,
    )
    environment = config["env"]
    if float(environment["teacher_action_blend"]) != 0.0:
        raise RuntimeError("handoff screen refuses teacher action blending")

    with torch.no_grad():
        legacy_prefix = torch.from_numpy(prefix["legacy_observation"]).to(device)[None]
        n294_out, n294_warm = n294.forward_chunk_outputs(
            legacy_prefix, n294.initial_state(1, device)
        )
    replayed_suffix, suffix_warm = warm_suffix_stepwise(
        suffix, prefix["visual_observation"], device=device
    )
    if suffix_candidate == "sf066":
        suffix_reference = prefix["suffix_action"]
    else:
        suffix_reference, _ = warm_suffix_stepwise(
            suffix, prefix["visual_observation"], device=device
        )
    n294_replay_error = float(
        np.max(
            np.abs(
                np.clip(n294_out[0, :, :ACTION_SIZE].cpu().numpy(), -1.0, 1.0)
                - prefix["n294_action"]
            )
        )
    )
    suffix_replay_error = float(
        np.max(np.abs(replayed_suffix - suffix_reference))
    )
    if max(n294_replay_error, suffix_replay_error) > 5e-5:
        raise RuntimeError("source-locked prefix recurrent replay diverged")
    n294_state = _repeat_state(n294_warm, agents)
    suffix_state = _repeat_state(suffix_warm, agents)

    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != agents or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native visual vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (agents, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (agents,), torch.float32)
    actions_cpu = torch.zeros((agents, ACTION_SIZE), dtype=torch.float32)
    adapter = LegacyN294ObservationAdapter(
        agents,
        dt_s=DT_SECONDS,
        time_limit_s=TIME_LIMIT_SECONDS,
        sample_interval_steps=4,
        dropout_range_m=4.25,
        dropout_from_gate_index=0,
    )
    history = np.zeros((agents, 3, ACTION_SIZE), dtype=np.float32)
    history[:, 0] = measured_previous_action
    history[:, 1] = prefix["selected_action"][-2]
    history[:, 2] = prefix["selected_action"][-3]
    held_phase = np.full(agents, FIRST_GATE_PHASE, dtype=np.float32)
    reached = np.zeros(agents, dtype=bool)
    failed = np.zeros(agents, dtype=bool)
    selected_suffix = np.zeros(agents, dtype=np.int32)
    executed_action_max_error = 0.0
    action_envelope_violations = 0
    nonfinite_action = False
    inference_seconds = 0.0
    native_log: dict[str, Any] = {}
    vector_steps = 0
    trace: dict[str, list[Any]] = {
        "selected_action": [],
        "raw_phase": [],
        "held_phase": [],
        "mask_mass": [],
        "mask_centroid_xy": [],
        "alias_zoom": [],
    }
    started = time.perf_counter()
    try:
        vector.reset()
        adapter.reset()
        with torch.no_grad():
            for local_step in range(int(environment["max_steps"])):
                current = observations.numpy().copy()
                current[:, ACTION_HISTORY] = history.reshape(agents, -1)
                raw_phase = current[:, PHASE_PRIVILEGED_INDEX].astype(
                    np.float32, copy=True
                )
                global_step = len(prefix["selected_action"]) + local_step
                held_phase, _ = update_held_phase(
                    raw_phase, held_phase, step=global_step
                )
                reached |= held_phase >= SECOND_GATE_PHASE - PHASE_EPSILON
                active_cpu = ~(reached | failed)
                if not active_cpu.any():
                    break

                legacy = adapter.observe(
                    current, raw_phase, held_phase, step=global_step
                )
                suffix_input = visual_suffix_observation(current, held_phase)
                step_zoom = alias_zoom if local_step < alias_hold_steps else 1.0
                suffix_input = centered_mask_zoom(suffix_input, step_zoom)
                active = torch.from_numpy(active_cpu).to(device)
                inference_started = time.perf_counter()
                n294_output, n294_candidate = n294.forward_chunk_outputs(
                    torch.from_numpy(legacy).to(device).unsqueeze(1), n294_state
                )
                suffix_output, suffix_candidate_state = suffix.forward_step(
                    torch.from_numpy(suffix_input).to(device), suffix_state
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - inference_started
                n294_state = preserve_frozen_state(n294_state, n294_candidate, active)
                suffix_state = preserve_frozen_state(
                    suffix_state, suffix_candidate_state, active
                )
                n294_action = torch.clamp(
                    n294_output[:, 0, :ACTION_SIZE], -1.0, 1.0
                ).cpu().numpy()
                suffix_action = suffix_output.mean.cpu().numpy()
                selected, suffix_rows = select_complete_actions(
                    n294_action, suffix_action, held_phase
                )
                selected[~active_cpu] = 0.0
                selected_suffix += (active_cpu & suffix_rows).astype(np.int32)
                if not np.isfinite(selected[active_cpu]).all():
                    nonfinite_action = True
                    break
                action_envelope_violations += int(
                    np.any(np.abs(selected[active_cpu]) > 1.0 + 1e-6, axis=1).sum()
                )

                if agents:
                    mask = suffix_input[0, :MASK_SIZE].reshape(64, 64)
                    mass = float(mask.sum())
                    if mass > 1e-8:
                        yy, xx = np.indices((64, 64), dtype=np.float32)
                        centroid = [float((mask * xx).sum() / mass), float((mask * yy).sum() / mass)]
                    else:
                        centroid = [31.5, 31.5]
                    trace["selected_action"].append(selected[0].copy())
                    trace["raw_phase"].append(float(raw_phase[0]))
                    trace["held_phase"].append(float(held_phase[0]))
                    trace["mask_mass"].append(mass)
                    trace["mask_centroid_xy"].append(centroid)
                    trace["alias_zoom"].append(float(step_zoom))

                actions_cpu.copy_(torch.from_numpy(selected))
                executed_reference = selected.copy()
                vector.cpu_step(actions_cpu.data_ptr())
                vector_steps = local_step + 1
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(
                        np.max(
                            np.abs(executed[active_cpu] - executed_reference[active_cpu])
                        )
                    ),
                )
                history[:, 2] = history[:, 1]
                history[:, 1] = history[:, 0]
                history[:, 0] = selected
                terminal_now = (terminals.numpy() > 0.5) & active_cpu
                failed |= terminal_now
        native_log = dict(vector.log())
    finally:
        vector.close()

    metrics = flatten_log(pufferl, native_log)
    diagnostic_valid = bool(
        n294_replay_error <= 5e-5
        and suffix_replay_error <= 5e-5
        and executed_action_max_error <= 5e-5
        and action_envelope_violations == 0
        and not nonfinite_action
        and int((selected_suffix > 0).sum()) == agents
    )
    output.mkdir(parents=True, exist_ok=False)
    trace_path = output / "trace_agent0.npz"
    np.savez_compressed(
        trace_path,
        **{name: np.asarray(values) for name, values in trace.items()},
    )
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/eval_vq2_n294_visual_suffix_composite.py",
        ROOT / "pufferlib/vq2_n294_visual_composite.py",
        PREFIX_REPORT,
        PREFIX_TRACE,
        N404_REPORT,
        N404_TRACE,
        *suffix_source_paths,
    )
    report = {
        "schema": "vq2_measured_visual_suffix_handoff_v1",
        "tag": tag,
        "diagnostic_valid": diagnostic_valid,
        "next_gate_passed": bool(reached.all()),
        "agents": agents,
        "seed": seed,
        "device": device_name,
        "suffix_candidate": suffix_candidate,
        "suffix_checkpoint_sha256": (
            CANDIDATE_ARTIFACTS[suffix_candidate][1]
            if suffix_candidate in CANDIDATE_ARTIFACTS
            else suffix_payload.get("checkpoint_sha256", "")
        ),
        "warm_prefix_records": int(len(prefix["selected_action"])),
        "n294_prefix_replay_max_error": n294_replay_error,
        "suffix_prefix_replay_max_error": suffix_replay_error,
        "measured_previous_action": measured_previous_action.tolist(),
        "measured_start_elapsed_seconds": MEASURED_START_ELAPSED_SECONDS,
        "measured_start_velocity": MEASURED_START_VELOCITY.tolist(),
        "measured_start_quaternion": MEASURED_START_QUATERNION.tolist(),
        "measured_start_body_rates": MEASURED_START_BODY_RATES.tolist(),
        "measured_gate2": MEASURED_GATE2.tolist(),
        "measured_alias_vector": MEASURED_ALIAS_VECTOR.tolist(),
        "measured_alias_zoom": MEASURED_ALIAS_ZOOM,
        "applied_alias_zoom": alias_zoom,
        "alias_hold_steps": alias_hold_steps,
        "camera_pitch_rad": camera_pitch_rad,
        "steps": vector_steps,
        "reached_next_gate": int(reached.sum()),
        "failed_before_next_gate": int(failed.sum()),
        "selected_suffix_steps_min_max": [
            int(selected_suffix.min()),
            int(selected_suffix.max()),
        ],
        "executed_action_max_error": executed_action_max_error,
        "action_envelope_violations": action_envelope_violations,
        "nonfinite_action": nonfinite_action,
        "inference_seconds": inference_seconds,
        "composite_inference_steps_per_second": (
            vector_steps / inference_seconds if inference_seconds > 0.0 else 0.0
        ),
        "wall_time_seconds": time.perf_counter() - started,
        "plant_action_contract": {
            "complete_puffer_output_only": True,
            "action_blend": False,
            "analytic_action": False,
            "visual_suffix_owns_all_handoff_actions": True,
        },
        "safety": {
            "flight_sim_packets_sent": 0,
            "teacher_actions_executed": 0,
            "student_updates": 0,
            "submission_actions": 0,
        },
        "loader_overrides": overrides,
        "native_metrics_diagnostic_only": metrics,
        "trace": {
            "path": str(trace_path.relative_to(ROOT)),
            "sha256": sha256_path(trace_path),
            "records": len(trace["selected_action"]),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--agents", type=int, default=128)
    parser.add_argument("--seed", type=int, default=43003)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--camera-pitch-rad", type=float, default=0.0)
    parser.add_argument("--alias-zoom", type=float, default=1.0)
    parser.add_argument("--alias-hold-steps", type=int, default=0)
    parser.add_argument(
        "--suffix-candidate",
        choices=("sf066", "c007", "c009", "c012"),
        default="sf066",
    )
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args()
    report = run_handoff(
        output=args.output,
        agents=args.agents,
        seed=args.seed,
        device_name=args.device,
        camera_pitch_rad=args.camera_pitch_rad,
        alias_zoom=args.alias_zoom,
        alias_hold_steps=args.alias_hold_steps,
        suffix_candidate=args.suffix_candidate,
        tag=args.tag,
    )
    return 0 if report["diagnostic_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
