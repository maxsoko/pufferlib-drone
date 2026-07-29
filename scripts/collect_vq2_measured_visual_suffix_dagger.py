#!/usr/bin/env python3
"""Collect one legal suffix-only DAgger dataset from the C005 handoff."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.torch_pufferl import _cpu_tensor
from pufferlib.vq2_informed import ACTION_HISTORY, ENV_OBS_SIZE, MASK_SIZE
from pufferlib.vq2_n294_visual_composite import (
    ACTION_SIZE,
    LegacyN294ObservationAdapter,
    select_complete_actions,
    visual_suffix_observation,
)
from pufferlib.vq2_oracle import alignment_oracle_action
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.collect_vq2_public_phase_dagger import (
    PHASE_PRIVILEGED_INDEX,
    PHASE_TAIL_SIZE,
    PhaseDatasetWriter,
    update_held_phase,
)
from scripts.eval_vq2_measured_visual_suffix_handoff import (
    FIRST_GATE_PHASE,
    MEASURED_ALIAS_ZOOM,
    centered_mask_zoom,
    load_measured_previous_action,
    load_warm_prefix,
    measured_suffix_config,
    warm_suffix_stepwise,
)
from scripts.eval_vq2_n294_visual_suffix_composite import (
    BACKEND_ENV_NAME,
    DT_SECONDS,
    SECOND_GATE_PHASE,
    TIME_LIMIT_SECONDS,
    flatten_log,
    load_n294,
    load_suffix,
)
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state


TAG = "vq2_c006_measured_visual_suffix_dagger_128"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
AGENTS = 128
SEED = 43006
ALIAS_START_AGENT = AGENTS // 2
ALIAS_HOLD_STEPS = 16
PHASE_EPSILON = np.float32(1e-6)
C005_TRUE_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c005a_measured_handoff_true_range_exact128/report.json"
)
C005_TRUE_REPORT_SHA256 = "f8d34cd8d7f11bea624061f7dbe9c6e4e962090bc9fa8c0e176e7d0886c153c3"
C005_ALIAS_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_c005b_measured_handoff_live_alias_exact128/report.json"
)
C005_ALIAS_REPORT_SHA256 = "ce537f45509defd4a48295c5b7e68225bdff937f9aeb11487363a5668a8b4428"
SF009_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf009_native_oracle_alignment_governed_64/report.json"
)
SF009_REPORT_SHA256 = "c76c4f12045cc5782c6b24f4e5d39eee52eeb1855e77f3272526b5d48a100f33"
SF011_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf011_native_oracle_randomized_course_4096/report.json"
)
SF011_REPORT_SHA256 = "f9b39ba627a3c4c1daf79672175ec35989e29acb8f6ae7ef5ecff212fe7c2a1e"


def apply_fixed_alias_group(
    observations: np.ndarray,
    *,
    local_step: int,
    alias_start_agent: int = ALIAS_START_AGENT,
) -> np.ndarray:
    """Apply the C005 alias only to the preregistered trailing agent group."""

    values = np.asarray(observations, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != MASK_SIZE + PHASE_TAIL_SIZE:
        raise ValueError("phase-legal observation ABI changed")
    if not 0 <= alias_start_agent <= values.shape[0] or local_step < 0:
        raise ValueError("alias group or step is invalid")
    result = values.copy()
    if local_step < ALIAS_HOLD_STEPS and alias_start_agent < values.shape[0]:
        result[alias_start_agent:] = centered_mask_zoom(
            result[alias_start_agent:], MEASURED_ALIAS_ZOOM
        )
    return result


def storage_parts(actor_observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode exactly the legal actor input without its training-only tail."""

    values = np.asarray(actor_observation, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != MASK_SIZE + PHASE_TAIL_SIZE:
        raise ValueError("phase-legal actor input ABI changed")
    if not np.isfinite(values).all():
        raise RuntimeError("actor input contains nonfinite values")
    raw_mask = values[:, :MASK_SIZE]
    if raw_mask.min(initial=0.0) < -1e-6 or raw_mask.max(initial=1.0) > 1.0 + 1e-6:
        raise RuntimeError("soft mask escaped [0,1]")
    mask = np.rint(np.clip(raw_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    tail = values[:, MASK_SIZE:].astype(np.float32, copy=True)
    return mask, tail


def collection_admitted(
    *,
    lengths: np.ndarray,
    terminal_count: np.ndarray,
    terminal_is_last: np.ndarray,
    records: int,
    n294_replay_error: float,
    suffix_replay_error: float,
    executed_action_error: float,
    action_envelope_violations: int,
    nonfinite_action: bool,
    phase_decreases: int,
) -> bool:
    return bool(
        lengths.shape == (AGENTS,)
        and np.all(lengths > 0)
        and np.all(terminal_count == 1)
        and np.all(terminal_is_last)
        and records == int(lengths.sum())
        and n294_replay_error <= 5e-5
        and suffix_replay_error <= 5e-5
        and executed_action_error <= 5e-5
        and action_envelope_violations == 0
        and not nonfinite_action
        and phase_decreases == 0
    )


def collect(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frozen = {
        C005_TRUE_REPORT: C005_TRUE_REPORT_SHA256,
        C005_ALIAS_REPORT: C005_ALIAS_REPORT_SHA256,
        SF009_REPORT: SF009_REPORT_SHA256,
        SF011_REPORT: SF011_REPORT_SHA256,
    }
    for path, expected in frozen.items():
        actual = sha256_path(path)
        if actual != expected:
            raise RuntimeError(f"source-lock mismatch for {path}: {actual}")
    for path in (C005_TRUE_REPORT, C005_ALIAS_REPORT):
        if not json.loads(path.read_text()).get("diagnostic_valid"):
            raise RuntimeError("C005 fixture did not pass its contract admission")
    for path in (SF009_REPORT, SF011_REPORT):
        if not json.loads(path.read_text()).get("baseline_passed"):
            raise RuntimeError("alignment oracle admission is missing")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled backend is not drone_race_vision")

    prefix = load_warm_prefix()
    measured_previous_action = load_measured_previous_action()
    device = torch.device(device_name)
    suffix, _ = load_suffix(device)
    n294 = load_n294(device)
    config, overrides = measured_suffix_config(
        pufferl, agents=AGENTS, seed=SEED, camera_pitch_rad=0.0
    )
    environment = config["env"]
    if float(environment["teacher_action_blend"]) != 0.0:
        raise RuntimeError("DAgger plant must remain student-only")

    with torch.no_grad():
        legacy_prefix = torch.from_numpy(prefix["legacy_observation"]).to(device)[None]
        n294_out, n294_warm = n294.forward_chunk_outputs(
            legacy_prefix, n294.initial_state(1, device)
        )
    replayed_suffix, suffix_warm = warm_suffix_stepwise(
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
        np.max(np.abs(replayed_suffix - prefix["suffix_action"]))
    )
    if max(n294_replay_error, suffix_replay_error) > 5e-5:
        raise RuntimeError("prefix replay diverged before collection")
    n294_state = n294_warm.repeat(1, AGENTS, 1)
    suffix_state = suffix_warm.repeat(1, AGENTS, 1)

    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native visual vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions_cpu = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    adapter = LegacyN294ObservationAdapter(
        AGENTS,
        dt_s=DT_SECONDS,
        time_limit_s=TIME_LIMIT_SECONDS,
        sample_interval_steps=4,
        dropout_range_m=4.25,
        dropout_from_gate_index=0,
    )
    history = np.zeros((AGENTS, 3, ACTION_SIZE), dtype=np.float32)
    history[:, 0] = measured_previous_action
    history[:, 1] = prefix["selected_action"][-2]
    history[:, 2] = prefix["selected_action"][-3]
    held_phase = np.full(AGENTS, FIRST_GATE_PHASE, dtype=np.float32)
    previous_phase = held_phase.copy()
    done = np.zeros(AGENTS, dtype=bool)
    lengths = np.zeros(AGENTS, dtype=np.int32)
    tracked_terminal_count = np.zeros(AGENTS, dtype=np.int32)
    records = 0
    executed_action_max_error = 0.0
    action_envelope_violations = 0
    nonfinite_action = False
    phase_decreases = 0
    oracle_sum = np.zeros(ACTION_SIZE, dtype=np.float64)
    oracle_square_sum = np.zeros(ACTION_SIZE, dtype=np.float64)
    oracle_min = np.full(ACTION_SIZE, np.inf, dtype=np.float64)
    oracle_max = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
    native_log: dict[str, Any] = {}
    manifest: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        vector.reset()
        adapter.reset()
        with tempfile.TemporaryDirectory(
            prefix=f".{TAG}_staging_", dir=output.parent
        ) as temporary:
            writer = PhaseDatasetWriter(
                Path(temporary) / "arrays",
                max_steps=int(environment["max_steps"]),
                agents=AGENTS,
            )
            with torch.no_grad():
                for local_step in range(int(environment["max_steps"])):
                    active = ~done
                    if not active.any():
                        break
                    current = observations.numpy().copy()
                    current[:, ACTION_HISTORY] = history.reshape(AGENTS, -1)
                    raw_phase = current[:, PHASE_PRIVILEGED_INDEX].astype(
                        np.float32, copy=True
                    )
                    global_step = len(prefix["selected_action"]) + local_step
                    held_phase, _ = update_held_phase(
                        raw_phase, held_phase, step=global_step
                    )
                    phase_decreases += int(
                        (held_phase[active] + 1e-7 < previous_phase[active]).sum()
                    )
                    previous_phase = held_phase.copy()
                    legacy = adapter.observe(
                        current, raw_phase, held_phase, step=global_step
                    )
                    actor_input = visual_suffix_observation(current, held_phase)
                    actor_input = apply_fixed_alias_group(
                        actor_input, local_step=local_step
                    )
                    oracle = alignment_oracle_action(current)
                    mask, tail = storage_parts(actor_input)
                    active_device = torch.from_numpy(active).to(device)
                    n294_output, n294_candidate = n294.forward_chunk_outputs(
                        torch.from_numpy(legacy).to(device).unsqueeze(1), n294_state
                    )
                    suffix_output, suffix_candidate = suffix.forward_step(
                        torch.from_numpy(actor_input).to(device), suffix_state
                    )
                    n294_state = preserve_frozen_state(
                        n294_state, n294_candidate, active_device
                    )
                    suffix_state = preserve_frozen_state(
                        suffix_state, suffix_candidate, active_device
                    )
                    n294_action = torch.clamp(
                        n294_output[:, 0, :ACTION_SIZE], -1.0, 1.0
                    ).cpu().numpy()
                    suffix_action = suffix_output.mean.cpu().numpy()
                    selected, suffix_rows = select_complete_actions(
                        n294_action, suffix_action, held_phase
                    )
                    if not np.all(suffix_rows[active]):
                        raise RuntimeError("N294 unexpectedly owned a suffix action")
                    selected[~active] = 0.0
                    if not np.isfinite(selected[active]).all():
                        nonfinite_action = True
                        raise RuntimeError("student action became nonfinite")
                    action_envelope_violations += int(
                        np.any(np.abs(selected[active]) > 1.0 + 1e-6, axis=1).sum()
                    )

                    actions_cpu.copy_(torch.from_numpy(selected))
                    vector.cpu_step(actions_cpu.data_ptr())
                    next_environment = observations.numpy()
                    executed = next_environment[
                        :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                    ]
                    executed_action_max_error = max(
                        executed_action_max_error,
                        float(np.max(np.abs(executed[active] - selected[active]))),
                    )
                    next_raw_phase = next_environment[:, PHASE_PRIVILEGED_INDEX]
                    native_terminal = (terminals.numpy() > 0.5) & active
                    crossed = (
                        next_raw_phase >= SECOND_GATE_PHASE - PHASE_EPSILON
                    ) & active
                    terminal = native_terminal | crossed
                    lengths[active] += 1
                    tracked_terminal_count += terminal.astype(np.int32)
                    records += int(active.sum())
                    values = oracle[active].astype(np.float64)
                    oracle_sum += values.sum(axis=0)
                    oracle_square_sum += (values * values).sum(axis=0)
                    oracle_min = np.minimum(oracle_min, values.min(axis=0))
                    oracle_max = np.maximum(oracle_max, values.max(axis=0))
                    mask[~active] = 0
                    tail[~active] = 0.0
                    oracle[~active] = 0.0
                    writer.append(
                        mask=mask,
                        tail=tail,
                        action=oracle,
                        terminal=terminal.astype(np.uint8),
                        valid=active.astype(np.uint8),
                    )
                    history[:, 2] = history[:, 1]
                    history[:, 1] = history[:, 0]
                    history[:, 0] = selected
                    done |= terminal
            native_log = dict(vector.log())
            terminal_count, terminal_is_last = writer.validate_episode_layout(lengths)
            if not np.array_equal(terminal_count, tracked_terminal_count):
                raise RuntimeError("tracked and staged terminals differ")
            admitted = collection_admitted(
                lengths=lengths,
                terminal_count=terminal_count,
                terminal_is_last=terminal_is_last,
                records=records,
                n294_replay_error=n294_replay_error,
                suffix_replay_error=suffix_replay_error,
                executed_action_error=executed_action_max_error,
                action_envelope_violations=action_envelope_violations,
                nonfinite_action=nonfinite_action,
                phase_decreases=phase_decreases,
            )
            if not admitted:
                raise RuntimeError("C006 staged dataset failed admission")
            manifest = writer.finalize(output)
    finally:
        vector.close()

    mean = oracle_sum / records
    variance = np.maximum(oracle_square_sum / records - mean * mean, 0.0)
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/eval_vq2_measured_visual_suffix_handoff.py",
        ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "scripts/collect_vq2_public_phase_dagger.py",
        C005_TRUE_REPORT,
        C005_ALIAS_REPORT,
        SF009_REPORT,
        SF011_REPORT,
    )
    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    metadata = {
        "schema": "vq2_measured_visual_suffix_dagger_time_major_v1",
        "tag": TAG,
        "agents": AGENTS,
        "true_range_agents": ALIAS_START_AGENT,
        "live_alias_agents": AGENTS - ALIAS_START_AGENT,
        "records": records,
        "episode_lengths": lengths.tolist(),
        "observation": {
            "mask_width": MASK_SIZE,
            "tail_width": PHASE_TAIL_SIZE,
            "stored_training_only_privileged_values_per_record": 0,
            "public_progress": "held_active_gate_index_div_6",
            "previous_action": "complete_action_selected_for_plant",
        },
        "action": {
            "width": ACTION_SIZE,
            "label_source": "training_only_alignment_oracle_query",
            "plant_source": "sf066_deterministic_mean",
            "teacher_blend": 0.0,
        },
        "alias": {
            "zoom": MEASURED_ALIAS_ZOOM,
            "hold_steps": ALIAS_HOLD_STEPS,
            "start_agent": ALIAS_START_AGENT,
        },
        "files": manifest,
        "source_sha256": source_sha256,
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    report = {
        "schema": "vq2_measured_visual_suffix_dagger_report_v1",
        "tag": TAG,
        "admitted": True,
        "records": records,
        "length_min_max": [int(lengths.min()), int(lengths.max())],
        "terminal_count_min_max": [
            int(tracked_terminal_count.min()),
            int(tracked_terminal_count.max()),
        ],
        "n294_prefix_replay_max_error": n294_replay_error,
        "suffix_prefix_replay_max_error": suffix_replay_error,
        "executed_action_max_error": executed_action_max_error,
        "action_envelope_violations": action_envelope_violations,
        "nonfinite_action": nonfinite_action,
        "phase_decreases": phase_decreases,
        "oracle_action": {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": oracle_min.tolist(),
            "max": oracle_max.tolist(),
        },
        "metadata_sha256": sha256_path(metadata_path),
        "files": manifest,
        "loader_overrides": overrides,
        "native_metrics_diagnostic_only": flatten_log(pufferl, native_log),
        "source_sha256": source_sha256,
        "wall_time_seconds": time.perf_counter() - started,
        "safety": {
            "flight_sim_packets_sent": 0,
            "teacher_actions_executed": 0,
            "student_plant_actions": records,
            "student_updates": 0,
            "submission_actions": 0,
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    report = collect(output=args.output, device_name=args.device)
    return 0 if report["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
