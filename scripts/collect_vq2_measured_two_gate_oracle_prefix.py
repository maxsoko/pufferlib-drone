#!/usr/bin/env python3
"""Collect exact measured two-gate oracle prefixes with held public phase."""

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
from pufferlib.vq2_informed import ACTION_HISTORY, ENV_OBS_SIZE, LEGAL_OBS_SIZE
from pufferlib.vq2_oracle import alignment_oracle_action
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.collect_vq2_recurrent_dagger import SF016_REPORT, SF016_REPORT_SHA256
import scripts.collect_vq2_public_phase_dagger as phase_data
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log, load_fixed_config


TAG = "vq2_sf047_measured_two_gate_oracle_prefix_64"
SCHEMA = "vq2_public_phase_oracle_prefix_time_major_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf047_measured_two_gate_oracle_prefix_preregistration_2026-07-28.md"
)
AGENTS = 64
EPISODES = 64
SEED = 42047
PREFIX_STEPS = 768
MAX_EXECUTED_LABEL_ERROR = 5e-5
MEASURED_GATES = (
    (10.78, 0.084, 0.56),
    (25.13, 8.96, 1.65),
)


def measured_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = load_fixed_config(
        pufferl_module,
        agents=AGENTS,
        episodes=EPISODES,
        seed=SEED,
        controller="governed",
        target_speed_m_s=2.0,
        episode_seconds=180.0,
        randomized_course=False,
    )
    environment = config["env"]
    environment.update(
        {
            "num_gates": 6,
            "use_custom_start": 0,
            "use_custom_gate_layout": 1,
            "reset_position_noise_xy": 0.0,
            "reset_position_noise_z": 0.0,
            "gate_position_domain_randomize": 0,
            "course_geometry_scale_randomize": 0,
            "sitl_plant_domain_randomize": 0,
            "gate_radius": 0.75,
            "gate0_radius": 0.75,
            "gate1_radius": 0.75,
            "teacher_action_blend": 1.0,
            "teacher_course_spline": 0,
            "teacher_segment_minimum_jerk": 0,
            "teacher_alignment_governor": 1,
            "w_action_teacher": 0.0,
        }
    )
    for index, (x, y, z) in enumerate(MEASURED_GATES):
        environment[f"gate{index}_x"] = x
        environment[f"gate{index}_y"] = y
        environment[f"gate{index}_z"] = z
    return config, overrides


def collect(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if sha256_path(SF016_REPORT) != SF016_REPORT_SHA256:
        raise RuntimeError("SF016 source-lock mismatch")
    if not json.loads(SF016_REPORT.read_text()).get("parity_passed"):
        raise RuntimeError("SF016 did not admit the oracle query")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled backend is not drone_race_vision")
    config, overrides = measured_config(pufferl)
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native vector ABI differs from SF047")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    zero_actions = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    done = np.zeros(AGENTS, dtype=bool)
    lengths = np.zeros(AGENTS, dtype=np.int32)
    terminal_counts = np.zeros(AGENTS, dtype=np.int32)
    held_phase = np.zeros(AGENTS, dtype=np.float32)
    previous_phase = np.zeros(AGENTS, dtype=np.float32)
    maximum_phase = np.zeros(AGENTS, dtype=np.float32)
    phase_changes_off_tick = 0
    phase_decreases = 0
    executed_label_max_error = 0.0
    labels = 0
    native_log: dict[str, Any] = {}
    manifest: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        vector.reset()
        with tempfile.TemporaryDirectory(
            prefix=f".{TAG}_staging_", dir=output.parent
        ) as temporary:
            writer = phase_data.PhaseDatasetWriter(
                Path(temporary) / "arrays",
                max_steps=PREFIX_STEPS,
                agents=AGENTS,
            )
            for step in range(PREFIX_STEPS):
                active = ~done
                current = observations.numpy()
                raw_phase = current[:, phase_data.PHASE_PRIVILEGED_INDEX]
                held_phase, sampled = phase_data.update_held_phase(
                    raw_phase, held_phase, step=step
                )
                changed = np.abs(held_phase - previous_phase) > 1e-7
                if changed.any() and not sampled:
                    phase_changes_off_tick += int(changed.sum())
                phase_decreases += int(
                    (held_phase[active] + 1e-7 < previous_phase[active]).sum()
                )
                previous_phase = held_phase.copy()
                maximum_phase = np.maximum(maximum_phase, held_phase)
                mask, tail = phase_data.phase_storage_parts(current, held_phase)
                oracle_label = alignment_oracle_action(current)
                vector.cpu_step(zero_actions.data_ptr())
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                if active.any():
                    executed_label_max_error = max(
                        executed_label_max_error,
                        float(np.max(np.abs(executed[active] - oracle_label[active]))),
                    )
                terminal = (terminals.numpy() > 0.5) & active
                lengths[active] += 1
                terminal_counts += terminal.astype(np.int32)
                labels += int(active.sum())
                mask[~active] = 0
                tail[~active] = 0.0
                oracle_label[~active] = 0.0
                writer.append(
                    mask=mask,
                    tail=tail,
                    action=oracle_label,
                    terminal=terminal.astype(np.uint8),
                    valid=active.astype(np.uint8),
                )
                done |= terminal
            native_log = dict(vector.log())
            staged_terminal, terminal_is_last = writer.validate_episode_layout(
                lengths
            )
            if not np.array_equal(staged_terminal, terminal_counts):
                raise RuntimeError("staged and tracked terminals differ")
            metrics = flatten_log(pufferl, native_log)
            admitted = bool(
                np.all(lengths > 0)
                and np.all(lengths <= PREFIX_STEPS)
                and np.all(terminal_counts <= 1)
                and np.all((terminal_counts == 0) | terminal_is_last)
                and labels == int(lengths.sum())
                and np.all(maximum_phase >= np.float32(2.0 / 6.0) - 1e-7)
                and executed_label_max_error <= MAX_EXECUTED_LABEL_ERROR
                and phase_changes_off_tick == 0
                and phase_decreases == 0
                and metrics.get("env/crash", 0.0) == 0.0
                and metrics.get("env/missed_gate", 0.0) == 0.0
                and metrics.get("env/out_of_order", 0.0) == 0.0
                and metrics.get("env/action_envelope_violation", 0.0) == 0.0
                and metrics.get("env/wire_rate_envelope_violation", 0.0) == 0.0
                and metrics.get("env/thrust_envelope_violation", 0.0) == 0.0
            )
            if not admitted:
                raise RuntimeError("SF047 measured oracle prefix failed admission")
            manifest = writer.finalize(output)
    finally:
        vector.close()

    source_paths = (
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "scripts/collect_vq2_public_phase_dagger.py",
        SF016_REPORT,
    )
    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    metadata = {
        "schema": SCHEMA,
        "tag": TAG,
        "time_steps": PREFIX_STEPS,
        "agents": AGENTS,
        "episodes": EPISODES,
        "records": labels,
        "episode_lengths": lengths.tolist(),
        "bounded_prefix_dataset": True,
        "bounded_prefix_steps": PREFIX_STEPS,
        "measured_gates_ned": [list(gate) for gate in MEASURED_GATES],
        "observation": {
            "stored_legal_width": PHASE_LEGAL_OBS_SIZE,
            "legacy_legal_width": LEGAL_OBS_SIZE,
            "mask_width": phase_data.MASK_SIZE,
            "legal_tail_width": phase_data.PHASE_TAIL_SIZE,
            "public_status_values_per_record": 1,
            "public_status_source": "held_active_gate_index_div_6",
            "public_status_rate_hz": phase_data.STATUS_RATE_HZ,
            "stored_training_only_privileged_values_per_record": 0,
        },
        "action": {
            "width": ACTION_SIZE,
            "source": "sf016_alignment_oracle",
            "plant_action_source": "sf016_alignment_oracle",
        },
        "files": manifest,
        "source_sha256": source_sha256,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    metrics = flatten_log(pufferl, native_log)
    report = {
        "schema": "vq2_measured_two_gate_oracle_prefix_report_v1",
        "tag": TAG,
        "admitted": True,
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "bounded_prefix_steps": PREFIX_STEPS,
        "records": labels,
        "measured_gates_ned": [list(gate) for gate in MEASURED_GATES],
        "maximum_phase_min": float(maximum_phase.min()),
        "maximum_phase_max": float(maximum_phase.max()),
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases,
        "executed_label_max_error": executed_label_max_error,
        "executed_label_error_threshold": MAX_EXECUTED_LABEL_ERROR,
        "metadata_sha256": sha256_path(output / "metadata.json"),
        "files": manifest,
        "metrics": metrics,
        "loader_overrides": overrides,
        "wall_time_seconds": time.perf_counter() - started,
        "source_sha256": source_sha256,
        "safety": {
            "oracle_actions_executed": labels,
            "student_actions_executed": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(collect(args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
