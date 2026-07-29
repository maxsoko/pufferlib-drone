#!/usr/bin/env python3
"""Collect a legal-only, variable-gate oracle corpus for recurrent BC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
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
from pufferlib.vq2_informed import (
    ACTION_HISTORY,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    MASK_SIZE,
    PRIVILEGED_NAMES,
)
from pufferlib.vq2_public_phase import (
    ENGINE_GATE_CAP,
    PUBLIC_STATUS_HZ,
    PUBLIC_STATUS_INTERVAL_STEPS,
)
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.collect_vq2_oracle_bc_dataset import (
    TimeMajorDatasetWriter,
    executed_action_labels,
    legal_storage_parts,
)
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log
from scripts.eval_vq2_variable_gate_oracle import (
    current_runtime_manifest,
    load_variable_config,
    mixed_variable_environment,
    sha256_path,
    write_json_atomic,
)


TAG = "vq2_vg003_variable_gate_legal_bc_dataset_256"
SCHEMA = "vq2_variable_gate_oracle_legal_bc_time_major_v1"
AGENTS = 256
EPISODES = 256
SEED = 429030
EPISODE_SECONDS = 360.0
MINIMUM_RECORDS = 1_500_000
MASK_SCALE = 1.0 / 255.0
LEGAL_TAIL_SIZE = LEGAL_OBS_SIZE - MASK_SIZE
PHASE_TAIL_SIZE = PHASE_LEGAL_OBS_SIZE - MASK_SIZE
PHASE_PRIVILEGED_INDEX = LEGAL_OBS_SIZE + PRIVILEGED_NAMES.index(
    "ordered_gate_phase"
)
EXPECTED_PHASE_INCREMENTS = sum(count - 1 for count in range(5, 13)) * (
    AGENTS // 8
)
DEFAULT_OUTPUT = (
    ROOT / "logs" / "drone_race_full_policy_six_gate_bootstrap" / TAG
)
PREREGISTRATION = (
    ROOT / "docs/vq2_vg003_variable_gate_legal_bc_preregistration_2026-07-29.md"
)
VG002_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg002_variable_gate_oracle_admission/report.json"
)
VG002_REPORT_SHA256 = (
    "e649a20acd92a0c341426325dce281fc989a3cdbae8d634f3a36c14202b8bb27"
)

# Exact executable inputs to VG003. The collector itself, preregistration,
# compiled extension, and admitted VG002 aggregate are added at runtime.
FROZEN_SOURCE_SHA256: dict[str, str] = {
    "ocean/drone_race/drone_race.c":
        "523aa40a4578e0ec4b8808686faa5ba2d58d1b681091d49425456c1b39b1657b",
    "ocean/drone_race/drone_race.h":
        "58615ed1d7dd4738d5fcf04d7cdf4b43a6a6cbab9dccc393c0d665820f7f5b06",
    "ocean/drone_race/binding.c":
        "409ed4689f145b9a1f17c858535ffd8842fe9aaef5cf2e67c82945e51effb273",
    "pufferlib/torch_pufferl.py":
        "705515235d386646bc90945f5f44d6ba435798a077e53f06cd301cbf7d91dd81",
    "pufferlib/vq2_informed.py":
        "35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a",
    "pufferlib/vq2_public_phase.py":
        "6749a689b10e44a6b4f3b291155cf74e91cb557ac1a6ed38607daeda5dba7338",
    "pufferlib/vq2_recurrent_phase.py":
        "a15e28b2147ea485ea2ce7d3a889027fe0878e4baaa146b8f71d5b86f18a05aa",
    "config/drone_race_vq2_informed_dreamer.ini":
        "5ed3c6d59694c4c710cdd2c578b10d1070364f3c81d5f7e5443e3021042aec8b",
    "scripts/eval_vq2_native_oracle.py":
        "98e099a3286173322b56054a98b095faa4366cedd72359516ead606cdb37863f",
    "scripts/eval_vq2_variable_gate_oracle.py":
        "ec65b09c614b16240717266f32fd49221f7f7075b5540eb802c6008f2acd93c6",
    "scripts/collect_vq2_oracle_bc_dataset.py":
        "84fdfea354c50c807bce53b5a45a211fa2378f0ace25ebbc394b2f52cee87868",
    "scripts/run_vq2_vg003_vast.sh":
        "3392af1d4eceabc417fa575f807d269b82a36ff351d7dcd818257ee75efe4499",
}


class VariableGateDatasetWriter(TimeMajorDatasetWriter):
    """SF012 time-major storage with one held public phase value."""

    _SPECS = {
        "mask": (np.uint8, (MASK_SIZE,)),
        "tail": (np.float32, (PHASE_TAIL_SIZE,)),
        "action": (np.float32, (ACTION_SIZE,)),
        "terminal": (np.uint8, ()),
        "valid": (np.uint8, ()),
    }

    def append(
        self,
        *,
        mask: np.ndarray,
        tail: np.ndarray,
        action: np.ndarray,
        terminal: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        if self.steps >= self.max_steps:
            raise RuntimeError("dataset writer exceeded its fixed horizon")
        expected = {
            "mask": (self.agents, MASK_SIZE),
            "tail": (self.agents, PHASE_TAIL_SIZE),
            "action": (self.agents, ACTION_SIZE),
            "terminal": (self.agents,),
            "valid": (self.agents,),
        }
        supplied = {
            "mask": mask,
            "tail": tail,
            "action": action,
            "terminal": terminal,
            "valid": valid,
        }
        for name, value in supplied.items():
            if value.shape != expected[name]:
                raise ValueError(f"{name} batch shape {value.shape} != {expected[name]}")
            self.arrays[name][self.steps] = value
        self.steps += 1


def phase_storage_parts(
    environment_observation: np.ndarray,
    held_phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Persist the legal 4,118 values plus one held `/16` public scalar."""

    mask, legal_tail = legal_storage_parts(environment_observation)
    phase = np.asarray(held_phase, dtype=np.float32)
    if phase.shape != (environment_observation.shape[0],):
        raise ValueError("held public phase does not align with observations")
    tail = np.concatenate((legal_tail, phase[:, None]), axis=1)
    if tail.shape != (environment_observation.shape[0], PHASE_TAIL_SIZE):
        raise RuntimeError("variable-gate phase tail width changed")
    return mask, tail


def update_held_phase(
    raw_phase: np.ndarray,
    held_phase: np.ndarray,
    *,
    step: int,
) -> tuple[np.ndarray, bool, float]:
    """Sample the fixed-cap native mirror at the public 4 Hz cadence."""

    raw = np.asarray(raw_phase, dtype=np.float32)
    held = np.asarray(held_phase, dtype=np.float32).copy()
    if raw.shape != held.shape:
        raise ValueError("raw and held phase batches do not align")
    if not np.isfinite(raw).all() or raw.min(initial=0.0) < -1e-7 or raw.max(
        initial=1.0
    ) > 1.0 + 1e-7:
        raise RuntimeError("native public-phase mirror escaped [0,1]")
    scaled = raw.astype(np.float64) * ENGINE_GATE_CAP
    encoding_error = float(np.max(np.abs(scaled - np.rint(scaled)), initial=0.0))
    if encoding_error > 1e-6:
        raise RuntimeError("native public-phase mirror is not an index divided by 16")
    sampled = step % PUBLIC_STATUS_INTERVAL_STEPS == 0
    if sampled:
        held[:] = raw
    return held, sampled, encoding_error


def variable_collection_config(
    pufferl_module: Any,
) -> tuple[dict[str, Any], list[str]]:
    config, overrides = load_variable_config(
        pufferl_module,
        agents=AGENTS,
        episodes=EPISODES,
        seed=SEED,
        num_gates=12,
    )
    config["env"].update(mixed_variable_environment(seed=SEED))
    return config, overrides


def verify_frozen_sources() -> tuple[dict[str, str], str]:
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG003 preregistration is missing")
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        digest = sha256_path(ROOT / relative)
        if digest != expected:
            raise RuntimeError(
                f"source-lock mismatch for {relative}: {digest} != {expected}"
            )
        observed[relative] = digest
    if sha256_path(VG002_REPORT) != VG002_REPORT_SHA256:
        raise RuntimeError("VG002 aggregate report changed before collection")
    vg002 = json.loads(VG002_REPORT.read_text())
    if not vg002.get("admitted") or vg002.get("completed_counts") != [5, 8, 11, 12]:
        raise RuntimeError("VG002 did not admit the required variable counts")
    observed[str(VG002_REPORT.relative_to(ROOT))] = VG002_REPORT_SHA256
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return observed, source_commit


def prepare_collection_state(
    *,
    output: Path,
    state_path: Path,
    staging: Path,
    state_identity: dict[str, Any],
    source_sha256: dict[str, str],
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Create or resume one source-locked attempt without retrying rejection.

    A process interruption leaves status ``collecting`` and is recoverable by
    rebuilding only the scoped staging/output directories.  Admission failure
    writes status ``rejected`` and is terminal for the tag.
    """

    state: dict[str, Any] | None = None
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for key, expected in state_identity.items():
            if state.get(key) != expected:
                raise RuntimeError(f"VG003 resume mismatch for {key}")

    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        if state is None or state.get("status") != "admitted":
            raise RuntimeError("VG003 immutable report has no admitted run state")
        report = json.loads(report_path.read_text())
        if not report.get("admitted") or report.get("source_sha256") != source_sha256:
            raise RuntimeError("existing VG003 report does not match current sources")
        return state, report

    if state is None:
        if output.exists():
            raise RuntimeError("existing VG003 output has no source-locked state")
        if staging.exists():
            raise RuntimeError("VG003 staging exists without source-locked state")
        state = dict(state_identity)
        state["resume_count"] = 0
    else:
        if not resume:
            raise FileExistsError(f"VG003 state already exists at {state_path}")
        status = state.get("status")
        if status == "rejected":
            raise RuntimeError("VG003 rejection is terminal; unchanged retry forbidden")
        if status == "admitted":
            raise RuntimeError("VG003 admitted state is missing its immutable report")
        if status != "collecting":
            raise RuntimeError(f"VG003 cannot resume state status {status!r}")
        # Only exact, source-locked collecting state reaches these removals.
        # Both targets are uniquely scoped to this VG003 tag.
        if output.exists():
            shutil.rmtree(output)
        if staging.exists():
            shutil.rmtree(staging)
        state["resume_count"] = int(state.get("resume_count", 0)) + 1

    state.update({"status": "collecting", "records": 0, "vector_steps": 0})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(state_path, state)
    return state, None


def variable_dataset_admission_passes(
    metrics: dict[str, float],
    *,
    lengths: np.ndarray,
    terminal_count: np.ndarray,
    terminal_is_last: np.ndarray,
    label_count: int,
    history_shift_max_error: float,
    phase_changes_off_tick: int,
    phase_decreases: int,
    phase_increments: int,
    raw_phase_encoding_max_error: float,
) -> bool:
    if metrics.get("env/n") != float(EPISODES):
        return False
    if metrics.get("env/success_rate") != 1.0:
        return False
    required_zero = (
        "crash",
        "timeout",
        "missed_gate",
        "out_of_order",
        "crossing_margin_violation",
        "action_envelope_violation",
        "wire_rate_envelope_violation",
        "thrust_envelope_violation",
    )
    if any(metrics.get(f"env/{name}", math.inf) != 0.0 for name in required_zero):
        return False
    if metrics.get("env/valid_run_rate") != 1.0:
        return False
    if abs(metrics.get("env/gates_passed", 0.0) - 8.5) > 1e-7:
        return False
    for count in range(1, 17):
        expected = 0.125 if 5 <= count <= 12 else 0.0
        if abs(metrics.get(f"env/gate_count{count}_episode", -1.0) - expected) > 1e-12:
            return False
        if abs(metrics.get(f"env/gate_count{count}_success", -1.0) - expected) > 1e-12:
            return False
    for gate in range(12):
        expected_sampled = 1.0 if gate < 5 else (12 - gate) / 8.0
        if abs(metrics.get(f"env/ordered_gate{gate}_sampled", -1.0) - expected_sampled) > 1e-7:
            return False
        if metrics.get(f"env/ordered_gate{gate}_radial", math.inf) > 0.10:
            return False
    return (
        lengths.shape == (AGENTS,)
        and bool(np.all(lengths > 0))
        and bool(np.all(terminal_count == 1))
        and bool(np.all(terminal_is_last))
        and label_count == int(lengths.sum())
        and label_count >= MINIMUM_RECORDS
        and history_shift_max_error <= 1e-7
        and phase_changes_off_tick == 0
        and phase_decreases == 0
        and phase_increments == EXPECTED_PHASE_INCREMENTS
        and raw_phase_encoding_max_error <= 1e-6
    )


def collect(output: Path = DEFAULT_OUTPUT, *, resume: bool = False) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    frozen_sources, source_commit = verify_frozen_sources()
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("VG003 requires the drone_race_vision backend")
    if getattr(_C, "precision_bytes", None) != 4:
        raise RuntimeError("VG003 requires a float32 binding")
    extension_path = Path(_C.__file__).resolve()
    source_sha256 = {
        **frozen_sources,
        str(Path(__file__).resolve().relative_to(ROOT)): sha256_path(Path(__file__)),
        str(PREREGISTRATION.relative_to(ROOT)): sha256_path(PREREGISTRATION),
        "compiled_extension": sha256_path(extension_path),
    }
    runtime = current_runtime_manifest()
    state_path = output.with_name(f"{output.name}_state.json")
    staging = output.with_name(f".{output.name}_staging")
    state_identity = {
        "schema": "vq2_variable_gate_legal_bc_run_state_v1",
        "tag": TAG,
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "minimum_records": MINIMUM_RECORDS,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "compiled_extension_path": str(extension_path),
        "runtime": runtime,
        "safety": {
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "student_updates": 0,
            "student_checkpoints_written": 0,
            "submission_authorized": False,
        },
    }
    state, existing_report = prepare_collection_state(
        output=output,
        state_path=state_path,
        staging=staging,
        state_identity=state_identity,
        source_sha256=source_sha256,
        resume=resume,
    )
    if existing_report is not None:
        return existing_report

    config, loader_overrides = variable_collection_config(pufferl)
    environment = config["env"]
    if int(environment["evaluation_episode_limit"]) != 1:
        raise RuntimeError("VG003 requires one episode per vector instance")
    max_steps = int(environment["max_steps"])
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native vector ABI differs from VG003")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    zero_actions = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    done = np.zeros(AGENTS, dtype=bool)
    lengths = np.zeros(AGENTS, dtype=np.int32)
    tracked_terminal_count = np.zeros(AGENTS, dtype=np.int32)
    held_phase = np.zeros(AGENTS, dtype=np.float32)
    previous_stored_phase = np.zeros(AGENTS, dtype=np.float32)
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_increments = 0
    phase_samples = 0
    raw_phase_encoding_max_error = 0.0
    label_count = 0
    history_shift_max_error = 0.0
    action_sum = np.zeros(ACTION_SIZE, dtype=np.float64)
    action_square_sum = np.zeros(ACTION_SIZE, dtype=np.float64)
    action_min = np.full(ACTION_SIZE, np.inf, dtype=np.float64)
    action_max = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
    native_log: dict[str, Any] = {}
    manifest: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    try:
        vector.reset()
        writer = VariableGateDatasetWriter(
            staging, max_steps=max_steps, agents=AGENTS
        )
        for step in range(max_steps):
            active = ~done
            if not active.any():
                break
            current_environment = observations.numpy()
            current_legal = current_environment[:, :LEGAL_OBS_SIZE].copy()
            raw_phase = current_environment[:, PHASE_PRIVILEGED_INDEX]
            held_phase, sampled, encoding_error = update_held_phase(
                raw_phase, held_phase, step=step
            )
            raw_phase_encoding_max_error = max(
                raw_phase_encoding_max_error, encoding_error
            )
            changed = np.abs(held_phase - previous_stored_phase) > 1e-7
            if changed.any() and not sampled:
                phase_changes_off_tick += int(changed.sum())
            phase_decreases += int(
                (held_phase[active] + 1e-7 < previous_stored_phase[active]).sum()
            )
            phase_increments += int(
                (
                    held_phase[active]
                    > previous_stored_phase[active] + 1e-7
                ).sum()
            )
            if sampled:
                phase_samples += int(active.sum())
            previous_stored_phase = held_phase.copy()
            mask, tail = phase_storage_parts(current_environment, held_phase)

            vector.cpu_step(zero_actions.data_ptr())
            next_environment = observations.numpy()
            labels, shift_error = executed_action_labels(
                current_legal, next_environment, active
            )
            history_shift_max_error = max(history_shift_max_error, shift_error)
            terminal = (terminals.numpy() > 0.5) & active
            selected = labels[active].astype(np.float64)
            action_sum += selected.sum(axis=0)
            action_square_sum += np.square(selected).sum(axis=0)
            action_min = np.minimum(action_min, selected.min(axis=0))
            action_max = np.maximum(action_max, selected.max(axis=0))
            label_count += int(active.sum())
            lengths[active] += 1
            tracked_terminal_count += terminal.astype(np.int32)
            mask[~active] = 0
            tail[~active] = 0.0
            labels[~active] = 0.0
            writer.append(
                mask=mask,
                tail=tail,
                action=labels,
                terminal=terminal.astype(np.uint8),
                valid=active.astype(np.uint8),
            )
            done |= terminal

        native_log = dict(vector.log())
        metrics = flatten_log(pufferl, native_log)
        terminal_count, terminal_is_last = writer.validate_episode_layout(lengths)
        if not np.array_equal(terminal_count, tracked_terminal_count):
            raise RuntimeError("staged and tracked terminal counts differ")
        admitted = variable_dataset_admission_passes(
            metrics,
            lengths=lengths,
            terminal_count=terminal_count,
            terminal_is_last=terminal_is_last,
            label_count=label_count,
            history_shift_max_error=history_shift_max_error,
            phase_changes_off_tick=phase_changes_off_tick,
            phase_decreases=phase_decreases,
            phase_increments=phase_increments,
            raw_phase_encoding_max_error=raw_phase_encoding_max_error,
        )
        if not admitted:
            state.update({
                "status": "rejected",
                "records": int(label_count),
                "vector_steps": int(lengths.max(initial=0)),
            })
            write_json_atomic(state_path, state)
            raise RuntimeError("VG003 collection failed admission; no corpus finalized")
        manifest = writer.finalize(output)
    finally:
        vector.close()

    action_mean = action_sum / label_count
    action_variance = np.maximum(
        action_square_sum / label_count - action_mean**2, 0.0
    )
    metrics = flatten_log(pufferl, native_log)
    metadata = {
        "schema": SCHEMA,
        "tag": TAG,
        "time_steps": int(lengths.max()),
        "agents": AGENTS,
        "episodes": EPISODES,
        "records": int(label_count),
        "episode_lengths": lengths.tolist(),
        "observation": {
            "schema": "vq2_visual_ctbr_public_phase_v1",
            "native_width": ENV_OBS_SIZE,
            "stored_legal_width": PHASE_LEGAL_OBS_SIZE,
            "mask_width": MASK_SIZE,
            "mask_dtype": "uint8",
            "mask_decode_scale": MASK_SCALE,
            "legal_sensor_history_width": LEGAL_TAIL_SIZE,
            "public_phase_width": 1,
            "public_phase_tail_index": LEGAL_TAIL_SIZE,
            "public_phase_encoding": "clamp(active_gate_index,0,16)/16",
            "public_phase_rate_hz": PUBLIC_STATUS_HZ,
            "public_phase_hold_steps": PUBLIC_STATUS_INTERVAL_STEPS,
            "stored_training_only_privileged_values_per_record": 0,
            "stored_total_gate_count_values_per_record": 0,
        },
        "action": {
            "schema": "normalized_attitude_ctbr_v1",
            "width": ACTION_SIZE,
            "dtype": "float32",
            "source": "next_observation_newest_executed_action_history",
        },
        "files": manifest,
        "source_sha256": source_sha256,
    }
    write_json_atomic(output / "metadata.json", metadata)
    metadata_sha256 = sha256_path(output / "metadata.json")
    report = {
        "schema": "vq2_variable_gate_legal_bc_collection_report_v1",
        "tag": TAG,
        "admitted": True,
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "records": int(label_count),
        "vector_steps": int(lengths.max()),
        "wall_time_seconds": time.perf_counter() - started,
        "loader_overrides": loader_overrides,
        "fixed_environment": environment,
        "metrics": metrics,
        "dataset": {
            "metadata_sha256": metadata_sha256,
            "files": manifest,
            "episode_length_min": int(lengths.min()),
            "episode_length_mean": float(lengths.mean()),
            "episode_length_max": int(lengths.max()),
            "terminal_count_per_episode_min": int(terminal_count.min()),
            "terminal_count_per_episode_max": int(terminal_count.max()),
            "history_shift_max_error": history_shift_max_error,
            "phase_changes_off_tick": phase_changes_off_tick,
            "phase_decreases": phase_decreases,
            "phase_increments": phase_increments,
            "phase_samples": phase_samples,
            "raw_phase_encoding_max_error": raw_phase_encoding_max_error,
            "action_min": action_min.tolist(),
            "action_max": action_max.tolist(),
            "action_mean": action_mean.tolist(),
            "action_std": np.sqrt(action_variance).tolist(),
            "stored_training_only_privileged_values_per_record": 0,
            "stored_total_gate_count_values_per_record": 0,
        },
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "compiled_extension_path": str(extension_path),
        "runtime": runtime,
        "safety": state_identity["safety"],
    }
    write_json_atomic(output / "report.json", report)
    state.update({
        "status": "admitted",
        "records": int(label_count),
        "vector_steps": int(lengths.max()),
        "metadata_sha256": metadata_sha256,
        "report_sha256": sha256_path(output / "report.json"),
        "files": manifest,
    })
    write_json_atomic(state_path, state)
    shutil.rmtree(staging)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(args.output.resolve(), resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
