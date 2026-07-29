#!/usr/bin/env python3
"""Collect SF033-driven DAgger labels with one held public phase scalar."""

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
from pufferlib.vq2_informed import (
    ACTION_HISTORY,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    PRIVILEGED_NAMES,
)
from pufferlib.vq2_oracle import alignment_oracle_action
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.collect_vq2_oracle_bc_dataset import (
    MASK_SIZE,
    TimeMajorDatasetWriter,
    legal_storage_parts,
    sha256_path,
)
from scripts.collect_vq2_recurrent_dagger import SF016_REPORT, SF016_REPORT_SHA256
from scripts.collect_vq2_recurrent_dagger_next import next_dagger_collection_passes
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log, load_fixed_config
from scripts.eval_vq2_recurrent_roll_priority import load_sf033


TAG = "vq2_sf039_public_phase_dagger_512"
SCHEMA = "vq2_public_phase_dagger_time_major_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf039_public_phase_dagger_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf033_recurrent_roll_priority_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "91088ca96f432f57f1b85bbc521ddab29ba422fadf2b79b8d642ee426f09bea7"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "f54bf8512490a4629c08bf35ceae7c7fe07514e3424f89433f680536d96a0279"
)
SCREEN_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf034_recurrent_roll_priority_teacher_free_512/report.json"
)
SCREEN_REPORT_SHA256 = (
    "97ce3f49b18ee4d0e6fcb498f3d9098b5f0fbe03436cb26982488ae6c561ac7a"
)
AGENTS = 512
EPISODES = 512
SEED = 42039
COLLECTION_STEP_LIMIT = 2048
STATUS_RATE_HZ = 4
POLICY_RATE_HZ = 64
STATUS_HOLD_STEPS = POLICY_RATE_HZ // STATUS_RATE_HZ
PHASE_PRIVILEGED_INDEX = LEGAL_OBS_SIZE + PRIVILEGED_NAMES.index(
    "ordered_gate_phase"
)
PHASE_TAIL_SIZE = PHASE_LEGAL_OBS_SIZE - MASK_SIZE


class PhaseDatasetWriter(TimeMajorDatasetWriter):
    """Time-major writer with one public status value after the old legal tail."""

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
            raise RuntimeError("dataset writer exceeded its preregistered horizon")
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


def update_held_phase(
    raw_phase: np.ndarray,
    held_phase: np.ndarray,
    *,
    step: int,
) -> tuple[np.ndarray, bool]:
    raw = np.asarray(raw_phase, dtype=np.float32)
    held = np.asarray(held_phase, dtype=np.float32).copy()
    if raw.shape != held.shape:
        raise ValueError("raw and held public phase must align")
    sampled = step % STATUS_HOLD_STEPS == 0
    if sampled:
        held[:] = raw
    if not np.isfinite(held).all() or held.min(initial=0.0) < -1e-6 or held.max(
        initial=1.0
    ) > 1.0 + 1e-6:
        raise RuntimeError("held public phase escaped [0,1]")
    return held, sampled


def phase_storage_parts(
    environment_observation: np.ndarray,
    held_phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask, old_tail = legal_storage_parts(environment_observation)
    phase = np.asarray(held_phase, dtype=np.float32)
    if phase.shape != (environment_observation.shape[0],):
        raise ValueError("held phase does not align with native observations")
    tail = np.concatenate((old_tail, phase[:, None]), axis=1)
    if tail.shape[1] != PHASE_TAIL_SIZE:
        raise RuntimeError("phase-legal tail width changed")
    return mask, tail


def phase_dagger_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = load_fixed_config(
        pufferl_module,
        agents=AGENTS,
        episodes=EPISODES,
        seed=SEED,
        controller="governed",
        target_speed_m_s=2.0,
        episode_seconds=180.0,
        randomized_course=True,
    )
    config["env"].update(
        {
            "teacher_action_blend": 0.0,
            "teacher_course_spline": 0,
            "teacher_segment_minimum_jerk": 0,
            "teacher_alignment_governor": 0,
            "w_action_teacher": 0.0,
        }
    )
    return config, overrides


def collect(
    output: Path = DEFAULT_OUTPUT,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frozen = {
        CHECKPOINT: CHECKPOINT_SHA256,
        TRAIN_REPORT: TRAIN_REPORT_SHA256,
        SCREEN_REPORT: SCREEN_REPORT_SHA256,
        SF016_REPORT: SF016_REPORT_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    if not json.loads(SF016_REPORT.read_text()).get("parity_passed"):
        raise RuntimeError("SF016 did not admit the oracle query")
    if json.loads(SCREEN_REPORT.read_text()).get("full_course_passed"):
        raise RuntimeError("phase DAgger is unnecessary after a passing SF034")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF039 preregisters CUDA inference")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled backend is not drone_race_vision")

    device = torch.device(device_name)
    model, checkpoint = load_sf033(device)
    config, overrides = phase_dagger_config(pufferl)
    environment = config["env"]
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native vector ABI differs from SF039")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions_cpu = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    state = model.initial_state(AGENTS, device=device)
    done = np.zeros(AGENTS, dtype=bool)
    lengths = np.zeros(AGENTS, dtype=np.int32)
    tracked_terminal_count = np.zeros(AGENTS, dtype=np.int32)
    held_phase = np.zeros(AGENTS, dtype=np.float32)
    previous_stored_phase = np.zeros(AGENTS, dtype=np.float32)
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_samples = 0
    label_count = 0
    executed_action_max_error = 0.0
    native_log: dict[str, Any] = {}
    manifest: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        vector.reset()
        with tempfile.TemporaryDirectory(
            prefix=f".{TAG}_staging_", dir=output.parent
        ) as temporary:
            writer = PhaseDatasetWriter(
                Path(temporary) / "arrays",
                max_steps=COLLECTION_STEP_LIMIT,
                agents=AGENTS,
            )
            with torch.no_grad():
                for step in range(COLLECTION_STEP_LIMIT):
                    active = ~done
                    if not active.any():
                        break
                    current = observations.numpy()
                    raw_phase = current[:, PHASE_PRIVILEGED_INDEX]
                    held_phase, sampled = update_held_phase(
                        raw_phase, held_phase, step=step
                    )
                    changed = np.abs(held_phase - previous_stored_phase) > 1e-7
                    if changed.any() and not sampled:
                        phase_changes_off_tick += int(changed.sum())
                    phase_decreases += int(
                        (held_phase[active] + 1e-7 < previous_stored_phase[active]).sum()
                    )
                    if sampled:
                        phase_samples += int(active.sum())
                    previous_stored_phase = held_phase.copy()
                    mask, tail = phase_storage_parts(current, held_phase)
                    query_label = alignment_oracle_action(current)
                    legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                    actor_output, candidate_state = model.forward_step(legal, state)
                    active_device = torch.from_numpy(active).to(device)
                    state = torch.where(
                        active_device[None, :, None], candidate_state, state
                    )
                    student_action = torch.where(
                        active_device[:, None],
                        actor_output.mean,
                        torch.zeros_like(actor_output.mean),
                    )
                    if not bool(torch.isfinite(student_action[active_device]).all()):
                        raise RuntimeError("SF033 produced a non-finite action")
                    student_np = student_action.detach().cpu().numpy().astype(
                        np.float32, copy=False
                    )
                    if not np.isfinite(query_label[active]).all():
                        raise RuntimeError("oracle query produced a non-finite label")
                    actions_cpu.copy_(student_action.detach().cpu())
                    vector.cpu_step(actions_cpu.data_ptr())
                    executed = observations.numpy()[
                        :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                    ]
                    executed_action_max_error = max(
                        executed_action_max_error,
                        float(np.max(np.abs(executed[active] - student_np[active]))),
                    )
                    terminal = (terminals.numpy() > 0.5) & active
                    lengths[active] += 1
                    tracked_terminal_count += terminal.astype(np.int32)
                    label_count += int(active.sum())
                    mask[~active] = 0
                    tail[~active] = 0.0
                    query_label[~active] = 0.0
                    writer.append(
                        mask=mask,
                        tail=tail,
                        action=query_label,
                        terminal=terminal.astype(np.uint8),
                        valid=active.astype(np.uint8),
                    )
                    done |= terminal
            native_log = dict(vector.log())
            terminal_count, terminal_is_last = writer.validate_episode_layout(
                lengths
            )
            if not np.array_equal(terminal_count, tracked_terminal_count):
                raise RuntimeError("staged and tracked terminals differ")
            metrics = flatten_log(pufferl, native_log)
            admitted = next_dagger_collection_passes(
                metrics,
                lengths=lengths,
                terminal_count=terminal_count,
                terminal_is_last=terminal_is_last,
                labels=label_count,
                executed_action_max_error=executed_action_max_error,
            )
            admitted = (
                admitted
                and phase_changes_off_tick == 0
                and phase_decreases == 0
                and phase_samples > 0
            )
            if not admitted:
                raise RuntimeError(
                    "SF039 collection failed admission; data were not finalized"
                )
            manifest = writer.finalize(output)
    finally:
        vector.close()

    source_paths = (
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "scripts/collect_vq2_oracle_bc_dataset.py",
        CHECKPOINT,
        TRAIN_REPORT,
        SCREEN_REPORT,
        SF016_REPORT,
    )
    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    metadata = {
        "schema": SCHEMA,
        "tag": TAG,
        "time_steps": int(lengths.max()),
        "agents": AGENTS,
        "episodes": EPISODES,
        "records": int(label_count),
        "episode_lengths": lengths.tolist(),
        "observation": {
            "stored_legal_width": PHASE_LEGAL_OBS_SIZE,
            "legacy_legal_width": LEGAL_OBS_SIZE,
            "mask_width": MASK_SIZE,
            "legal_tail_width": PHASE_TAIL_SIZE,
            "public_status_values_per_record": 1,
            "public_status_source": "held_active_gate_index_div_6",
            "public_status_rate_hz": STATUS_RATE_HZ,
            "stored_training_only_privileged_values_per_record": 0,
        },
        "action": {
            "width": ACTION_SIZE,
            "source": "sf016_alignment_oracle_query_at_student_state",
            "plant_action_source": "sf033_recurrent_actor_mean",
        },
        "files": manifest,
        "source_sha256": source_sha256,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    metrics = flatten_log(pufferl, native_log)
    report = {
        "schema": "vq2_public_phase_dagger_collection_report_v1",
        "tag": TAG,
        "admitted": True,
        "checkpoint_best_epoch": checkpoint["best_epoch"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "screen_report_sha256": SCREEN_REPORT_SHA256,
        "sf016_report_sha256": SF016_REPORT_SHA256,
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "collection_step_limit": COLLECTION_STEP_LIMIT,
        "vector_steps": int(lengths.max()),
        "records": int(label_count),
        "episode_length_min": int(lengths.min()),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_max": int(lengths.max()),
        "status_hold_steps": STATUS_HOLD_STEPS,
        "status_rate_hz": STATUS_RATE_HZ,
        "phase_samples": phase_samples,
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases,
        "teacher_action_blend": 0.0,
        "executed_action_max_error": executed_action_max_error,
        "stored_training_only_privileged_values_per_record": 0,
        "metadata_sha256": sha256_path(output / "metadata.json"),
        "files": manifest,
        "metrics": metrics,
        "loader_overrides": overrides,
        "wall_time_seconds": time.perf_counter() - started,
        "source_sha256": source_sha256,
        "safety": {
            "student_actions_executed": label_count,
            "teacher_actions_executed": 0,
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
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.output.resolve(), device_name=args.device),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

