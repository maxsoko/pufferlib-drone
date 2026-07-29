#!/usr/bin/env python3
"""Collect a broad legal SF014-visited DAgger dataset for Gate-2 coverage."""

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
from scripts.collect_vq2_oracle_bc_dataset import (
    ACTION_SIZE,
    TimeMajorDatasetWriter,
    legal_storage_parts,
    sha256_path,
)
from scripts.collect_vq2_recurrent_dagger import (
    SF016_REPORT,
    SF016_REPORT_SHA256,
)
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log, load_fixed_config
from scripts.eval_vq2_recurrent_policy import CHECKPOINT, CHECKPOINT_SHA256, load_actor


TAG = "vq2_sf021_recurrent_dagger_broad_512"
SCHEMA = "vq2_recurrent_dagger_time_major_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf021_recurrent_dagger_broad_preregistration_2026-07-28.md"
)
AGENTS = 512
EPISODES = 512
SEED = 42021
EPISODE_SECONDS = 180.0
TARGET_SPEED_M_S = 2.0
COLLECTION_STEP_LIMIT = 512
ORACLE_QUERY_SHA256 = (
    "877e7935b368414488fbaa93ff7bd6b3dce65f5686f224e442fb317a141d7694"
)
RECURRENT_ACTOR_SHA256 = (
    "5ccbee9ce7c98a3a9e9eae627d9f5ea86b3d1c29707eef1921d8c5f56c196199"
)


def broad_dagger_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = load_fixed_config(
        pufferl_module,
        agents=AGENTS,
        episodes=EPISODES,
        seed=SEED,
        controller="governed",
        target_speed_m_s=TARGET_SPEED_M_S,
        episode_seconds=EPISODE_SECONDS,
        randomized_course=True,
    )
    environment = config["env"]
    environment.update(
        {
            "teacher_action_blend": 0.0,
            "teacher_course_spline": 0,
            "teacher_segment_minimum_jerk": 0,
            "teacher_alignment_governor": 0,
            "w_action_teacher": 0.0,
        }
    )
    return config, overrides


def broad_dagger_collection_passes(
    metrics: dict[str, float],
    *,
    lengths: np.ndarray,
    terminal_count: np.ndarray,
    terminal_is_last: np.ndarray,
    labels: int,
    executed_action_max_error: float,
) -> bool:
    return (
        metrics.get("env/n", 0.0) == float(EPISODES)
        and metrics.get("env/crash", 0.0) == 0.0
        and metrics.get("env/timeout", 0.0) == 0.0
        and metrics.get("env/out_of_order", 0.0) == 0.0
        and metrics.get("env/action_envelope_violation", 0.0) == 0.0
        and metrics.get("env/wire_rate_envelope_violation", 0.0) == 0.0
        and metrics.get("env/thrust_envelope_violation", 0.0) == 0.0
        and lengths.shape == (AGENTS,)
        and bool(np.all(lengths > 0))
        and bool(np.all(lengths <= COLLECTION_STEP_LIMIT))
        and bool(np.all(terminal_count == 1))
        and bool(np.all(terminal_is_last))
        and labels == int(lengths.sum())
        and executed_action_max_error <= 1e-7
    )


def collect(
    output: Path = DEFAULT_OUTPUT,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if sha256_path(SF016_REPORT) != SF016_REPORT_SHA256:
        raise RuntimeError("SF016 parity report hash mismatch")
    sf016 = json.loads(SF016_REPORT.read_text())
    if not sf016.get("parity_passed"):
        raise RuntimeError("SF016 did not admit the Python oracle query")
    if sha256_path(ROOT / "pufferlib/vq2_oracle.py") != ORACLE_QUERY_SHA256:
        raise RuntimeError("SF016-admitted oracle query source changed")
    if sha256_path(ROOT / "pufferlib/vq2_recurrent.py") != RECURRENT_ACTOR_SHA256:
        raise RuntimeError("SF014 recurrent actor source changed")
    if sha256_path(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SF014 checkpoint hash mismatch")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF021 preregisters CUDA inference")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled backend is not drone_race_vision")

    device = torch.device(device_name)
    model, checkpoint = load_actor(CHECKPOINT, device=device)
    config, overrides = broad_dagger_config(pufferl)
    environment = config["env"]
    if int(environment["max_steps"]) < COLLECTION_STEP_LIMIT:
        raise RuntimeError("native episode horizon is shorter than the collection cap")

    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native vector ABI differs from SF021")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions_cpu = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    state = model.initial_state(AGENTS, device=device)
    done = np.zeros(AGENTS, dtype=bool)
    lengths = np.zeros(AGENTS, dtype=np.int32)
    tracked_terminal_count = np.zeros(AGENTS, dtype=np.int32)
    label_min = np.full(ACTION_SIZE, np.inf, dtype=np.float64)
    label_max = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
    student_action_min = np.full(ACTION_SIZE, np.inf, dtype=np.float64)
    student_action_max = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
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
            writer = TimeMajorDatasetWriter(
                Path(temporary) / "arrays",
                max_steps=COLLECTION_STEP_LIMIT,
                agents=AGENTS,
            )
            with torch.no_grad():
                for _ in range(COLLECTION_STEP_LIMIT):
                    active = ~done
                    if not active.any():
                        break
                    current = observations.numpy()
                    mask, tail = legal_storage_parts(current)
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
                        raise RuntimeError("student produced a non-finite DAgger action")
                    student_np = student_action.detach().cpu().numpy().astype(
                        np.float32, copy=False
                    )
                    selected_label = query_label[active].astype(np.float64)
                    selected_student = student_np[active].astype(np.float64)
                    label_min = np.minimum(label_min, selected_label.min(axis=0))
                    label_max = np.maximum(label_max, selected_label.max(axis=0))
                    student_action_min = np.minimum(
                        student_action_min, selected_student.min(axis=0)
                    )
                    student_action_max = np.maximum(
                        student_action_max, selected_student.max(axis=0)
                    )
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
                raise RuntimeError("staged and tracked DAgger terminals differ")
            metrics = flatten_log(pufferl, native_log)
            admitted = broad_dagger_collection_passes(
                metrics,
                lengths=lengths,
                terminal_count=terminal_count,
                terminal_is_last=terminal_is_last,
                labels=label_count,
                executed_action_max_error=executed_action_max_error,
            )
            if not admitted:
                raise RuntimeError(
                    "SF021 collection failed admission; permanent labels were not written"
                )
            manifest = writer.finalize(output)
    finally:
        vector.close()

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "scripts/collect_vq2_oracle_bc_dataset.py",
        ROOT / "scripts/eval_vq2_recurrent_policy.py",
        PREREGISTRATION,
        CHECKPOINT,
        SF016_REPORT,
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    metadata = {
        "schema": SCHEMA,
        "tag": TAG,
        "time_steps": int(lengths.max()),
        "collection_step_limit": COLLECTION_STEP_LIMIT,
        "agents": AGENTS,
        "episodes": EPISODES,
        "records": int(label_count),
        "episode_lengths": lengths.tolist(),
        "observation": {
            "stored_legal_width": LEGAL_OBS_SIZE,
            "mask_width": 4096,
            "mask_dtype": "uint8",
            "mask_decode_scale": 1.0 / 255.0,
            "legal_tail_width": LEGAL_OBS_SIZE - 4096,
            "legal_tail_dtype": "float32",
            "native_privileged_width": ENV_OBS_SIZE - LEGAL_OBS_SIZE,
            "stored_privileged_values_per_record": 0,
        },
        "action": {
            "width": ACTION_SIZE,
            "dtype": "float32",
            "source": "sf016_admitted_alignment_oracle_query_at_student_state",
            "plant_action_source": "sf014_recurrent_actor_mean",
        },
        "files": manifest,
        "source_sha256": source_hashes,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    metrics = flatten_log(pufferl, native_log)
    report = {
        "schema": "vq2_recurrent_dagger_collection_report_v1",
        "tag": TAG,
        "admitted": True,
        "checkpoint_best_epoch": checkpoint["best_epoch"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
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
        "teacher_action_blend": float(environment["teacher_action_blend"]),
        "executed_action_max_error": executed_action_max_error,
        "query_label_min": label_min.tolist(),
        "query_label_max": label_max.tolist(),
        "student_action_min": student_action_min.tolist(),
        "student_action_max": student_action_max.tolist(),
        "stored_privileged_values_per_record": 0,
        "metadata_sha256": sha256_path(output / "metadata.json"),
        "files": manifest,
        "metrics": metrics,
        "loader_overrides": overrides,
        "wall_time_seconds": time.perf_counter() - started,
        "source_sha256": source_hashes,
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
    report = collect(args.output.resolve(), device_name=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
