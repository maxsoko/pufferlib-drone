#!/usr/bin/env python3
"""Collect oracle labels on the fresh SF042 public-phase state distribution."""

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
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.collect_vq2_recurrent_dagger import SF016_REPORT, SF016_REPORT_SHA256
from scripts.collect_vq2_recurrent_dagger_next import next_dagger_collection_passes
import scripts.collect_vq2_public_phase_dagger as sf039
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log


TAG = "vq2_sf043_public_phase_dagger_next_512"
SCHEMA = "vq2_public_phase_dagger_time_major_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf043_public_phase_dagger_next_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf042_public_phase_recurrent_fit_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "fe5cd944a588b574e3106c72c7859d0e044efd827871f47358b21473bcddb03c"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "08f76283b79f1be0370ff6403a96e85a33d042e2bbe3d95b9d6a461cb067e800"
)
AGENTS = 512
EPISODES = 512
SEED = 42043
COLLECTION_STEP_LIMIT = 2048


def phase_actor_observation(
    environment_observation: torch.Tensor,
    held_phase: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    if tuple(environment_observation.shape) != (AGENTS, ENV_OBS_SIZE):
        raise ValueError("native observation batch changed")
    phase = torch.from_numpy(np.asarray(held_phase, dtype=np.float32).copy())
    phase = phase[:, None].to(device)
    return torch.cat(
        (environment_observation[:, :LEGAL_OBS_SIZE].to(device), phase), dim=1
    )


def load_actor(device: torch.device) -> tuple[VQ2PhaseResidualActor, dict[str, Any]]:
    if sha256_path(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SF042 checkpoint hash mismatch")
    if sha256_path(TRAIN_REPORT) != TRAIN_REPORT_SHA256:
        raise RuntimeError("SF042 report hash mismatch")
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != "vq2_public_phase_recurrent_checkpoint_v1"
        or contract.get("class") != "VQ2PhaseResidualActor"
        or contract.get("legal_observation_size") != PHASE_LEGAL_OBS_SIZE
        or contract.get("action_size") != ACTION_SIZE
    ):
        raise RuntimeError("SF042 actor contract changed")
    if payload.get("safety", {}).get(
        "stored_training_only_privileged_values_per_actor_record"
    ) != 0:
        raise RuntimeError("SF042 does not prove a legal actor boundary")
    model = VQ2PhaseResidualActor(
        hidden_size=int(contract["hidden_size"]),
        initial_std=float(contract["initial_std"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


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
        SF016_REPORT: SF016_REPORT_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    if not json.loads(SF016_REPORT.read_text()).get("parity_passed"):
        raise RuntimeError("SF016 did not admit the oracle query")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF043 preregisters CUDA inference")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("compiled backend is not drone_race_vision")

    device = torch.device(device_name)
    model, checkpoint = load_actor(device)
    sf039.SEED = SEED
    config, overrides = sf039.phase_dagger_config(pufferl)
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native vector ABI differs from SF043")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    actions_cpu = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)
    state = model.initial_state(AGENTS, device=device)
    done = np.zeros(AGENTS, dtype=bool)
    lengths = np.zeros(AGENTS, dtype=np.int32)
    terminal_counts = np.zeros(AGENTS, dtype=np.int32)
    held_phase = np.zeros(AGENTS, dtype=np.float32)
    previous_phase = np.zeros(AGENTS, dtype=np.float32)
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_record_counts: dict[str, int] = {}
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
            writer = sf039.PhaseDatasetWriter(
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
                    raw_phase = current[:, sf039.PHASE_PRIVILEGED_INDEX]
                    held_phase, sampled = sf039.update_held_phase(
                        raw_phase, held_phase, step=step
                    )
                    changed = np.abs(held_phase - previous_phase) > 1e-7
                    if changed.any() and not sampled:
                        phase_changes_off_tick += int(changed.sum())
                    phase_decreases += int(
                        (held_phase[active] + 1e-7 < previous_phase[active]).sum()
                    )
                    previous_phase = held_phase.copy()
                    for value in np.unique(held_phase[active]):
                        key = f"{float(value):.9f}"
                        phase_record_counts[key] = phase_record_counts.get(key, 0) + int(
                            (held_phase[active] == value).sum()
                        )
                    mask, tail = sf039.phase_storage_parts(current, held_phase)
                    label = alignment_oracle_action(current)
                    actor_observation = phase_actor_observation(
                        observations, held_phase, device=device
                    )
                    actor_output, candidate_state = model.forward_step(
                        actor_observation, state
                    )
                    active_device = torch.from_numpy(active).to(device)
                    state = torch.where(
                        active_device[None, :, None], candidate_state, state
                    )
                    action = torch.where(
                        active_device[:, None],
                        actor_output.mean,
                        torch.zeros_like(actor_output.mean),
                    )
                    if not bool(torch.isfinite(action[active_device]).all()):
                        raise RuntimeError("SF042 produced a non-finite action")
                    action_np = action.detach().cpu().numpy().astype(
                        np.float32, copy=False
                    )
                    if not np.isfinite(label[active]).all():
                        raise RuntimeError("oracle query produced a non-finite label")
                    actions_cpu.copy_(action.detach().cpu())
                    vector.cpu_step(actions_cpu.data_ptr())
                    executed = observations.numpy()[
                        :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                    ]
                    executed_action_max_error = max(
                        executed_action_max_error,
                        float(np.max(np.abs(executed[active] - action_np[active]))),
                    )
                    terminal = (terminals.numpy() > 0.5) & active
                    lengths[active] += 1
                    terminal_counts += terminal.astype(np.int32)
                    label_count += int(active.sum())
                    mask[~active] = 0
                    tail[~active] = 0.0
                    label[~active] = 0.0
                    writer.append(
                        mask=mask,
                        tail=tail,
                        action=label,
                        terminal=terminal.astype(np.uint8),
                        valid=active.astype(np.uint8),
                    )
                    done |= terminal
            native_log = dict(vector.log())
            staged_counts, terminal_is_last = writer.validate_episode_layout(lengths)
            if not np.array_equal(staged_counts, terminal_counts):
                raise RuntimeError("staged and tracked terminals differ")
            metrics = flatten_log(pufferl, native_log)
            admitted = next_dagger_collection_passes(
                metrics,
                lengths=lengths,
                terminal_count=staged_counts,
                terminal_is_last=terminal_is_last,
                labels=label_count,
                executed_action_max_error=executed_action_max_error,
            )
            admitted = (
                admitted
                and phase_changes_off_tick == 0
                and phase_decreases == 0
            )
            if not admitted:
                raise RuntimeError(
                    "SF043 collection failed admission; data were not finalized"
                )
            manifest = writer.finalize(output)
    finally:
        vector.close()

    source_paths = (
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "scripts/collect_vq2_public_phase_dagger.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "pufferlib/vq2_oracle.py",
        CHECKPOINT,
        TRAIN_REPORT,
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
            "mask_width": sf039.MASK_SIZE,
            "legal_tail_width": sf039.PHASE_TAIL_SIZE,
            "public_status_values_per_record": 1,
            "public_status_source": "held_active_gate_index_div_6",
            "public_status_rate_hz": sf039.STATUS_RATE_HZ,
            "stored_training_only_privileged_values_per_record": 0,
        },
        "action": {
            "width": ACTION_SIZE,
            "source": "sf016_alignment_oracle_query_at_student_state",
            "plant_action_source": "sf042_phase_recurrent_actor_mean",
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
        "train_report_sha256": TRAIN_REPORT_SHA256,
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "collection_step_limit": COLLECTION_STEP_LIMIT,
        "vector_steps": int(lengths.max()),
        "records": int(label_count),
        "episode_length_min": int(lengths.min()),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_max": int(lengths.max()),
        "phase_record_counts": phase_record_counts,
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

