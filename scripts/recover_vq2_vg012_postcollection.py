#!/usr/bin/env python3
"""Recover VG012 native metrics by replaying its finalized action sequence."""

from __future__ import annotations

import argparse
import json
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
)
from pufferlib.vq2_public_phase import ENGINE_GATE_CAP
from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.collect_vq2_variable_gate_oracle_bc_dataset import (
    PHASE_PRIVILEGED_INDEX,
    PHASE_TAIL_SIZE,
    phase_storage_parts,
    update_held_phase,
)
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME, flatten_log
from scripts.eval_vq2_variable_gate_oracle import (
    current_runtime_manifest,
    sha256_path,
    write_json_atomic,
    write_json_once,
)
from scripts.eval_vq2_variable_gate_recurrent_policy import preserve_frozen_state
from scripts.train_vq2_variable_gate_recurrent_bc import audit_public_phase_layout

import scripts.collect_vq2_variable_gate_dagger as collector
import scripts.collect_vq2_vg012_variable_gate_dagger as vg012


TAG = "vq2_vg013_vg012_postcollection_recovery_001"
SCHEMA = "vq2_vg012_postcollection_recovery_report_v1"
ORIGINAL_SOURCE_COMMIT = "107963410014d8bb8009e1fb4ded2dd79adeab11"
ORIGINAL_DATASET = vg012.DEFAULT_OUTPUT
ORIGINAL_STATE = ORIGINAL_DATASET.with_name(f"{ORIGINAL_DATASET.name}_state.json")
ORIGINAL_METADATA_SHA256 = (
    "e7bfdba11d1a1c65187c820203bb3a675d51b93cdff9270498ce8280aa25caac"
)
ORIGINAL_STATE_SHA256 = (
    "75a89042b24c0d3be4e13c529ecf1119df24e07972f1cf06ae802970e6214520"
)
ORIGINAL_RUN_LOG_SHA256 = (
    "c860ba57ad1d0b22bb800b5c54cbf44cb17f2c9c22eebd8d225e1312ef940d9b"
)
ORIGINAL_ARCHIVE_SHA256 = (
    "c2f868dcb401700d39119c4a5bfd4b94316fc9a73c7cd8ea329780223be506a0"
)
ORIGINAL_COMPILED_EXTENSION_SHA256 = (
    "52d60c205cd08cca1033276ea723ada43305ff3d8feb3f1e3948f4effd6cabb0"
)
ORIGINAL_SOURCE_SHA256 = {
    "scripts/collect_vq2_variable_gate_dagger.py": (
        "2ad03c788bdac170c114eb3c5ba5272afc2c185f4169dd2c870829d90ab242b5"
    ),
    "scripts/collect_vq2_vg012_variable_gate_dagger.py": (
        "baeb921da78f2d7510b76fddca2905203dac15d64445d1c46db79a0ed2646ccb"
    ),
    "scripts/run_vq2_vg012_vast.sh": (
        "08759b2a724f927571a8f3ae6400f752f56d28fef065d68de9f1c61f2d3a3b30"
    ),
    "docs/vq2_vg012_variable_gate_dagger_round2_preregistration_2026-07-30.md": (
        "94dcd090b4e6d3aa92902fc8a2fb2129f688588b27d84c604157c3e77cd5f2f8"
    ),
}
ORIGINAL_ARRAY_SHA256 = {
    "action.npy": "a7e5c358e1cebb8d4c0a2be8abc8026e4800d42565411833de3592b8c1a2ada9",
    "mask.npy": "1f2c8590f1894f8cd86095dbdd023ec1a1ca68c3c737c9c083f717b8c97a5a54",
    "tail.npy": "80b8fabfac393623153073c8d7a2c408b4fb954bcdc49212ed6d347f3bfdff4a",
    "terminal.npy": "a17bc26a493ed25e0ab12fb7c115b45f190150ca093d275fbf60418b2b02f565",
    "valid.npy": "3c18a5a880ca254491aa4f1d8d6b15a386706b58298b6d17030c6ffb749bf1d0",
}
PREREGISTRATION = (
    ROOT / "docs/vq2_vg013_postcollection_recovery_preregistration_2026-07-30.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg013_recovery_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
DEFAULT_ARCHIVE = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg012_failed_postcollection_corpus.tgz"
)
DEFAULT_RUN_LOG = ROOT.parent / "vq2_vg012_collection_001.log"


def json_ready(value: Any) -> Any:
    """Convert NumPy scalars recursively before strict evidence serialization."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def verify_failed_run(
    *,
    dataset: Path,
    state_path: Path,
    run_log: Path,
    archive: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Verify the exact postcollection failure and every immutable array."""

    fixed_hashes = {
        state_path: ORIGINAL_STATE_SHA256,
        dataset / "metadata.json": ORIGINAL_METADATA_SHA256,
        run_log: ORIGINAL_RUN_LOG_SHA256,
        archive: ORIGINAL_ARCHIVE_SHA256,
    }
    observed = {}
    for path, expected in fixed_hashes.items():
        digest = sha256_path(path)
        if digest != expected:
            raise RuntimeError(f"VG013 evidence hash mismatch: {path}")
        observed[source_label(path)] = digest

    if (dataset / "report.json").exists():
        raise RuntimeError("VG012 unexpectedly has an immutable collection report")
    state = json.loads(state_path.read_text())
    metadata = json.loads((dataset / "metadata.json").read_text())
    log_text = run_log.read_text()
    if (
        state.get("schema") != "vq2_vg012_variable_gate_dagger_collection_state_v1"
        or state.get("tag") != vg012.TAG
        or state.get("source_commit") != ORIGINAL_SOURCE_COMMIT
        or state.get("status") != "collecting"
        or state.get("resume_count") != 0
        or state.get("records") != 0
        or state.get("vector_steps") != 0
        or state.get("compiled_extension_path") is None
        or state.get("source_sha256", {}).get("compiled_extension")
        != ORIGINAL_COMPILED_EXTENSION_SHA256
        or state.get("admission_contract", {}).get("minimum_records") != 150_000
        or state.get("admission_contract", {}).get("crash_rate_is_admission")
        is not False
        or state.get("admission_contract", {}).get("gate_2_reach_is_admission")
        is not False
        or any(state.get("source_sha256", {}).get(key) != digest
               for key, digest in ORIGINAL_SOURCE_SHA256.items())
        or any(state.get("safety", {}).get(key) != expected for key, expected in {
            "teacher_actions_executed": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        }.items())
    ):
        raise RuntimeError("VG012 collecting state is not the source-locked failure")
    if (
        "drone_race native regressions ok" not in log_text
        or "drone_race vision native regressions ok" not in log_text
        or "36 passed" not in log_text
        or "TypeError: Object of type bool is not JSON serializable" not in log_text
        or "VG012 DAgger collection failed admission" in log_text
    ):
        raise RuntimeError("VG012 log does not prove the postcollection JSON failure")

    if (
        metadata.get("schema") != "vq2_variable_gate_dagger_time_major_v1"
        or metadata.get("tag") != vg012.TAG
        or metadata.get("time_steps") != 2048
        or metadata.get("agents") != 512
        or metadata.get("episodes") != 512
        or metadata.get("records") != 171_903
        or metadata.get("action", {}).get("plant_action_source")
        != "vg010_recurrent_actor_deterministic_mean"
        or metadata.get("observation", {}).get(
            "stored_training_only_privileged_values_per_record"
        )
        != 0
        or metadata.get("observation", {}).get(
            "stored_total_gate_count_values_per_record"
        )
        != 0
    ):
        raise RuntimeError("VG012 finalized metadata contract changed")
    for name, expected in ORIGINAL_ARRAY_SHA256.items():
        contract = metadata.get("files", {}).get(name, {})
        if contract.get("sha256") != expected:
            raise RuntimeError(f"VG012 metadata hash changed for {name}")
        digest = sha256_path(dataset / name)
        if digest != expected:
            raise RuntimeError(f"VG012 finalized array changed for {name}")
        observed[source_label(dataset / name)] = digest
    return state, metadata, observed


def audit_dataset(dataset: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Recompute every corpus predicate available without native replay."""

    mask = np.load(dataset / "mask.npy", mmap_mode="r")
    tail = np.load(dataset / "tail.npy", mmap_mode="r")
    action = np.load(dataset / "action.npy", mmap_mode="r")
    terminal = np.load(dataset / "terminal.npy", mmap_mode="r")
    valid = np.load(dataset / "valid.npy", mmap_mode="r")
    expected_shapes = {
        "mask": (2048, 512, MASK_SIZE),
        "tail": (2048, 512, PHASE_TAIL_SIZE),
        "action": (2048, 512, ACTION_SIZE),
        "terminal": (2048, 512),
        "valid": (2048, 512),
    }
    for name, expected in expected_shapes.items():
        if tuple(locals()[name].shape) != expected:
            raise RuntimeError(f"VG012 recovered {name} shape changed")
    lengths = np.asarray(valid.sum(axis=0, dtype=np.int64), dtype=np.int64)
    if lengths.tolist() != metadata.get("episode_lengths"):
        raise RuntimeError("VG012 recovered lengths differ from metadata")
    terminal_count = np.asarray(terminal.sum(axis=0), dtype=np.int64)
    terminal_is_last = np.zeros(512, dtype=bool)
    for agent, raw_length in enumerate(lengths):
        length = int(raw_length)
        if (
            length <= 0
            or length > 2048
            or not np.all(valid[:length, agent] == 1)
            or np.any(valid[length:, agent] != 0)
            or not np.all(terminal[: length - 1, agent] == 0)
            or terminal[length - 1, agent] != 1
            or np.any(terminal[length:, agent] != 0)
        ):
            raise RuntimeError(f"VG012 recovered episode {agent} layout changed")
        terminal_is_last[agent] = True
    phase_audit = audit_public_phase_layout(tail, valid, lengths)
    valid_rows = np.asarray(valid, dtype=bool)
    phase_index = np.rint(np.asarray(tail[:, :, -1]) * ENGINE_GATE_CAP).astype(
        np.int32
    )
    phase_records = np.bincount(
        phase_index[valid_rows], minlength=ENGINE_GATE_CAP + 1
    ).astype(np.int64)
    maximum_phase = np.zeros(512, dtype=np.int32)
    for agent, raw_length in enumerate(lengths):
        maximum_phase[agent] = int(phase_index[: int(raw_length), agent].max())
    selected_action = np.asarray(action[valid_rows], dtype=np.float32)
    query_nonfinite = int((~np.isfinite(selected_action)).sum())
    query_envelope = int(
        np.any(np.abs(selected_action) > 1.0 + 1e-6, axis=1).sum()
    )
    return {
        "records": int(lengths.sum()),
        "lengths": lengths,
        "terminal_count": terminal_count,
        "terminal_is_last": terminal_is_last,
        "episode_length_min": int(lengths.min()),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_max": int(lengths.max()),
        "phase_audit": phase_audit,
        "phase_records": phase_records,
        "gate_1_reach": int((maximum_phase >= 1).sum()),
        "gate_2_reach": int((maximum_phase >= 2).sum()),
        "query_nonfinite": query_nonfinite,
        "query_action_envelope_violations": query_envelope,
        "query_action_min": selected_action.min(axis=0).tolist(),
        "query_action_max": selected_action.max(axis=0).tolist(),
    }


def recovery_predicates(
    *,
    replay: dict[str, Any],
    original_predicates: dict[str, bool],
) -> dict[str, bool]:
    predicates = {
        "original_collection_predicates": all(original_predicates.values()),
        "stored_active_layout_exact": replay["active_mismatches"] == 0,
        "stored_observation_mask_exact": replay["mask_max_byte_error"] == 0,
        "stored_observation_tail_exact": replay["tail_max_error"] <= 1e-7,
        "vg010_action_reproduction": (
            replay["actor_recorded_action_max_error"] <= 1e-7
        ),
        "recorded_action_execution_exact": (
            replay["executed_action_max_error"] <= 1e-7
        ),
        "terminal_layout_reproduced": replay["terminal_mismatches"] == 0,
        "all_episodes_replayed": replay["episodes_replayed"] == 512,
        "all_records_replayed": replay["replayed_actions"] == 171_903,
        "all_terminal_actions_derived": replay["derived_terminal_actions"] == 512,
        "all_nonterminal_actions_recorded": (
            replay["recorded_nonterminal_actions"] == 171_391
        ),
        "stored_phase_records_reproduced": (
            replay["stored_phase_records_match"] is True
        ),
        "actor_actions_finite": replay["actor_nonfinite"] == 0,
        "actor_actions_in_envelope": replay["actor_envelope_violations"] == 0,
    }
    return {name: bool(value) for name, value in predicates.items()}


def replay_native(
    *,
    dataset: Path,
    audit: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, float], list[str]]:
    """Replay recorded actions; derive only each unavailable terminal action."""

    from pufferlib import _C, pufferl

    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("VG013 requires drone_race_vision backend")
    if getattr(_C, "precision_bytes", None) != 4:
        raise RuntimeError("VG013 requires a float32 native binding")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VG013 preregisters exact CUDA actor replay")
    vg012.configure_collector()
    config, overrides = collector.dagger_config(pufferl)
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != 512 or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("VG013 native vector ABI changed")
    observations = _cpu_tensor(
        vector.obs_ptr, (512, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (512,), torch.float32)
    actions_cpu = torch.zeros((512, ACTION_SIZE), dtype=torch.float32)
    actor, _ = collector.recurrent_evaluator.load_actor(device)
    recurrent = actor.initial_state(512, device=device)
    mask = np.load(dataset / "mask.npy", mmap_mode="r")
    tail = np.load(dataset / "tail.npy", mmap_mode="r")
    terminal = np.load(dataset / "terminal.npy", mmap_mode="r")
    valid = np.load(dataset / "valid.npy", mmap_mode="r")

    done = np.zeros(512, dtype=bool)
    held_phase = np.zeros(512, dtype=np.float32)
    previous_held_phase = np.zeros(512, dtype=np.float32)
    phase_records = np.zeros(ENGINE_GATE_CAP + 1, dtype=np.int64)
    active_mismatches = 0
    terminal_mismatches = 0
    phase_changes_off_tick = 0
    phase_decreases = 0
    phase_skips = 0
    raw_phase_encoding_max_error = 0.0
    mask_max_byte_error = 0
    tail_max_error = 0.0
    actor_recorded_action_max_error = 0.0
    executed_action_max_error = 0.0
    actor_nonfinite = 0
    actor_envelope_violations = 0
    derived_terminal_actions = 0
    replayed_actions = 0
    history_offset = ACTION_HISTORY.start - MASK_SIZE
    started = time.perf_counter()
    native_log: dict[str, Any] = {}
    try:
        vector.reset()
        with torch.no_grad():
            for step in range(2048):
                active = ~done
                stored_active = np.asarray(valid[step], dtype=bool)
                active_mismatches += int(np.count_nonzero(active ^ stored_active))
                if active_mismatches:
                    raise RuntimeError("VG013 active layout diverged from VG012")
                if not active.any():
                    break
                current = observations.numpy()
                raw_phase = current[:, PHASE_PRIVILEGED_INDEX]
                held_phase, sampled, encoding_error = update_held_phase(
                    raw_phase, held_phase, step=step
                )
                raw_phase_encoding_max_error = max(
                    raw_phase_encoding_max_error, encoding_error
                )
                changed = np.abs(held_phase - previous_held_phase) > 1e-7
                if changed.any() and not sampled:
                    phase_changes_off_tick += int(changed.sum())
                delta = held_phase[active] - previous_held_phase[active]
                phase_decreases += int((delta < -1e-7).sum())
                phase_skips += int(
                    (delta > 1.0 / ENGINE_GATE_CAP + 1e-7).sum()
                )
                previous_held_phase = held_phase.copy()
                phase_index = np.rint(
                    held_phase * ENGINE_GATE_CAP
                ).astype(np.int32)
                phase_records += np.bincount(
                    phase_index[active], minlength=ENGINE_GATE_CAP + 1
                )

                fresh_mask, fresh_tail = phase_storage_parts(current, held_phase)
                mask_max_byte_error = max(
                    mask_max_byte_error,
                    int(np.max(np.abs(
                        fresh_mask[active].astype(np.int16)
                        - np.asarray(mask[step, active], dtype=np.int16)
                    ), initial=0)),
                )
                tail_max_error = max(
                    tail_max_error,
                    float(np.max(np.abs(
                        fresh_tail[active]
                        - np.asarray(tail[step, active], dtype=np.float32)
                    ), initial=0.0)),
                )
                if mask_max_byte_error or tail_max_error > 1e-7:
                    raise RuntimeError("VG013 observation replay diverged from VG012")

                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                phase_tensor = torch.from_numpy(held_phase[:, None]).to(device)
                actor_input = torch.cat((legal, phase_tensor), dim=1)
                active_device = torch.from_numpy(active).to(device)
                actor_result, candidate = actor.forward_step(
                    actor_input, recurrent
                )
                recurrent = preserve_frozen_state(
                    recurrent, candidate, active_device
                )
                predicted = torch.where(
                    active_device[:, None],
                    actor_result.mean,
                    torch.zeros_like(actor_result.mean),
                ).detach().cpu().numpy().astype(np.float32, copy=False)
                selected_prediction = predicted[active]
                actor_nonfinite += int((~np.isfinite(selected_prediction)).sum())
                actor_envelope_violations += int(np.any(
                    np.abs(selected_prediction) > 1.0 + 1e-6, axis=1
                ).sum())

                ending = (np.asarray(terminal[step], dtype=bool) & active)
                continuing = active & ~ending
                chosen = predicted.copy()
                if continuing.any():
                    if step + 1 >= 2048:
                        raise RuntimeError("VG013 continuing row escaped storage")
                    if not np.array_equal(
                        np.asarray(valid[step + 1], dtype=bool), continuing
                    ):
                        raise RuntimeError("VG013 next-row validity changed")
                    recorded = np.asarray(
                        tail[
                            step + 1,
                            :,
                            history_offset : history_offset + ACTION_SIZE,
                        ],
                        dtype=np.float32,
                    )
                    actor_recorded_action_max_error = max(
                        actor_recorded_action_max_error,
                        float(np.max(np.abs(
                            predicted[continuing] - recorded[continuing]
                        ), initial=0.0)),
                    )
                    chosen[continuing] = recorded[continuing]
                if actor_recorded_action_max_error > 1e-7:
                    raise RuntimeError("VG013 actor does not reproduce recorded actions")
                derived_terminal_actions += int(ending.sum())
                replayed_actions += int(active.sum())
                actions_cpu.copy_(torch.from_numpy(chosen))
                vector.cpu_step(actions_cpu.data_ptr())
                executed = observations.numpy()[
                    :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
                ]
                executed_action_max_error = max(
                    executed_action_max_error,
                    float(np.max(np.abs(
                        executed[active] - chosen[active]
                    ), initial=0.0)),
                )
                observed_terminal = (terminals.numpy() > 0.5) & active
                terminal_mismatches += int(
                    np.count_nonzero(observed_terminal ^ ending)
                )
                if terminal_mismatches:
                    raise RuntimeError("VG013 terminal replay diverged from VG012")
                done |= ending
        native_log = dict(vector.log())
    finally:
        vector.close()

    replay = {
        "vector_steps": int(step + 1),
        "episodes_replayed": int(done.sum()),
        "replayed_actions": replayed_actions,
        "recorded_nonterminal_actions": replayed_actions - derived_terminal_actions,
        "derived_terminal_actions": derived_terminal_actions,
        "active_mismatches": active_mismatches,
        "terminal_mismatches": terminal_mismatches,
        "mask_max_byte_error": mask_max_byte_error,
        "tail_max_error": tail_max_error,
        "actor_recorded_action_max_error": actor_recorded_action_max_error,
        "executed_action_max_error": executed_action_max_error,
        "actor_nonfinite": actor_nonfinite,
        "actor_envelope_violations": actor_envelope_violations,
        "phase_records": phase_records.tolist(),
        "phase_changes_off_tick": phase_changes_off_tick,
        "phase_decreases": phase_decreases,
        "phase_skips": phase_skips,
        "raw_phase_encoding_max_error": raw_phase_encoding_max_error,
        "wall_time_seconds": time.perf_counter() - started,
        "stored_phase_records_match": bool(np.array_equal(
            phase_records, audit["phase_records"]
        )),
    }
    return replay, flatten_log(pufferl, native_log), overrides


def current_source_identity(
    *,
    dataset: Path,
    state_path: Path,
    run_log: Path,
    archive: Path,
) -> tuple[str, dict[str, str]]:
    paths = (
        Path(__file__).resolve(),
        PREREGISTRATION,
        RUNNER,
        collector.GOAL_PROMPT,
        dataset / "metadata.json",
        state_path,
        run_log,
        archive,
        vg012.CHECKPOINT,
        vg012.TRAIN_REPORT,
        vg012.VG010_ADMISSION,
        vg012.VG011_REPORT,
        vg012.VG011_REJECTION,
        ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "scripts/collect_vq2_variable_gate_dagger.py",
        ROOT / "scripts/collect_vq2_vg012_variable_gate_dagger.py",
    )
    source_sha256 = {source_label(path): sha256_path(path) for path in paths}
    from pufferlib import _C

    source_sha256["compiled_extension"] = sha256_path(Path(_C.__file__).resolve())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, source_sha256


def run_recovery(
    *,
    output: Path,
    dataset: Path,
    state_path: Path,
    run_log: Path,
    archive: Path,
    device_name: str,
    resume: bool,
) -> dict[str, Any]:
    failed_state, metadata, observed_hashes = verify_failed_run(
        dataset=dataset,
        state_path=state_path,
        run_log=run_log,
        archive=archive,
    )
    audit = audit_dataset(dataset, metadata)
    source_commit, source_sha256 = current_source_identity(
        dataset=dataset,
        state_path=state_path,
        run_log=run_log,
        archive=archive,
    )
    identity = {
        "schema": "vq2_vg012_postcollection_recovery_state_v1",
        "tag": TAG,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "original_source_commit": ORIGINAL_SOURCE_COMMIT,
        "original_state_sha256": ORIGINAL_STATE_SHA256,
        "original_metadata_sha256": ORIGINAL_METADATA_SHA256,
        "original_run_log_sha256": ORIGINAL_RUN_LOG_SHA256,
        "original_archive_sha256": ORIGINAL_ARCHIVE_SHA256,
        "runtime": current_runtime_manifest(),
        "safety": {
            "new_course_samples": 0,
            "teacher_actions_executed": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    recovery_state_path = output / "state.json"
    report_path = output / "report.json"
    if report_path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {report_path}")
        state = json.loads(recovery_state_path.read_text())
        report = json.loads(report_path.read_text())
        if (
            state.get("status") != "admitted"
            or any(state.get(key) != value for key, value in identity.items())
            or not report.get("admitted")
            or report.get("source_sha256") != source_sha256
        ):
            raise RuntimeError("VG013 completed recovery identity changed")
        return report
    if recovery_state_path.exists():
        if not resume:
            raise FileExistsError(f"VG013 recovery state exists: {recovery_state_path}")
        recovery_state = json.loads(recovery_state_path.read_text())
        if any(recovery_state.get(key) != value for key, value in identity.items()):
            raise RuntimeError("VG013 recovery resume identity changed")
        if recovery_state.get("status") != "replaying":
            raise RuntimeError("VG013 recovery is not resumable")
        resume_count = int(recovery_state.get("resume_count", 0)) + 1
    else:
        resume_count = 0
    recovery_state = {
        **identity,
        "status": "replaying",
        "resume_count": resume_count,
    }
    write_json_atomic(recovery_state_path, json_ready(recovery_state))

    replay, metrics, overrides = replay_native(
        dataset=dataset,
        audit=audit,
        device=torch.device(device_name),
    )
    predicate_arguments = {
        "lengths": audit["lengths"],
        "terminal_count": audit["terminal_count"],
        "terminal_is_last": audit["terminal_is_last"],
        "labels": audit["records"],
        "executed_action_max_error": replay["executed_action_max_error"],
        "phase_changes_off_tick": replay["phase_changes_off_tick"],
        "phase_decreases": replay["phase_decreases"],
        "phase_skips": replay["phase_skips"],
        "raw_phase_encoding_max_error": replay["raw_phase_encoding_max_error"],
        "phase_records": audit["phase_records"],
        "query_nonfinite": audit["query_nonfinite"],
        "query_action_envelope_violations": (
            audit["query_action_envelope_violations"]
        ),
    }
    original_predicates = collector.dagger_collection_predicates(
        metrics, **predicate_arguments
    )
    replay_predicates = recovery_predicates(
        replay=replay, original_predicates=original_predicates
    )
    admitted = all(replay_predicates.values())
    report = {
        "schema": SCHEMA,
        "tag": TAG,
        "completed": True,
        "admitted": admitted,
        "original_tag": vg012.TAG,
        "original_failure": {
            "stage": "postcollection_report_serialization",
            "exception": "TypeError: Object of type bool is not JSON serializable",
            "unchanged_rollout_retried": False,
            "source_commit": ORIGINAL_SOURCE_COMMIT,
            "state_sha256": ORIGINAL_STATE_SHA256,
            "metadata_sha256": ORIGINAL_METADATA_SHA256,
            "run_log_sha256": ORIGINAL_RUN_LOG_SHA256,
            "archive_sha256": ORIGINAL_ARCHIVE_SHA256,
        },
        "dataset": {
            "records": audit["records"],
            "episode_length_min": audit["episode_length_min"],
            "episode_length_mean": audit["episode_length_mean"],
            "episode_length_max": audit["episode_length_max"],
            "phase_audit": audit["phase_audit"],
            "phase_records": audit["phase_records"].tolist(),
            "gate_1_reach": audit["gate_1_reach"],
            "gate_2_reach": audit["gate_2_reach"],
            "query_nonfinite": audit["query_nonfinite"],
            "query_action_envelope_violations": (
                audit["query_action_envelope_violations"]
            ),
            "query_action_min": audit["query_action_min"],
            "query_action_max": audit["query_action_max"],
            "array_sha256": ORIGINAL_ARRAY_SHA256,
        },
        "native_replay": replay,
        "metrics": metrics,
        "original_collection_predicates": original_predicates,
        "recovery_predicates": replay_predicates,
        "failed_recovery_predicates": [
            name for name, passed in replay_predicates.items() if not passed
        ],
        "loader_overrides": overrides,
        "observed_original_hashes": observed_hashes,
        "original_state_source_sha256": failed_state["source_sha256"],
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": identity["runtime"],
        "resume_count": resume_count,
        "safety": {
            **identity["safety"],
            "recorded_native_actions_replayed": replay["replayed_actions"],
            "recorded_nonterminal_actions": replay[
                "recorded_nonterminal_actions"
            ],
            "source_derived_terminal_actions": replay[
                "derived_terminal_actions"
            ],
        },
    }
    write_json_once(report_path, json_ready(report))
    recovery_state.update({
        "status": "admitted" if admitted else "rejected",
        "report_sha256": sha256_path(report_path),
        "admitted": admitted,
    })
    write_json_atomic(recovery_state_path, json_ready(recovery_state))
    if not admitted:
        raise RuntimeError("VG013 failed postcollection recovery admission")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=ORIGINAL_DATASET)
    parser.add_argument("--original-state", type=Path, default=ORIGINAL_STATE)
    parser.add_argument("--run-log", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run_recovery(
        output=args.output.resolve(),
        dataset=args.dataset.resolve(),
        state_path=args.original_state.resolve(),
        run_log=args.run_log.resolve(),
        archive=args.archive.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(json_ready(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
