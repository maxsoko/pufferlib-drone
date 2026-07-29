#!/usr/bin/env python3
"""Collect the source-locked SF009 oracle into a legal-only BC dataset.

The native environment necessarily exposes a trailing training-only state
slice, but this collector never passes that slice to its writer.  Each saved
row contains the legal observation at time ``t`` and the complete action that
actually drove the plant, recovered from the newest action-history slot in the
native observation at ``t + 1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    MASK_SIZE,
)
from scripts.eval_vq2_native_oracle import (
    BACKEND_ENV_NAME,
    flatten_log,
    load_fixed_config,
    oracle_passes,
)


TAG = "vq2_sf012_oracle_legal_bc_dataset_64"
SCHEMA = "vq2_oracle_legal_bc_time_major_v1"
AGENTS = 64
EPISODES = 64
SEED = 42012
EPISODE_SECONDS = 180.0
TARGET_SPEED_M_S = 2.0
MASK_SCALE = 1.0 / 255.0
LEGAL_TAIL_SIZE = LEGAL_OBS_SIZE - MASK_SIZE
ACTION_SIZE = 4
DEFAULT_OUTPUT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG
)
PREREGISTRATION = ROOT / "docs/vq2_sf012_oracle_bc_dataset_preregistration_2026-07-28.md"
SF011_REPORT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf011_native_oracle_randomized_course_4096/report.json"
)

# These are the exact admitted SF011 controller/evaluator inputs.  Collection
# fails closed if any one changes before labels are materialized.
FROZEN_SOURCE_SHA256 = {
    "ocean/drone_race/drone_race.c":
        "47f5c40d0cd04e59caf0e3aad604847a8273deb7c4e64f6f683c2a813b2ab5b4",
    "ocean/drone_race/drone_race.h":
        "b202dbf4ce4a142da16abe044f84e63e76313cf04d351147b757525608ef2df8",
    "ocean/drone_race/binding.c":
        "e30b734ee467ac8f5ce2d79f00df9a8d2738b871097317126c6a3c7f242729e2",
    "scripts/eval_vq2_native_oracle.py":
        "98e099a3286173322b56054a98b095faa4366cedd72359516ead606cdb37863f",
    "config/drone_race_vq2_informed_dreamer.ini":
        "5ed3c6d59694c4c710cdd2c578b10d1070364f3c81d5f7e5443e3021042aec8b",
    "pufferlib/vq2_informed.py":
        "35b63ed73891aa7e13ecc3888507894534c8fdf701ebfb7f6f8a52470ec28c2a",
}
FROZEN_SF011_REPORT_SHA256 = (
    "f9b39ba627a3c4c1daf79672175ec35989e29acb8f6ae7ef5ecff212fe7c2a1e"
)
FROZEN_COMPILED_EXTENSION_SHA256 = (
    "2a61ff24489e01c85b678317684ae8fe8607ac6960ee46f5efa48b5ff265ea62"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_sources(root: Path = ROOT) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        digest = sha256_path(root / relative)
        if digest != expected:
            raise RuntimeError(
                f"source-lock mismatch for {relative}: {digest} != {expected}"
            )
        observed[relative] = digest
    sf011_digest = sha256_path(SF011_REPORT)
    if sf011_digest != FROZEN_SF011_REPORT_SHA256:
        raise RuntimeError(
            "SF011 admission report changed before collection: "
            f"{sf011_digest} != {FROZEN_SF011_REPORT_SHA256}"
        )
    observed[str(SF011_REPORT.relative_to(root))] = sf011_digest
    return observed


def legal_storage_parts(
    environment_observation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Copy only the legal mask and 22-value legal sensor tail."""

    if environment_observation.ndim != 2:
        raise ValueError("native observation batch must be rank two")
    if environment_observation.shape[1] != ENV_OBS_SIZE:
        raise ValueError(
            f"expected native width {ENV_OBS_SIZE}, got "
            f"{environment_observation.shape[1]}"
        )
    legal = environment_observation[:, :LEGAL_OBS_SIZE]
    raw_mask = legal[:, :MASK_SIZE]
    tail = legal[:, MASK_SIZE:].astype(np.float32, copy=True)
    if tail.shape[1] != LEGAL_TAIL_SIZE:
        raise RuntimeError("legal tail width changed")
    if not np.isfinite(raw_mask).all() or not np.isfinite(tail).all():
        raise RuntimeError("non-finite legal observation")
    if raw_mask.min(initial=0.0) < -1e-6 or raw_mask.max(initial=1.0) > 1.0 + 1e-6:
        raise RuntimeError("visual mask escaped [0, 1]")
    mask = np.rint(np.clip(raw_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    return mask, tail


def executed_action_labels(
    current_legal: np.ndarray,
    next_environment_observation: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Recover ``a_t`` from ``O_(t+1)`` and audit the history shift."""

    if current_legal.ndim != 2 or current_legal.shape[1] != LEGAL_OBS_SIZE:
        raise ValueError("current legal observation has the wrong ABI")
    if (
        next_environment_observation.ndim != 2
        or next_environment_observation.shape[1] != ENV_OBS_SIZE
        or next_environment_observation.shape[0] != current_legal.shape[0]
    ):
        raise ValueError("next native observation has the wrong ABI")
    active = np.asarray(active, dtype=bool)
    if active.shape != (current_legal.shape[0],):
        raise ValueError("active mask does not align with the observation batch")

    start = ACTION_HISTORY.start
    labels = next_environment_observation[:, start : start + ACTION_SIZE].astype(
        np.float32, copy=True
    )
    if active.any():
        shifted = next_environment_observation[
            active, start + ACTION_SIZE : ACTION_HISTORY.stop
        ]
        previous = current_legal[active, start : ACTION_HISTORY.stop - ACTION_SIZE]
        shift_error = float(np.max(np.abs(shifted - previous), initial=0.0))
        selected = labels[active]
        if not np.isfinite(selected).all():
            raise RuntimeError("oracle emitted a non-finite action label")
        if selected.min(initial=-1.0) < -1.0 - 1e-6 or selected.max(
            initial=1.0
        ) > 1.0 + 1e-6:
            raise RuntimeError("oracle action label escaped [-1, 1]")
    else:
        shift_error = 0.0
    return labels, shift_error


class TimeMajorDatasetWriter:
    """Bounded time-major staging with an exact used-length final copy."""

    _SPECS = {
        "mask": (np.uint8, (MASK_SIZE,)),
        "tail": (np.float32, (LEGAL_TAIL_SIZE,)),
        "action": (np.float32, (ACTION_SIZE,)),
        "terminal": (np.uint8, ()),
        "valid": (np.uint8, ()),
    }

    def __init__(self, staging: Path, *, max_steps: int, agents: int) -> None:
        if max_steps <= 0 or agents <= 0:
            raise ValueError("writer dimensions must be positive")
        staging.mkdir(parents=True, exist_ok=False)
        self.staging = staging
        self.max_steps = max_steps
        self.agents = agents
        self.steps = 0
        self.arrays: dict[str, np.memmap] = {}
        for name, (dtype, trailing) in self._SPECS.items():
            self.arrays[name] = np.lib.format.open_memmap(
                staging / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(max_steps, agents, *trailing),
            )

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
            "tail": (self.agents, LEGAL_TAIL_SIZE),
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

    def finalize(self, output: Path) -> dict[str, dict[str, Any]]:
        if self.steps <= 0:
            raise RuntimeError("cannot finalize an empty dataset")
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.mkdir(parents=True)
        manifest: dict[str, dict[str, Any]] = {}
        for name, (dtype, trailing) in self._SPECS.items():
            source = self.arrays[name]
            destination = np.lib.format.open_memmap(
                output / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(self.steps, self.agents, *trailing),
            )
            for begin in range(0, self.steps, 64):
                end = min(begin + 64, self.steps)
                destination[begin:end] = source[begin:end]
            destination.flush()
            del destination
            path = output / f"{name}.npy"
            manifest[path.name] = {
                "shape": [self.steps, self.agents, *trailing],
                "dtype": np.dtype(dtype).name,
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            }
        for array in self.arrays.values():
            array.flush()
        self.arrays.clear()
        return manifest

    def validate_episode_layout(
        self, lengths: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate every valid prefix and terminal directly from staged rows."""

        lengths = np.asarray(lengths)
        if lengths.shape != (self.agents,):
            raise ValueError("episode lengths do not align with writer agents")
        terminal_count = np.zeros(self.agents, dtype=np.int32)
        terminal_is_last = np.zeros(self.agents, dtype=bool)
        valid = self.arrays["valid"][: self.steps]
        terminal = self.arrays["terminal"][: self.steps]
        for agent, raw_length in enumerate(lengths):
            length = int(raw_length)
            if length <= 0 or length > self.steps:
                raise RuntimeError("episode length escaped the staged time range")
            if not np.all(valid[:length, agent] == 1):
                raise RuntimeError("dataset contains a hole inside an episode")
            if np.any(valid[length:, agent] != 0):
                raise RuntimeError("dataset contains valid rows after episode end")
            terminal_count[agent] = int(terminal[:, agent].sum())
            terminal_is_last[agent] = bool(terminal[length - 1, agent] == 1)
            if np.any(terminal[: length - 1, agent] != 0):
                terminal_is_last[agent] = False
            if np.any(terminal[length:, agent] != 0):
                terminal_is_last[agent] = False
        return terminal_count, terminal_is_last


def dataset_admission_passes(
    metrics: dict[str, float],
    *,
    lengths: np.ndarray,
    terminal_count: np.ndarray,
    terminal_is_last: np.ndarray,
    label_count: int,
    history_shift_max_error: float,
) -> bool:
    if not oracle_passes(metrics, episodes=EPISODES):
        return False
    for gate in range(6):
        if metrics.get(f"env/ordered_gate{gate}_sampled", 0.0) != 1.0:
            return False
        if metrics.get(f"env/ordered_gate{gate}_radial", math.inf) > 0.10:
            return False
    return (
        lengths.shape == (AGENTS,)
        and bool(np.all(lengths > 0))
        and bool(np.all(terminal_count == 1))
        and bool(np.all(terminal_is_last))
        and label_count == int(lengths.sum())
        and history_shift_max_error <= 1e-7
    )


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def collect(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frozen_sources = verify_frozen_sources()
    if not PREREGISTRATION.is_file():
        raise RuntimeError("SF012 preregistration is missing")
    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError(
            f"compiled backend is {getattr(_C, 'env_name', None)!r}; "
            f"expected {BACKEND_ENV_NAME!r}"
        )
    compiled_extension = Path(_C.__file__).resolve()
    compiled_digest = sha256_path(compiled_extension)
    if compiled_digest != FROZEN_COMPILED_EXTENSION_SHA256:
        raise RuntimeError(
            "compiled native extension changed before collection: "
            f"{compiled_digest} != {FROZEN_COMPILED_EXTENSION_SHA256}"
        )

    config, loader_overrides = load_fixed_config(
        pufferl,
        agents=AGENTS,
        episodes=EPISODES,
        seed=SEED,
        controller="governed",
        target_speed_m_s=TARGET_SPEED_M_S,
        episode_seconds=EPISODE_SECONDS,
        randomized_course=True,
    )
    environment = config["env"]
    if int(environment["evaluation_episode_limit"]) != 1:
        raise RuntimeError("SF012 requires exactly one episode per vector agent")
    max_steps = int(environment["max_steps"])

    output.parent.mkdir(parents=True, exist_ok=True)
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("native vector ABI differs from the SF012 contract")
    observations = _cpu_tensor(
        vector.obs_ptr, (AGENTS, ENV_OBS_SIZE), torch.float32
    )
    terminals = _cpu_tensor(vector.terminals_ptr, (AGENTS,), torch.float32)
    zero_actions = torch.zeros((AGENTS, ACTION_SIZE), dtype=torch.float32)

    done = np.zeros(AGENTS, dtype=bool)
    lengths = np.zeros(AGENTS, dtype=np.int32)
    terminal_count = np.zeros(AGENTS, dtype=np.int32)
    action_sum = np.zeros(ACTION_SIZE, dtype=np.float64)
    action_square_sum = np.zeros(ACTION_SIZE, dtype=np.float64)
    action_min = np.full(ACTION_SIZE, np.inf, dtype=np.float64)
    action_max = np.full(ACTION_SIZE, -np.inf, dtype=np.float64)
    label_count = 0
    history_shift_max_error = 0.0
    started = time.perf_counter()
    native_log: dict[str, Any] = {}
    manifest: dict[str, dict[str, Any]] = {}

    try:
        vector.reset()
        with tempfile.TemporaryDirectory(
            prefix=f".{TAG}_staging_", dir=output.parent
        ) as temporary:
            writer = TimeMajorDatasetWriter(
                Path(temporary) / "arrays", max_steps=max_steps, agents=AGENTS
            )
            for _ in range(max_steps):
                active = ~done
                if not active.any():
                    break
                current_environment = observations.numpy()
                current_legal = current_environment[:, :LEGAL_OBS_SIZE].copy()
                mask, tail = legal_storage_parts(current_environment)

                vector.cpu_step(zero_actions.data_ptr())
                next_environment = observations.numpy()
                labels, shift_error = executed_action_labels(
                    current_legal, next_environment, active
                )
                history_shift_max_error = max(
                    history_shift_max_error, shift_error
                )
                terminal = (terminals.numpy() > 0.5) & active

                selected = labels[active].astype(np.float64)
                action_sum += selected.sum(axis=0)
                action_square_sum += np.square(selected).sum(axis=0)
                action_min = np.minimum(action_min, selected.min(axis=0))
                action_max = np.maximum(action_max, selected.max(axis=0))
                label_count += int(active.sum())
                lengths[active] += 1
                terminal_count += terminal.astype(np.int32)

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
            staged_terminal_count, terminal_is_last = (
                writer.validate_episode_layout(lengths)
            )
            if not np.array_equal(staged_terminal_count, terminal_count):
                raise RuntimeError("staged terminal counts differ from collection state")
            admitted = dataset_admission_passes(
                metrics,
                lengths=lengths,
                terminal_count=terminal_count,
                terminal_is_last=terminal_is_last,
                label_count=label_count,
                history_shift_max_error=history_shift_max_error,
            )
            if not admitted:
                raise RuntimeError(
                    "SF012 collection failed admission; permanent labels were not written"
                )
            manifest = writer.finalize(output)
    finally:
        vector.close()

    wall_time = time.perf_counter() - started
    action_mean = action_sum / label_count
    action_variance = np.maximum(action_square_sum / label_count - action_mean**2, 0.0)
    source_hashes = {
        **frozen_sources,
        str(Path(__file__).resolve().relative_to(ROOT)): sha256_path(Path(__file__)),
        str(PREREGISTRATION.relative_to(ROOT)): sha256_path(PREREGISTRATION),
        str(compiled_extension): sha256_path(compiled_extension),
    }
    metrics = flatten_log(pufferl, native_log)
    metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "tag": TAG,
        "time_steps": int(max(lengths)),
        "agents": AGENTS,
        "episodes": EPISODES,
        "records": int(label_count),
        "episode_lengths": lengths.tolist(),
        "observation": {
            "schema": "vq2_visual_ctbr_v2",
            "native_width": ENV_OBS_SIZE,
            "stored_legal_width": LEGAL_OBS_SIZE,
            "mask_width": MASK_SIZE,
            "mask_dtype": "uint8",
            "mask_decode_scale": MASK_SCALE,
            "legal_tail_width": LEGAL_TAIL_SIZE,
            "legal_tail_dtype": "float32",
            "native_privileged_width": ENV_OBS_SIZE - LEGAL_OBS_SIZE,
            "stored_privileged_values_per_record": 0,
        },
        "action": {
            "schema": "normalized_attitude_ctbr_v1",
            "width": ACTION_SIZE,
            "dtype": "float32",
            "source": "next_observation_newest_executed_action_history",
        },
        "files": manifest,
        "source_sha256": source_hashes,
    }
    _json_dump(output / "metadata.json", metadata)
    metadata_sha = sha256_path(output / "metadata.json")
    report: dict[str, Any] = {
        "schema": "vq2_oracle_legal_bc_collection_report_v1",
        "tag": TAG,
        "admitted": True,
        "agents": AGENTS,
        "episodes": EPISODES,
        "seed": SEED,
        "vector_steps": int(max(lengths)),
        "records": int(label_count),
        "wall_time_seconds": wall_time,
        "loader_overrides": loader_overrides,
        "fixed_environment": {
            key: environment[key]
            for key in sorted(environment)
            if key.startswith("teacher_")
            or key.startswith("gate_position_")
            or key.startswith("course_geometry_")
            or key in {
                "dt",
                "num_gates",
                "gate_radius",
                "max_steps",
                "time_limit_seconds",
                "evaluation_episode_limit",
                "evaluation_episode_offset",
                "sitl_plant_domain_randomize",
                "use_custom_start",
            }
        },
        "metrics": metrics,
        "dataset": {
            "metadata_sha256": metadata_sha,
            "files": manifest,
            "episode_length_min": int(lengths.min()),
            "episode_length_mean": float(lengths.mean()),
            "episode_length_max": int(lengths.max()),
            "terminal_count_per_episode_min": int(terminal_count.min()),
            "terminal_count_per_episode_max": int(terminal_count.max()),
            "history_shift_max_error": history_shift_max_error,
            "action_min": action_min.tolist(),
            "action_max": action_max.tolist(),
            "action_mean": action_mean.tolist(),
            "action_std": np.sqrt(action_variance).tolist(),
            "stored_privileged_values_per_record": 0,
        },
        "source_sha256": source_hashes,
        "safety": {
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "student_updates": 0,
            "student_checkpoints_written": 0,
            "submission_authorized": False,
        },
    }
    _json_dump(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = collect(args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
