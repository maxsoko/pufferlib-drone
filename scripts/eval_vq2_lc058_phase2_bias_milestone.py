#!/usr/bin/env python3
"""Fast paired Gate-3 milestone screen for constant phase-2 Puffer biases."""

from __future__ import annotations

import argparse
import json
import os
import platform
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
from pufferlib.vq2_informed import ACTION_HISTORY, ENV_OBS_SIZE, LEGAL_OBS_SIZE
from pufferlib.vq2_public_phase import OFFICIAL_PROGRESS_SCALE, PUBLIC_STATUS_INTERVAL_STEPS
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2UnboundedProgressMLPResidualActor
from scripts.eval_vq2_lc001_long_course_oracle import load_config
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME
from scripts.eval_vq2_recurrent_policy import preserve_frozen_state
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.eval_vq2_lc015_restore_vg071_head1 import state_sha256
import scripts.eval_vq2_lc007_converted_vg071_long_course as core


TAG = "vq2_lc058_phase2_bias_milestone_001"
SCHEMA = "vq2_lc058_phase2_bias_milestone_report_v1"
GROUP_SIZE = 32
BIAS_CANDIDATES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("baseline", (0.0, 0.0, 0.0, 0.0)),
    ("pitch_m0p0025", (-0.0025, 0.0, 0.0, 0.0)),
    ("pitch_m0p0050", (-0.0050, 0.0, 0.0, 0.0)),
    ("pitch_m0p0100", (-0.0100, 0.0, 0.0, 0.0)),
    ("pitch_m0p0200", (-0.0200, 0.0, 0.0, 0.0)),
    ("roll_p0p0100", (0.0, 0.0100, 0.0, 0.0)),
    ("pitch_m0p0050_roll_p0p0100", (-0.0050, 0.0100, 0.0, 0.0)),
    ("pitch_p0p0050", (0.0050, 0.0, 0.0, 0.0)),
)
GROUPS = len(BIAS_CANDIDATES)
TOTAL_AGENTS = GROUP_SIZE * GROUPS
EPISODES = TOTAL_AGENTS
THREADS = 32
SEED = 431580
ENV_SEED_GROUP_SIZE = GROUP_SIZE
ENV_SEED_INDEX_OFFSET = 0
NUM_GATES = 24
MAX_STEPS = 3500
TARGET_RAW_INDEX = 3
TARGET_PHASE = 2
MINIMUM_PASS_GAIN = 1
PARENT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc048_phase6_sequential_interpolation_bracket_001"
)
PARENT_CHECKPOINT = PARENT / "a0p003/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "5a2046a856aa29e8c8bff81789664ff68cd44d67e5ad6e6bf90ba02cf9dec084"
PARENT_REPORT = PARENT / "report.json"
PARENT_REPORT_SHA256 = "0f5849a41f7554b88565caff758ec9a60e25d3c359d3c9603b45646c1dceb669"
PARENT_CHILD_REPORT = PARENT / "a0p003/report.json"
PARENT_CHILD_REPORT_SHA256 = "b2fb203ca9a49e74371694da71f8bd72757a1d0aab3d930ce5c2c8afdd85b3de"
PREREGISTRATION = ROOT / "docs/vq2_lc058_phase2_bias_milestone_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc058_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc058_phase2_bias_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()
NEXT_AUTHORITY_SELECTED = (
    "Run one independent-seed Gate-3 milestone confirmation of the selected bias; "
    "do not promote from this screen alone."
)
NEXT_AUTHORITY_NONE = "Reject the constant-bias family and retain LC048."
VECTORIZED_CHECKPOINT_SURGERY = "phase-2 indexed output bias before tanh"


def group_slice(group: int) -> slice:
    if not 0 <= group < GROUPS:
        raise ValueError("candidate group is outside the LC058 bracket")
    return slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)


def bias_matrix(*, device: torch.device | str = "cpu") -> torch.Tensor:
    rows = []
    for _, values in BIAS_CANDIDATES:
        rows.extend([values] * GROUP_SIZE)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def apply_phase2_bias(
    pre_tanh: torch.Tensor,
    held_progress: torch.Tensor,
    deltas: torch.Tensor,
) -> torch.Tensor:
    """Vectorize exact phase-2 output-bias checkpoint surgery."""

    if pre_tanh.shape != deltas.shape or pre_tanh.shape[-1] != ACTION_SIZE:
        raise ValueError("LC058 action bias tensors do not align")
    if held_progress.shape != (pre_tanh.shape[0], 1):
        raise ValueError("LC058 held progress tensor does not align")
    phase = torch.round(held_progress[:, 0] * OFFICIAL_PROGRESS_SCALE).to(torch.long)
    phase2 = phase == TARGET_PHASE
    return torch.tanh(pre_tanh + phase2[:, None] * deltas)


def candidate_state(
    parent_state: dict[str, torch.Tensor], bias: tuple[float, float, float, float]
) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    row = state["indexed_phase_residual_output_bias"]
    row[TARGET_PHASE].add_(torch.tensor(bias, dtype=row.dtype))
    return state


def build_candidate_context(
    payload: dict[str, Any], *, device: torch.device
) -> Any:
    del payload
    return bias_matrix(device=device)


def apply_candidate_actions(
    actor_output: Any,
    next_recurrent: torch.Tensor,
    held_progress: torch.Tensor,
    context: Any,
) -> torch.Tensor:
    del next_recurrent
    return apply_phase2_bias(actor_output.pre_tanh_mean, held_progress, context)


def initialize_actor_execution(
    payload: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    actor = load_actor(payload, device)
    return {
        "actor": actor,
        "recurrent": actor.initial_state(TOTAL_AGENTS, device=device),
    }


def execute_actor_actions(
    execution: dict[str, Any],
    actor_input: torch.Tensor,
    active: torch.Tensor,
    held_progress: torch.Tensor,
    candidate_context: Any,
) -> torch.Tensor:
    actor_output, next_recurrent = execution["actor"].forward_step(
        actor_input, execution["recurrent"]
    )
    execution["recurrent"] = preserve_frozen_state(
        execution["recurrent"], next_recurrent, active
    )
    return apply_candidate_actions(
        actor_output, next_recurrent, held_progress, candidate_context
    )


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    return candidate_state(parent_state, BIAS_CANDIDATES[candidate_index][1])


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    bias = BIAS_CANDIDATES[candidate_index][1]
    return {
        "bias": list(bias),
        "bias_l2": float(np.linalg.norm(np.asarray(bias, dtype=np.float64))),
    }


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline = items[0]
    eligible = [
        item for item in items[1:]
        if item["transport_pass"]
        and item["gate3_passes"] >= baseline["gate3_passes"] + MINIMUM_PASS_GAIN
        and item["pre_gate3_terminals"] <= baseline["pre_gate3_terminals"]
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            item["gate3_passes"],
            -item["pre_gate3_terminals"],
            -item["bias_l2"],
            -item["candidate_index"],
        ),
    )


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        PARENT_CHILD_REPORT: PARENT_CHILD_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC058 bound input changed: {path}")
    aggregate = json.loads(PARENT_REPORT.read_text())
    child = json.loads(PARENT_CHILD_REPORT.read_text())
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        aggregate.get("schema") != "vq2_lc048_phase6_sequential_interpolation_bracket_report_v1"
        or not aggregate.get("numerically_admitted")
        or aggregate.get("selected", {}).get("alpha") != 0.0025
        or aggregate.get("selected", {}).get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or child.get("schema") != "vq2_lc048_phase6_interpolation_screen_v1"
        or not child.get("diagnostic_valid")
        or payload.get("schema") != "vq2_lc048_phase6_interpolation_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or payload.get("model", {}).get("class") != "VQ2UnboundedProgressMLPResidualActor"
    ):
        raise RuntimeError("LC048 does not authorize the LC058 offline milestone screen")
    return payload


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, PARENT_CHILD_REPORT,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/eval_vq2_lc007_converted_vg071_long_course.py",
        ROOT / "src/vecenv.h", ROOT / "src/bindings.cu", ROOT / "src/bindings_cpu.cpp",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
        *EXTRA_SOURCE_PATHS,
    )
    extension = Path(_C.__file__).resolve()
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {
            **{str(path.relative_to(ROOT)): sha256_path(path) for path in paths},
            "compiled_extension": sha256_path(extension),
        },
        "compiled_extension_path": str(extension),
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
    }


def load_actor(
    payload: dict[str, Any], device: torch.device
) -> VQ2UnboundedProgressMLPResidualActor:
    contract = payload["model"]
    actor = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=int(contract["hidden_size"]),
        residual_size=int(contract["residual_size"]),
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(payload["model_state"])
    actor.eval()
    return actor


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    from pufferlib import _C, pufferl

    payload = verify_inputs()
    if _C.env_name != BACKEND_ENV_NAME or _C.precision_bytes != 4:
        raise RuntimeError("LC058 requires the float32 drone_race_vision backend")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC058 preregisters CUDA inference")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"LC058 requires OMP_NUM_THREADS={THREADS}")
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("LC058 requires OMP_DYNAMIC=FALSE")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC058 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)

    identity = source_identity()
    device = torch.device(device_name)
    actor_execution = initialize_actor_execution(payload, device=device)
    config, overrides = load_config(
        pufferl, num_gates=NUM_GATES, agents=TOTAL_AGENTS,
        episodes=EPISODES, seed=SEED, threads=THREADS,
    )
    config["vec"]["env_seed_group_size"] = ENV_SEED_GROUP_SIZE
    if ENV_SEED_INDEX_OFFSET:
        config["vec"]["env_seed_index_offset"] = ENV_SEED_INDEX_OFFSET
    environment = config["env"]
    environment.update({
        "evaluation_episode_limit": 1,
        "max_steps": MAX_STEPS,
        "time_limit_seconds": MAX_STEPS / 64.0,
        "teacher_action_blend": 0.0,
        "teacher_roll_until_gate_index": 0,
        "teacher_course_spline": 0,
        "teacher_segment_minimum_jerk": 0,
        "teacher_alignment_governor": 0,
        "w_action_teacher": 0.0,
    })
    vector = _C.create_vec(config, gpu=0)
    if vector.total_agents != TOTAL_AGENTS or vector.obs_size != ENV_OBS_SIZE:
        vector.close()
        raise RuntimeError("LC058 native vector ABI changed")
    observations = _cpu_tensor(vector.obs_ptr, (TOTAL_AGENTS, ENV_OBS_SIZE), torch.float32)
    terminals = _cpu_tensor(vector.terminals_ptr, (TOTAL_AGENTS,), torch.float32)
    actions_cpu = torch.zeros((TOTAL_AGENTS, ACTION_SIZE), dtype=torch.float32)
    candidate_context = build_candidate_context(payload, device=device)
    done = torch.zeros(TOTAL_AGENTS, dtype=torch.bool)
    resolved = np.zeros(TOTAL_AGENTS, dtype=bool)
    passed = np.zeros(TOTAL_AGENTS, dtype=bool)
    prepass_terminal = np.zeros(TOTAL_AGENTS, dtype=bool)
    held = np.zeros(TOTAL_AGENTS, dtype=np.float32)
    previous_held = held.copy()
    maximum_raw_index = np.zeros(TOTAL_AGENTS, dtype=np.int32)
    resolve_step = np.full(TOTAL_AGENTS, -1, dtype=np.int32)
    phase_changes_off_tick = np.zeros(GROUPS, dtype=np.int64)
    phase_decreases = np.zeros(GROUPS, dtype=np.int64)
    phase_skips = np.zeros(GROUPS, dtype=np.int64)
    raw_encoding_max_error = np.zeros(GROUPS, dtype=np.float64)
    action_envelope_violations = np.zeros(GROUPS, dtype=np.int64)
    executed_action_max_error = np.zeros(GROUPS, dtype=np.float64)
    initial_groups_exact = False
    nonfinite_action = False
    vector_steps = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        vector.reset()
        initial = observations.numpy()
        initial_groups_exact = all(
            np.array_equal(initial[group_slice(0)], initial[group_slice(group)])
            for group in range(1, GROUPS)
        )
        if not initial_groups_exact:
            raise RuntimeError("LC058 paired seed groups do not begin identically")
        with torch.no_grad():
            for step in range(MAX_STEPS):
                active_cpu = ~done
                if resolved.all() or not bool(active_cpu.any()):
                    break
                current = observations.numpy()
                raw = current[:, core.PHASE_PRIVILEGED_INDEX]
                held, sampled, _ = core.update_held_progress(raw, held, step=step)
                active_np = active_cpu.numpy()
                changed = np.abs(held - previous_held) > 1e-7
                delta_index = np.rint((held - previous_held) * OFFICIAL_PROGRESS_SCALE)
                raw_scaled = raw * OFFICIAL_PROGRESS_SCALE
                raw_indices = np.rint(raw_scaled).astype(np.int32)
                encoding_errors = np.abs(raw_scaled - raw_indices)
                for group in range(GROUPS):
                    selected = group_slice(group)
                    group_active = active_np[selected]
                    if changed[selected].any() and not sampled:
                        phase_changes_off_tick[group] += int(changed[selected].sum())
                    phase_decreases[group] += int(((delta_index[selected] < 0) & group_active).sum())
                    phase_skips[group] += int(((delta_index[selected] > 1) & group_active).sum())
                    raw_encoding_max_error[group] = max(
                        raw_encoding_max_error[group],
                        float(encoding_errors[selected].max(initial=0.0)),
                    )
                previous_held = held.copy()
                maximum_raw_index = np.maximum(maximum_raw_index, raw_indices)
                newly_passed = (~resolved) & (raw_indices >= TARGET_RAW_INDEX)
                passed |= newly_passed
                resolve_step[newly_passed] = step
                resolved |= newly_passed

                legal = observations[:, :LEGAL_OBS_SIZE].to(device)
                progress = torch.from_numpy(held[:, None]).to(device)
                actor_input = torch.cat((legal, progress), dim=1)
                active = active_cpu.to(device)
                inference_started = time.perf_counter()
                action = execute_actor_actions(
                    actor_execution, actor_input, active, progress, candidate_context
                )
                action = torch.where(active[:, None], action, torch.zeros_like(action))
                inference_seconds += time.perf_counter() - inference_started
                if not bool(torch.isfinite(action[active]).all()):
                    nonfinite_action = True
                    break
                action_envelope = (action[active].abs() > 1.0 + 1e-6).sum()
                if int(action_envelope):
                    for group in range(GROUPS):
                        selected = group_slice(group)
                        action_envelope_violations[group] += int(
                            (action[selected][active_cpu[selected].to(device)].abs() > 1.0 + 1e-6).sum()
                        )
                actions_cpu.copy_(action.detach().cpu())
                vector.cpu_step(actions_cpu.data_ptr())
                executed = observations.numpy()[:, ACTION_HISTORY.start:ACTION_HISTORY.start + ACTION_SIZE]
                action_np = actions_cpu.numpy()
                for group in range(GROUPS):
                    selected = group_slice(group)
                    group_active = active_np[selected]
                    executed_action_max_error[group] = max(
                        executed_action_max_error[group],
                        float(np.max(np.abs(executed[selected][group_active] - action_np[selected][group_active]), initial=0.0)),
                    )
                post_raw = observations.numpy()[:, core.PHASE_PRIVILEGED_INDEX]
                post_indices = np.rint(post_raw * OFFICIAL_PROGRESS_SCALE).astype(np.int32)
                maximum_raw_index = np.maximum(maximum_raw_index, post_indices)
                newly_passed = (~resolved) & (post_indices >= TARGET_RAW_INDEX)
                passed |= newly_passed
                resolve_step[newly_passed] = step + 1
                resolved |= newly_passed
                terminal_np = (terminals.numpy() > 0.5) & active_np
                newly_terminal = (~resolved) & terminal_np
                prepass_terminal |= newly_terminal
                resolve_step[newly_terminal] = step + 1
                resolved |= newly_terminal
                done |= terminals > 0.5
                vector_steps = step + 1
    finally:
        vector.close()
    wall = time.perf_counter() - started

    parent_state = payload["model_state"]
    items: list[dict[str, Any]] = []
    for group, (name, bias) in enumerate(BIAS_CANDIDATES):
        selected = group_slice(group)
        baseline_passed = passed[group_slice(0)]
        candidate_passed = passed[selected]
        candidate = candidate_state_for_index(parent_state, group)
        candidate_metadata = candidate_metadata_for_index(group)
        milestone_index = np.minimum(maximum_raw_index[selected], TARGET_RAW_INDEX)
        distribution = {
            str(index): int((milestone_index == index).sum())
            for index in range(TARGET_RAW_INDEX + 1)
        }
        transport = bool(
            not nonfinite_action
            and action_envelope_violations[group] == 0
            and executed_action_max_error[group] <= core.MAX_EXECUTED_ACTION_ERROR
            and phase_changes_off_tick[group] == 0
            and phase_decreases[group] == 0
            and phase_skips[group] == 0
            and raw_encoding_max_error[group] <= 1e-6
        )
        pass_steps = resolve_step[selected][passed[selected]]
        target_passes = int(passed[selected].sum())
        target_pass_rate = float(passed[selected].mean())
        paired_gains = int((candidate_passed & ~baseline_passed).sum())
        paired_losses = int((~candidate_passed & baseline_passed).sum())
        paired_net = int(candidate_passed.sum() - baseline_passed.sum())
        pre_target_terminals = int(prepass_terminal[selected].sum())
        resolve_step_mean = float(pass_steps.mean()) if pass_steps.size else None
        resolve_step_max = int(pass_steps.max()) if pass_steps.size else None
        items.append({
            "candidate_index": group, "name": name, **candidate_metadata,
            "candidate_state_sha256": state_sha256(candidate),
            "target_raw_index": TARGET_RAW_INDEX,
            "target_passes": target_passes,
            "target_pass_rate": target_pass_rate,
            "paired_target_gains_vs_baseline": paired_gains,
            "paired_target_losses_vs_baseline": paired_losses,
            "paired_target_net_vs_baseline": paired_net,
            "pre_target_terminals": pre_target_terminals,
            "target_resolve_step_mean": resolve_step_mean,
            "target_resolve_step_max": resolve_step_max,
            # Retain legacy names for the already-source-locked Gate-3 users.
            "gate3_passes": target_passes,
            "gate3_pass_rate": target_pass_rate,
            "paired_gate3_gains_vs_baseline": paired_gains,
            "paired_gate3_losses_vs_baseline": paired_losses,
            "paired_gate3_net_vs_baseline": paired_net,
            "pre_gate3_terminals": pre_target_terminals,
            "unresolved": int((~resolved[selected]).sum()),
            "gate3_resolve_step_mean": resolve_step_mean,
            "gate3_resolve_step_max": resolve_step_max,
            "maximum_raw_index_distribution": distribution,
            "transport_pass": transport,
            "action_envelope_violations": int(action_envelope_violations[group]),
            "executed_action_max_error": float(executed_action_max_error[group]),
            "phase_changes_off_tick": int(phase_changes_off_tick[group]),
            "phase_decreases": int(phase_decreases[group]),
            "phase_skips": int(phase_skips[group]),
            "raw_progress_encoding_max_error": float(raw_encoding_max_error[group]),
        })
    selected = choose_candidate(items)
    diagnostic_valid = bool(
        initial_groups_exact and not nonfinite_action
        and all(item["transport_pass"] for item in items)
        and items[0]["candidate_state_sha256"] == state_sha256(parent_state)
    )
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "diagnostic_valid": diagnostic_valid,
        "numerically_admitted": False,
        "causal_screen_selected": None if selected is None else selected,
        "selection_requires_independent_confirmation": selected is not None,
        "candidate_groups": GROUPS, "group_size": GROUP_SIZE,
        "minimum_pass_gain": MINIMUM_PASS_GAIN,
        "total_agents": TOTAL_AGENTS, "episodes": EPISODES, "threads": THREADS,
        "seed": SEED, "num_gates": NUM_GATES, "target_raw_index": TARGET_RAW_INDEX,
        "environment_seed_group_size": ENV_SEED_GROUP_SIZE,
        "environment_seed_index_offset": ENV_SEED_INDEX_OFFSET,
        "max_steps": MAX_STEPS, "vector_steps": vector_steps,
        "wall_time_seconds": wall, "inference_seconds": inference_seconds,
        "initial_seed_groups_exact": initial_groups_exact,
        "single_cuda_context": True, "single_native_vector": True,
        "vectorized_checkpoint_surgery": VECTORIZED_CHECKPOINT_SURGERY,
        "items": items, "loader_overrides": overrides,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "source_identity": identity,
        "safety": {
            "runtime_teacher_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            NEXT_AUTHORITY_SELECTED if selected is not None else NEXT_AUTHORITY_NONE
        ),
    }
    write_json_once(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
