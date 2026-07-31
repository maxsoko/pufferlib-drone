#!/usr/bin/env python3
"""Screen small VG033->VG042 checkpoint update fractions on one fresh fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase import VQ2PhaseRecurrentActor
from scripts.eval_vq2_native_oracle import BACKEND_ENV_NAME
from scripts.eval_vq2_variable_gate_oracle import (
    sha256_path,
    write_json_atomic,
    write_json_once,
)
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save
import scripts.compare_vq2_staged_count5_diagnostic as comparator
import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator


TAG = "vq2_vg044_vg033_to_vg042_checkpoint_line_bracket_001"
SCHEMA = "vq2_checkpoint_line_bracket_report_v1"
STATE_SCHEMA = "vq2_checkpoint_line_bracket_state_v1"
SEED = 429158
AGENTS = 64
EPISODES = 64
NUM_THREADS = 4
MAX_STEPS = 2560
ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50)

PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg033_variable_gate_eight_source_refit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60"
)
PARENT_ADMISSION = (
    ROOT / "docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
)
PARENT_ADMISSION_SHA256 = (
    "dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44"
)
UPDATE_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg042_variable_gate_phase_balanced_refit_001/policy_best.pt"
)
UPDATE_CHECKPOINT_SHA256 = (
    "876b0ecc4dae769f6a9dd04a0114e9623961cf46397499a41d326bdc74141ddc"
)
UPDATE_REPORT = UPDATE_CHECKPOINT.parent / "report.json"
UPDATE_REPORT_SHA256 = (
    "2f4c1c9b3055ec17445b2dacc5d0fcd3bf9e9387c0b340e06eefd74fd96b953f"
)
UPDATE_ADMISSION = (
    ROOT / "docs/vq2_vg042_phase_balanced_refit_admission_2026-07-31.json"
)
UPDATE_ADMISSION_SHA256 = (
    "4b18462aec48479acdc3250d20266e2754d46d2566c9e0f3373f1c46fb803cd8"
)
REJECTION = (
    ROOT
    / "docs/vq2_vg043_paired_count5_diagnostic_rejection_2026-07-31.json"
)
REJECTION_SHA256 = (
    "f396c3774b4bbf059ef79285481ed4a93da0be0c643d68a004312650fba2e87c"
)
GOAL_PROMPT = (
    ROOT / "docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"
)
GOAL_PROMPT_SHA256 = (
    "03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1"
)
PREREGISTRATION = (
    ROOT
    / "docs/vq2_vg044_checkpoint_line_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_vg044_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def runtime_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "numpy": np.__version__,
    }


def alpha_label(alpha: float) -> str:
    return f"{alpha:.3f}".replace(".", "p")


def verify_inputs() -> None:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        PARENT_ADMISSION: PARENT_ADMISSION_SHA256,
        UPDATE_CHECKPOINT: UPDATE_CHECKPOINT_SHA256,
        UPDATE_REPORT: UPDATE_REPORT_SHA256,
        UPDATE_ADMISSION: UPDATE_ADMISSION_SHA256,
        REJECTION: REJECTION_SHA256,
        GOAL_PROMPT: GOAL_PROMPT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG044 input hash mismatch: {path}")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG044 preregistration is missing")

    parent = json.loads(PARENT_REPORT.read_text())
    update = json.loads(UPDATE_REPORT.read_text())
    rejection = json.loads(REJECTION.read_text())
    if (
        parent.get("tag") != "vq2_vg033_variable_gate_eight_source_refit_001"
        or not parent.get("completed")
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG044 parent is not admitted")
    if (
        update.get("tag") != "vq2_vg042_variable_gate_phase_balanced_refit_001"
        or not update.get("completed")
        or not update.get("numerically_admitted")
        or update.get("checkpoint_sha256") != UPDATE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG044 update endpoint is not admitted numerically")
    if (
        rejection.get("schema")
        != "vq2_vg043_paired_count5_diagnostic_rejection_v1"
        or not rejection.get("completed")
        or rejection.get("qualified_for_count11_diagnostic")
        or rejection.get("parent", {}).get("actor") != "VG033"
        or rejection.get("candidate", {}).get("actor") != "VG042"
    ):
        raise RuntimeError("VG043 rejection does not authorize VG044")


def load_endpoints() -> tuple[dict[str, Any], dict[str, Any]]:
    parent = torch.load(
        PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    update = torch.load(
        UPDATE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if (
        parent.get("schema")
        != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or update.get("schema")
        != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or parent.get("model") != update.get("model")
        or parent.get("safety") != update.get("safety")
        or set(parent.get("model_state", {}))
        != set(update.get("model_state", {}))
    ):
        raise RuntimeError("VG044 endpoint actor contracts differ")
    return parent, update


def interpolate_state(
    parent_state: dict[str, torch.Tensor],
    update_state: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("VG044 alpha must be in [0,1]")
    result: dict[str, torch.Tensor] = {}
    for name, parent in parent_state.items():
        update = update_state[name]
        if parent.shape != update.shape or parent.dtype != update.dtype:
            raise RuntimeError(f"VG044 endpoint tensor mismatch: {name}")
        if parent.is_floating_point():
            value = parent.double() + alpha * (
                update.double() - parent.double()
            )
            result[name] = value.to(parent.dtype)
        else:
            if not torch.equal(parent, update):
                raise RuntimeError(
                    f"VG044 nonfloating endpoint differs: {name}"
                )
            result[name] = parent.clone()
    return result


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("VG044 requires drone_race_vision backend")
    if getattr(_C, "precision_bytes", None) != 4:
        raise RuntimeError("VG044 requires float32 native binding")
    extension = Path(_C.__file__).resolve()
    paths = (
        Path(__file__).resolve(),
        PREREGISTRATION,
        RUNNER,
        GOAL_PROMPT,
        PARENT_CHECKPOINT,
        PARENT_REPORT,
        PARENT_ADMISSION,
        UPDATE_CHECKPOINT,
        UPDATE_REPORT,
        UPDATE_ADMISSION,
        REJECTION,
        ROOT / "tests/test_eval_vq2_vg044_checkpoint_line_bracket.py",
        ROOT / "scripts/eval_vq2_variable_gate_recurrent_policy.py",
        ROOT / "scripts/compare_vq2_staged_count5_diagnostic.py",
        ROOT / "scripts/collect_vq2_variable_gate_oracle_bc_dataset.py",
        ROOT / "scripts/eval_vq2_variable_gate_oracle.py",
        ROOT / "scripts/eval_vq2_recurrent_policy.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
        ROOT / "config/drone_race_vq2_informed_dreamer.ini",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
    )
    hashes = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in paths
    }
    hashes["compiled_extension"] = sha256_path(extension)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "source_commit": commit,
        "source_sha256": hashes,
        "runtime": runtime_manifest(),
        "compiled_extension_path": str(extension),
    }


def configure_evaluator() -> Any:
    original_config = evaluator.teacher_free_config

    def bracket_config(
        pufferl_module: Any,
        *,
        num_gates: int,
    ) -> tuple[dict[str, Any], list[str]]:
        config, overrides = original_config(
            pufferl_module, num_gates=num_gates
        )
        config["vec"]["num_threads"] = NUM_THREADS
        config["env"]["max_steps"] = MAX_STEPS
        return config, [
            *overrides,
            "--vec.num-threads",
            str(NUM_THREADS),
            "--env.max-steps",
            str(MAX_STEPS),
        ]

    evaluator.SCHEMA = "vq2_staged_count5_component_v1"
    evaluator.AGENTS = AGENTS
    evaluator.EPISODES_PER_COUNT = EPISODES
    evaluator.TOTAL_EPISODES = EPISODES
    evaluator.SEEDS = {5: SEED}
    evaluator.REQUIRE_ZERO_CROSSING_MARGIN = False
    evaluator.teacher_free_config = bracket_config
    return original_config


def qualifies(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return bool(
        parent["hard_transport_pass"]
        and candidate["hard_transport_pass"]
        and int(candidate["gate_reach"]["1"])
        >= int(parent["gate_reach"]["1"])
        and int(candidate["gate_reach"]["2"])
        >= int(parent["gate_reach"]["2"])
        and candidate["crashes"] <= parent["crashes"]
        and (
            candidate["successes"] > parent["successes"]
            or int(candidate["gate_reach"]["3"])
            > int(parent["gate_reach"]["3"])
        )
    )


def selection_key(item: dict[str, Any]) -> tuple[float, ...]:
    summary = item["summary"]
    return (
        float(summary["successes"]),
        float(summary["gate_reach"]["4"]),
        float(summary["gate_reach"]["3"]),
        float(summary["gate_reach"]["2"]),
        float(summary["mean_gates_passed"]),
        -float(summary["crashes"]),
        -float(item["alpha"]),
    )


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG044 preregisters CUDA inference")
    identity = source_identity()
    state_path = output / "state.json"
    report_path = output / "report.json"
    policy_path = output / "policy_selected.pt"
    state_identity = {
        "schema": STATE_SCHEMA,
        "tag": TAG,
        "alphas": list(ALPHAS),
        "seed": SEED,
        "agents": AGENTS,
        "episodes": EPISODES,
        "num_threads": NUM_THREADS,
        "max_steps": MAX_STEPS,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "update_checkpoint_sha256": UPDATE_CHECKPOINT_SHA256,
        **identity,
        "safety": {
            "teacher_plant_actions": 0,
            "student_updates": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        report = json.loads(report_path.read_text())
        if report.get("source_identity") != identity:
            raise RuntimeError("VG044 completed report source mismatch")
        if report.get("numerically_admitted"):
            if sha256_path(policy_path) != report.get("checkpoint_sha256"):
                raise RuntimeError("VG044 selected checkpoint changed")
        return report
    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"VG044 state exists at {state_path}")
        state = json.loads(state_path.read_text())
        for key, expected in state_identity.items():
            if state.get(key) != expected:
                raise RuntimeError(f"VG044 resume mismatch for {key}")
    else:
        if output.exists():
            raise RuntimeError("VG044 output exists without state")
        output.mkdir(parents=True)
        state = {
            **state_identity,
            "status": "screening",
            "completed_alphas": [],
        }
        write_json_atomic(state_path, state)

    parent_payload, update_payload = load_endpoints()
    states = {
        alpha: interpolate_state(
            parent_payload["model_state"],
            update_payload["model_state"],
            alpha,
        )
        for alpha in ALPHAS
    }
    state_digests = {
        alpha: state_sha256(model_state)
        for alpha, model_state in states.items()
    }
    device = torch.device(device_name)
    original_config = configure_evaluator()
    items: list[dict[str, Any]] = []
    try:
        for alpha in ALPHAS:
            label = alpha_label(alpha)
            count_path = output / f"alpha_{label}.json"
            evaluator.TAG = f"{TAG}_a{label}"
            evaluator.CHECKPOINT_SHA256 = state_digests[alpha]

            def load_actor(
                target: torch.device,
                *,
                model_state: dict[str, torch.Tensor] = states[alpha],
                payload: dict[str, Any] = parent_payload,
                selected_alpha: float = alpha,
            ) -> tuple[VQ2PhaseRecurrentActor, dict[str, Any]]:
                contract = payload["model"]
                actor = VQ2PhaseRecurrentActor(
                    hidden_size=int(contract["hidden_size"]),
                    initial_std=float(contract["initial_std"]),
                ).to(target)
                actor.load_state_dict(model_state)
                actor.eval()
                return actor, {
                    **payload,
                    "tag": TAG,
                    "best_epoch": 0,
                    "interpolation_alpha": selected_alpha,
                }

            evaluator.load_actor = load_actor
            if count_path.is_file():
                count_report = json.loads(count_path.read_text())
                if (
                    count_report.get("source_identity") != identity
                    or count_report.get("checkpoint_sha256")
                    != state_digests[alpha]
                ):
                    raise RuntimeError(
                        f"VG044 completed alpha {alpha} changed"
                    )
            else:
                count_report = evaluator.run_count(
                    num_gates=5,
                    device=device,
                    source_identity=identity,
                )
                write_json_once(count_path, count_report)
            summary = comparator.summarize_count(
                count_report, num_gates=5
            )
            items.append({
                "alpha": alpha,
                "state_sha256": state_digests[alpha],
                "count_path": count_path.name,
                "count_sha256": sha256_path(count_path),
                "summary": summary,
            })
            state.update({
                "status": "screening",
                "completed_alphas": [item["alpha"] for item in items],
            })
            write_json_atomic(state_path, state)
    finally:
        evaluator.teacher_free_config = original_config

    parent = items[0]["summary"]
    qualified = [
        item for item in items[1:] if qualifies(parent, item["summary"])
    ]
    selected = max(qualified, key=selection_key) if qualified else None
    checkpoint_sha256: str | None = None
    if selected is not None:
        selected_alpha = float(selected["alpha"])
        payload = {
            **parent_payload,
            "tag": TAG,
            "model_state": states[selected_alpha],
            "train_config": {
                "operation": "checkpoint_linear_interpolation",
                "alpha": selected_alpha,
                "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
                "update_checkpoint_sha256": UPDATE_CHECKPOINT_SHA256,
            },
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_report_sha256": PARENT_REPORT_SHA256,
            "best_epoch": 0,
            "optimizer_updates": 0,
            "minimum_transition_window_exposure": 4.0,
            "equal_source_weight_audit": True,
            "numerically_admitted": True,
            "source_commit": identity["source_commit"],
            "source_sha256": identity["source_sha256"],
            "runtime": identity["runtime"],
            "safety": state_identity["safety"]
            | {
                "actor_input_privileged_values": 0,
                "teacher_blend": 0.0,
            },
            "interpolation": {
                "alpha": selected_alpha,
                "state_sha256": selected["state_sha256"],
            },
        }
        atomic_torch_save(policy_path, payload)
        checkpoint_sha256 = sha256_path(policy_path)

    report = {
        "schema": SCHEMA,
        "tag": TAG,
        "completed": True,
        "numerically_admitted": selected is not None,
        "best_epoch": 0,
        "optimizer_updates": 0,
        "minimum_transition_window_exposure": 4.0,
        "equal_source_weight_audit": True,
        "alphas": list(ALPHAS),
        "seed": SEED,
        "agents": AGENTS,
        "episodes": EPISODES,
        "num_threads": NUM_THREADS,
        "max_steps": MAX_STEPS,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "update_checkpoint_sha256": UPDATE_CHECKPOINT_SHA256,
        "checkpoint": (
            policy_path.name if selected is not None else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "selected_alpha": (
            float(selected["alpha"]) if selected is not None else None
        ),
        "parent": parent,
        "candidates": items[1:],
        "qualification": {
            "gate1_not_regressed": True,
            "gate2_not_regressed": True,
            "crash_count_not_regressed": True,
            "strict_gate3_or_finish_improvement": True,
            "hard_transport_required": True,
        },
        "source_identity": identity,
        "safety": state_identity["safety"]
        | {
            "actor_input_privileged_values": 0,
            "teacher_blend": 0.0,
        },
        "next_authority": (
            "Preregister one fresh paired confirmation of the selected alpha."
            if selected is not None
            else "Reject this line direction and start a causally distinct offline repair."
        ),
    }
    if any(
        not math.isfinite(value)
        for item in items
        for value in (
            float(item["summary"]["mean_gates_passed"]),
            float(item["summary"]["executed_action_max_error"]),
        )
    ):
        raise RuntimeError("VG044 report contains a nonfinite metric")
    write_json_once(report_path, report)
    state.update({
        "status": "admitted" if selected is not None else "rejected",
        "completed_alphas": list(ALPHAS),
        "selected_alpha": report["selected_alpha"],
        "report_sha256": sha256_path(report_path),
        "checkpoint_sha256": checkpoint_sha256,
    })
    write_json_atomic(state_path, state)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
