#!/usr/bin/env python3
"""Bracket VG033->VG053 fractions on new six-gate episode offsets."""

from __future__ import annotations

import argparse
import json
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
import scripts.compare_vq2_staged_count5_diagnostic as summarizer
import scripts.eval_vq2_variable_gate_recurrent_policy as evaluator
import scripts.eval_vq2_vg044_checkpoint_line_bracket as interpolation


TAG = "vq2_vg055_vg033_to_vg053_count6_fraction_bracket_001"
SCHEMA = "vq2_vg055_count6_fraction_bracket_report_v1"
STATE_SCHEMA = "vq2_vg055_count6_fraction_bracket_state_v1"
ALPHAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
OFFSETS = (32, 40, 48, 56)
SEEDS = (429175, 429176, 429177, 429178)
AGENTS = 32
EPISODES = 32
THREADS = 4
MAX_STEPS = 3072
NUM_GATES = 6

PARENT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg033_variable_gate_eight_source_refit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = "56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26"
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = "033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60"
PARENT_ADMISSION = ROOT / "docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
PARENT_ADMISSION_SHA256 = "dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44"
UPDATE_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg053_late_gate_trust_refit_001/policy_best.pt"
)
UPDATE_CHECKPOINT_SHA256 = "f0a9812f55ce95d437af06a043c64d75a3ea757ac6965df9e3cdf64f30563ce5"
UPDATE_REPORT = UPDATE_CHECKPOINT.parent / "report.json"
UPDATE_REPORT_SHA256 = "ec8b81b3d357fcbfb4105ceda1fd22b43ce921d08b238bea35224fc49485f8b6"
UPDATE_ADMISSION = ROOT / "docs/vq2_vg053_late_gate_trust_refit_admission_2026-07-31.json"
UPDATE_ADMISSION_SHA256 = "3838729ec1a5d2ffbb596aae436804099ce361035eccce91b2ad83145770b134"
REJECTION = ROOT / "docs/vq2_vg054_count6_multi_offset_rejection_2026-07-31.json"
REJECTION_SHA256 = "bb78f928927d3b790e58cb17c79c88bcbc49a14ea4edaa27b229e059b8ddbbf4"
GOAL = ROOT / "docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"
GOAL_SHA256 = "03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1"
PREREGISTRATION = ROOT / "docs/vq2_vg055_count6_fraction_bracket_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg055_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
ACTOR_CLASS = VQ2PhaseRecurrentActor
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()


def runtime_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "numpy": np.__version__,
    }


def verify_inputs() -> None:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        PARENT_ADMISSION: PARENT_ADMISSION_SHA256,
        UPDATE_CHECKPOINT: UPDATE_CHECKPOINT_SHA256,
        UPDATE_REPORT: UPDATE_REPORT_SHA256,
        UPDATE_ADMISSION: UPDATE_ADMISSION_SHA256,
        REJECTION: REJECTION_SHA256,
        GOAL: GOAL_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG055 evidence hash mismatch: {path}")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG055 preregistration is missing")
    update = json.loads(UPDATE_REPORT.read_text())
    admission = json.loads(UPDATE_ADMISSION.read_text())
    rejection = json.loads(REJECTION.read_text())
    if (
        not update.get("numerically_admitted")
        or update.get("checkpoint_sha256") != UPDATE_CHECKPOINT_SHA256
        or update.get("safety", {}).get("flight_sim_packets_sent") != 0
        or update.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("VG055 update endpoint is not admitted")
    if (
        not admission.get("numerically_admitted")
        or admission.get("artifact_sha256", {}).get("checkpoint")
        != UPDATE_CHECKPOINT_SHA256
        or admission.get("live_authority")
    ):
        raise RuntimeError("VG055 endpoint admission changed")
    if (
        rejection.get("schema") != "vq2_vg054_count6_multi_offset_rejection_v1"
        or not rejection.get("completed")
        or rejection.get("qualified")
        or rejection.get("qualification_predicates", {}).get("gate2_not_regressed")
        or not rejection.get("unchanged_retry_forbidden")
    ):
        raise RuntimeError("VG054 rejection does not authorize VG055")


def load_endpoints() -> tuple[dict[str, Any], dict[str, Any]]:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    update = torch.load(UPDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        parent.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or update.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or parent.get("model") != update.get("model")
        or set(parent.get("model_state", {})) != set(update.get("model_state", {}))
    ):
        raise RuntimeError("VG055 endpoint actor contracts differ")
    return parent, update


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    if getattr(_C, "env_name", None) != BACKEND_ENV_NAME:
        raise RuntimeError("VG055 requires drone_race_vision")
    if getattr(_C, "precision_bytes", None) != 4:
        raise RuntimeError("VG055 requires float32 native binding")
    extension = Path(_C.__file__).resolve()
    paths = (
        Path(__file__).resolve(),
        PREREGISTRATION,
        RUNNER,
        GOAL,
        PARENT_CHECKPOINT,
        PARENT_REPORT,
        PARENT_ADMISSION,
        UPDATE_CHECKPOINT,
        UPDATE_REPORT,
        UPDATE_ADMISSION,
        REJECTION,
        ROOT / "scripts/eval_vq2_vg044_checkpoint_line_bracket.py",
        ROOT / "scripts/eval_vq2_variable_gate_recurrent_policy.py",
        ROOT / "scripts/compare_vq2_staged_count5_diagnostic.py",
        ROOT / "scripts/collect_vq2_variable_gate_oracle_bc_dataset.py",
        ROOT / "scripts/eval_vq2_variable_gate_oracle.py",
        ROOT / "ocean/drone_race/drone_race.c",
        ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "ocean/drone_race/binding.c",
        ROOT / "config/drone_race_vq2_informed_dreamer.ini",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        *EXTRA_SOURCE_PATHS,
    )
    hashes = {str(path.relative_to(ROOT)): sha256_path(path) for path in paths}
    hashes["compiled_extension"] = sha256_path(extension)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "source_commit": commit,
        "source_sha256": hashes,
        "runtime": runtime_manifest(),
        "compiled_extension_path": str(extension),
        "episode_offsets": list(OFFSETS),
    }


def offset_config(original: Any, *, offset: int) -> Any:
    def config(pufferl_module: Any, *, num_gates: int) -> tuple[dict[str, Any], list[str]]:
        values, overrides = original(pufferl_module, num_gates=num_gates)
        values["vec"]["num_threads"] = THREADS
        values["env"]["max_steps"] = MAX_STEPS
        values["env"]["evaluation_episode_offset"] = offset
        return values, [*overrides, "--vec.num-threads", str(THREADS),
                        "--env.max-steps", str(MAX_STEPS),
                        "--env.evaluation-episode-offset", str(offset)]
    return config


def aggregate_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = EPISODES * len(OFFSETS)
    return {
        "episodes": total,
        "successes": sum(item["summary"]["successes"] for item in items),
        "crashes": sum(item["summary"]["crashes"] for item in items),
        "misses": sum(item["summary"]["misses"] for item in items),
        "gate_reach": {
            str(gate): sum(int(item["summary"]["gate_reach"][str(gate)]) for item in items)
            for gate in range(1, NUM_GATES + 1)
        },
        "mean_gates_passed": sum(item["summary"]["mean_gates_passed"] * EPISODES for item in items) / total,
        "all_hard_transport_pass": all(item["summary"]["hard_transport_pass"] for item in items),
    }


def qualifies(parent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    p, c = parent["gate_reach"], candidate["gate_reach"]
    downstream_parent = (parent["successes"], p["6"], p["5"], p["4"], p["3"], parent["mean_gates_passed"])
    downstream_candidate = (candidate["successes"], c["6"], c["5"], c["4"], c["3"], candidate["mean_gates_passed"])
    return bool(
        parent["all_hard_transport_pass"] and candidate["all_hard_transport_pass"]
        and c["1"] >= p["1"] and c["2"] >= p["2"]
        and candidate["crashes"] <= parent["crashes"]
        and downstream_candidate > downstream_parent
    )


def selection_key(item: dict[str, Any]) -> tuple[float, ...]:
    s, g = item["aggregate"], item["aggregate"]["gate_reach"]
    return (float(s["successes"]), float(g["6"]), float(g["5"]),
            float(g["4"]), float(g["3"]), float(g["2"]),
            float(s["mean_gates_passed"]), -float(s["crashes"]),
            -float(item["alpha"]))


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG055 requires CUDA inference")
    identity = source_identity()
    state_path, report_path = output / "state.json", output / "report.json"
    policy_path = output / "policy_selected.pt"
    state_identity = {
        "schema": STATE_SCHEMA, "tag": TAG, "alphas": list(ALPHAS),
        "offsets": list(OFFSETS), "seeds": list(SEEDS), "agents": AGENTS,
        "episodes_per_offset": EPISODES, "threads": THREADS,
        "max_steps": MAX_STEPS, "num_gates": NUM_GATES,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "update_checkpoint_sha256": UPDATE_CHECKPOINT_SHA256,
        "source_identity": identity,
    }
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        report = json.loads(report_path.read_text())
        if report.get("source_identity") != identity:
            raise RuntimeError("VG055 completed source changed")
        return report
    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"VG055 state exists at {output}")
        state = json.loads(state_path.read_text())
        for key, expected in state_identity.items():
            if state.get(key) != expected:
                raise RuntimeError(f"VG055 resume mismatch: {key}")
    else:
        if output.exists():
            raise RuntimeError("VG055 output exists without state")
        output.mkdir(parents=True)
        state = {**state_identity, "status": "screening", "completed_alphas": []}
        write_json_atomic(state_path, state)

    parent_payload, update_payload = load_endpoints()
    states = {alpha: interpolation.interpolate_state(parent_payload["model_state"], update_payload["model_state"], alpha) for alpha in ALPHAS}
    state_hashes = {alpha: interpolation.state_sha256(value) for alpha, value in states.items()}
    original = evaluator.teacher_free_config
    evaluator.SCHEMA = "vq2_staged_gate_count_component_v1"
    evaluator.AGENTS = AGENTS
    evaluator.EPISODES_PER_COUNT = EPISODES
    evaluator.TOTAL_EPISODES = EPISODES
    evaluator.MINIMUM_SUCCESS_RATE = 0.0
    evaluator.REQUIRE_ZERO_CROSSING_MARGIN = False
    results: list[dict[str, Any]] = []
    try:
        for alpha in ALPHAS:
            label = interpolation.alpha_label(alpha)
            alpha_dir = output / f"alpha_{label}"
            alpha_dir.mkdir(exist_ok=True)
            offset_items: list[dict[str, Any]] = []
            for offset, seed in zip(OFFSETS, SEEDS, strict=True):
                path = alpha_dir / f"offset_{offset}.json"
                evaluator.TAG = f"{TAG}_a{label}_o{offset}"
                evaluator.SEEDS = {NUM_GATES: seed}
                evaluator.CHECKPOINT_SHA256 = state_hashes[alpha]
                evaluator.teacher_free_config = offset_config(original, offset=offset)
                def load_actor(target: torch.device, *, model_state: dict[str, torch.Tensor] = states[alpha], selected_alpha: float = alpha) -> tuple[VQ2PhaseRecurrentActor, dict[str, Any]]:
                    contract = parent_payload["model"]
                    actor = ACTOR_CLASS(
                        hidden_size=int(contract["hidden_size"]),
                        initial_std=float(contract["initial_std"]),
                    ).to(target)
                    actor.load_state_dict(model_state)
                    actor.eval()
                    return actor, {**parent_payload, "tag": TAG, "best_epoch": 0, "interpolation_alpha": selected_alpha}
                evaluator.load_actor = load_actor
                if path.is_file():
                    count = json.loads(path.read_text())
                    if count.get("source_identity") != identity or count.get("checkpoint_sha256") != state_hashes[alpha]:
                        raise RuntimeError(f"VG055 alpha/offset changed: {alpha}/{offset}")
                else:
                    count = evaluator.run_count(num_gates=NUM_GATES, device=torch.device(device_name), source_identity=identity)
                    write_json_once(path, count)
                offset_items.append({"offset": offset, "seed": seed, "count_path": str(path.relative_to(output)), "count_sha256": sha256_path(path), "summary": summarizer.summarize_count(count, num_gates=NUM_GATES)})
            results.append({"alpha": alpha, "state_sha256": state_hashes[alpha], "aggregate": aggregate_items(offset_items), "items": offset_items})
            state["completed_alphas"] = [item["alpha"] for item in results]
            write_json_atomic(state_path, state)
    finally:
        evaluator.teacher_free_config = original

    parent = results[0]["aggregate"]
    qualified = [item for item in results[1:] if qualifies(parent, item["aggregate"])]
    selected = max(qualified, key=selection_key) if qualified else None
    checkpoint_sha256 = None
    if selected is not None:
        alpha = float(selected["alpha"])
        payload = {
            **parent_payload, "tag": TAG, "model_state": states[alpha],
            "train_config": {"operation": "checkpoint_linear_interpolation", "alpha": alpha,
                             "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
                             "update_checkpoint_sha256": UPDATE_CHECKPOINT_SHA256},
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_report_sha256": PARENT_REPORT_SHA256,
            "best_epoch": 0, "optimizer_updates": 0,
            "minimum_transition_window_exposure": 4.0,
            "equal_source_weight_audit": True, "numerically_admitted": True,
            "source_commit": identity["source_commit"],
            "source_sha256": identity["source_sha256"], "runtime": identity["runtime"],
            "safety": {"actor_input_privileged_values": 0, "teacher_blend": 0.0,
                       "teacher_plant_actions": 0, "flight_sim_packets_sent": 0,
                       "submission_authorized": False},
            "interpolation": {"alpha": alpha, "state_sha256": selected["state_sha256"]},
        }
        atomic_torch_save(policy_path, payload)
        checkpoint_sha256 = sha256_path(policy_path)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": selected is not None, "alphas": list(ALPHAS),
        "offsets": list(OFFSETS), "episodes_per_alpha": AGENTS * len(OFFSETS),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "update_checkpoint_sha256": UPDATE_CHECKPOINT_SHA256,
        "selected_alpha": None if selected is None else selected["alpha"],
        "checkpoint": None if selected is None else policy_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "parent": parent, "candidates": results[1:],
        "source_identity": identity,
        "safety": {"teacher_plant_actions": 0, "student_updates": 0,
                   "flight_sim_packets_sent": 0, "submission_authorized": False},
        "next_authority": ("Preregister a larger independent multi-offset six-gate screen." if selected is not None else "Reject this update line and pursue a causally distinct repair."),
    }
    write_json_once(report_path, report)
    state.update({"status": "admitted" if selected is not None else "rejected",
                  "completed_alphas": list(ALPHAS), "selected_alpha": report["selected_alpha"],
                  "report_sha256": sha256_path(report_path), "checkpoint_sha256": checkpoint_sha256})
    write_json_atomic(state_path, state)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
