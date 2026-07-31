#!/usr/bin/env python3
"""Run the paired LC046 phase-6 bracket concurrently across Vast CPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase_residual import (
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc015_restore_vg071_head1 as screen
import scripts.eval_vq2_lc024_phase2_interpolation_bracket as base


TAG = "vq2_lc047_phase6_parallel_interpolation_bracket_001"
SCHEMA = "vq2_lc047_phase6_parallel_interpolation_bracket_report_v1"
CHILD_SCHEMA = "vq2_lc047_phase6_interpolation_screen_v1"
CHECKPOINT_SCHEMA = "vq2_lc047_phase6_interpolation_checkpoint_v1"
ALPHAS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25)
SEED = 431470
TARGET_PHASE = 6
WORKER_THREADS = 32
WORKER_PROCESSES = len(ALPHAS)
LC043 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc043_phase5_interpolation_bracket_001"
)
LC043_REPORT = LC043 / "report.json"
LC043_REPORT_SHA256 = (
    "3dddf5bb2e3e6f3249a7215abe938a91625e3f9494c3f3c0984b22ddbbdaa115"
)
PARENT_CHECKPOINT = LC043 / "a0p005/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "d2f9c235cc0a966d8d96ddca7513026e6689cf14aeb8c0cff22891769b93ad52"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "35f14abb43d7e6a9db7cac4370102131dfcc6c94bde68bc83ccd123e6774366d"
)
LC045_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc045_index6_student_dagger_features_001/report.json"
)
LC045_REPORT_SHA256 = (
    "2abc1bf32d5e8812779fcfe4ed495f2e947d6098e99b07ed15332bfbbcfaebfa"
)
FIT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc046_phase6_student_head_001/policy_best.pt"
)
FIT_CHECKPOINT_SHA256 = (
    "9cc9f4111a7a520c4e73ddd0a62444fae47a7a3134185d719b3202639c0b4c14"
)
FIT_REPORT = FIT_CHECKPOINT.parent / "report.json"
FIT_REPORT_SHA256 = (
    "ed814d8b24a11b45e6c116fe910e02e1858d5ec0c6ef04828cd5f8dc1a9e426a"
)
PREREGISTRATION = (
    ROOT
    / "docs/vq2_lc047_phase6_parallel_interpolation_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc047_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CURRENT_ALPHA = 0.0


def verify_inputs() -> None:
    expected = {
        LC043_REPORT: LC043_REPORT_SHA256,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC045_REPORT: LC045_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC047 bound input changed: {path}")
    aggregate = json.loads(LC043_REPORT.read_text())
    dataset = json.loads(LC045_REPORT.read_text())
    fitted = json.loads(FIT_REPORT.read_text())
    if (
        aggregate.get("schema")
        != "vq2_lc043_phase5_interpolation_bracket_report_v1"
        or not aggregate.get("numerically_admitted")
        or aggregate.get("selected", {}).get("alpha") != 0.005
        or aggregate.get("selected", {}).get("checkpoint_sha256")
        != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema")
        != "vq2_lc045_index6_student_dagger_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_phase_records", [])[TARGET_PHASE] != 189_009
        or dataset.get("teacher_plant_actions_executed") != 0
        or fitted.get("schema") != "vq2_lc046_phase6_student_head_report_v1"
        or not fitted.get("numerically_admitted")
        or fitted.get("checkpoint_sha256") != FIT_CHECKPOINT_SHA256
        or fitted.get("selected_validation", {}).get("phases", {}).get(
            "6", {}
        ).get("improvement_factor", 0.0) < 1.33
        or fitted.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fitted.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC043/LC045/LC046 do not authorize LC047")


def build_candidate(
    device: torch.device,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any], str]:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    fitted = torch.load(FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = parent.get("model", {})
    if (
        parent.get("schema") != "vq2_lc043_phase5_interpolation_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or fitted.get("schema") != "vq2_lc046_phase6_student_head_checkpoint_v1"
        or not fitted.get("numerically_admitted")
        or contract.get("class") != "VQ2UnboundedProgressMLPResidualActor"
    ):
        raise RuntimeError("LC047 checkpoint contract changed")
    actor = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(parent["model_state"])
    with torch.no_grad():
        for name in screen.RESIDUAL_NAMES:
            start = parent["model_state"][name][TARGET_PHASE].to(device)
            end = fitted["model_state"][name][TARGET_PHASE].to(device)
            getattr(actor, name)[TARGET_PHASE].copy_(
                torch.lerp(start, end, CURRENT_ALPHA)
            )
    state = {name: value.detach().cpu() for name, value in actor.state_dict().items()}
    for name, value in state.items():
        if name in screen.RESIDUAL_NAMES:
            keep = torch.arange(value.shape[0]) != TARGET_PHASE
            if not torch.equal(value[keep], parent["model_state"][name][keep]):
                raise RuntimeError(f"LC047 changed a non-target row: {name}")
        elif not torch.equal(value, parent["model_state"][name]):
            raise RuntimeError(f"LC047 changed frozen Puffer state: {name}")
    actor.eval()
    payload = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": screen.TAG,
        "model_state": state,
        "best_epoch": fitted.get("best_epoch"),
        "optimizer_updates": 0,
        "numerically_admitted": False,
        "surgery": {
            "operation": "interpolate_phase_head_toward_student_state_fit",
            "phase": TARGET_PHASE,
            "alpha": CURRENT_ALPHA,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "fit_checkpoint_sha256": FIT_CHECKPOINT_SHA256,
        },
    }
    return actor, payload, screen.state_sha256(state)


def alpha_slug(alpha: float) -> str:
    return f"a{alpha:.3f}".replace(".", "p")


def configure(alpha: float) -> None:
    global CURRENT_ALPHA
    CURRENT_ALPHA = alpha
    screen.TAG = f"{TAG}_{alpha_slug(alpha)}"
    screen.SCHEMA = CHILD_SCHEMA
    screen.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    screen.SEED = SEED
    screen.PREREGISTRATION = PREREGISTRATION
    screen.RUNNER = RUNNER
    screen.RESTORED_PHASES = (TARGET_PHASE,)
    screen.MIN_MEAN_GATES = 3.75
    screen.MIN_MAXIMUM_INDEX = 8
    screen.MAX_CRASH_RATE = 0.50
    screen.CONVERSION_OPERATION = (
        f"interpolate phase-{TARGET_PHASE} head toward fit at alpha {alpha:.4f}"
    )
    screen.EXTRA_EVIDENCE_PATHS = (
        Path(__file__).resolve(), LC043_REPORT, PARENT_CHECKPOINT,
        PARENT_REPORT, LC045_REPORT, FIT_CHECKPOINT, FIT_REPORT,
    )
    screen.verify_inputs = verify_inputs
    screen.build_candidate = build_candidate


def run_worker(
    *, alpha: float, output: Path, device_name: str, resume: bool,
) -> dict[str, Any]:
    if alpha not in ALPHAS:
        raise ValueError(f"LC047 worker alpha is not preregistered: {alpha}")
    configure(alpha)
    return screen.run(
        output=output / alpha_slug(alpha), device_name=device_name, resume=resume
    )


def _item(alpha: float, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    child_output = output / alpha_slug(alpha)
    child = json.loads((child_output / "report.json").read_text())
    component = json.loads((child_output / "count_24.json").read_text())
    item = {
        "alpha": alpha,
        "candidate_state_sha256": child["candidate_state_sha256"],
        "mean_gates_passed": child["mean_gates_passed"],
        "maximum_raw_index": child["maximum_raw_index"],
        "maximum_raw_index_distribution": child["maximum_raw_index_distribution"],
        "crash_rate": child["crash_rate"],
        "miss_rate": child["miss_rate"],
        "timeout_rate": child["timeout_rate"],
        "diagnostic_valid": child["diagnostic_valid"],
        "worker_wall_time_seconds": component["wall_time_seconds"],
        "checkpoint": (
            str(Path(alpha_slug(alpha)) / child["checkpoint"])
            if child["checkpoint"] else None
        ),
        "checkpoint_sha256": child["checkpoint_sha256"],
        "child_report_sha256": sha256_path(child_output / "report.json"),
    }
    return item, child["source_identity"]


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {output}")

    started = time.perf_counter()
    processes: list[tuple[float, subprocess.Popen[str]]] = []
    for alpha in ALPHAS:
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker-alpha", repr(alpha), "--output", str(output),
            "--device", device_name,
        ]
        if resume:
            command.append("--resume")
        processes.append((alpha, subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )))
    failures: list[str] = []
    for alpha, process in processes:
        stdout, stderr = process.communicate()
        if process.returncode:
            failures.append(
                f"alpha {alpha} exited {process.returncode}; "
                f"stdout={stdout[-2000:]!r}; stderr={stderr[-4000:]!r}"
            )
    if failures:
        raise RuntimeError("LC047 parallel worker failure:\n" + "\n".join(failures))

    items: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        item, identity = _item(alpha, output)
        items.append(item)
        identities.append(identity)
    if any(identity != identities[0] for identity in identities[1:]):
        raise RuntimeError("LC047 source identity changed across workers")

    baseline = items[0]
    admitted = [
        item for item in items[1:]
        if item["checkpoint"] is not None
        and item["mean_gates_passed"] > baseline["mean_gates_passed"]
        and item["maximum_raw_index"] >= baseline["maximum_raw_index"]
        and item["crash_rate"] <= baseline["crash_rate"]
    ]
    selected = max(
        admitted,
        key=lambda item: (
            item["mean_gates_passed"], item["maximum_raw_index"],
            -item["crash_rate"], -item["alpha"],
        ),
        default=None,
    )
    wall = time.perf_counter() - started
    sequential_equivalent = sum(item["worker_wall_time_seconds"] for item in items)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "diagnostic_valid": all(item["diagnostic_valid"] for item in items),
        "numerically_admitted": selected is not None,
        "paired_seed": SEED,
        "episodes_per_candidate": screen.EPISODES,
        "num_gates": screen.NUM_GATES,
        "baseline": baseline, "items": items, "selected": selected,
        "wall_time_seconds": wall,
        "sequential_equivalent_worker_seconds": sequential_equivalent,
        "measured_parallel_speedup": sequential_equivalent / max(wall, 1e-12),
        "parallel_layout": {
            "worker_processes": WORKER_PROCESSES,
            "threads_per_worker": WORKER_THREADS,
            "requested_cpu_threads": WORKER_PROCESSES * WORKER_THREADS,
            "shared_cuda_device": device_name,
        },
        "source_identity": identities[0],
        "safety": {
            "teacher_plant_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Retain the selected phase-6 interpolation for a larger offline screen."
            if selected else "Reject LC047 and retain the unchanged LC043 Puffer."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json_once(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-alpha", type=float)
    args = parser.parse_args()
    if args.worker_alpha is not None:
        report = run_worker(
            alpha=args.worker_alpha, output=args.output.resolve(),
            device_name=args.device, resume=args.resume,
        )
    else:
        report = run(
            output=args.output.resolve(), device_name=args.device,
            resume=args.resume,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
