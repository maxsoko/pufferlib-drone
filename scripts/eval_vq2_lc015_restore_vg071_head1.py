#!/usr/bin/env python3
"""Restore selected VG071 heads in LC010S and run one bounded 24-gate screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase_residual import (
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save
import scripts.eval_vq2_lc007_converted_vg071_long_course as core


TAG = "vq2_lc015_restore_vg071_head1_001"
SCHEMA = "vq2_lc015_restore_vg071_head1_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc015_restore_vg071_head1_checkpoint_v1"
AGENTS = 32
EPISODES = 32
THREADS = 32
SEED = 431150
NUM_GATES = 24
MAX_STEPS = 12000
LC010S_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc010s_phase_independent_early_stop_001/policy_best.pt"
)
LC010S_CHECKPOINT_SHA256 = (
    "34747327c2f431a6153d200891bad45b8431a5a5d0fe2418108ae822aa716d4d"
)
LC010S_REPORT = LC010S_CHECKPOINT.parent / "report.json"
LC010S_REPORT_SHA256 = (
    "afae8fa795159824bd447472b7eb5047167f19a4848264c95593c08ab43af21f"
)
VG071_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg071_nonlinear_residual_count5_multi_offset_001/policy_selected.pt"
)
VG071_CHECKPOINT_SHA256 = (
    "60677e385cefb6d6af5c957071cb846de8396d8872880a90832d987b7494f010"
)
VG071_REPORT = VG071_CHECKPOINT.parent / "report.json"
VG071_REPORT_SHA256 = (
    "c1494b9c56bdf2cbbfead91dc34d51d45cb780cd6317bc924835a67f7cc6b5e4"
)
LC011_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc011_bounded_prefix_screen_001/report.json"
)
LC011_REPORT_SHA256 = (
    "ed82daf9a06babafffc7236820045a924ce3100faba5e948807aed27fdccffd6"
)
LC014_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc014_bounded_student_dagger_screen_001/report.json"
)
LC014_REPORT_SHA256 = (
    "d8200a913a3a8b5bfe6c8216decc01a57dbab01e741551f8e342de9c59eda486"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc015_restore_vg071_head1_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc015_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
RESIDUAL_NAMES = (
    "indexed_phase_residual_input",
    "indexed_phase_residual_input_bias",
    "indexed_phase_residual_output",
    "indexed_phase_residual_output_bias",
)
RESTORED_PHASES = (1,)
MIN_MEAN_GATES = 1.09375
MIN_MAXIMUM_INDEX = 2
MAX_CRASH_RATE = 0.50
EXTRA_EVIDENCE_PATHS: tuple[Path, ...] = ()
CONVERSION_OPERATION: str | None = None


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_inputs() -> None:
    expected = {
        LC010S_CHECKPOINT: LC010S_CHECKPOINT_SHA256,
        LC010S_REPORT: LC010S_REPORT_SHA256,
        VG071_CHECKPOINT: VG071_CHECKPOINT_SHA256,
        VG071_REPORT: VG071_REPORT_SHA256,
        LC011_REPORT: LC011_REPORT_SHA256,
        LC014_REPORT: LC014_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC015 bound input changed: {path}")
    old = json.loads(LC011_REPORT.read_text())
    failed = json.loads(LC014_REPORT.read_text())
    if (
        old.get("components", {}).get("24", {}).get("mean_gates_passed")
        != 1.09375
        or failed.get("components", {}).get("24", {}).get("mean_gates_passed")
        != 1.0
        or failed.get("components", {}).get("24", {}).get("crash_rate")
        != 0.90625
        or old.get("safety", {}).get("flight_sim_packets_sent") != 0
        or failed.get("safety", {}).get("flight_sim_packets_sent") != 0
        or failed.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC011/LC014 do not authorize the head-1 rollback")


def build_candidate(
    device: torch.device,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any], str]:
    parent = torch.load(LC010S_CHECKPOINT, map_location="cpu", weights_only=False)
    legacy = torch.load(VG071_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = parent.get("model", {})
    if (
        parent.get("schema")
        != "vq2_lc010s_phase_independent_early_stop_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or contract.get("class") != "VQ2UnboundedProgressMLPResidualActor"
        or legacy.get("schema") != "vq2_vg071_paired_synthetic_checkpoint_v1"
        or not legacy.get("numerically_admitted")
    ):
        raise RuntimeError("LC015 endpoint contract changed")
    actor = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    actor.load_state_dict(parent["model_state"])
    reference = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(contract["initial_std"]),
    )
    reference.load_converted_state(legacy["model_state"])
    restored_phases = tuple(sorted(set(int(phase) for phase in RESTORED_PHASES)))
    if not restored_phases:
        raise RuntimeError("at least one VG071 residual head must be restored")
    with torch.no_grad():
        for name in RESIDUAL_NAMES:
            target = getattr(actor, name)
            source = getattr(reference, name).to(device)
            if restored_phases[0] < 0 or restored_phases[-1] >= target.shape[0]:
                raise RuntimeError(f"restored phase is outside {name}: {restored_phases}")
            for phase in restored_phases:
                target[phase].copy_(source[phase])
    state = {name: value.detach().cpu() for name, value in actor.state_dict().items()}
    for name, value in state.items():
        if name in RESIDUAL_NAMES:
            keep = torch.ones(value.shape[0], dtype=torch.bool)
            keep[list(restored_phases)] = False
            if not torch.equal(value[keep], parent["model_state"][name][keep]):
                raise RuntimeError(f"head rollback changed a non-target row: {name}")
        elif not torch.equal(value, parent["model_state"][name]):
            raise RuntimeError(f"head rollback changed frozen Puffer state: {name}")
    actor.eval()
    payload = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": TAG,
        "model_state": state,
        "best_epoch": 0,
        "optimizer_updates": 0,
        "numerically_admitted": False,
        "surgery": {
            "operation": "restore_converted_vg071_residual_head",
            "restored_phases": list(restored_phases),
            "lc010s_checkpoint_sha256": LC010S_CHECKPOINT_SHA256,
            "vg071_checkpoint_sha256": VG071_CHECKPOINT_SHA256,
        },
    }
    return actor, payload, state_sha256(state)


def configure() -> None:
    core.TAG = TAG
    core.AGENTS = AGENTS
    core.EPISODES = EPISODES
    core.THREADS = THREADS
    core.SEEDS = {NUM_GATES: SEED}
    core.CHECKPOINT = LC010S_CHECKPOINT
    core.CHECKPOINT_SHA256 = LC010S_CHECKPOINT_SHA256
    core.PREREGISTRATION = PREREGISTRATION
    core.RUNNER = RUNNER
    core.MAX_STEPS_OVERRIDE = MAX_STEPS
    core.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), LC010S_REPORT, VG071_CHECKPOINT, VG071_REPORT,
        LC011_REPORT, LC014_REPORT, *EXTRA_EVIDENCE_PATHS,
    )


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    verify_inputs()
    configure()
    if output.exists():
        if not resume or not (output / "report.json").is_file():
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads((output / "report.json").read_text())
    device = torch.device(device_name)
    actor, payload, candidate_state_sha = build_candidate(device)
    identity = core.source_identity()

    def load_actor(
        target: torch.device,
    ) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any]]:
        if target != device:
            raise RuntimeError("head-rollback actor device changed")
        return actor, payload

    core.load_converted_actor = load_actor
    core.conversion_metadata = lambda _: {
        "operation": CONVERSION_OPERATION or (
            "restore converted VG071 residual heads "
            + ",".join(str(phase) for phase in RESTORED_PHASES)
        ),
        "candidate_state_sha256": candidate_state_sha,
        "whole_output_recurrent_puffer": True,
        "teacher_runtime_actions": 0,
    }
    component = core.run_count(
        num_gates=NUM_GATES, device=device, identity=identity
    )
    component["checkpoint_sha256"] = candidate_state_sha
    distribution = component["maximum_raw_index_distribution"]
    maximum_index = max(
        int(index) for index, count in distribution.items() if count
    )
    mean_gates = float(component["metrics"]["env/gates_passed"])
    crash_rate = float(component["metrics"]["env/crash"])
    admitted = bool(
        component["transport_pass"]
        and mean_gates > MIN_MEAN_GATES
        and maximum_index >= MIN_MAXIMUM_INDEX
        and crash_rate <= MAX_CRASH_RATE
    )
    output.mkdir(parents=True)
    write_json_once(output / "count_24.json", component)
    checkpoint_sha: str | None = None
    if admitted:
        payload["numerically_admitted"] = True
        payload["source_commit"] = identity["source_commit"]
        payload["source_sha256"] = identity["source_sha256"]
        payload["screen"] = {
            "mean_gates_passed": mean_gates,
            "maximum_raw_index": maximum_index,
            "crash_rate": crash_rate,
        }
        atomic_torch_save(output / "policy_selected.pt", payload)
        checkpoint_sha = sha256_path(output / "policy_selected.pt")
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "diagnostic_valid": component["transport_pass"],
        "numerically_admitted": admitted,
        "candidate_state_sha256": candidate_state_sha,
        "checkpoint": "policy_selected.pt" if admitted else None,
        "checkpoint_sha256": checkpoint_sha,
        "component_sha256": sha256_path(output / "count_24.json"),
        "mean_gates_passed": mean_gates,
        "maximum_raw_index": maximum_index,
        "maximum_raw_index_distribution": distribution,
        "crash_rate": crash_rate,
        "miss_rate": float(component["metrics"]["env/missed_gate"]),
        "timeout_rate": float(component["metrics"]["env/timeout"]),
        "source_identity": identity,
        "surgery": payload["surgery"],
        "safety": {
            "teacher_plant_actions": 0, "student_updates": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "One fresh bounded student-state collection at the first weak phase."
            if admitted else "Reject the rollback candidate and retain the prior frontier."
        ),
    }
    write_json_once(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
