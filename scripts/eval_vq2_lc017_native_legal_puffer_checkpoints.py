#!/usr/bin/env python3
"""Deterministically screen three LC016 native legal Puffer checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_native_legal_puffer import VQ2NativeLegalPufferActor
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.eval_vq2_lc007_converted_vg071_long_course as core


TAG = "vq2_lc017_native_legal_puffer_checkpoint_screen_001"
SCHEMA = "vq2_lc017_native_legal_puffer_checkpoint_screen_report_v1"
AGENTS = 16
EPISODES = 16
THREADS = 32
SEED = 431170
NUM_GATES = 24
MAX_STEPS = 5000
TRAINING = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc016_native_legal_puffer_ppo_001"
)
TRAIN_REPORT = TRAINING / "report.json"
TRAIN_REPORT_SHA256 = (
    "7c8d74a95c1905b3ca27956b85d7df8ac74ac389445346f5cb134adb5cf8d8ae"
)
CHECKPOINTS = (
    (2_015_232, "policy_000002015232.bin", "a220b4dbc8ef8e27f9d981b0601b955f6cb247b51eb17f09937858f6ab03d15a"),
    (6_012_928, "policy_000006012928.bin", "d1948c8146519bed708695c8c29bb9db5700c648bf6ed881d9cb1a59aad0a404"),
    (10_010_624, "policy_000010010624_final.bin", "50c27c12e6505d4bfb8ba108c5e2194604441041f475a600d7d73c0fd2d0aae9"),
)
LC015_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc015_restore_vg071_head1_001/report.json"
)
LC015_REPORT_SHA256 = (
    "5c38e8436bff1b5e80ea4a8e32f9e756d078e9df52154c46606e2b5bdbbef9a5"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc017_native_legal_puffer_checkpoint_screen_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc017_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    if sha256_path(TRAIN_REPORT) != TRAIN_REPORT_SHA256:
        raise RuntimeError("LC017 training report changed")
    if sha256_path(LC015_REPORT) != LC015_REPORT_SHA256:
        raise RuntimeError("LC017 LC015 frontier report changed")
    report = json.loads(TRAIN_REPORT.read_text())
    frontier = json.loads(LC015_REPORT.read_text())
    if (
        not report.get("training_admitted")
        or report.get("privileged_policy_input_max_abs") != 0.0
        or report.get("public_progress_encoding_max_error") != 0.0
        or report.get("agent_steps_per_second", 0.0) < 150_000
        or frontier.get("mean_gates_passed") != 2.4375
        or frontier.get("maximum_raw_index") != 4
        or frontier.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("LC016/LC015 do not authorize LC017")
    manifest = {
        (item["agent_steps"], item["path"], item["sha256"])
        for item in report.get("checkpoints", [])
    }
    for steps, name, digest in CHECKPOINTS:
        path = TRAINING / name
        if (steps, name, digest) not in manifest or sha256_path(path) != digest:
            raise RuntimeError(f"LC017 checkpoint changed: {path}")


def configure() -> None:
    core.TAG = TAG
    core.AGENTS = AGENTS
    core.EPISODES = EPISODES
    core.THREADS = THREADS
    core.SEEDS = {NUM_GATES: SEED}
    core.CHECKPOINT = TRAINING / CHECKPOINTS[0][1]
    core.CHECKPOINT_SHA256 = CHECKPOINTS[0][2]
    core.PREREGISTRATION = PREREGISTRATION
    core.RUNNER = RUNNER
    core.MAX_STEPS_OVERRIDE = MAX_STEPS
    core.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), TRAIN_REPORT, LC015_REPORT,
        ROOT / "pufferlib/vq2_native_legal_puffer.py",
        ROOT / "tests/test_vq2_native_legal_puffer.py",
        *(TRAINING / item[1] for item in CHECKPOINTS),
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
    identity = core.source_identity()
    output.mkdir(parents=True)
    device = torch.device(device_name)
    items: list[dict[str, Any]] = []
    for steps, name, digest in CHECKPOINTS:
        path = TRAINING / name
        actor = VQ2NativeLegalPufferActor.from_binary(path).to(device).eval()

        def load_actor(
            target: torch.device,
            *, selected: VQ2NativeLegalPufferActor = actor,
            selected_steps: int = steps,
        ) -> tuple[VQ2NativeLegalPufferActor, dict[str, Any]]:
            if target != device:
                raise RuntimeError("LC017 actor device changed")
            return selected, {
                "schema": "vq2_native_legal_puffer_binary_v1",
                "best_epoch": selected_steps,
            }

        core.load_converted_actor = load_actor
        core.CHECKPOINT_SHA256 = digest
        core.conversion_metadata = lambda _, s=steps: {
            "operation": "native PuffeRL binary to legal PyTorch actor",
            "agent_steps": s, "whole_output_recurrent_puffer": True,
            "native_privileged_columns_dropped": 34,
        }
        component = core.run_count(
            num_gates=NUM_GATES, device=device, identity=identity
        )
        component_path = output / f"steps_{steps}.json"
        write_json_once(component_path, component)
        distribution = component["maximum_raw_index_distribution"]
        maximum = max(int(k) for k, count in distribution.items() if count)
        items.append({
            "agent_steps": steps, "checkpoint": name,
            "checkpoint_sha256": digest,
            "component": component_path.name,
            "component_sha256": sha256_path(component_path),
            "transport_pass": component["transport_pass"],
            "mean_gates_passed": float(component["metrics"]["env/gates_passed"]),
            "maximum_raw_index": maximum,
            "crash_rate": float(component["metrics"]["env/crash"]),
            "miss_rate": float(component["metrics"]["env/missed_gate"]),
            "timeout_rate": float(component["metrics"]["env/timeout"]),
        })
    eligible = [
        item for item in items
        if item["transport_pass"] and item["mean_gates_passed"] > 2.4375
        and item["maximum_raw_index"] >= 4 and item["crash_rate"] <= 0.50
    ]
    selected = max(
        eligible,
        key=lambda item: (
            item["maximum_raw_index"], item["mean_gates_passed"],
            -item["crash_rate"], -item["agent_steps"],
        ),
        default=None,
    )
    selected_sha: str | None = None
    if selected is not None:
        shutil.copy2(TRAINING / selected["checkpoint"], output / "policy_selected.bin")
        selected_sha = sha256_path(output / "policy_selected.bin")
        if selected_sha != selected["checkpoint_sha256"]:
            raise RuntimeError("LC017 selected checkpoint copy changed")
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "diagnostic_valid": all(item["transport_pass"] for item in items),
        "numerically_admitted": selected is not None,
        "seed": SEED, "agents_per_checkpoint": AGENTS,
        "max_steps": MAX_STEPS, "items": items,
        "selected_agent_steps": selected["agent_steps"] if selected else None,
        "checkpoint": "policy_selected.bin" if selected else None,
        "checkpoint_sha256": selected_sha,
        "source_identity": identity,
        "safety": {
            "actor_input_privileged_values": 0, "teacher_plant_actions": 0,
            "student_updates": 0, "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "One larger fresh-seed full-start screen of the selected native Puffer."
            if selected else "Reject LC016 policy quality and revise its PPO curriculum."
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
