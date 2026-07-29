#!/usr/bin/env python3
"""Run the fresh teacher-free full-course SF034 screen for SF033."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE, VQ2RecurrentActor
import scripts.eval_vq2_recurrent_dagger_broad as screen
from scripts.collect_vq2_oracle_bc_dataset import sha256_path


TAG = "vq2_sf034_recurrent_roll_priority_teacher_free_512"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT
    / "docs/vq2_sf034_recurrent_roll_priority_teacher_free_preregistration_2026-07-28.md"
)
CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf033_recurrent_roll_priority_001/policy_best.pt"
)
CHECKPOINT_SHA256 = (
    "91088ca96f432f57f1b85bbc521ddab29ba422fadf2b79b8d642ee426f09bea7"
)
TRAIN_REPORT = CHECKPOINT.parent / "report.json"
TRAIN_REPORT_SHA256 = (
    "f54bf8512490a4629c08bf35ceae7c7fe07514e3424f89433f680536d96a0279"
)
AGENTS = 512
EPISODES = 512
SEED = 42034


def load_sf033(device: torch.device) -> tuple[VQ2RecurrentActor, dict]:
    if sha256_path(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SF033 checkpoint hash mismatch")
    if sha256_path(TRAIN_REPORT) != TRAIN_REPORT_SHA256:
        raise RuntimeError("SF033 report hash mismatch")
    report = json.loads(TRAIN_REPORT.read_text())
    if not report.get("numerically_admitted"):
        raise RuntimeError("SF033 did not pass numerical admission")
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if payload.get("schema") != "vq2_recurrent_roll_priority_checkpoint_v1":
        raise RuntimeError("unsupported SF033 checkpoint schema")
    contract = payload.get("model", {})
    if (
        contract.get("class") != "VQ2RecurrentActor"
        or contract.get("legal_observation_size") != LEGAL_OBS_SIZE
        or contract.get("action_size") != ACTION_SIZE
    ):
        raise RuntimeError("SF033 actor contract changed")
    if payload.get("safety", {}).get(
        "stored_privileged_values_per_actor_record"
    ) != 0:
        raise RuntimeError("SF033 does not prove legal-only actor records")
    model = VQ2RecurrentActor(
        hidden_size=int(contract["hidden_size"]),
        initial_std=float(contract["initial_std"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def configure_screen_module() -> None:
    screen.TAG = TAG
    screen.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    screen.PREREGISTRATION = PREREGISTRATION
    screen.CHECKPOINT = CHECKPOINT
    screen.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    screen.TRAIN_REPORT = TRAIN_REPORT
    screen.TRAIN_REPORT_SHA256 = TRAIN_REPORT_SHA256
    screen.AGENTS = AGENTS
    screen.EPISODES = EPISODES
    screen.SEED = SEED
    screen.load_sf022 = load_sf033


def run_screen(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda") -> dict:
    configure_screen_module()
    report = screen.run_screen(output=output, device_name=device_name)
    wrapper_key = str(Path(__file__).resolve().relative_to(ROOT))
    wrapper_sha256 = sha256_path(Path(__file__).resolve())
    report["source_sha256"][wrapper_key] = wrapper_sha256
    report["screen_wrapper_sha256"] = wrapper_sha256
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    report = run_screen(output=args.output.resolve(), device_name=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["full_course_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

