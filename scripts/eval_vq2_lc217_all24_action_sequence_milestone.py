#!/usr/bin/env python3
"""Teacher-free exact-context all-24 screen of LC216 against LC213."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_lc178_phase15_recurrent_adapter_milestone import (
    load_actor as shared_load_actor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc214_phase16_17_action_sequence_milestone as base


BASE_CONFIGURE = base.configure
TAG = "vq2_lc217_all24_action_sequence_milestone_001"
SCHEMA = "vq2_lc217_all24_action_sequence_milestone_report_v1"
GROUP_SIZE = 128
PAIR_SIZE = 256
TARGET_PHASE = 23
TARGET_RAW_INDEX = 24
MAX_STEPS = 45_000
SEED = 432_217
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
MINIMUM_PASS_GAIN = GROUP_SIZE
BIASES = (
    ("parent_lc213_raw18", (0.0,) * 4),
    ("lc216_all24_action_sequence", (0.0,) * 4),
)
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc213_phase16_17_action_sequence_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "3ab6bebed9c1350601e4a664ef78caed65c755f84f92b2b0dbd99686cf99f156"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "e5d2f309ca259c68d3f11d61636bcd438423de43b05dadd6a8a35ccd99013b39"
CANDIDATE_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc216_all24_action_sequence_001"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "policy_selected.pt"
CANDIDATE_CHECKPOINT_SHA256 = "672af0c4b014bb7d7399e2d709f268a487a83294b7b2a72ebfc79a48896a95b9"
CANDIDATE_REPORT = CANDIDATE_DIR / "report.json"
CANDIDATE_REPORT_SHA256 = "b112413c778d984f369717643cdd8d69e0ec984e28c47e1b36b20aed18503617"
PREREGISTRATION = ROOT / "docs/vq2_lc217_all24_action_sequence_milestone_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc217_vast.sh"
TEST = ROOT / "tests/test_eval_vq2_lc217_all24_action_sequence_milestone.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def parent_payload() -> dict[str, Any]:
    return torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_payload() -> dict[str, Any]:
    return torch.load(CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)


def candidate_state_for_index(
    parent_state: dict[str, torch.Tensor], candidate_index: int
) -> dict[str, torch.Tensor]:
    if candidate_index not in (0, 1):
        raise IndexError("LC217 has only LC213 and LC216")
    source = parent_state if candidate_index == 0 else candidate_payload()["model_state"]
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def candidate_metadata_for_index(candidate_index: int) -> dict[str, Any]:
    return {
        "all24_action_sequence_candidate": candidate_index == 1,
        "checkpoint_sha256": (
            CANDIDATE_CHECKPOINT_SHA256 if candidate_index == 1
            else PARENT_CHECKPOINT_SHA256
        ),
        "sequence_length": 9_592 if candidate_index == 1 else 2_329,
        "bias_l2": 0.0,
        "endpoint_phases": list(range(16, 24)) if candidate_index == 1 else [16, 17],
    }


def load_actor(payload: dict[str, Any], device: torch.device) -> Any:
    state = payload["model_state"]
    if "phase_action_sequence" not in state:
        return shared_load_actor(payload, device)
    length = int(state["phase_action_sequence"].shape[0])
    if length == 2_329:
        contract = parent_payload()["model"]
    elif length == 9_592:
        contract = candidate_payload()["model"]
    else:
        raise RuntimeError("LC217 encountered an unknown sequence length")
    corrected = {
        **payload,
        "model": {
            **payload["model"],
            "class": contract["class"],
            "sequence_phase_min": contract["sequence_phase_min"],
            "sequence_phase_max_exclusive": contract[
                "sequence_phase_max_exclusive"
            ],
            "sequence_length": contract["sequence_length"],
        },
    }
    return shared_load_actor(corrected, device)


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC217 bound input changed: {path}")
    parent = parent_payload()
    candidate = candidate_payload()
    parent_report = json.loads(PARENT_REPORT.read_text())
    construction = json.loads(CANDIDATE_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc213_phase16_17_action_sequence_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent.get("model", {}).get("sequence_length") != 2_329
        or parent_report.get("schema")
        != "vq2_lc213_phase16_17_action_sequence_report_v1"
        or not parent_report.get("numerically_admitted")
        or candidate.get("schema") != "vq2_lc216_all24_action_sequence_checkpoint_v1"
        or not candidate.get("numerically_admitted")
        or candidate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or candidate.get("model", {}).get("sequence_phase_min") != 16
        or candidate.get("model", {}).get("sequence_phase_max_exclusive") != 24
        or candidate.get("model", {}).get("sequence_length") != 9_592
        or construction.get("schema")
        != "vq2_lc216_all24_action_sequence_report_v1"
        or not construction.get("numerically_admitted")
        or not construction.get("frozen_non_sequence_state_exact")
        or not construction.get("frozen_lc213_prefix_exact")
        or not construction.get("status_gap_exact")
        or construction.get("checkpoint_sha256") != CANDIDATE_CHECKPOINT_SHA256
        or construction.get("continuation", {}).get("duplicate_action_max_error")
        != 0.0
        or construction.get("continuation", {}).get("final_terminal_agents") != 256
        or construction.get("safety", {}).get("flight_sim_packets_sent") != 0
        or construction.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC213/LC216 do not authorize LC217")
    for name, value in parent["model_state"].items():
        other = candidate["model_state"][name]
        if name == "phase_action_sequence":
            if not torch.equal(value, other[: value.shape[0]]):
                raise RuntimeError("LC216 changed LC213's sequence prefix")
        elif not torch.equal(value, other):
            raise RuntimeError(f"LC216 changed frozen tensor {name}")
    return parent


def choose_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline, candidate = items
    if (
        baseline["target_passes"] == 0
        and baseline.get("maximum_raw_index_distribution", {}).get("18")
        == GROUP_SIZE
        and candidate["target_passes"] == GROUP_SIZE
        and candidate["paired_target_gains_vs_baseline"] == GROUP_SIZE
        and candidate["paired_target_losses_vs_baseline"] == 0
        and candidate["transport_pass"]
        and candidate["pre_target_terminals"] <= baseline["pre_target_terminals"]
    ):
        return candidate
    return None


def configure() -> None:
    BASE_CONFIGURE()
    base.milestone.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), CANDIDATE_CHECKPOINT, CANDIDATE_REPORT,
        ROOT / "scripts/build_vq2_lc216_all24_action_sequence.py",
        ROOT / "scripts/collect_vq2_lc215_raw18_24_action_sequence.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    base.milestone.NEXT_AUTHORITY_SELECTED = (
        "Promote LC216 as the offline all-24 source; run broad deterministic admission and zero-command shadow before any bounded FlightSim authority."
    )
    base.milestone.NEXT_AUTHORITY_NONE = (
        "Reject LC216 and retain LC213 at raw 18; do not run FlightSim."
    )
    base.milestone.VECTORIZED_CHECKPOINT_SURGERY = (
        "two complete saved-form action-sequence Puffers, each executing a full "
        "256-row numerical context; LC216 carries the exact LC213 prefix"
    )


def configure_outer() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "GROUP_SIZE", "PAIR_SIZE", "TARGET_PHASE",
        "TARGET_RAW_INDEX", "MAX_STEPS", "SEED", "ENV_SEED_GROUP_SIZE",
        "ENV_SEED_INDEX_OFFSET", "MINIMUM_PASS_GAIN", "BIASES",
        "PARENT_CHECKPOINT", "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT",
        "PARENT_REPORT_SHA256", "CANDIDATE_CHECKPOINT",
        "CANDIDATE_CHECKPOINT_SHA256", "CANDIDATE_REPORT",
        "CANDIDATE_REPORT_SHA256", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "parent_payload", "candidate_payload",
        "candidate_state_for_index", "candidate_metadata_for_index",
        "load_actor", "verify_inputs", "choose_candidate", "configure",
    )
    originals = tuple(getattr(base, name) for name in names)
    values = (
        TAG, SCHEMA, GROUP_SIZE, PAIR_SIZE, TARGET_PHASE, TARGET_RAW_INDEX,
        MAX_STEPS, SEED, ENV_SEED_GROUP_SIZE, ENV_SEED_INDEX_OFFSET,
        MINIMUM_PASS_GAIN, BIASES, PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT, PARENT_REPORT_SHA256, CANDIDATE_CHECKPOINT,
        CANDIDATE_CHECKPOINT_SHA256, CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256,
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT, parent_payload,
        candidate_payload, candidate_state_for_index,
        candidate_metadata_for_index, load_actor, verify_inputs,
        choose_candidate, configure,
    )
    for name, value in zip(names, values):
        setattr(base, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(base, name, value)


def run(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure_outer()
    try:
        return base.run(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("diagnostic_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
