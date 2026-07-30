#!/usr/bin/env python3
"""Continue VG020 with parity-gated bitwise device-side mask decode."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent_phase import (
    PHASE_LEGAL_OBS_SIZE,
    VQ2PhaseRecurrentActor,
)
from scripts.eval_vq2_variable_gate_oracle import write_json_atomic
from scripts.train_vq2_recurrent_bc import (
    ActionAccumulator,
    temporal_smoothness,
    weighted_action_mse,
)
from scripts.train_vq2_variable_gate_dagger_refit import RecurrentChunkStream
from scripts.train_vq2_variable_gate_recurrent_bc import (
    VariableGateBCDataset,
    atomic_torch_save,
    evaluate,
    reconstruct_phase_batch,
    sha256_path,
    source_label,
)
import scripts.train_vq2_variable_gate_four_source_refit as four_source
import scripts.train_vq2_variable_gate_three_source_refit as three_source


TAG = "vq2_vg022_five_source_device_decode_continuation_001"
CHECKPOINT_SCHEMA = "vq2_variable_gate_recurrent_bc_checkpoint_v1"
REPORT_SCHEMA = "vq2_variable_gate_five_source_device_decode_report_v1"
STATE_SCHEMA = "vq2_variable_gate_five_source_device_decode_state_v1"
SEED = 429094
MIGRATION_STATE_SHA256 = (
    "3d4694baaeac55184d3672675b4c159ee339adbff45c4b46246a58118ecbb14b"
)
MIGRATION_SOURCE_COMMIT = "83a48e63e1982aa636ff2b5136a9ea2451e6ff7d"
MIGRATION_TAG = "vq2_vg020_variable_gate_five_source_refit_001"
MIGRATION_SCHEMA = "vq2_variable_gate_five_source_refit_state_v1"

CLEAN_DATASET = four_source.CLEAN_DATASET
CLEAN_REPORT_SHA256 = four_source.CLEAN_REPORT_SHA256
CLEAN_METADATA_SHA256 = four_source.CLEAN_METADATA_SHA256
DAGGER1_DATASET = four_source.DAGGER1_DATASET
DAGGER1_REPORT_SHA256 = four_source.DAGGER1_REPORT_SHA256
DAGGER1_METADATA_SHA256 = four_source.DAGGER1_METADATA_SHA256
DAGGER1_ADMISSION = four_source.DAGGER1_ADMISSION
DAGGER1_ADMISSION_SHA256 = four_source.DAGGER1_ADMISSION_SHA256
DAGGER2_DATASET = four_source.DAGGER2_DATASET
DAGGER2_METADATA_SHA256 = four_source.DAGGER2_METADATA_SHA256
DAGGER2_RECOVERY_REPORT = four_source.DAGGER2_RECOVERY_REPORT
DAGGER2_RECOVERY_REPORT_SHA256 = four_source.DAGGER2_RECOVERY_REPORT_SHA256
DAGGER2_ADMISSION = four_source.DAGGER2_ADMISSION
DAGGER2_ADMISSION_SHA256 = four_source.DAGGER2_ADMISSION_SHA256
DAGGER3_DATASET = four_source.DAGGER3_DATASET
DAGGER3_REPORT_SHA256 = four_source.DAGGER3_REPORT_SHA256
DAGGER3_METADATA_SHA256 = four_source.DAGGER3_METADATA_SHA256
DAGGER3_ADMISSION = four_source.DAGGER3_ADMISSION
DAGGER3_ADMISSION_SHA256 = four_source.DAGGER3_ADMISSION_SHA256
DAGGER4_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg019_variable_gate_dagger_round4_vg017_visited_512"
)
DAGGER4_REPORT_SHA256 = (
    "2079ce9de0ff96bab8e218ba249307988799a7c7062d7175185d86608c213e82"
)
DAGGER4_METADATA_SHA256 = (
    "773273fc221e472d57d71a0407bdd3c579713aa91aab309a033c286c22541f22"
)
DAGGER4_ADMISSION = (
    ROOT / "docs/vq2_vg019_variable_gate_dagger_round4_admission_2026-07-30.json"
)
DAGGER4_ADMISSION_SHA256 = (
    "9b110d50fb4f36c750725eb0f2f77afc8506d6eaeab739577b10e02aa715b286"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg017_variable_gate_four_source_refit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "6769476f309beb4bb502c64a83c4f992b47da2b1a8d8b6913398b4b19b3ab4a8"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "476c730649c0f127feb4975fe2ad44f66be64b0ea8dde4d48c5ada7bb0d4c86b"
)
PARENT_ADMISSION = ROOT / "docs/vq2_vg017_four_source_refit_admission_2026-07-30.json"
PARENT_ADMISSION_SHA256 = (
    "df664cbac28ee676cbd5e056cb1f06745bf3e622710b3faa085eca90425741b6"
)
GOAL_PROMPT = four_source.GOAL_PROMPT
GOAL_PROMPT_SHA256 = four_source.GOAL_PROMPT_SHA256
VG020_PREREGISTRATION = (
    ROOT / "docs/vq2_vg020_five_source_refit_preregistration_2026-07-30.md"
)
VG020_SUPERSESSION = (
    ROOT / "docs/vq2_vg020_epoch3_throughput_supersession_2026-07-30.json"
)
VG020_SUPERSESSION_SHA256 = (
    "5beb1c64077779727d48e100e2db7db9d5d23561b9196c5cdc324879e0edd34d"
)
VG021_PREREGISTRATION = (
    ROOT / "docs/vq2_vg021_accelerated_continuation_preregistration_2026-07-30.md"
)
VG021_REJECTION = ROOT / "docs/vq2_vg021_local_batching_rejection_2026-07-30.json"
VG021_REJECTION_SHA256 = (
    "442c33c334793cb6c2a6f676db4bf75674ab0ee0e2d388dbfe59eb5f2a0c2527"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_vg022_device_decode_continuation_preregistration_2026-07-30.md"
)
VG022_BOOTSTRAP_FAILURE = (
    ROOT / "docs/vq2_vg022_bootstrap_001_preflight_failure_2026-07-30.json"
)
VG022_BOOTSTRAP_FAILURE_SHA256 = (
    "a84ba484737bf4a572f1d165f4abe1e96a7fe578e95aeedfe8b5d3a5234e0c99"
)
RUNNER = ROOT / "scripts/run_vq2_vg022_vast.sh"
PARITY_CHECKER = ROOT / "scripts/check_vq2_vg022_device_decode_parity.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
SOURCE_WEIGHTS = {
    "clean": 0.35,
    "dagger1": 0.10,
    "dagger2": 0.10,
    "dagger3": 0.15,
    "dagger4": 0.30,
}


@dataclass(frozen=True)
class FiveSourceConfig:
    seed: int = SEED
    hidden_size: int = 256
    initial_std: float = 0.15
    epochs: int = 12
    source_agent_batch_size: int = 4
    agent_batch_size: int = 8
    sequence_chunk: int = 256
    clean_validation_agents: int = 32
    dagger1_validation_agents: int = 64
    dagger2_validation_agents: int = 64
    dagger3_validation_agents: int = 64
    dagger4_validation_agents: int = 64
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    smoothness_weight: float = 1e-4
    transition_window_exposure: int = 4
    clean_objective_weight: float = 0.35
    dagger1_objective_weight: float = 0.10
    dagger2_objective_weight: float = 0.10
    dagger3_objective_weight: float = 0.15
    dagger4_objective_weight: float = 0.30
    action_weights: tuple[float, float, float, float] = (1.0, 1.0, 4.0, 1.0)
    maximum_clean_validation_weighted_mse: float = 0.02
    maximum_dagger1_validation_weighted_mse: float = 0.02
    maximum_dagger2_validation_weighted_mse: float = 0.10
    maximum_dagger3_validation_weighted_mse: float = 0.10
    maximum_dagger4_validation_weighted_mse: float = 0.10


def source_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        PREREGISTRATION,
        VG021_PREREGISTRATION,
        VG021_REJECTION,
        VG022_BOOTSTRAP_FAILURE,
        VG020_PREREGISTRATION,
        VG020_SUPERSESSION,
        RUNNER,
        PARITY_CHECKER,
        GOAL_PROMPT,
        DAGGER1_ADMISSION,
        DAGGER2_ADMISSION,
        DAGGER3_ADMISSION,
        DAGGER4_ADMISSION,
        PARENT_ADMISSION,
        ROOT / "scripts/train_vq2_variable_gate_four_source_refit.py",
        ROOT / "scripts/train_vq2_variable_gate_three_source_refit.py",
        ROOT / "scripts/train_vq2_variable_gate_dagger_refit.py",
        ROOT / "scripts/train_vq2_variable_gate_recurrent_bc.py",
        ROOT / "scripts/train_vq2_recurrent_bc.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        CLEAN_DATASET / "report.json",
        CLEAN_DATASET / "metadata.json",
        DAGGER1_DATASET / "report.json",
        DAGGER1_DATASET / "metadata.json",
        DAGGER2_RECOVERY_REPORT,
        DAGGER2_DATASET / "metadata.json",
        DAGGER3_DATASET / "report.json",
        DAGGER3_DATASET / "metadata.json",
        DAGGER4_DATASET / "report.json",
        DAGGER4_DATASET / "metadata.json",
        PARENT_CHECKPOINT,
        PARENT_REPORT,
    ]


def current_source_identity() -> tuple[str, dict[str, str]]:
    hashes = {source_label(path): sha256_path(path) for path in source_paths()}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, hashes


def objective_weights(config: FiveSourceConfig) -> dict[str, float]:
    return {
        "clean": config.clean_objective_weight,
        "dagger1": config.dagger1_objective_weight,
        "dagger2": config.dagger2_objective_weight,
        "dagger3": config.dagger3_objective_weight,
        "dagger4": config.dagger4_objective_weight,
    }


def five_source_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    valid: dict[str, torch.Tensor],
    *,
    weights: torch.Tensor,
    previous_predictions: dict[str, torch.Tensor | None],
    previous_valid: dict[str, torch.Tensor | None],
    config: FiveSourceConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components: dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None
    for name, source_weight in objective_weights(config).items():
        action, _ = weighted_action_mse(
            predictions[name], targets[name], valid[name], weights
        )
        smoothness = temporal_smoothness(
            predictions[name],
            valid[name],
            previous_prediction=previous_predictions[name],
            previous_valid=previous_valid[name],
        )
        source_total = action + config.smoothness_weight * smoothness
        components[f"{name}_action"] = action
        components[f"{name}_smoothness"] = smoothness
        components[f"{name}_total"] = source_total
        weighted = source_weight * source_total
        total = weighted if total is None else total + weighted
    if total is None:
        raise RuntimeError("VG020 has no source objective")
    return total, components


def source_weight_audit(
    sums: dict[str, float], updates: int, config: FiveSourceConfig
) -> bool:
    expected = {
        name: updates * weight for name, weight in objective_weights(config).items()
    }
    return all(abs(sums[name] - expected[name]) <= 1e-9 for name in expected)


def verify_inputs() -> None:
    four_source.verify_inputs()
    expected = {
        GOAL_PROMPT: GOAL_PROMPT_SHA256,
        VG020_SUPERSESSION: VG020_SUPERSESSION_SHA256,
        VG021_REJECTION: VG021_REJECTION_SHA256,
        VG022_BOOTSTRAP_FAILURE: VG022_BOOTSTRAP_FAILURE_SHA256,
        DAGGER4_ADMISSION: DAGGER4_ADMISSION_SHA256,
        PARENT_ADMISSION: PARENT_ADMISSION_SHA256,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        DAGGER4_DATASET / "report.json": DAGGER4_REPORT_SHA256,
        DAGGER4_DATASET / "metadata.json": DAGGER4_METADATA_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG022 source evidence hash mismatch: {path}")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG022 preregistration is missing")
    parent = json.loads(PARENT_REPORT.read_text())
    parent_admission = json.loads(PARENT_ADMISSION.read_text())
    dagger4 = json.loads((DAGGER4_DATASET / "report.json").read_text())
    dagger4_admission = json.loads(DAGGER4_ADMISSION.read_text())
    if (
        parent.get("schema") != "vq2_variable_gate_four_source_refit_report_v1"
        or parent.get("tag") != "vq2_vg017_variable_gate_four_source_refit_001"
        or not parent.get("completed")
        or not parent.get("numerically_admitted")
        or parent.get("best_epoch") != 11
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG017 parent is not admitted for VG020")
    if (
        parent_admission.get("schema")
        != "vq2_vg017_four_source_refit_admission_v1"
        or not parent_admission.get("numerically_admitted")
        or parent_admission.get("artifact_sha256", {}).get("checkpoint")
        != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("VG017 admission evidence does not bind VG020")
    if (
        dagger4.get("schema")
        != "vq2_vg019_variable_gate_dagger_collection_report_v1"
        or not dagger4.get("admitted")
        or dagger4.get("records") != 475_453
        or dagger4.get("failed_admission_predicates")
    ):
        raise RuntimeError("VG019 report does not admit its dataset")
    if (
        dagger4_admission.get("schema")
        != "vq2_vg019_variable_gate_dagger_admission_v1"
        or not dagger4_admission.get("admitted")
        or dagger4_admission.get("artifact_sha256", {}).get("report")
        != DAGGER4_REPORT_SHA256
    ):
        raise RuntimeError("VG019 admission evidence does not bind VG020")


def _load_parent(model: VQ2PhaseRecurrentActor) -> dict[str, Any]:
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("tag") != "vq2_vg017_variable_gate_four_source_refit_001"
        or contract.get("legal_observation_size") != PHASE_LEGAL_OBS_SIZE
        or contract.get("hidden_size") != model.hidden_size
        or not payload.get("numerically_admitted")
    ):
        raise RuntimeError("VG017 parent actor contract changed")
    model.load_state_dict(payload["model_state"])
    return payload


def verify_completed_output(output: Path, report: dict[str, Any]) -> None:
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("tag") != TAG
        or not report.get("completed")
    ):
        raise RuntimeError("existing VG020 report identity changed")
    source_commit, source_sha256 = current_source_identity()
    if (
        report.get("source_commit") != source_commit
        or report.get("source_sha256") != source_sha256
        or sha256_path(output / "policy_best.pt")
        != report.get("checkpoint_sha256")
    ):
        raise RuntimeError("completed VG020 source/checkpoint identity changed")
    state = torch.load(
        output / "training_state.pt", map_location="cpu", weights_only=False
    )
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("tag") != TAG
        or state.get("status") != "completed"
        or state.get("source_commit") != source_commit
        or state.get("source_sha256") != source_sha256
        or state.get("checkpoint_sha256") != report.get("checkpoint_sha256")
        or state.get("report_sha256") != sha256_path(output / "report.json")
    ):
        raise RuntimeError("completed VG020 state does not bind the report")


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: FiveSourceConfig = FiveSourceConfig(),
    resume: bool = False,
    migration_state: Path | None = None,
    parity_report: Path | None = None,
) -> dict[str, Any]:
    if config.sequence_chunk < 256 or config.transition_window_exposure < 4:
        raise RuntimeError("VG022 requires 256-step BPTT and 4x transitions")
    if objective_weights(config) != SOURCE_WEIGHTS:
        raise RuntimeError("VG022 requires exact 0.35/0.10/0.10/0.15/0.30 weights")
    verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG022 preregisters CUDA training")

    report_path = output / "report.json"
    state_path = output / "training_state.pt"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        report = json.loads(report_path.read_text())
        verify_completed_output(output, report)
        return report
    if output.exists() and not state_path.is_file():
        raise RuntimeError("VG022 output exists without resumable state")
    if state_path.is_file() and migration_state is not None:
        raise RuntimeError("VG022 cannot migrate over an existing state")

    device = torch.device(device_name)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    datasets = {
        "clean": VariableGateBCDataset(
            CLEAN_DATASET,
            expected_report_sha256=CLEAN_REPORT_SHA256,
            expected_metadata_sha256=CLEAN_METADATA_SHA256,
        ),
        "dagger1": VariableGateBCDataset(
            DAGGER1_DATASET,
            expected_report_sha256=DAGGER1_REPORT_SHA256,
            expected_metadata_sha256=DAGGER1_METADATA_SHA256,
        ),
        "dagger2": VariableGateBCDataset(
            DAGGER2_DATASET,
            report_path=DAGGER2_RECOVERY_REPORT,
            expected_report_sha256=DAGGER2_RECOVERY_REPORT_SHA256,
            expected_metadata_sha256=DAGGER2_METADATA_SHA256,
        ),
        "dagger3": VariableGateBCDataset(
            DAGGER3_DATASET,
            expected_report_sha256=DAGGER3_REPORT_SHA256,
            expected_metadata_sha256=DAGGER3_METADATA_SHA256,
        ),
        "dagger4": VariableGateBCDataset(
            DAGGER4_DATASET,
            expected_report_sha256=DAGGER4_REPORT_SHA256,
            expected_metadata_sha256=DAGGER4_METADATA_SHA256,
        ),
    }
    split_counts = {
        "clean": config.clean_validation_agents,
        "dagger1": config.dagger1_validation_agents,
        "dagger2": config.dagger2_validation_agents,
        "dagger3": config.dagger3_validation_agents,
        "dagger4": config.dagger4_validation_agents,
    }
    splits = {
        name: three_source._dataset_split(dataset.agents, split_counts[name])
        for name, dataset in datasets.items()
    }

    model = VQ2PhaseRecurrentActor(
        hidden_size=config.hidden_size, initial_std=config.initial_std
    ).to(device)
    parent = _load_parent(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config.seed)

    source_commit, source_sha256 = current_source_identity()
    if parity_report is None or not parity_report.is_file():
        raise RuntimeError("VG022 requires its admitted device-decode parity report")
    parity = json.loads(parity_report.read_text())
    if (
        parity.get("schema") != "vq2_vg022_device_decode_parity_report_v1"
        or parity.get("tag") != TAG
        or parity.get("source_commit") != source_commit
        or parity.get("migration_state_sha256") != MIGRATION_STATE_SHA256
        or not parity.get("admitted")
        or parity.get("optimizer_steps") != 0
        or parity.get("state_writes") != 0
        or not parity.get("decode_bitwise_equal")
        or float(parity.get("mean_max_abs_error", float("inf"))) != 0.0
        or float(parity.get("pre_tanh_max_abs_error", float("inf"))) != 0.0
        or float(parity.get("next_state_max_abs_error", float("inf"))) != 0.0
        or float(parity.get("loss_abs_error", float("inf"))) != 0.0
        or float(parity.get("gradient_max_abs_error", float("inf"))) != 0.0
        or float(parity.get("speedup", 0.0)) < 1.15
        or parity.get("runtime") != three_source.runtime_manifest()
    ):
        raise RuntimeError("VG022 device-decode parity report is not admitted")
    parity_identity = {
        "report_sha256": sha256_path(parity_report),
        "mean_max_abs_error": parity["mean_max_abs_error"],
        "pre_tanh_max_abs_error": parity["pre_tanh_max_abs_error"],
        "next_state_max_abs_error": parity["next_state_max_abs_error"],
        "loss_abs_error": parity["loss_abs_error"],
        "gradient_max_abs_error": parity["gradient_max_abs_error"],
        "speedup": parity["speedup"],
    }
    identity = {
        "schema": STATE_SCHEMA,
        "tag": TAG,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": three_source.runtime_manifest(),
        "train_config": asdict(config),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "migration": {
            "state_sha256": MIGRATION_STATE_SHA256,
            "source_commit": MIGRATION_SOURCE_COMMIT,
            "tag": MIGRATION_TAG,
            "schema": MIGRATION_SCHEMA,
            "completed_epoch": 3,
            "optimizer_updates": 31_742,
        },
        "acceleration_parity": parity_identity,
        "dataset_phase_audits": {
            name: dataset.phase_audit for name, dataset in datasets.items()
        },
        "source_balancing": {
            "per_optimizer_step": dict(SOURCE_WEIGHTS),
            "raw_record_count_does_not_set_source_weight": True,
            "training_agents": {name: len(splits[name][0]) for name in datasets},
            "validation_agents": {
                name: len(splits[name][1]) for name in datasets
            },
        },
        "safety": {
            "actor_input_privileged_values": 0,
            "teacher_blend": 0.0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }

    baseline_validation: dict[str, Any] = {}
    best_epoch = 0
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    optimizer_updates = 0
    completed_epoch = 0

    def restore_dynamic_state(saved: dict[str, Any]) -> None:
        nonlocal history, baseline_validation, best_epoch, best_score
        nonlocal best_state, optimizer_updates, completed_epoch
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        random.setstate(saved["python_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        history = saved["history"]
        baseline_validation = saved["baseline_validation"]
        best_epoch = int(saved["best_epoch"])
        best_score = float(saved["best_source_balanced_validation"])
        best_state = saved["best_state"]
        optimizer_updates = int(saved["optimizer_updates"])
        completed_epoch = int(saved["completed_epoch"])

    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"VG022 state already exists at {state_path}")
        saved = torch.load(state_path, map_location=device, weights_only=False)
        for key, expected in identity.items():
            if saved.get(key) != expected:
                raise RuntimeError(f"VG022 resume mismatch for {key}")
        if saved.get("status") != "training":
            raise RuntimeError("VG022 can resume only active training state")
        restore_dynamic_state(saved)
    else:
        if resume:
            raise RuntimeError("VG022 --resume requested without state")
        if migration_state is None or not migration_state.is_file():
            raise RuntimeError("VG022 requires the frozen VG020 migration state")
        if sha256_path(migration_state) != MIGRATION_STATE_SHA256:
            raise RuntimeError("VG022 migration state hash mismatch")
        saved = torch.load(migration_state, map_location=device, weights_only=False)
        required = {
            "schema": MIGRATION_SCHEMA,
            "tag": MIGRATION_TAG,
            "source_commit": MIGRATION_SOURCE_COMMIT,
            "status": "training",
            "completed_epoch": 3,
            "optimizer_updates": 31_742,
            "runtime": identity["runtime"],
            "train_config": identity["train_config"],
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_report_sha256": PARENT_REPORT_SHA256,
            "dataset_phase_audits": identity["dataset_phase_audits"],
            "source_balancing": identity["source_balancing"],
            "safety": identity["safety"],
        }
        for key, expected in required.items():
            if saved.get(key) != expected:
                raise RuntimeError(f"VG022 migration mismatch for {key}")
        if [int(item["epoch"]) for item in saved.get("history", [])] != [1, 2, 3]:
            raise RuntimeError("VG022 migration history is not exact epochs 1..3")
        if int(saved.get("best_epoch", -1)) != 3:
            raise RuntimeError("VG022 migration best epoch changed")
        restore_dynamic_state(saved)
        output.mkdir(parents=True)
        atomic_torch_save(
            state_path,
            three_source._state_payload(
                identity=identity,
                status="training",
                completed_epoch=completed_epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                rng=rng,
                history=history,
                baseline_validation=baseline_validation,
                best_epoch=best_epoch,
                best_score=best_score,
                best_state=best_state,
                optimizer_updates=optimizer_updates,
            ),
        )

    baseline_by_source = {name: baseline_validation[name] for name in datasets}

    action_weights = torch.tensor(config.action_weights, device=device)
    fixed_weights = objective_weights(config)
    started = time.perf_counter()
    for epoch in range(completed_epoch + 1, config.epochs + 1):
        model.train()
        epoch_updates_start = optimizer_updates
        streams = {
            name: RecurrentChunkStream(
                dataset,
                splits[name][0],
                batch_size=config.source_agent_batch_size,
                sequence_chunk=config.sequence_chunk,
                rng=rng,
                device=device,
                cyclic=name != "clean",
            )
            for name, dataset in datasets.items()
        }
        train_metrics = {name: ActionAccumulator() for name in datasets}
        source_weight_sums = {name: 0.0 for name in datasets}
        source_loss_sums = {name: 0.0 for name in datasets}
        total_loss_sum = 0.0
        paired_chunks = 0
        transition_chunks = 0
        transition_exposures = 0
        while True:
            clean_item = streams["clean"].next(model)
            if clean_item is None:
                break
            items = {"clean": clean_item}
            for name in ("dagger1", "dagger2", "dagger3", "dagger4"):
                item = streams[name].next(model)
                if item is None:
                    raise RuntimeError(f"VG020 cyclic {name} stream ended")
                items[name] = item
            transition_present = any(item.transition.any() for item in items.values())
            repetitions = config.transition_window_exposure if transition_present else 1
            transition_chunks += int(transition_present)
            final_outputs: dict[str, Any] = {}
            final_states: dict[str, torch.Tensor] = {}
            for _ in range(repetitions):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    outputs: dict[str, Any] = {}
                    states: dict[str, torch.Tensor] = {}
                    for name, item in items.items():
                        outputs[name], states[name] = model.forward_sequence(
                            item.observation, item.start_state
                        )
                    loss, components = five_source_loss(
                        {name: output.mean for name, output in outputs.items()},
                        {name: item.target for name, item in items.items()},
                        {name: item.valid for name, item in items.items()},
                        weights=action_weights,
                        previous_predictions={
                            name: item.previous_prediction for name, item in items.items()
                        },
                        previous_valid={
                            name: item.previous_valid for name, item in items.items()
                        },
                        config=config,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer_updates += 1
                transition_exposures += int(transition_present)
                total_loss_sum += float(loss.detach().item())
                for name in datasets:
                    source_weight_sums[name] += fixed_weights[name]
                    source_loss_sums[name] += float(
                        components[f"{name}_total"].detach().item()
                    )
                final_outputs = outputs
                final_states = states
            if set(final_outputs) != set(datasets):
                raise RuntimeError("VG020 emitted no complete source update")
            for name, item in items.items():
                train_metrics[name].add(
                    final_outputs[name].mean, item.target, item.valid
                )
                streams[name].commit(
                    final_outputs[name].mean, final_states[name], item.valid
                )
            paired_chunks += 1

        validation = {
            name: evaluate(model, dataset, splits[name][1], config, device)
            for name, dataset in datasets.items()
        }
        validation_score = sum(
            fixed_weights[name] * validation[name]["weighted_mse"]
            for name in datasets
        )
        epoch_updates = optimizer_updates - epoch_updates_start
        weights_exact = source_weight_audit(source_weight_sums, epoch_updates, config)
        if not weights_exact:
            raise RuntimeError("VG020 source objective weights diverged")
        minimum_transition_exposure = (
            transition_exposures / transition_chunks if transition_chunks else 0.0
        )
        epoch_report = {
            "epoch": epoch,
            "paired_chunks": paired_chunks,
            "optimizer_updates": optimizer_updates,
            "epoch_optimizer_updates": epoch_updates,
            "transition_paired_chunks": transition_chunks,
            "transition_update_exposures": transition_exposures,
            "minimum_transition_window_exposure": minimum_transition_exposure,
            "source_objective_weight_sums": source_weight_sums,
            "source_weight_audit": weights_exact,
            "stream_chunks": {
                name: stream.source_chunks for name, stream in streams.items()
            },
            "stream_records": {
                name: stream.source_records for name, stream in streams.items()
            },
            "cycles_started": {
                name: stream.cycles_started for name, stream in streams.items()
            },
            "mean_total_loss_per_update": total_loss_sum / max(epoch_updates, 1),
            "mean_source_loss_per_update": {
                name: source_loss_sums[name] / max(epoch_updates, 1)
                for name in datasets
            },
            "train": {
                name: accumulator.result()
                for name, accumulator in train_metrics.items()
            },
            "validation": validation,
            "source_balanced_validation_weighted_mse": validation_score,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        if np.isfinite(validation_score) and validation_score < best_score:
            best_score = float(validation_score)
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        completed_epoch = epoch
        atomic_torch_save(
            state_path,
            three_source._state_payload(
                identity=identity,
                status="training",
                completed_epoch=completed_epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                rng=rng,
                history=history,
                baseline_validation=baseline_validation,
                best_epoch=best_epoch,
                best_score=best_score,
                best_state=best_state,
                optimizer_updates=optimizer_updates,
            ),
        )

    if [int(item["epoch"]) for item in history] != list(range(1, 13)):
        raise RuntimeError("VG022 final history is not exact epochs 1..12")
    minimum_transition_exposure = min(
        float(item["minimum_transition_window_exposure"]) for item in history
    )
    source_balance_audit = all(bool(item["source_weight_audit"]) for item in history)
    if minimum_transition_exposure + 1e-12 < config.transition_window_exposure:
        raise RuntimeError("VG020 transition exposure contract failed")
    if not source_balance_audit:
        raise RuntimeError("VG020 source balance audit failed")
    selected_validation = (
        baseline_by_source if best_epoch == 0 else history[best_epoch - 1]["validation"]
    )
    numerically_admitted = bool(
        best_epoch > 0
        and np.isfinite(best_score)
        and best_score < float(baseline_validation["source_balanced_weighted_mse"])
        and selected_validation["dagger4"]["weighted_mse"]
        < baseline_validation["dagger4"]["weighted_mse"]
        and selected_validation["clean"]["weighted_mse"]
        <= config.maximum_clean_validation_weighted_mse
        and selected_validation["dagger1"]["weighted_mse"]
        <= config.maximum_dagger1_validation_weighted_mse
        and selected_validation["dagger2"]["weighted_mse"]
        <= config.maximum_dagger2_validation_weighted_mse
        and selected_validation["dagger3"]["weighted_mse"]
        <= config.maximum_dagger3_validation_weighted_mse
        and selected_validation["dagger4"]["weighted_mse"]
        <= config.maximum_dagger4_validation_weighted_mse
        and source_balance_audit
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "tag": TAG,
        "model": parent["model"],
        "model_state": best_state,
        "train_config": asdict(config),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "migration": identity["migration"],
        "acceleration_parity": identity["acceleration_parity"],
        "source_splits": {
            name: {
                "train": splits[name][0].tolist(),
                "validation": splits[name][1].tolist(),
            }
            for name in datasets
        },
        "dataset_phase_audits": identity["dataset_phase_audits"],
        "baseline_validation": baseline_validation,
        "best_epoch": best_epoch,
        "best_source_balanced_validation": best_score,
        "selected_validation": selected_validation,
        "history": history,
        "optimizer_updates": optimizer_updates,
        "minimum_transition_window_exposure": minimum_transition_exposure,
        "equal_source_weight_audit": source_balance_audit,
        "numerically_admitted": numerically_admitted,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": identity["runtime"],
        "safety": identity["safety"],
    }
    checkpoint_path = output / "policy_best.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": REPORT_SCHEMA,
        "tag": TAG,
        "completed": True,
        "numerically_admitted": numerically_admitted,
        "checkpoint": source_label(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "migration": identity["migration"],
        "acceleration_parity": identity["acceleration_parity"],
        "best_epoch": best_epoch,
        "best_source_balanced_validation": best_score,
        "baseline_validation": baseline_validation,
        "selected_validation": selected_validation,
        "minimum_transition_window_exposure": minimum_transition_exposure,
        "equal_source_weight_audit": source_balance_audit,
        "optimizer_updates": optimizer_updates,
        "history": history,
        "train_config": asdict(config),
        "dataset_phase_audits": identity["dataset_phase_audits"],
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": identity["runtime"],
        "wall_time_seconds": time.perf_counter() - started,
        "safety": identity["safety"],
    }
    write_json_atomic(report_path, report)
    final_state = three_source._state_payload(
        identity=identity,
        status="completed",
        completed_epoch=completed_epoch,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        rng=rng,
        history=history,
        baseline_validation=baseline_validation,
        best_epoch=best_epoch,
        best_score=best_score,
        best_state=best_state,
        optimizer_updates=optimizer_updates,
    )
    final_state["checkpoint_sha256"] = report["checkpoint_sha256"]
    final_state["report_sha256"] = sha256_path(report_path)
    atomic_torch_save(state_path, final_state)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--migration-state", type=Path)
    parser.add_argument("--parity-report", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
        migration_state=(
            None if args.migration_state is None else args.migration_state.resolve()
        ),
        parity_report=args.parity_report.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
