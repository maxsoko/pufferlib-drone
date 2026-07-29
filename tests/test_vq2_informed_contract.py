from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

from pufferlib.vq2_informed import (
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    MASK_SIZE,
    OBSERVATION_SCHEMA,
    PRIVILEGED_NAMES,
    PRIVILEGED_SIZE,
    LegalActorAdapter,
    VQ2VisualEncoder,
    configure_full_start_evaluation,
    split_environment_observation,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "ocean" / "drone_race" / "drone_race.h"


def test_python_schema_matches_native_abi() -> None:
    assert MASK_SIZE == 4096
    assert LEGAL_OBS_SIZE == 4118
    assert PRIVILEGED_SIZE == 34
    assert ENV_OBS_SIZE == 4152
    text = HEADER.read_text(encoding="utf-8")
    match = re.search(r"#define DRONE_RACE_VISUAL_PRIVILEGED_SIZE\s+(\d+)", text)
    assert match is not None and int(match.group(1)) == PRIVILEGED_SIZE
    assert len(set(PRIVILEGED_NAMES)) == PRIVILEGED_SIZE
    assert OBSERVATION_SCHEMA == "vq2_visual_ctbr_v2"
    assert PRIVILEGED_NAMES[26:32] == (
        "ctbr_hover_thrust",
        "ctbr_vertical_accel_per_thrust",
        "ctbr_max_thrust",
        "ctbr_rate_lag_time_constant",
        "ctbr_gravity",
        "ctbr_linear_drag",
    )


def test_split_is_exact_and_rejects_wrong_abi() -> None:
    observation = torch.arange(2 * ENV_OBS_SIZE, dtype=torch.float32).reshape(
        2, ENV_OBS_SIZE
    )
    split = split_environment_observation(observation)
    assert split.legal.shape == (2, LEGAL_OBS_SIZE)
    assert split.privileged.shape == (2, PRIVILEGED_SIZE)
    assert split.legal.data_ptr() == observation.data_ptr()
    assert torch.equal(split.privileged, observation[:, LEGAL_OBS_SIZE:])
    with pytest.raises(ValueError, match="expected a 4152-value"):
        split_environment_observation(observation[:, :-1])


def test_visual_encoder_structurally_rejects_privileged_tail() -> None:
    encoder = VQ2VisualEncoder(output_size=32)
    legal = torch.zeros(3, LEGAL_OBS_SIZE)
    assert encoder(legal).shape == (3, 32)
    with pytest.raises(ValueError, match="competition-legal slice"):
        encoder(torch.zeros(3, ENV_OBS_SIZE))


def test_privileged_intervention_cannot_change_actor_action() -> None:
    torch.manual_seed(5)
    actor = nn.Sequential(VQ2VisualEncoder(32), nn.Linear(32, 4), nn.Tanh())
    adapter = LegalActorAdapter(actor).eval()
    observation = torch.randn(4, ENV_OBS_SIZE)
    changed = observation.clone()
    changed[:, LEGAL_OBS_SIZE:] = 1000.0 * torch.randn(
        4, PRIVILEGED_SIZE
    )
    with torch.no_grad():
        before = adapter(observation)
        after = adapter(changed)
    assert torch.equal(before, after)


def test_evaluation_contract_disables_every_local_start_curriculum() -> None:
    config = {
        "env": {
            "gate_local_start_curriculum": 1,
            "gate_local_start_probability": 0.7,
            "mixed_start_curriculum": 1,
            "segment_start_probability": 0.5,
            "use_custom_start": 0,
        }
    }
    configured = configure_full_start_evaluation(
        config, episodes_per_agent=2, episode_offset=17
    )
    environment = configured["env"]
    assert environment["gate_local_start_curriculum"] == 0
    assert environment["gate_local_start_probability"] == 0.0
    assert environment["mixed_start_curriculum"] == 0
    assert environment["segment_start_probability"] == 0.0
    assert environment["evaluation_episode_limit"] == 2
    assert environment["evaluation_episode_offset"] == 17


def test_evaluation_contract_rejects_late_custom_segment_start() -> None:
    config = {"env": {"use_custom_start": 1, "start_gate_index": 2}}
    with pytest.raises(RuntimeError, match="custom start after Gate 1"):
        configure_full_start_evaluation(
            config, episodes_per_agent=1, episode_offset=0
        )
