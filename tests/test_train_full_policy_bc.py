import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_full_policy_bc.py"
SCRIPTS = MODULE_PATH.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("train_full_policy_bc", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_race_phase_supports_six_gate_progress_observation():
    observations = np.zeros((3, module.OBSERVATIONS), dtype=np.float32)
    observations[:, 23] = (0.0, 3.0 / 6.0, 5.0 / 6.0)

    phase = module._race_phase(
        observations, gate_progress_observation=True)

    assert np.allclose(phase, (0.0, 0.5, 5.0 / 6.0))


def test_six_gate_dataset_requires_progress_phase_contract():
    legacy = np.zeros((2, module.RECORD_WIDTH), dtype=np.float32)
    six_gate = legacy.copy()
    six_gate[1, 28] = 1.0

    assert not module.requires_gate_progress_observation([legacy])
    assert module.requires_gate_progress_observation([six_gate])


def test_restore_unselected_encoder_features_preserves_only_selected_columns():
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    encoder = source + 100.0

    module.restore_unselected_encoder_features(encoder, source, (1, 4))

    assert torch.equal(encoder[:, 0], source[:, 0])
    assert torch.equal(encoder[:, 2], source[:, 2])
    assert torch.equal(encoder[:, 3], source[:, 3])
    assert torch.equal(encoder[:, 5], source[:, 5])
    assert torch.equal(encoder[:, 1], source[:, 1] + 100.0)
    assert torch.equal(encoder[:, 4], source[:, 4] + 100.0)


def test_source_relative_targets_adds_only_requested_action_offsets():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    decoder = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [9.0, 9.0],
    ])
    offset = torch.tensor([0.0, 0.25, 0.0, -0.5])

    targets = module.source_relative_targets(hidden, decoder, offset)

    assert torch.equal(
        targets,
        torch.tensor([[1.0, 2.25, 3.0, 0.5], [3.0, 4.25, 7.0, 0.5]]),
    )


def test_offset_source_actions_broadcasts_over_sequence_predictions():
    source = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
    )
    offset = torch.tensor([0.0, -0.25, 0.5, 0.0])

    targets = module.offset_source_actions(source, offset)

    assert torch.equal(
        targets,
        torch.tensor(
            [[[1.0, 1.75, 3.5, 4.0], [5.0, 5.75, 7.5, 8.0]]]
        ),
    )


def test_source_anchor_outside_active_uses_every_non_target_record():
    mask = torch.tensor([[True, True, True, False]])
    active = torch.tensor([[False, True, False, False]])
    observations = torch.zeros((1, 4, module.OBSERVATIONS))
    race_phase = torch.tensor([[0.0, 2 / 3, 1.0, 0.0]])

    anchor = module.source_anchor_mask(
        mask,
        active,
        observations,
        race_phase,
        outside_active=True,
        max_race_phase=None,
        until_elapsed_fraction=0.23,
    )

    assert torch.equal(anchor, torch.tensor([[True, False, True, False]]))
