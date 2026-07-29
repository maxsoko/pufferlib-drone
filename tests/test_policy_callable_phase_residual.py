import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

CHECKPOINT_MODULE_PATH = SCRIPTS / "policy_callable_checkpoint.py"
CHECKPOINT_SPEC = importlib.util.spec_from_file_location(
    "phase_test_checkpoint_policy", CHECKPOINT_MODULE_PATH
)
checkpoint_policy = importlib.util.module_from_spec(CHECKPOINT_SPEC)
sys.modules[CHECKPOINT_SPEC.name] = checkpoint_policy
CHECKPOINT_SPEC.loader.exec_module(checkpoint_policy)

PHASE_MODULE_PATH = SCRIPTS / "policy_callable_phase_residual.py"
PHASE_SPEC = importlib.util.spec_from_file_location(
    "policy_callable_phase_residual", PHASE_MODULE_PATH
)
phase_policy = importlib.util.module_from_spec(PHASE_SPEC)
sys.modules[PHASE_SPEC.name] = phase_policy
PHASE_SPEC.loader.exec_module(phase_policy)


def _append_aligned(storage: list[float], values: list[float]) -> None:
    storage.extend(values)
    while len(storage) % 8 != 0:
        storage.append(0.0)


def _build_checkpoint(path: Path, *, decoder_offset: float = 0.0) -> None:
    input_dim = 23
    hidden_dim = 2
    num_actions = 4
    values: list[float] = []

    encoder = np.linspace(-0.2, 0.3, hidden_dim * input_dim, dtype=np.float32)
    _append_aligned(values, encoder.tolist())

    decoder = np.linspace(
        -0.4, 0.5, (num_actions + 1) * hidden_dim, dtype=np.float32
    ).reshape(num_actions + 1, hidden_dim)
    decoder[:num_actions] += np.float32(decoder_offset)
    _append_aligned(values, decoder.reshape(-1).tolist())
    _append_aligned(values, [0.0] * num_actions)

    mingru = np.linspace(
        -0.3, 0.25, 3 * hidden_dim * hidden_dim, dtype=np.float32
    )
    _append_aligned(values, mingru.tolist())
    np.asarray(values, dtype=np.float32).tofile(path)


def _configure(monkeypatch, base: Path, residual: Path) -> None:
    monkeypatch.setenv("PUFFER_POLICY_BASE_CHECKPOINT_PATH", str(base))
    monkeypatch.setenv("PUFFER_POLICY_RESIDUAL_CHECKPOINT_PATH", str(residual))
    monkeypatch.delenv("PUFFER_POLICY_FINAL_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("PUFFER_POLICY_GATE_HEADS_JSON", raising=False)
    monkeypatch.delenv(
        "PUFFER_POLICY_GATE_HEAD_MIN_CONFIDENCE_JSON", raising=False
    )
    monkeypatch.delenv(
        "PUFFER_POLICY_RESIDUAL_LOW_CONFIDENCE_CHECKPOINT_PATH", raising=False
    )
    monkeypatch.delenv(
        "PUFFER_POLICY_FINAL_LOW_CONFIDENCE_CHECKPOINT_PATH", raising=False
    )
    monkeypatch.setenv("PUFFER_POLICY_INPUT_DIM", "23")
    monkeypatch.setenv("PUFFER_POLICY_HIDDEN_DIM", "2")
    monkeypatch.setenv("PUFFER_POLICY_NUM_LAYERS", "1")
    monkeypatch.setenv("PUFFER_POLICY_NUM_ACTIONS", "4")
    monkeypatch.setenv("PUFFER_POLICY_RACE_PHASE_DENOMINATOR", "3")
    phase_policy._BASE = None
    phase_policy._HEAD_DECODERS = None
    phase_policy._LOW_CONFIDENCE_HEAD_DECODERS = None
    phase_policy._KEY = None


def test_phase_policy_exactly_preserves_base_through_gate_two(tmp_path, monkeypatch):
    base_path = tmp_path / "base.bin"
    residual_path = tmp_path / "residual.bin"
    _build_checkpoint(base_path)
    _build_checkpoint(residual_path, decoder_offset=0.2)
    _configure(monkeypatch, base_path, residual_path)

    base = checkpoint_policy.CheckpointPolicy.load(
        str(base_path), input_dim=23, hidden_dim=2, num_layers=1, num_actions=4
    )
    phase_policy.reset()
    last_yaw = 0.0
    for phase in (0.0, np.float32(1.0 / 3.0), np.float32(2.0 / 3.0)):
        official_observation = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
        official_observation[22] = phase
        recurrent_observation = official_observation.copy()
        recurrent_observation[22] = last_yaw
        expected = base.infer(recurrent_observation)
        actual = phase_policy.infer(official_observation)
        assert actual == expected
        last_yaw = expected[3]


def test_phase_policy_uses_residual_decoder_from_gate_two(tmp_path, monkeypatch):
    base_path = tmp_path / "base.bin"
    residual_path = tmp_path / "residual.bin"
    _build_checkpoint(base_path)
    _build_checkpoint(residual_path, decoder_offset=0.2)
    _configure(monkeypatch, base_path, residual_path)

    residual = checkpoint_policy.CheckpointPolicy.load(
        str(residual_path), input_dim=23, hidden_dim=2, num_layers=1, num_actions=4
    )
    phase_policy.reset()
    official_observation = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
    official_observation[22] = 1.0
    recurrent_observation = official_observation.copy()
    recurrent_observation[22] = 0.0

    assert phase_policy.infer(official_observation) == pytest.approx(
        residual.infer(recurrent_observation), abs=1e-7
    )


def test_phase_policy_rejects_residual_recurrent_changes(tmp_path, monkeypatch):
    base_path = tmp_path / "base.bin"
    residual_path = tmp_path / "residual.bin"
    _build_checkpoint(base_path)
    _build_checkpoint(residual_path, decoder_offset=0.2)
    residual_values = np.fromfile(residual_path, dtype=np.float32)
    # The aligned MinGRU tensor begins at float offset 72 for this fixture.
    residual_values[72] += 1.0
    residual_values.tofile(residual_path)
    _configure(monkeypatch, base_path, residual_path)

    with pytest.raises(ValueError, match="MinGRU layer"):
        phase_policy.reset()


def test_phase_policy_selects_separate_gate_two_and_three_heads(tmp_path, monkeypatch):
    base_path = tmp_path / "base.bin"
    gate_two_path = tmp_path / "gate-two.bin"
    gate_three_path = tmp_path / "gate-three.bin"
    _build_checkpoint(base_path)
    _build_checkpoint(gate_two_path, decoder_offset=0.2)
    _build_checkpoint(gate_three_path, decoder_offset=-0.15)
    _configure(monkeypatch, base_path, gate_two_path)
    monkeypatch.setenv("PUFFER_POLICY_FINAL_CHECKPOINT_PATH", str(gate_three_path))

    phase_policy.reset()
    observation = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
    observation[22] = np.float32(2.0 / 3.0)
    gate_two = checkpoint_policy.CheckpointPolicy.load(
        str(gate_two_path), input_dim=23, hidden_dim=2, num_layers=1, num_actions=4
    )
    recurrent_observation = observation.copy()
    recurrent_observation[22] = 0.0
    assert phase_policy.infer(observation) == pytest.approx(
        gate_two.infer(recurrent_observation), abs=1e-7
    )

    phase_policy.reset()
    observation[22] = 1.0
    gate_three = checkpoint_policy.CheckpointPolicy.load(
        str(gate_three_path), input_dim=23, hidden_dim=2, num_layers=1, num_actions=4
    )
    recurrent_observation[22] = 0.0
    assert phase_policy.infer(observation) == pytest.approx(
        gate_three.infer(recurrent_observation), abs=1e-7
    )


def test_phase_policy_selects_named_gate_one_head(tmp_path, monkeypatch):
    base_path = tmp_path / "base.bin"
    gate_one_path = tmp_path / "gate-one.bin"
    gate_two_path = tmp_path / "gate-two.bin"
    gate_three_path = tmp_path / "gate-three.bin"
    _build_checkpoint(base_path)
    _build_checkpoint(gate_one_path, decoder_offset=0.05)
    _build_checkpoint(gate_two_path, decoder_offset=0.2)
    _build_checkpoint(gate_three_path, decoder_offset=-0.15)
    _configure(monkeypatch, base_path, gate_two_path)
    monkeypatch.setenv("PUFFER_POLICY_PREVIOUS_CHECKPOINT_PATH", str(gate_one_path))
    monkeypatch.setenv("PUFFER_POLICY_FINAL_CHECKPOINT_PATH", str(gate_three_path))

    observation = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
    observation[22] = np.float32(1.0 / 3.0)
    recurrent_observation = observation.copy()
    recurrent_observation[22] = 0.0
    gate_one = checkpoint_policy.CheckpointPolicy.load(
        str(gate_one_path), input_dim=23, hidden_dim=2, num_layers=1, num_actions=4
    )

    phase_policy.reset()
    assert phase_policy.infer(observation) == pytest.approx(
        gate_one.infer(recurrent_observation), abs=1e-7
    )


def test_phase_policy_falls_back_to_base_below_gate_confidence(tmp_path, monkeypatch):
    base_path = tmp_path / "base.bin"
    gate_two_path = tmp_path / "gate-two.bin"
    gate_three_path = tmp_path / "gate-three.bin"
    _build_checkpoint(base_path)
    _build_checkpoint(gate_two_path, decoder_offset=0.2)
    _build_checkpoint(gate_three_path, decoder_offset=-0.15)
    _configure(monkeypatch, base_path, gate_two_path)
    monkeypatch.setenv("PUFFER_POLICY_FINAL_CHECKPOINT_PATH", str(gate_three_path))
    monkeypatch.setenv("PUFFER_POLICY_GATE_HEAD_MIN_CONFIDENCE_JSON", '{"2": 0.6}')

    observation = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
    observation[17] = 0.2
    observation[22] = np.float32(2.0 / 3.0)
    recurrent_observation = observation.copy()
    recurrent_observation[22] = 0.0
    base = checkpoint_policy.CheckpointPolicy.load(
        str(base_path), input_dim=23, hidden_dim=2, num_layers=1, num_actions=4
    )

    phase_policy.reset()
    assert phase_policy.infer(observation) == base.infer(recurrent_observation)


def test_phase_policy_selects_low_confidence_expert(tmp_path, monkeypatch):
    base_path = tmp_path / "base.bin"
    gate_two_path = tmp_path / "gate-two.bin"
    gate_three_path = tmp_path / "gate-three.bin"
    low_gate_two_path = tmp_path / "low-gate-two.bin"
    _build_checkpoint(base_path)
    _build_checkpoint(gate_two_path, decoder_offset=0.2)
    _build_checkpoint(gate_three_path, decoder_offset=-0.15)
    _build_checkpoint(low_gate_two_path, decoder_offset=0.05)
    _configure(monkeypatch, base_path, gate_two_path)
    monkeypatch.setenv("PUFFER_POLICY_FINAL_CHECKPOINT_PATH", str(gate_three_path))
    monkeypatch.setenv(
        "PUFFER_POLICY_RESIDUAL_LOW_CONFIDENCE_CHECKPOINT_PATH",
        str(low_gate_two_path),
    )
    monkeypatch.setenv(
        "PUFFER_POLICY_FINAL_LOW_CONFIDENCE_CHECKPOINT_PATH",
        str(gate_three_path),
    )
    monkeypatch.setenv("PUFFER_POLICY_GATE_HEAD_MIN_CONFIDENCE_JSON", '{"2": 0.4}')

    observation = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
    observation[17] = 0.1
    observation[22] = np.float32(2.0 / 3.0)
    recurrent_observation = observation.copy()
    recurrent_observation[22] = 0.0
    low_head = checkpoint_policy.CheckpointPolicy.load(
        str(low_gate_two_path),
        input_dim=23,
        hidden_dim=2,
        num_layers=1,
        num_actions=4,
    )

    phase_policy.reset()
    assert phase_policy.infer(observation) == pytest.approx(
        low_head.infer(recurrent_observation), abs=1e-7
    )
