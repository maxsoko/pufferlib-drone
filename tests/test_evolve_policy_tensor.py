import json
from pathlib import Path

import numpy as np

from scripts.convert_policy_checkpoint_layout import (
    pack_layout,
    tensor_counts,
    unpack_layout,
)
from scripts.evolve_policy_tensor import form_fitness_direction, generate_antithetic


def _checkpoint(path: Path) -> tuple[list[int], list[np.ndarray]]:
    counts = tensor_counts(input_dim=4, hidden_dim=8, num_layers=3, num_actions=4)
    rng = np.random.default_rng(7)
    tensors = [rng.normal(0.0, 0.1, count).astype(np.float32) for count in counts]
    pack_layout(tensors, counts, precision_bytes=4).tofile(path)
    return counts, tensors


def _read(path: Path, counts: list[int]) -> list[np.ndarray]:
    return unpack_layout(
        np.fromfile(path, dtype=np.float32), counts, precision_bytes=4
    )


def _report(path: Path, *, gates: float, radial: float) -> None:
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "env/gates_passed": gates,
                    "env/avg_terminal_crossing_radial": radial,
                    "env/crash": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )


def test_antithetic_generation_changes_only_selected_tensor(tmp_path):
    parent_path = tmp_path / "parent.bin"
    counts, parent = _checkpoint(parent_path)
    records = generate_antithetic(
        parent_path,
        tmp_path / "probes",
        tensor_index=5,
        radius=1e-4,
        num_directions=2,
        seed=3385,
        input_dim=4,
        hidden_dim=8,
        layout_precision_bytes=4,
    )
    assert len(records) == 4
    plus = _read(tmp_path / "probes/direction0_plus.bin", counts)
    minus = _read(tmp_path / "probes/direction0_minus.bin", counts)
    for index in range(5):
        np.testing.assert_array_equal(plus[index], parent[index])
        np.testing.assert_array_equal(minus[index], parent[index])
    plus_delta = plus[5] - parent[5]
    minus_delta = minus[5] - parent[5]
    assert np.linalg.norm(plus_delta) > 0.0
    np.testing.assert_allclose(plus_delta, -minus_delta, atol=1e-8)


def test_fitness_direction_uses_antithetic_report_differences(tmp_path):
    parent_path = tmp_path / "parent.bin"
    counts, parent = _checkpoint(parent_path)
    probes = tmp_path / "probes"
    generate_antithetic(
        parent_path,
        probes,
        tensor_index=5,
        radius=1e-4,
        num_directions=2,
        seed=3385,
        input_dim=4,
        hidden_dim=8,
        layout_precision_bytes=4,
    )
    _report(probes / "direction0_plus_floor6_exact1024.json", gates=1.0, radial=0.8)
    _report(probes / "direction0_minus_floor6_exact1024.json", gates=1.0, radial=1.2)
    _report(probes / "direction1_plus_floor6_exact1024.json", gates=1.0, radial=1.3)
    _report(probes / "direction1_minus_floor6_exact1024.json", gates=1.0, radial=0.9)

    records = form_fitness_direction(
        parent_path,
        probes,
        tmp_path / "fitness",
        tensor_index=5,
        radius=1e-4,
        num_directions=2,
        input_dim=4,
        hidden_dim=8,
        layout_precision_bytes=4,
    )
    assert records[0]["fitness_difference"] > 0.0
    assert records[1]["fitness_difference"] < 0.0
    plus = _read(tmp_path / "fitness/fitness_direction_plus.bin", counts)
    for index in range(5):
        np.testing.assert_array_equal(plus[index], parent[index])
    assert np.linalg.norm(plus[5] - parent[5]) > 0.0
