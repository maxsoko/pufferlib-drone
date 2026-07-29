import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "scale_policy_encoder_hidden_rows.py"
)
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "scale_policy_encoder_hidden_rows", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_four_row_latent_blocks_use_full_32_observation_stride():
    assert MODULE.encoder_row_bounds(
        input_dim=32, hidden_dim=128, row_start=96, row_count=4
    ) == (3072, 3200)
    assert MODULE.encoder_row_bounds(
        input_dim=32, hidden_dim=128, row_start=100, row_count=4
    ) == (3200, 3328)


def test_encoder_row_bounds_rejects_out_of_range_blocks():
    try:
        MODULE.encoder_row_bounds(
            input_dim=32, hidden_dim=128, row_start=126, row_count=4
        )
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("out-of-range encoder block was accepted")
