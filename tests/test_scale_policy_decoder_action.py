import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "scale_policy_decoder_action.py"
)
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "scale_policy_decoder_action", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_policy_decoder_rows_start_after_all_32_encoder_columns():
    assert MODULE.decoder_action_bounds(
        input_dim=32, hidden_dim=128, action_index=0
    ) == (4096, 4224)
    assert MODULE.decoder_action_bounds(
        input_dim=32, hidden_dim=128, action_index=1
    ) == (4224, 4352)
    assert MODULE.decoder_action_bounds(
        input_dim=32, hidden_dim=128, action_index=2
    ) == (4352, 4480)
