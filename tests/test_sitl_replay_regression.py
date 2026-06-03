import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sitl_replay_regression.py"
SCRIPTS = MODULE_PATH.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("sitl_replay_regression", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_compare_modes_policy_non_inferior():
    baseline = {"valid_rate": 0.9}
    policy = {"valid_rate": 0.9}
    out = module.compare_modes(baseline, policy)
    assert out["policy_non_inferior_to_baseline"] is True


def test_compare_modes_policy_inferior():
    baseline = {"valid_rate": 0.95}
    policy = {"valid_rate": 0.80}
    out = module.compare_modes(baseline, policy)
    assert out["policy_non_inferior_to_baseline"] is False
