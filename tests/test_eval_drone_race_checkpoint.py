import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_drone_race_checkpoint.py"
SCRIPTS = MODULE_PATH.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("eval_drone_race_checkpoint", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_validate_backend_env_accepts_exact_env_match():
    module._validate_backend_env("drone_race_competition", "drone_race_competition", "drone_race")


def test_validate_backend_env_accepts_backend_match():
    module._validate_backend_env("drone_race", "drone_race_competition", "drone_race")


def test_validate_backend_env_rejects_mismatch():
    with pytest.raises(RuntimeError, match="backend mismatch"):
        module._validate_backend_env("drone", "drone_race_competition", "drone_race")

