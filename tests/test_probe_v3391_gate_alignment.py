from argparse import Namespace
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "probe_v3391_gate_alignment.py"
SPEC = importlib.util.spec_from_file_location("probe_v3391_gate_alignment", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_probe_rejects_nonpositive_duration(tmp_path):
    args = Namespace(duration_s=0.0, output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="duration-s"):
        module.run_probe(args)
