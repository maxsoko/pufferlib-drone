import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_gate3_terminal_level_n191.py"


def test_verifier_module_loads_and_candidate_parser_is_exact():
    spec = importlib.util.spec_from_file_location("verify_n192", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._candidate(Path("policy_bounded_051_attempt_001.json")) == "051"
