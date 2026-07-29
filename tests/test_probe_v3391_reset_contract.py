import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_v3391_reset_contract.py"
SPEC = importlib.util.spec_from_file_location("probe_v3391_reset_contract", MODULE_PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC.loader.exec_module(probe)


def status(**overrides):
    values = {
        "sim_boot_time_ms": 3000,
        "race_start_boot_time_ms": 3300,
        "race_finish_time_ns": -1,
        "active_gate_index": 0,
        "last_gate_race_time": -1,
    }
    values.update(overrides)
    return probe.RaceStatus(**values)


def test_accepts_only_large_boot_rollback_and_fresh_unfinished_gate_zero():
    assert probe.accepted_reset_status(200_000, status())
    assert not probe.accepted_reset_status(3500, status())
    assert not probe.accepted_reset_status(200_000, status(race_start_boot_time_ms=-1))
    assert not probe.accepted_reset_status(200_000, status(active_gate_index=1))
    assert not probe.accepted_reset_status(200_000, status(race_finish_time_ns=1))


def test_missing_status_is_not_accepted():
    assert not probe.accepted_reset_status(200_000, None)
