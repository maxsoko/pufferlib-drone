import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "wait_official_race_start.py"
SPEC = importlib.util.spec_from_file_location("wait_official_race_start", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

ADAPTER_SPEC = importlib.util.spec_from_file_location("drone_sitl_adapter", SCRIPTS / "drone_sitl_adapter.py")
sitl = importlib.util.module_from_spec(ADAPTER_SPEC)
sys.modules[ADAPTER_SPEC.name] = sitl
ADAPTER_SPEC.loader.exec_module(sitl)


class FakeTelemetry:
    def __init__(self):
        self.state = sitl.TelemetryState()
        self.metrics = sitl.TelemetryMetrics()


class FakeAdapter:
    def __init__(self, _endpoint, dropout_after_s=1.0, *, start_after_polls=1):
        self.telemetry = FakeTelemetry()
        self.start_after_polls = start_after_polls
        self.polls = 0
        self.closed = False

    def poll_telemetry(self, *, timeout_s=0.0):
        self.polls += 1
        self.telemetry.metrics.messages_seen += 1
        self.telemetry.metrics.heartbeats += 1
        self.telemetry.metrics.race_statuses += 1
        if self.polls >= self.start_after_polls:
            self.telemetry.state.race_status = sitl.RaceStatus(
                sim_boot_time_ms=100,
                race_start_boot_time_ms=10,
                race_finish_time_ns=-1,
                active_gate_index=0,
                last_gate_race_time=-1,
            )
            self.telemetry.state.base_mode = 193
            self.telemetry.state.system_status = 4

    def close(self):
        self.closed = True


def test_wait_for_race_start_reports_started():
    args = SimpleNamespace(
        endpoint="udpin:0.0.0.0:14550",
        duration=1.0,
        poll_timeout_s=0.0,
        telemetry_dropout_s=1.0,
    )

    report = module.wait_for_race_start(
        args,
        adapter_factory=lambda *a, **kw: FakeAdapter(*a, **kw, start_after_polls=1),
    )

    assert report["race_started"] is True
    assert report["race_status"]["race_start_boot_time_ms"] == 10
    assert report["base_mode"] == 193
    assert report["heartbeats_seen"] == 1


def test_wait_for_race_start_validates_duration():
    args = SimpleNamespace(
        endpoint="udpin:0.0.0.0:14550",
        duration=0.0,
        poll_timeout_s=0.0,
        telemetry_dropout_s=1.0,
    )

    with pytest.raises(ValueError, match="duration"):
        module.wait_for_race_start(args, adapter_factory=FakeAdapter)
