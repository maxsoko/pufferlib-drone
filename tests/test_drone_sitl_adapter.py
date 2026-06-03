import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "drone_sitl_adapter.py"
SPEC = importlib.util.spec_from_file_location("drone_sitl_adapter", MODULE_PATH)
sitl = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sitl
SPEC.loader.exec_module(sitl)


class FakeMavlinkMessage:
    def __init__(self, msg_type, **attrs):
        self._msg_type = msg_type
        for key, value in attrs.items():
            setattr(self, key, value)

    def get_type(self):
        return self._msg_type


def test_telemetry_parser_tracks_ts002_messages():
    parser = sitl.MavlinkTelemetryParser(dropout_after_s=1.0)

    assert parser.ingest(
        FakeMavlinkMessage("HEARTBEAT", system_status=4, base_mode=1, custom_mode=99),
        now_s=10.0,
    ) == "HEARTBEAT"
    assert parser.ingest(
        FakeMavlinkMessage(
            "ATTITUDE",
            time_boot_ms=123,
            roll=0.1,
            pitch=-0.2,
            yaw=0.3,
            rollspeed=0.01,
            pitchspeed=0.02,
            yawspeed=0.03,
        ),
        now_s=10.1,
    ) == "ATTITUDE"
    assert parser.ingest(
        FakeMavlinkMessage(
            "HIGHRES_IMU",
            time_usec=123456,
            xacc=1.0,
            yacc=2.0,
            zacc=3.0,
            xgyro=0.4,
            ygyro=0.5,
            zgyro=0.6,
            xmag=7.0,
            ymag=8.0,
            zmag=9.0,
            abs_pressure=1000.0,
            diff_pressure=0.1,
            pressure_alt=42.0,
            temperature=25.0,
            fields_updated=0xFFFF,
        ),
        now_s=10.2,
    ) == "HIGHRES_IMU"
    assert parser.ingest({"type": "TIMESYNC", "tc1": 111, "ts1": 222}, now_s=10.3) == "TIMESYNC"

    assert parser.metrics.messages_seen == 4
    assert parser.metrics.heartbeats == 1
    assert parser.metrics.attitudes == 1
    assert parser.metrics.highres_imus == 1
    assert parser.metrics.timesyncs == 1
    assert parser.state.system_status == 4
    assert parser.state.roll == 0.1
    assert parser.state.zacc == 3.0
    assert parser.state.timesync_tc1 == 111


def test_telemetry_parser_counts_unknown_malformed_and_dropouts():
    parser = sitl.MavlinkTelemetryParser(dropout_after_s=0.5)

    assert parser.ingest(FakeMavlinkMessage("UNKNOWN"), now_s=1.0) == "UNKNOWN"
    assert parser.metrics.unknown_messages == 1

    assert parser.ingest(object(), now_s=1.1) is None
    assert parser.metrics.malformed_messages == 1

    parser.update_ages(now_s=1.7)
    assert parser.metrics.telemetry_dropouts == 1
    parser.update_ages(now_s=2.5)
    assert parser.metrics.telemetry_dropouts == 1

    parser.ingest(FakeMavlinkMessage("HEARTBEAT"), now_s=2.6)
    parser.update_ages(now_s=3.2)
    assert parser.metrics.telemetry_dropouts == 2


def test_validate_rates_enforces_ts002_bounds():
    sitl.validate_rates(2.0, 50.0)
    sitl.validate_rates(5.0, 99.9)

    for heartbeat_hz, command_hz in [(1.9, 50.0), (2.0, 49.9), (2.0, 100.0)]:
        try:
            sitl.validate_rates(heartbeat_hz, command_hz)
        except ValueError:
            pass
        else:
            raise AssertionError("expected rate validation to fail")


def test_dry_run_writes_report(tmp_path):
    report_path = tmp_path / "dry.json"
    args = type(
        "Args",
        (),
        {
            "endpoint": "udpout:127.0.0.1:14540",
            "mode": "dry-run",
            "heartbeat_hz": 2.0,
            "command_hz": 50.0,
            "command_kind": "local_ned_velocity",
            "json_path": str(report_path),
        },
    )()

    sitl.run_dry(args)
    report = json.loads(report_path.read_text())

    assert report["endpoint"] == "udpout:127.0.0.1:14540"
    assert report["mode"] == "dry-run"
    assert report["command_hz"] == 50.0
    assert report["telemetry"]["messages_seen"] == 0
