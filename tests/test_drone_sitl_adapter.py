import importlib.util
import json
import struct
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


def test_telemetry_parser_tracks_example_extended_messages():
    parser = sitl.MavlinkTelemetryParser(dropout_after_s=1.0)

    assert parser.ingest(
        FakeMavlinkMessage(
            "LOCAL_POSITION_NED",
            x=1.0,
            y=2.0,
            z=-3.0,
            vx=0.4,
            vy=0.5,
            vz=-0.6,
        ),
        now_s=20.0,
    ) == "LOCAL_POSITION_NED"
    assert parser.state.local_position_ned_m == (1.0, 2.0, -3.0)
    assert parser.state.linear_velocity_m_s == (0.4, 0.5, -0.6)

    assert parser.ingest(
        FakeMavlinkMessage(
            "ODOMETRY",
            x=4.0,
            y=5.0,
            z=-6.0,
            q=[1.0, 0.0, 0.1, 0.2],
            vx=0.7,
            vy=0.8,
            vz=-0.9,
        ),
        now_s=20.1,
    ) == "ODOMETRY"
    assert parser.state.odometry_position_ned_m == (4.0, 5.0, -6.0)
    assert parser.state.odometry_quaternion_wxyz == (1.0, 0.0, 0.1, 0.2)

    assert parser.ingest(
        FakeMavlinkMessage("ACTUATOR_OUTPUT_STATUS", actuator=[1, 2, 3, 4]),
        now_s=20.2,
    ) == "ACTUATOR_OUTPUT_STATUS"
    assert parser.state.actuator_outputs == (1.0, 2.0, 3.0, 4.0)

    assert parser.ingest(
        FakeMavlinkMessage(
            "COLLISION",
            id=1001,
            threat_level=2,
            horizontal_minimum_delta=12.5,
        ),
        now_s=20.3,
    ) == "COLLISION"
    assert parser.state.collision_id == 1001
    assert parser.metrics.collisions == 1


def test_telemetry_parser_decodes_example_race_status_and_track_data():
    parser = sitl.MavlinkTelemetryParser(dropout_after_s=1.0)

    race_payload = struct.pack(
        "<BQqqIq",
        sitl.ENCAPSULATED_RACE_STATUS_MSG_ID,
        1234,
        1000,
        -1,
        2,
        987654321,
    )
    race_payload = race_payload + bytes(253 - len(race_payload))
    parser.ingest(FakeMavlinkMessage("ENCAPSULATED_DATA", data=race_payload), now_s=30.0)
    assert parser.metrics.encapsulated_data == 1
    assert parser.metrics.race_statuses == 1
    assert parser.state.race_status.active_gate_index == 2
    assert parser.state.race_status.last_gate_race_time == 987654321

    track_payload = struct.pack(
        "<H",
        1,
    ) + struct.pack(
        "<Hfffffffff",
        7,
        1.0,
        2.0,
        -3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.5,
        1.5,
    )
    parser.ingest(
        FakeMavlinkMessage("DATA_TRANSMISSION_HANDSHAKE", width=42, packets=1),
        now_s=30.1,
    )
    packet = struct.pack("<BH", sitl.ENCAPSULATED_TRACK_INFO_MSG_ID, 42) + track_payload
    parser.ingest(FakeMavlinkMessage("ENCAPSULATED_DATA", seqnr=0, data=packet), now_s=30.2)

    assert parser.metrics.data_handshakes == 1
    assert parser.metrics.track_infos == 1
    assert len(parser.state.track_gates) == 1
    gate = parser.state.track_gates[0]
    assert gate.gate_id == 7
    assert gate.position_ned_z == -3.0
    assert gate.width_m == 1.5


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
