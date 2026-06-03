import importlib.util
import struct
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sitl_stream_probe.py"
SPEC = importlib.util.spec_from_file_location("sitl_stream_probe", MODULE_PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_classify_mavlink_datagram():
    assert probe.classify_mavlink_datagram(b"") is None
    assert probe.classify_mavlink_datagram(bytes([0xFD, 0x01])) == "v2"
    assert probe.classify_mavlink_datagram(bytes([0xFE, 0x01])) == "v1"
    assert probe.classify_mavlink_datagram(bytes([0x00, 0x01])) is None


def test_ts002_camera_packet_heuristic():
    payload = b"\xff\xd8synthetic\xff\xd9"
    header = struct.pack(
        probe.TS002_CAMERA_HEADER_FORMAT,
        10,  # frame_id
        0,   # chunk_id
        1,   # total_chunks
        len(payload),
        len(payload),
        123456789,  # sim_time_ns
    )
    packet = header + payload
    assert probe.looks_like_ts002_camera_packet(packet) is True

    bad = bytearray(packet)
    bad[8:12] = (0).to_bytes(4, "little")  # jpeg_size = 0
    assert probe.looks_like_ts002_camera_packet(bytes(bad)) is False


def test_evaluate_probe_requirements():
    report = probe.ProbeReport(
        duration_s=1.0,
        host="0.0.0.0",
        mavlink_port=14540,
        camera_port=5600,
        mavlink=probe.MavlinkProbeStats(packets_seen=0),
        camera=probe.CameraProbeStats(packets_seen=0, ts002_header_packets=0),
        requirements_met=True,
        blockers=[],
    )
    met, blockers = probe.evaluate_probe_requirements(
        report,
        require_mavlink=True,
        require_camera=True,
        require_ts002_header=True,
    )
    assert met is False
    assert "no_mavlink_packets" in blockers
    assert "no_camera_packets" in blockers
    assert "no_ts002_camera_headers" in blockers
