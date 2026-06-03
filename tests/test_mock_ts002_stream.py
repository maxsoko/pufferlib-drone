import importlib.util
import struct
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mock_ts002_stream.py"
SPEC = importlib.util.spec_from_file_location("mock_ts002_stream", MODULE_PATH)
mock = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mock
SPEC.loader.exec_module(mock)


def test_scheduled_gate_size_progression():
    w0, h0 = mock.scheduled_gate_size(0.0, 20.0)
    w1, h1 = mock.scheduled_gate_size(14.0, 20.0)
    assert (w0, h0) == (120, 120)
    assert w1 >= 520 - 1
    assert h1 >= 340 - 1


def test_make_camera_packets_chunking_and_headers():
    jpeg = b"\xff\xd8" + (b"A" * 3000) + b"\xff\xd9"
    packets = mock.make_camera_packets(
        frame_id=42,
        jpeg=jpeg,
        sim_time_ns=1234567890,
        max_payload_bytes=1200,
    )
    assert len(packets) == 3

    reassembled = bytearray()
    for index, packet in enumerate(packets):
        frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack(
            mock.TS002_CAMERA_HEADER_FORMAT,
            packet[:mock.TS002_CAMERA_HEADER_SIZE],
        )
        payload = packet[mock.TS002_CAMERA_HEADER_SIZE:]
        assert frame_id == 42
        assert chunk_id == index
        assert total_chunks == 3
        assert jpeg_size == len(jpeg)
        assert payload_size == len(payload)
        assert sim_time_ns == 1234567890
        reassembled.extend(payload)
    assert bytes(reassembled) == jpeg


def test_scheduled_gate_size_for_phase_endpoints():
    assert mock.scheduled_gate_size_for_phase(0.0) == (120, 120)
    w, h = mock.scheduled_gate_size_for_phase(1.0)
    assert w >= 520 - 1
    assert h >= 340 - 1
