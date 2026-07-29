import importlib.util
import socket
import sys
import time
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "drone_camera_receiver.py"
SPEC = importlib.util.spec_from_file_location("drone_camera_receiver", MODULE_PATH)
camera = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = camera
SPEC.loader.exec_module(camera)


def make_packet(frame_id, chunk_id, total_chunks, jpeg, sim_time_ns):
    chunk_size = (len(jpeg) + total_chunks - 1) // total_chunks
    payload = jpeg[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
    header = camera.CameraChunkHeader(
        frame_id=frame_id,
        chunk_id=chunk_id,
        total_chunks=total_chunks,
        jpeg_size=len(jpeg),
        payload_size=len(payload),
        sim_time_ns=sim_time_ns,
    )
    return header.to_bytes() + payload


def ingest_frame(reassembler, frame_id, jpeg, sim_time_ns, total_chunks=2):
    frame = None
    for chunk_id in range(total_chunks):
        frame = reassembler.ingest_packet(
            make_packet(frame_id, chunk_id, total_chunks, jpeg, sim_time_ns)
        ) or frame
    return frame


def test_camera_chunk_header_round_trip():
    jpeg = b"\xff\xd8synthetic-jpeg\xff\xd9"
    packet = make_packet(
        frame_id=17,
        chunk_id=1,
        total_chunks=3,
        jpeg=jpeg,
        sim_time_ns=123456789,
    )

    header, payload = camera.CameraChunkHeader.parse(packet)

    assert camera.TS002_CAMERA_HEADER_SIZE == 24
    assert header.frame_id == 17
    assert header.chunk_id == 1
    assert header.total_chunks == 3
    assert header.jpeg_size == len(jpeg)
    assert header.payload_size == len(payload)
    assert header.sim_time_ns == 123456789


def test_reassembles_out_of_order_chunks():
    jpeg = b"\xff\xd8chunked-synthetic-jpeg-payload\xff\xd9"
    packets = [
        make_packet(4, 1, 3, jpeg, 1_000_000_000),
        make_packet(4, 2, 3, jpeg, 1_000_000_000),
        make_packet(4, 0, 3, jpeg, 1_000_000_000),
    ]
    reassembler = camera.JpegFrameReassembler()

    assert reassembler.ingest_packet(packets[0]) is None
    assert reassembler.ingest_packet(packets[1]) is None
    frame = reassembler.ingest_packet(packets[2])

    assert frame is not None
    assert frame.frame_id == 4
    assert frame.sim_time_ns == 1_000_000_000
    assert frame.jpeg == jpeg
    assert frame.received_chunks == 3
    assert frame.total_chunks == 3
    assert frame.calibration.intrinsics.fx == 320.0
    # v3379 live calibration: rendered camera is level even though TS-002
    # documents a 20-degree uptilt.
    assert frame.calibration.extrinsics.camera_uptilt_deg == 0.0
    assert reassembler.metrics.completed_frames == 1
    assert reassembler.pending_frames == 0


def test_evict_stale_incomplete_frame_counts_loss():
    jpeg = b"\xff\xd8missing-middle-and-end\xff\xd9"
    reassembler = camera.JpegFrameReassembler()

    reassembler.ingest_packet(
        make_packet(8, 0, 3, jpeg, 10_000),
        arrival_time_ns=100,
    )
    evicted = reassembler.evict_stale(now_ns=200, stale_after_ns=100)

    assert evicted == 1
    assert reassembler.metrics.chunk_loss == 2
    assert reassembler.metrics.reconstruction_failures == 1
    assert reassembler.metrics.incomplete_frames_evicted == 1
    assert reassembler.pending_frames == 0


def test_malformed_and_duplicate_packets_update_metrics():
    jpeg = b"\xff\xd8duplicate-test\xff\xd9"
    packet = make_packet(11, 0, 2, jpeg, 20_000)
    reassembler = camera.JpegFrameReassembler()

    assert reassembler.ingest_packet(b"short") is None
    assert reassembler.ingest_packet(packet) is None
    assert reassembler.ingest_packet(packet) is None

    assert reassembler.metrics.packets_seen == 3
    assert reassembler.metrics.malformed_packets == 1
    assert reassembler.metrics.duplicate_chunks == 1
    assert reassembler.metrics.chunks_accepted == 1


def test_suppresses_complete_duplicate_frame_copy():
    jpeg = b"\xff\xd8vq2-complete-frame-copy\xff\xd9"
    reassembler = camera.JpegFrameReassembler()

    first = ingest_frame(reassembler, 23, jpeg, 4_000_000_000, total_chunks=3)
    duplicate = ingest_frame(
        reassembler, 23, jpeg, 4_000_000_000, total_chunks=3
    )

    assert first is not None
    assert duplicate is None
    assert reassembler.metrics.completed_frames == 1
    assert reassembler.metrics.duplicate_chunks == 3
    assert reassembler.metrics.duplicate_completed_frame_chunks == 3
    assert reassembler.metrics.out_of_order_frames == 0
    assert reassembler.pending_frames == 0


def test_completed_identity_cache_is_bounded_and_allows_old_key_reuse():
    jpeg = b"\xff\xd8bounded-cache\xff\xd9"
    reassembler = camera.JpegFrameReassembler(recent_completed_capacity=2)

    assert ingest_frame(reassembler, 1, jpeg, 1_000, total_chunks=1) is not None
    assert ingest_frame(reassembler, 2, jpeg, 2_000, total_chunks=1) is not None
    assert ingest_frame(reassembler, 3, jpeg, 3_000, total_chunks=1) is not None
    reused = ingest_frame(reassembler, 1, jpeg, 1_000, total_chunks=1)

    assert reused is not None
    assert reassembler.metrics.completed_frames == 4


def test_frame_fps_jitter_and_stall_metrics():
    reassembler = camera.JpegFrameReassembler()
    jpeg = b"\xff\xd8fps-test\xff\xd9"

    assert ingest_frame(reassembler, 1, jpeg, 0) is not None
    assert ingest_frame(reassembler, 2, jpeg, camera.EXPECTED_FRAME_INTERVAL_NS) is not None
    assert ingest_frame(reassembler, 3, jpeg, camera.EXPECTED_FRAME_INTERVAL_NS * 3) is not None

    assert reassembler.metrics.completed_frames == 3
    assert round(reassembler.metrics.last_frame_fps, 1) == 15.0
    assert round(reassembler.metrics.average_frame_fps, 1) == 22.5
    assert reassembler.metrics.average_jitter_ns > 0
    assert reassembler.metrics.frame_stalls == 1
    assert reassembler.metrics.max_frame_gap_ns == camera.EXPECTED_FRAME_INTERVAL_NS * 2


def test_latest_frame_receiver_drains_udp_off_loop():
    receiver = camera.LatestFrameCameraReceiver(host="127.0.0.1", port=0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        port = receiver._receiver._socket.getsockname()[1]
        jpeg = b"\xff\xd8latest-frame-test\xff\xd9"
        for chunk_id in range(2):
            sender.sendto(
                make_packet(1, chunk_id, 2, jpeg, 1_000),
                ("127.0.0.1", port),
            )
        time.sleep(0.08)
        for chunk_id in range(2):
            sender.sendto(
                make_packet(2, chunk_id, 2, jpeg, 2_000),
                ("127.0.0.1", port),
            )

        deadline = time.monotonic() + 1.0
        frames = []
        while time.monotonic() < deadline:
            frames = receiver.poll_frames(max_packets=1)
            if frames and frames[0].frame_id == 2:
                break
            time.sleep(0.01)

        assert frames[0].frame_id == 2
        assert receiver.reassembler.metrics.completed_frames == 2
        assert receiver.poll_frames(max_packets=1) == []
    finally:
        sender.close()
        receiver.close()
