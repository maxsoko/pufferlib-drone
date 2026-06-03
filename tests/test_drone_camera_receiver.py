import importlib.util
import sys
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
    assert frame.calibration.extrinsics.camera_uptilt_deg == 20.0
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
