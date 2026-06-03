import base64
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "scripts"

CAPTURE_PATH = ROOT / "sitl_udp_capture.py"
CAPTURE_SPEC = importlib.util.spec_from_file_location("sitl_udp_capture", CAPTURE_PATH)
capture = importlib.util.module_from_spec(CAPTURE_SPEC)
sys.modules[CAPTURE_SPEC.name] = capture
CAPTURE_SPEC.loader.exec_module(capture)

REPLAY_PATH = ROOT / "sitl_udp_replay.py"
REPLAY_SPEC = importlib.util.spec_from_file_location("sitl_udp_replay", REPLAY_PATH)
replay = importlib.util.module_from_spec(REPLAY_SPEC)
sys.modules[REPLAY_SPEC.name] = replay
REPLAY_SPEC.loader.exec_module(replay)


def test_event_encode_parse_round_trip():
    payload = b"\x01\x02hello"
    event = capture.UdpEvent.from_payload(dt_ns=12345, stream=capture.STREAM_MAVLINK, payload=payload)
    parsed = replay.parse_event_line(json.dumps(event.to_dict()))
    assert parsed.dt_ns == 12345
    assert parsed.stream == replay.STREAM_MAVLINK
    assert parsed.payload == payload


def test_load_events_sorted_by_time(tmp_path):
    path = tmp_path / "events.jsonl"
    lines = [
        {"dt_ns": 30, "stream": "camera", "payload_b64": base64.b64encode(b"c").decode("ascii")},
        {"dt_ns": 10, "stream": "mavlink", "payload_b64": base64.b64encode(b"m").decode("ascii")},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    events = replay.load_events(str(path))
    assert [event.dt_ns for event in events] == [10, 30]
    assert [event.stream for event in events] == [replay.STREAM_MAVLINK, replay.STREAM_CAMERA]


def test_capture_metadata_hash_matches_file(tmp_path):
    output_path = tmp_path / "capture.jsonl"
    output_path.write_text('{"dt_ns":1,"payload_b64":"AA==","stream":"mavlink"}\n')
    metrics = capture.CaptureMetrics(duration_s=1.0, events_written=1)
    metadata = capture.build_capture_metadata(
        output_path=str(output_path),
        metrics=metrics,
        host="0.0.0.0",
        mavlink_port=14540,
        camera_port=5600,
        timing_profile="normal",
        generator_params={"source": "unit-test"},
    )
    assert metadata.output_sha256 == capture.sha256_file(str(output_path))
    assert metadata.capture_id


def test_materialize_impaired_schedule_drops_and_reorders():
    events = [
        replay.ReplayEvent(dt_ns=10, stream=replay.STREAM_MAVLINK, payload=b"a"),
        replay.ReplayEvent(dt_ns=20, stream=replay.STREAM_MAVLINK, payload=b"b"),
        replay.ReplayEvent(dt_ns=30, stream=replay.STREAM_CAMERA, payload=b"c"),
    ]
    schedule, dropped, swaps = replay.materialize_impaired_schedule(
        events,
        speed=1.0,
        loss_rate=0.2,
        reorder_rate=0.5,
        latency_ms=5.0,
        jitter_ms=0.0,
        seed=123,
    )
    assert dropped >= 0
    assert swaps >= 0
    assert len(schedule) + dropped == len(events)
    assert schedule == sorted(schedule, key=lambda item: (item[0], item[1]))
