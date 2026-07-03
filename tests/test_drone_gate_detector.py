import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "drone_gate_detector.py"
SPEC = importlib.util.spec_from_file_location("drone_gate_detector", MODULE_PATH)
detector_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = detector_module
SPEC.loader.exec_module(detector_module)


cv2 = getattr(detector_module, "cv2", None)
np = getattr(detector_module, "np", None)


def test_detect_jpeg_returns_none_for_invalid_bytes():
    detector = detector_module.SquareGateDetector()
    assert detector.detect_jpeg(b"not-a-jpeg") is None
    assert detector.metrics.frames_seen == 1
    assert detector.metrics.decode_failures == 1


@pytest.mark.skipif(cv2 is None or np is None, reason="opencv/numpy unavailable")
def test_detects_synthetic_square_gate():
    image = np.zeros((360, 640), dtype=np.uint8)
    cv2.rectangle(image, (210, 80), (430, 300), color=255, thickness=14)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok

    detector = detector_module.SquareGateDetector(min_area_px=5000.0)
    detection = detector.detect_jpeg(jpeg.tobytes())

    assert detection is not None
    assert detection.confidence > 0.0
    assert detection.area_px > 5000.0
    assert len(detection.corners) == 4
    assert detector.metrics.detections == 1


@pytest.mark.skipif(cv2 is None or np is None, reason="opencv/numpy unavailable")
def test_color_required_rejects_white_square_gate():
    image = np.zeros((360, 640), dtype=np.uint8)
    cv2.rectangle(image, (210, 80), (430, 300), color=255, thickness=14)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok

    detector = detector_module.SquareGateDetector(min_area_px=5000.0, allow_grayscale_fallback=False)
    detection = detector.detect_jpeg(jpeg.tobytes())

    assert detection is None
    assert detector.metrics.no_quad_found == 1


@pytest.mark.skipif(cv2 is None or np is None, reason="opencv/numpy unavailable")
def test_colored_gate_beats_white_grid_square():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (250, 5), (325, 55), color=(255, 255, 255), thickness=2)
    cv2.rectangle(image, (210, 170), (285, 245), color=(20, 20, 255), thickness=-1)
    cv2.rectangle(image, (228, 188), (267, 227), color=(0, 0, 0), thickness=-1)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok

    detector = detector_module.SquareGateDetector(
        min_area_px=300.0,
        max_aspect_error=0.8,
        min_fill_ratio=0.1,
    )
    detection = detector.detect_jpeg(jpeg.tobytes())

    assert detection is not None
    xs = [point[0] for point in detection.corners]
    ys = [point[1] for point in detection.corners]
    assert 200 <= sum(xs) / len(xs) <= 295
    assert 160 <= sum(ys) / len(ys) <= 255
    assert detector.metrics.detections == 1


@pytest.mark.skipif(cv2 is None or np is None, reason="opencv/numpy unavailable")
def test_colored_detector_returns_inner_aperture_when_visible():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 50), (320, 270), color=(20, 20, 255), thickness=-1)
    cv2.rectangle(image, (155, 105), (265, 215), color=(0, 0, 0), thickness=-1)
    cv2.rectangle(image, (112, 72), (128, 88), color=(0, 0, 0), thickness=-1)
    cv2.rectangle(image, (292, 232), (306, 246), color=(0, 0, 0), thickness=-1)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok

    detector = detector_module.SquareGateDetector(
        min_area_px=500.0,
        max_aspect_error=0.8,
        min_fill_ratio=0.1,
        allow_grayscale_fallback=False,
    )
    detection = detector.detect_jpeg(jpeg.tobytes())

    assert detection is not None
    xs = [point[0] for point in detection.corners]
    ys = [point[1] for point in detection.corners]
    assert min(xs) == pytest.approx(155.0, abs=6.0)
    assert max(xs) == pytest.approx(266.0, abs=6.0)
    assert min(ys) == pytest.approx(105.0, abs=6.0)
    assert max(ys) == pytest.approx(216.0, abs=6.0)
    assert detection.bounding_width_px < 140.0
    assert detection.bounding_height_px < 140.0


@pytest.mark.skipif(cv2 is None or np is None, reason="opencv/numpy unavailable")
def test_colored_detector_prefers_front_gate_over_square_rear_gate():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (305, 170), (346, 195), color=(20, 20, 255), thickness=-1)
    cv2.rectangle(image, (326, 209), (349, 231), color=(20, 20, 255), thickness=-1)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok

    detector = detector_module.SquareGateDetector(
        min_area_px=300.0,
        max_aspect_error=0.8,
        min_fill_ratio=0.1,
    )
    detection = detector.detect_jpeg(jpeg.tobytes())

    assert detection is not None
    ys = [point[1] for point in detection.corners]
    assert sum(ys) / len(ys) < 205.0
