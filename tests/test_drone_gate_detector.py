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
