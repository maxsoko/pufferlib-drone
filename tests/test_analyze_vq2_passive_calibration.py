import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_vq2_passive_calibration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_vq2_passive_calibration", MODULE_PATH
)
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def test_weighted_median_uses_weights_and_rejects_empty_input():
    assert calibration.weighted_median([(0.0, 1.0), (10.0, 5.0)]) == 10.0
    with pytest.raises(ValueError):
        calibration.weighted_median([])


@pytest.mark.skipif(calibration.cv2 is None, reason="OpenCV unavailable")
def test_hangar_vanishing_point_recovers_synthetic_horizon():
    cv2 = calibration.cv2
    np = calibration.np
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    target = (320, 190)
    for endpoint in (
        (0, 20),
        (0, 70),
        (0, 120),
        (100, 0),
        (170, 0),
        (639, 15),
        (639, 65),
        (639, 115),
        (540, 0),
        (470, 0),
    ):
        cv2.line(image, endpoint, target, (210, 210, 210), 2)

    result = calibration.estimate_horizontal_vanishing_point(image)

    assert abs(result["vanishing_point_px"][0] - target[0]) < 5.0
    assert abs(result["vanishing_point_px"][1] - target[1]) < 5.0
    assert abs(result["camera_uptilt_deg"] - 1.7899) < 1.0
