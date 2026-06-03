import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODULE_PATH = SCRIPTS / "detector_stress_suite.py"
SPEC = importlib.util.spec_from_file_location("detector_stress_suite", MODULE_PATH)
stress = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stress
SPEC.loader.exec_module(stress)


@pytest.mark.skipif(stress.cv2 is None, reason="opencv unavailable")
def test_run_scenario_noise_returns_metrics():
    detector = stress.SquareGateDetector(min_area_px=800.0)
    result = stress.run_scenario(detector, "noise", frames=6, seed=123)
    assert result.scenario == "noise"
    assert result.frames == 6
    assert 0.0 <= result.detection_rate <= 1.0
    assert result.detections >= 0


def test_evaluate_respects_thresholds():
    results = [
        stress.ScenarioMetrics("noise", frames=10, detections=10, detection_rate=1.0, average_confidence=0.5, passed=False),
        stress.ScenarioMetrics("blur", frames=10, detections=5, detection_rate=0.5, average_confidence=0.05, passed=False),
    ]
    threshold_cfg = {
        "thresholds": {
            "noise": {"min_detection_rate": 0.9, "min_average_confidence": 0.2},
            "blur": {"min_detection_rate": 0.9, "min_average_confidence": 0.2},
        }
    }
    passed, blockers = stress.evaluate(results, threshold_cfg)
    assert passed is False
    assert any("blur" in blocker for blocker in blockers)
    assert results[0].passed is True
