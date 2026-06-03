#!/usr/bin/env python3
"""Deterministic detector stress suite for noise/blur/compression/occlusion/scale."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from drone_gate_detector import SquareGateDetector, cv2
from mock_ts002_stream import build_gate_image, encode_jpeg


@dataclass
class ScenarioMetrics:
    scenario: str
    frames: int
    detections: int
    detection_rate: float
    average_confidence: float
    passed: bool


def load_thresholds(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("threshold config must decode to JSON object")
    return payload


def _apply_noise(image, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=18.0, size=image.shape).astype(np.float32)
    out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def _apply_blur(image) -> np.ndarray:
    return cv2.GaussianBlur(image, (11, 11), 2.5)


def _apply_compression(image) -> bytes:
    return encode_jpeg(image, quality=30)


def _apply_occlusion(image) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    x0 = int(round(0.42 * w))
    y0 = int(round(0.42 * h))
    x1 = int(round(0.63 * w))
    y1 = int(round(0.63 * h))
    out[y0:y1, x0:x1] = 0
    return out


def _scale_dims(index: int, frames: int) -> tuple[int, int]:
    t = index / max(1, frames - 1)
    width = int(round(120 + t * (520 - 120)))
    height = int(round(120 + t * (340 - 120)))
    return width, height


def run_scenario(detector: SquareGateDetector, scenario: str, *, frames: int, seed: int) -> ScenarioMetrics:
    if cv2 is None:
        raise RuntimeError("opencv is required for detector stress suite")
    rng = np.random.default_rng(seed)
    detections = 0
    confidence_sum = 0.0
    for i in range(frames):
        width, height = _scale_dims(i, frames)
        image = build_gate_image(width, height, border_px=10)
        jpeg_bytes: bytes
        if scenario == "noise":
            jpeg_bytes = encode_jpeg(_apply_noise(image, rng), quality=90)
        elif scenario == "blur":
            jpeg_bytes = encode_jpeg(_apply_blur(image), quality=90)
        elif scenario == "compression":
            jpeg_bytes = _apply_compression(image)
        elif scenario == "occlusion":
            jpeg_bytes = encode_jpeg(_apply_occlusion(image), quality=90)
        elif scenario == "scale":
            jpeg_bytes = encode_jpeg(image, quality=90)
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        detection = detector.detect_jpeg(jpeg_bytes)
        if detection is not None:
            detections += 1
            confidence_sum += float(detection.confidence)

    detection_rate = detections / max(1, frames)
    average_confidence = confidence_sum / max(1, detections)
    return ScenarioMetrics(
        scenario=scenario,
        frames=frames,
        detections=detections,
        detection_rate=detection_rate,
        average_confidence=average_confidence,
        passed=False,
    )


def evaluate(results: list[ScenarioMetrics], threshold_config: dict) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    thresholds = threshold_config.get("thresholds", {})
    for result in results:
        limits = thresholds.get(result.scenario, {})
        min_detection_rate = float(limits.get("min_detection_rate", 0.0))
        min_avg_conf = float(limits.get("min_average_confidence", 0.0))
        result.passed = result.detection_rate >= min_detection_rate and result.average_confidence >= min_avg_conf
        if not result.passed:
            blockers.append(
                f"{result.scenario}:rate={result.detection_rate:.3f} conf={result.average_confidence:.3f}"
            )
    return len(blockers) == 0, blockers


def write_json(path: str, payload: dict) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic detector stress suite")
    parser.add_argument("--threshold-config", default=os.path.join("config", "detector_stress_thresholds.json"))
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--json-path", default="")
    parser.add_argument("--detector-min-area-px", type=float, default=1200.0)
    parser.add_argument("--detector-max-aspect-error", type=float, default=0.5)
    parser.add_argument("--detector-min-fill-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("frames must be positive")
    threshold_config = load_thresholds(args.threshold_config)
    required = threshold_config.get("required_scenarios", ["noise", "blur", "compression", "occlusion", "scale"])
    if not isinstance(required, list):
        raise ValueError("required_scenarios must be a list")

    detector = SquareGateDetector(
        min_area_px=args.detector_min_area_px,
        max_aspect_error=args.detector_max_aspect_error,
        min_fill_ratio=args.detector_min_fill_ratio,
    )
    results = [
        run_scenario(detector, str(name), frames=args.frames, seed=args.seed + i)
        for i, name in enumerate(required)
    ]
    passed, blockers = evaluate(results, threshold_config)
    payload = {
        "threshold_config": os.path.abspath(args.threshold_config),
        "frames": args.frames,
        "seed": args.seed,
        "detector": {
            "min_area_px": args.detector_min_area_px,
            "max_aspect_error": args.detector_max_aspect_error,
            "min_fill_ratio": args.detector_min_fill_ratio,
        },
        "results": [asdict(result) for result in results],
        "suite_passed": passed,
        "blockers": blockers,
    }
    write_json(args.json_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("detector stress suite failed: " + ", ".join(blockers))


if __name__ == "__main__":
    main()
