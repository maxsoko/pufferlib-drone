#!/usr/bin/env python3
"""Evaluate cyan course-guidance detection on saved simulator frames."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from drone_course_controller import CyanGuidanceDetector, cv2


def evaluate(patterns: list[str]) -> dict:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(pattern, recursive=True))
    paths = sorted({path.resolve() for path in paths if path.is_file()})

    detector = CyanGuidanceDetector()
    samples: list[dict] = []
    detected = 0
    abs_heading_sum = 0.0
    for path in paths:
        image = None if cv2 is None else cv2.imread(str(path), cv2.IMREAD_COLOR)
        detection = detector.detect_bgr(image)
        if detection is None:
            samples.append({"path": str(path), "detected": False})
            continue
        detected += 1
        abs_heading_sum += abs(detection.heading_error_rad)
        samples.append(
            {
                "path": str(path),
                "detected": True,
                "center_x_norm": round(detection.center_x_norm, 6),
                "heading_error_rad": round(detection.heading_error_rad, 6),
                "confidence": round(detection.confidence, 6),
                "supporting_rows": detection.supporting_rows,
                "pixel_count": detection.pixel_count,
            }
        )

    total = len(paths)
    return {
        "source_count": total,
        "detected_count": detected,
        "detection_rate": round(detected / total, 6) if total else 0.0,
        "mean_abs_heading_error_rad": (
            round(abs_heading_sum / detected, 6) if detected else None
        ),
        "detector_available": detector.available,
        "metrics": detector.metrics.__dict__,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patterns",
        nargs="*",
        default=[
            "logs/sitl/agent_sprint_*_frames/first_detection.jpg",
            "logs/sitl/agent_sprint_*_frames/closest_detection.jpg",
        ],
    )
    parser.add_argument("--json-path", default="logs/sitl/course_guidance_offline.json")
    parser.add_argument("--min-detection-rate", type=float, default=0.75)
    args = parser.parse_args()

    report = evaluate(args.patterns)
    path = Path(args.json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["detector_available"]:
        return 2
    return 0 if report["detection_rate"] >= args.min_detection_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
