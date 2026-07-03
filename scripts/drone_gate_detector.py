#!/usr/bin/env python3
"""First-pass square-gate detector over TS-002 JPEG frames."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - runtime fallback
    cv2 = None
    np = None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclasses.dataclass(frozen=True)
class GateDetection:
    corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    area_px: float
    bounding_width_px: float
    bounding_height_px: float
    fill_ratio: float
    confidence: float


@dataclasses.dataclass
class GateDetectorMetrics:
    frames_seen: int = 0
    detections: int = 0
    decode_failures: int = 0
    contour_rejections: int = 0
    no_quad_found: int = 0


class SquareGateDetector:
    """Detect dominant quadrilateral gate contour from a JPEG frame."""

    def __init__(
        self,
        *,
        min_area_px: float = 1200.0,
        max_aspect_error: float = 0.5,
        min_fill_ratio: float = 0.15,
        allow_grayscale_fallback: bool = True,
    ) -> None:
        if min_area_px <= 0.0:
            raise ValueError("min_area_px must be positive")
        if not 0.0 <= max_aspect_error <= 1.0:
            raise ValueError("max_aspect_error must be in [0, 1]")
        if not 0.0 < min_fill_ratio <= 1.0:
            raise ValueError("min_fill_ratio must be in (0, 1]")
        self.min_area_px = min_area_px
        self.max_aspect_error = max_aspect_error
        self.min_fill_ratio = min_fill_ratio
        self.allow_grayscale_fallback = bool(allow_grayscale_fallback)
        self.metrics = GateDetectorMetrics()
        self.available = cv2 is not None and np is not None

    def detect_jpeg(self, jpeg: bytes) -> GateDetection | None:
        self.metrics.frames_seen += 1
        if not self.available:
            self.metrics.decode_failures += 1
            return None
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.metrics.decode_failures += 1
            return None
        return self.detect_bgr(image)

    def detect_bgr(self, image) -> GateDetection | None:
        if not self.available:
            self.metrics.decode_failures += 1
            return None
        color_candidate = self._detect_red_gate(image)
        if color_candidate is not None:
            self.metrics.detections += 1
            return color_candidate
        if not self.allow_grayscale_fallback:
            self.metrics.no_quad_found += 1
            return None
        return self.detect_grayscale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    def detect_grayscale(self, image) -> GateDetection | None:
        if not self.available:
            self.metrics.decode_failures += 1
            return None
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=40, threshold2=120)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidate = None
        candidate_score = -math.inf

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                self.metrics.contour_rejections += 1
                continue
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) != 4:
                self.metrics.contour_rejections += 1
                continue
            if not cv2.isContourConvex(approx):
                self.metrics.contour_rejections += 1
                continue
            area = abs(float(cv2.contourArea(approx)))
            if area < self.min_area_px:
                self.metrics.contour_rejections += 1
                continue

            x, y, width, height = cv2.boundingRect(approx)
            if width <= 0 or height <= 0:
                self.metrics.contour_rejections += 1
                continue

            aspect_error = abs((width / height) - 1.0)
            if aspect_error > self.max_aspect_error:
                self.metrics.contour_rejections += 1
                continue

            fill_ratio = area / (width * height)
            if fill_ratio < self.min_fill_ratio:
                self.metrics.contour_rejections += 1
                continue

            squareness = 1.0 - clamp(aspect_error / max(self.max_aspect_error, 1e-6), 0.0, 1.0)
            confidence = clamp(fill_ratio * squareness, 0.0, 1.0)
            score = confidence * area
            if score <= candidate_score:
                continue

            points = approx.reshape(-1, 2).astype(float)
            corners = tuple(order_quad_corners(points))
            candidate = GateDetection(
                corners=corners,
                area_px=area,
                bounding_width_px=float(width),
                bounding_height_px=float(height),
                fill_ratio=fill_ratio,
                confidence=confidence,
            )
            candidate_score = score

        if candidate is None:
            self.metrics.no_quad_found += 1
            return None
        self.metrics.detections += 1
        return candidate

    def _detect_red_gate(self, image) -> GateDetection | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        saturated = (saturation > 60) & (value > 80)
        red_like = ((hue < 15) | (hue > 145)) & saturated
        mask = red_like.astype(np.uint8) * 255
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        min_color_area_px = max(40.0, self.min_area_px * 0.2)
        aperture_candidate = None
        aperture_score = -math.inf
        candidate = None
        candidate_score = -math.inf
        hierarchy_rows = [] if hierarchy is None else hierarchy[0]
        children_by_parent: dict[int, list[int]] = {}
        for idx, row in enumerate(hierarchy_rows):
            parent = int(row[3])
            if parent >= 0:
                children_by_parent.setdefault(parent, []).append(idx)

        for contour_idx, contour in enumerate(contours):
            if hierarchy is not None and int(hierarchy_rows[contour_idx][3]) >= 0:
                continue

            area = abs(float(cv2.contourArea(contour)))
            if area < min_color_area_px:
                self.metrics.contour_rejections += 1
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0:
                self.metrics.contour_rejections += 1
                continue

            aspect_error = abs((width / height) - 1.0)
            if aspect_error > self.max_aspect_error:
                self.metrics.contour_rejections += 1
                continue

            fill_ratio = area / (width * height)
            if fill_ratio < self.min_fill_ratio:
                self.metrics.contour_rejections += 1
                continue

            squareness = 1.0 - clamp(aspect_error / max(self.max_aspect_error, 1e-6), 0.0, 1.0)
            confidence = clamp(fill_ratio * (0.5 + 0.5 * squareness), 0.0, 1.0)
            score = (-float(y) * 10000.0) + (confidence * area)
            if score > candidate_score:
                gate_x = float(x)
                gate_y = float(y)
                gate_width = float(width)
                gate_height = float(height)
                if width > height * 1.25:
                    gate_height = float(width)
                elif height > width * 1.25:
                    gate_width = float(height)

                max_height, max_width = image.shape[:2]
                gate_width = min(gate_width, float(max_width) - gate_x)
                gate_height = min(gate_height, float(max_height) - gate_y)
                points = np.array(
                    [
                        [gate_x, gate_y],
                        [gate_x + gate_width, gate_y],
                        [gate_x + gate_width, gate_y + gate_height],
                        [gate_x, gate_y + gate_height],
                    ],
                    dtype=float,
                )
                candidate = GateDetection(
                    corners=tuple(order_quad_corners(points)),
                    area_px=area,
                    bounding_width_px=float(width),
                    bounding_height_px=float(height),
                    fill_ratio=fill_ratio,
                    confidence=confidence,
                )
                candidate_score = score

            aperture = self._best_inner_aperture(
                contours,
                children_by_parent.get(contour_idx, []),
                parent_x=x,
                parent_y=y,
                parent_width=width,
                parent_height=height,
            )
            if aperture is None:
                continue
            aperture_detection, child_score = aperture
            combined_score = (-float(y) * 10000.0) + child_score + area * 0.01
            if combined_score > aperture_score:
                aperture_candidate = aperture_detection
                aperture_score = combined_score

        return aperture_candidate if aperture_candidate is not None else candidate

    def _best_inner_aperture(
        self,
        contours,
        child_indices: list[int],
        *,
        parent_x: int,
        parent_y: int,
        parent_width: int,
        parent_height: int,
    ) -> tuple[GateDetection, float] | None:
        parent_box_area = max(1, parent_width * parent_height)
        min_child_box_area = max(36.0, parent_box_area * 0.05)
        best_detection = None
        best_score = -math.inf

        for child_idx in child_indices:
            child = contours[child_idx]
            area = abs(float(cv2.contourArea(child)))
            x, y, width, height = cv2.boundingRect(child)
            if width <= 0 or height <= 0:
                self.metrics.contour_rejections += 1
                continue
            box_area = width * height
            if box_area < min_child_box_area:
                continue
            if x <= parent_x or y <= parent_y:
                continue
            if x + width >= parent_x + parent_width or y + height >= parent_y + parent_height:
                continue

            aspect_error = abs((width / height) - 1.0)
            max_aperture_aspect_error = min(0.9, self.max_aspect_error + 0.25)
            if aspect_error > max_aperture_aspect_error:
                self.metrics.contour_rejections += 1
                continue

            fill_ratio = area / box_area
            if fill_ratio < max(0.25, self.min_fill_ratio):
                self.metrics.contour_rejections += 1
                continue

            squareness = 1.0 - clamp(aspect_error / max(max_aperture_aspect_error, 1e-6), 0.0, 1.0)
            confidence = clamp(fill_ratio * (0.6 + 0.4 * squareness), 0.0, 1.0)
            score = confidence * box_area
            if score <= best_score:
                continue

            points = np.array(
                [
                    [float(x), float(y)],
                    [float(x + width), float(y)],
                    [float(x + width), float(y + height)],
                    [float(x), float(y + height)],
                ],
                dtype=float,
            )
            best_detection = GateDetection(
                corners=tuple(order_quad_corners(points)),
                area_px=area,
                bounding_width_px=float(width),
                bounding_height_px=float(height),
                fill_ratio=fill_ratio,
                confidence=confidence,
            )
            best_score = score

        if best_detection is None:
            return None
        return best_detection, best_score


def order_quad_corners(points) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    center = points.mean(axis=0)
    angles = [math.atan2(point[1] - center[1], point[0] - center[0]) for point in points]
    ordered = [tuple(float(v) for v in point) for _angle, point in sorted(zip(angles, points), key=lambda x: x[0])]
    top_left_index = min(range(4), key=lambda i: ordered[i][0] + ordered[i][1])
    return tuple(ordered[(top_left_index + i) % 4] for i in range(4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect TS-002 square gate corners from JPEG")
    parser.add_argument("--jpeg-path", required=True)
    parser.add_argument("--json-path", default="")
    args = parser.parse_args()

    detector = SquareGateDetector()
    with open(args.jpeg_path, "rb") as f:
        detection = detector.detect_jpeg(f.read())

    report = {
        "available": detector.available,
        "detection": dataclasses.asdict(detection) if detection is not None else None,
        "metrics": dataclasses.asdict(detector.metrics),
    }
    if args.json_path:
        directory = os.path.dirname(args.json_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
