#!/usr/bin/env python3
"""Causal full-FOV red evidence for the VQ2 recurrent Puffer actor.

The public transform intentionally stops at dense per-pixel evidence. It does
not choose a gate or derive geometric features. The official ``640x360`` frame
is scaled isotropically to ``64x36`` and vertically letterboxed so its camera
intrinsics match the native ``64x64`` visual environment exactly.
"""

from __future__ import annotations

import dataclasses

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in incomplete runtimes
    cv2 = None
    np = None


OFFICIAL_WIDTH = 640
OFFICIAL_HEIGHT = 360
OFFICIAL_FOCAL_X_PX = 320.0
OFFICIAL_FOCAL_Y_PX = 320.0
OFFICIAL_PRINCIPAL_X_PX = 320.0
OFFICIAL_PRINCIPAL_Y_PX = 180.0

MASK_WIDTH = 64
MASK_HEIGHT = 64
MASK_SIZE = MASK_WIDTH * MASK_HEIGHT
LETTERBOX_SCALE = MASK_WIDTH / OFFICIAL_WIDTH
RESIZED_WIDTH = MASK_WIDTH
RESIZED_HEIGHT = int(round(OFFICIAL_HEIGHT * LETTERBOX_SCALE))
LETTERBOX_TOP = (MASK_HEIGHT - RESIZED_HEIGHT) // 2
LETTERBOX_BOTTOM = MASK_HEIGHT - RESIZED_HEIGHT - LETTERBOX_TOP

NORMALIZED_FOCAL_X_PX = OFFICIAL_FOCAL_X_PX * LETTERBOX_SCALE
NORMALIZED_FOCAL_Y_PX = OFFICIAL_FOCAL_Y_PX * LETTERBOX_SCALE
NORMALIZED_PRINCIPAL_X_PX = OFFICIAL_PRINCIPAL_X_PX * LETTERBOX_SCALE
NORMALIZED_PRINCIPAL_Y_PX = (
    OFFICIAL_PRINCIPAL_Y_PX * LETTERBOX_SCALE + LETTERBOX_TOP
)


@dataclasses.dataclass(frozen=True)
class SoftRedMaskConfig:
    """Fixed soft-membership thresholds in OpenCV's HSV convention."""

    hue_full_distance: float = 15.0
    hue_zero_distance: float = 35.0
    saturation_zero: float = 40.0
    saturation_full: float = 60.0
    value_zero: float = 40.0
    value_full: float = 80.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.hue_full_distance < self.hue_zero_distance <= 90.0:
            raise ValueError("hue distances must satisfy 0 <= full < zero <= 90")
        if not 0.0 <= self.saturation_zero < self.saturation_full <= 255.0:
            raise ValueError("saturation thresholds must satisfy 0 <= zero < full <= 255")
        if not 0.0 <= self.value_zero < self.value_full <= 255.0:
            raise ValueError("value thresholds must satisfy 0 <= zero < full <= 255")


DEFAULT_CONFIG = SoftRedMaskConfig()


def _require_dependencies() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for VQ2 mask preprocessing")


def _rising_ramp(values, zero: float, full: float):
    return np.clip((values - zero) / (full - zero), 0.0, 1.0)


def red_response_bgr(image, *, config: SoftRedMaskConfig = DEFAULT_CONFIG):
    """Return independent soft red membership for every source pixel."""

    _require_dependencies()
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 BGR samples")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    hue_distance = np.minimum(hue, 180.0 - hue)
    hue_weight = np.clip(
        (config.hue_zero_distance - hue_distance)
        / (config.hue_zero_distance - config.hue_full_distance),
        0.0,
        1.0,
    )
    saturation_weight = _rising_ramp(
        saturation, config.saturation_zero, config.saturation_full
    )
    value_weight = _rising_ramp(value, config.value_zero, config.value_full)
    response = hue_weight * saturation_weight * value_weight
    return np.ascontiguousarray(response, dtype=np.float32)


def soft_red_mask_from_bgr(
    image, *, config: SoftRedMaskConfig = DEFAULT_CONFIG
):
    """Map one official decoded frame to the flat 4,096-value actor mask."""

    response = red_response_bgr(image, config=config)
    if response.shape != (OFFICIAL_HEIGHT, OFFICIAL_WIDTH):
        raise ValueError(
            f"expected an official {OFFICIAL_WIDTH}x{OFFICIAL_HEIGHT} frame, "
            f"received {response.shape[1]}x{response.shape[0]}"
        )

    resized = cv2.resize(
        response,
        (RESIZED_WIDTH, RESIZED_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((MASK_HEIGHT, MASK_WIDTH), dtype=np.float32)
    canvas[LETTERBOX_TOP : LETTERBOX_TOP + RESIZED_HEIGHT, :] = resized
    np.clip(canvas, 0.0, 1.0, out=canvas)
    return np.ascontiguousarray(canvas.reshape(MASK_SIZE), dtype=np.float32)


def soft_red_mask_from_jpeg(
    jpeg: bytes | bytearray | memoryview,
    *,
    config: SoftRedMaskConfig = DEFAULT_CONFIG,
):
    """Decode one JPEG and apply :func:`soft_red_mask_from_bgr` fail-closed."""

    _require_dependencies()
    try:
        encoded = np.frombuffer(jpeg, dtype=np.uint8)
    except (TypeError, ValueError) as exc:
        raise TypeError("jpeg must support the Python buffer protocol") from exc
    if encoded.size == 0:
        raise ValueError("jpeg payload is empty")
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("jpeg payload could not be decoded")
    return soft_red_mask_from_bgr(image, config=config)


__all__ = [
    "DEFAULT_CONFIG",
    "LETTERBOX_BOTTOM",
    "LETTERBOX_SCALE",
    "LETTERBOX_TOP",
    "MASK_HEIGHT",
    "MASK_SIZE",
    "MASK_WIDTH",
    "NORMALIZED_FOCAL_X_PX",
    "NORMALIZED_FOCAL_Y_PX",
    "NORMALIZED_PRINCIPAL_X_PX",
    "NORMALIZED_PRINCIPAL_Y_PX",
    "OFFICIAL_HEIGHT",
    "OFFICIAL_WIDTH",
    "RESIZED_HEIGHT",
    "RESIZED_WIDTH",
    "SoftRedMaskConfig",
    "red_response_bgr",
    "soft_red_mask_from_bgr",
    "soft_red_mask_from_jpeg",
]
