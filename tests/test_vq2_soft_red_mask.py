from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import vq2_soft_red_mask as mask_module


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_FRAME_DIR = ROOT / "logs" / "sitl" / "n254_v3391_gate_alignment"
OFFICIAL_FIXTURE = OFFICIAL_FRAME_DIR / "frame_15613.jpg"
OFFICIAL_FIXTURE_SHA256 = (
    "a93d3238490c33a33f1878a2dc47f0bbeb6885d39a9a086b5f4a0d4cb65e2ea0"
)


def _official_image(color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    image = np.empty(
        (mask_module.OFFICIAL_HEIGHT, mask_module.OFFICIAL_WIDTH, 3),
        dtype=np.uint8,
    )
    image[:] = color
    return image


def _bgr_from_hsv(rows: list[tuple[int, int, int]]) -> np.ndarray:
    hsv = np.asarray(rows, dtype=np.uint8).reshape(1, len(rows), 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_invalid_jpeg_and_wrong_shapes_fail_closed() -> None:
    with pytest.raises(ValueError, match="could not be decoded"):
        mask_module.soft_red_mask_from_jpeg(b"not a jpeg")
    with pytest.raises(ValueError, match="empty"):
        mask_module.soft_red_mask_from_jpeg(b"")
    with pytest.raises(TypeError, match="buffer protocol"):
        mask_module.soft_red_mask_from_jpeg(object())
    with pytest.raises(ValueError, match="official 640x360"):
        mask_module.soft_red_mask_from_bgr(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        mask_module.soft_red_mask_from_bgr(np.zeros((360, 640), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        mask_module.soft_red_mask_from_bgr(
            np.zeros((360, 640, 3), dtype=np.float32)
        )


def test_output_abi_range_contiguity_and_bit_determinism() -> None:
    image = _official_image()
    image[80:190, 20:95] = (12, 18, 240)
    first = mask_module.soft_red_mask_from_bgr(image)
    second = mask_module.soft_red_mask_from_bgr(image.copy())
    assert first.shape == (4096,)
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert np.isfinite(first).all()
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
    assert np.array_equal(first, second)


def test_letterbox_preserves_intrinsics_and_has_exact_zero_padding() -> None:
    mask = mask_module.soft_red_mask_from_bgr(_official_image((0, 0, 255))).reshape(
        64, 64
    )
    assert mask_module.RESIZED_WIDTH == 64
    assert mask_module.RESIZED_HEIGHT == 36
    assert mask_module.LETTERBOX_TOP == 14
    assert mask_module.LETTERBOX_BOTTOM == 14
    assert mask_module.NORMALIZED_FOCAL_X_PX == 32.0
    assert mask_module.NORMALIZED_FOCAL_Y_PX == 32.0
    assert mask_module.NORMALIZED_PRINCIPAL_X_PX == 32.0
    assert mask_module.NORMALIZED_PRINCIPAL_Y_PX == 32.0
    assert np.count_nonzero(mask[:14]) == 0
    assert np.count_nonzero(mask[50:]) == 0
    assert np.array_equal(mask[14:50], np.ones((36, 64), dtype=np.float32))


def test_nonred_primary_colors_are_rejected() -> None:
    colors = np.asarray(
        [
            (0, 0, 0),
            (128, 128, 128),
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
            (0, 0, 255),
        ],
        dtype=np.uint8,
    ).reshape(1, 6, 3)
    response = mask_module.red_response_bgr(colors)[0]
    assert np.array_equal(response[:5], np.zeros(5, dtype=np.float32))
    assert response[5] == 1.0


def test_soft_hue_saturation_and_value_ramps_are_monotonic() -> None:
    hue = mask_module.red_response_bgr(
        _bgr_from_hsv([(0, 255, 255), (15, 255, 255), (25, 255, 255),
                       (35, 255, 255), (45, 255, 255)])
    )[0]
    assert hue[0] == pytest.approx(1.0, abs=0.02)
    assert hue[1] == pytest.approx(1.0, abs=0.02)
    assert hue[2] == pytest.approx(0.5, abs=0.04)
    assert hue[3] == pytest.approx(0.0, abs=0.02)
    assert hue[4] == 0.0
    assert np.all(np.diff(hue) <= 0.0)

    saturation = mask_module.red_response_bgr(
        _bgr_from_hsv([(0, 0, 255), (0, 40, 255), (0, 50, 255),
                       (0, 60, 255), (0, 255, 255)])
    )[0]
    assert saturation[0] == 0.0
    assert saturation[1] == pytest.approx(0.0, abs=0.03)
    assert saturation[2] == pytest.approx(0.5, abs=0.06)
    assert saturation[3] == pytest.approx(1.0, abs=0.03)
    assert saturation[4] == 1.0
    assert np.all(np.diff(saturation) >= 0.0)

    value = mask_module.red_response_bgr(
        _bgr_from_hsv([(0, 255, 0), (0, 255, 40), (0, 255, 60),
                       (0, 255, 80), (0, 255, 255)])
    )[0]
    assert value[0] == 0.0
    assert value[1] == pytest.approx(0.0, abs=0.03)
    assert value[2] == pytest.approx(0.5, abs=0.04)
    assert value[3] == pytest.approx(1.0, abs=0.03)
    assert value[4] == 1.0
    assert np.all(np.diff(value) >= 0.0)


def test_full_horizontal_field_and_all_candidates_are_preserved() -> None:
    sparse = _official_image()
    sparse[90:150, 0:30] = (10, 10, 255)
    sparse[240:300, 610:640] = (10, 10, 255)
    sparse_mask = mask_module.soft_red_mask_from_bgr(sparse).reshape(64, 64)

    crowded = sparse.copy()
    crowded[130:250, 180:460] = (10, 10, 255)
    crowded_mask = mask_module.soft_red_mask_from_bgr(crowded).reshape(64, 64)

    assert float(sparse_mask[23:29, :3].min()) > 0.9
    assert float(sparse_mask[38:44, 61:].min()) > 0.9
    assert np.array_equal(sparse_mask[23:29, :3], crowded_mask[23:29, :3])
    assert np.array_equal(sparse_mask[38:44, 61:], crowded_mask[38:44, 61:])
    assert float(crowded_mask[27:39, 18:46].mean()) > 0.9


def test_jpeg_and_bgr_paths_agree_on_synthetic_scene() -> None:
    image = _official_image((20, 25, 30))
    cv2.rectangle(image, (12, 44), (118, 172), (15, 15, 250), thickness=-1)
    cv2.rectangle(image, (276, 90), (440, 310), (30, 45, 220), thickness=17)
    cv2.rectangle(image, (582, 205), (639, 359), (5, 10, 255), thickness=-1)
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    assert ok
    direct = mask_module.soft_red_mask_from_bgr(image)
    decoded = mask_module.soft_red_mask_from_jpeg(encoded.tobytes())
    assert float(np.abs(direct - decoded).mean()) <= 0.01


def test_source_has_no_selection_or_geometry_calls() -> None:
    source_path = Path(mask_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {
        "findContours",
        "connectedComponents",
        "morphologyEx",
        "dilate",
        "erode",
        "boundingRect",
        "approxPolyDP",
        "minAreaRect",
        "HoughLines",
        "HoughLinesP",
        "solvePnP",
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint(forbidden)


def test_retained_official_renderer_frames_are_nonempty_and_finite() -> None:
    assert OFFICIAL_FIXTURE.is_file()
    assert hashlib.sha256(OFFICIAL_FIXTURE.read_bytes()).hexdigest() == (
        OFFICIAL_FIXTURE_SHA256
    )
    paths = sorted(OFFICIAL_FRAME_DIR.glob("frame_*.jpg"))
    assert len(paths) == 8
    for path in paths:
        mask = mask_module.soft_red_mask_from_jpeg(path.read_bytes())
        assert mask.shape == (4096,)
        assert np.isfinite(mask).all()
        assert float(mask.max()) > 0.5
        assert int(np.count_nonzero(mask)) > 0
