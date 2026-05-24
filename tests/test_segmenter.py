import numpy as np
import cv2
import pytest


def _make_key_image(gray_val: int = 80) -> bytes:
    """White 640×480 background with a 300×60 key-shaped rectangle."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (170, 210), (470, 270), (gray_val, gray_val, gray_val), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_blank_image() -> bytes:
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_square_blob() -> bytes:
    """Square blob — aspect ratio 1:1, should fail bad_aspect_ratio gate."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (220, 140), (420, 340), (80, 80, 80), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_segment_valid_key_returns_224x224():
    from app.segmenter import segment_key
    result = segment_key(_make_key_image())
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_segment_valid_key_has_white_background():
    from app.segmenter import segment_key
    result = segment_key(_make_key_image())
    # Corners should be white (255) — part of the white canvas
    assert result[0, 0, 0] == 255


def test_segment_blank_raises_segmentation_failed():
    from app.segmenter import segment_key, SegmentationError
    with pytest.raises(SegmentationError) as exc_info:
        segment_key(_make_blank_image())
    assert exc_info.value.code == "segmentation_failed"


def test_segment_square_blob_raises_bad_aspect_ratio():
    from app.segmenter import segment_key, SegmentationError
    with pytest.raises(SegmentationError) as exc_info:
        segment_key(_make_square_blob())
    assert exc_info.value.code == "bad_aspect_ratio"


def test_check_blur_sharp_image_returns_true():
    from app.segmenter import check_blur
    # Checkerboard has high-frequency content → high Laplacian variance
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    assert check_blur(img) is True


def test_check_blur_uniform_image_returns_false():
    from app.segmenter import check_blur
    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    assert check_blur(img) is False


# ── segment_key_with_metrics ────────────────────────────────────────────────

def test_segment_key_with_metrics_returns_correct_shape():
    from app.segmenter import segment_key_with_metrics
    crop, metrics = segment_key_with_metrics(_make_key_image())
    assert crop.shape == (224, 224, 3)
    assert crop.dtype == np.uint8


def test_segment_key_with_metrics_has_all_metric_keys():
    from app.segmenter import segment_key_with_metrics
    _, metrics = segment_key_with_metrics(_make_key_image())
    assert "blur_score" in metrics
    assert "contour_area_pct" in metrics
    assert "aspect_ratio" in metrics
    assert "bbox" in metrics
    assert set(metrics["bbox"].keys()) == {"x", "y", "w", "h"}


def test_segment_key_with_metrics_values_in_valid_range():
    from app.segmenter import segment_key_with_metrics
    _, metrics = segment_key_with_metrics(_make_key_image())
    assert metrics["blur_score"] >= 100
    assert 0.05 <= metrics["contour_area_pct"] <= 0.80
    assert 2.0 <= metrics["aspect_ratio"] <= 8.0


def test_segment_key_with_metrics_raises_on_blank_image():
    from app.segmenter import segment_key_with_metrics, SegmentationError
    with pytest.raises(SegmentationError) as exc_info:
        segment_key_with_metrics(_make_blank_image())
    assert exc_info.value.code == "segmentation_failed"


def test_segment_key_unchanged_after_refactor():
    from app.segmenter import segment_key
    result = segment_key(_make_key_image())
    assert result.shape == (224, 224, 3)
