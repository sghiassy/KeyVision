import cv2
import numpy as np

ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 8.0


class SegmentationError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def check_blur(image_rgb: np.ndarray) -> bool:
    """Returns True if Laplacian variance >= 100 (sharp enough to embed)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return bool(cv2.Laplacian(gray, cv2.CV_64F).var() >= 100)


def contour_overlay(image_bytes: bytes) -> np.ndarray:
    """
    Returns the input image (resized to 640px longest edge) with the detected
    key contour drawn as a red stroke. Call only after segment_key() has
    already validated the image — this skips quality gates.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        return np.ones((480, 640, 3), dtype=np.uint8) * 255
    img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = img.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(overlay, [largest], -1, (255, 0, 0), thickness=3)
        _, _, bw, bh = cv2.boundingRect(largest)
        if bh > bw:
            overlay = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)

    return overlay


def _segment_key_internal(image_bytes: bytes):
    """
    Core segmentation logic. Returns (canvas, metrics).
    Raises SegmentationError if any quality gate fails.
    """
    # Step 1: decode + resize longest edge to 640px
    arr = np.frombuffer(image_bytes, np.uint8)
    if arr.size == 0:
        raise SegmentationError("segmentation_failed", "Empty image bytes")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise SegmentationError("segmentation_failed", "Could not decode image bytes")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Step 2: grayscale + Gaussian blur
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny edges + dilation to close outline gaps
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Step 4: find contours, select largest
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SegmentationError("segmentation_failed", "No contours found in image")
    largest = max(contours, key=cv2.contourArea)

    # Step 5: contour size gate
    img_area = img.shape[0] * img.shape[1]
    contour_area = cv2.contourArea(largest)
    contour_area_pct = contour_area / img_area
    if contour_area_pct < 0.05 or contour_area_pct > 0.80:
        raise SegmentationError(
            "segmentation_failed",
            f"Contour area {contour_area_pct:.1%} outside 5%–80% range "
            f"(contour_area={contour_area:.0f}, img_area={img_area})",
        )

    # Step 6: aspect ratio gate
    x, y, bw, bh = cv2.boundingRect(largest)
    ratio = max(bw, bh) / min(bw, bh)
    if ratio < ASPECT_RATIO_MIN or ratio > ASPECT_RATIO_MAX:
        raise SegmentationError(
            "bad_aspect_ratio",
            f"Bounding rect ratio {ratio:.2f} outside {ASPECT_RATIO_MIN}:1–{ASPECT_RATIO_MAX}:1 range "
            f"(x={x}, y={y}, w={bw}, h={bh})",
        )

    # Step 7: crop with 10% padding + canonical orientation + letterbox onto 224×224
    pad_x = int(bw * 0.10)
    pad_y = int(bh * 0.10)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + bw + pad_x)
    y2 = min(img.shape[0], y + bh + pad_y)
    crop = img[y1:y2, x1:x2]

    ch, cw = crop.shape[:2]
    if cw < ch:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        ch, cw = crop.shape[:2]

    canvas = np.ones((224, 224, 3), dtype=np.uint8) * 255
    fit_scale = min(224 / cw, 224 / ch)
    new_w, new_h = int(cw * fit_scale), int(ch * fit_scale)
    resized = cv2.resize(crop, (new_w, new_h))
    ox = (224 - new_w) // 2
    oy = (224 - new_h) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized

    # Step 8: blur gate on final crop — capture the actual variance value
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(canvas_gray, cv2.CV_64F).var())
    if blur_score < 100:
        raise SegmentationError(
            "too_blurry",
            "Laplacian variance below threshold — image too blurry to embed reliably",
        )

    metrics = {
        "blur_score": round(blur_score, 2),
        "contour_area_pct": round(float(contour_area_pct), 4),
        "aspect_ratio": round(float(ratio), 3),
        "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
    }
    return canvas, metrics


def segment_key(image_bytes: bytes) -> np.ndarray:
    """
    Returns a 224×224 RGB numpy array with the key centered on a white background.
    Raises SegmentationError with a code if any quality gate fails.
    """
    canvas, _ = _segment_key_internal(image_bytes)
    return canvas


def segment_key_with_metrics(image_bytes: bytes):
    """
    Same as segment_key() but also returns a metrics dict:
      blur_score, contour_area_pct, aspect_ratio, bbox.
    Returns (canvas, metrics). Raises SegmentationError on failure.
    """
    return _segment_key_internal(image_bytes)
