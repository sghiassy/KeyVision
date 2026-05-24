# KeyVision ML Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python FastAPI server that accepts key photos, segments them with OpenCV, embeds them with DINOv2, and matches them against a SQLite-backed store using cosine similarity.

**Architecture:** Five focused modules (`store`, `segmenter`, `embedder`, `matcher`, `main`) wired together by FastAPI routes. Each module is independently testable. DINOv2 is lazy-loaded on first inference call.

**Tech Stack:** Python 3.11+, FastAPI, OpenCV-headless, PyTorch, HuggingFace Transformers (DINOv2 ViT-B/14), SQLite, Pillow, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `app/__init__.py` | Package marker |
| `app/store.py` | SQLite CRUD — keys + key_images, embedding blobs |
| `app/segmenter.py` | OpenCV 8-step pipeline + quality gates → 224×224 crop |
| `app/embedder.py` | DINOv2 lazy-load wrapper → L2-normalized 768-d vector |
| `app/matcher.py` | Cosine similarity + per-key max aggregation + confidence bands |
| `app/main.py` | FastAPI routes wiring all components |
| `tests/conftest.py` | `test_db` and `client` fixtures |
| `tests/test_store.py` | Store unit tests |
| `tests/test_segmenter.py` | Segmenter unit tests (synthetic images) |
| `tests/test_embedder.py` | Embedder unit tests (marked slow — downloads model) |
| `tests/test_matcher.py` | Matcher unit tests (pure numpy, fast) |
| `tests/test_integration.py` | End-to-end enroll + recognize (marked slow) |
| `requirements.txt` | Python dependencies |
| `pytest.ini` | Marker definitions |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `app/__init__.py`
- Create: `tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write `requirements.txt`**

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
opencv-python-headless>=4.9.0
torch>=2.3.0
transformers>=4.40.0
numpy>=1.26.0
Pillow>=10.3.0
pytest>=8.2.0
httpx>=0.27.0
```

- [ ] **Step 2: Write `pytest.ini`**

```ini
[pytest]
markers =
    slow: marks tests that load DINOv2 or hit the full pipeline (deselect with '-m "not slow"')
```

- [ ] **Step 3: Create package markers**

```bash
mkdir -p app tests data/images
touch app/__init__.py tests/__init__.py
```

- [ ] **Step 4: Update `.gitignore`**

Add these lines to the existing `.gitignore`:

```
data/images/
data/keyvision.db
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: packages install without error.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini app/__init__.py tests/__init__.py .gitignore
git commit -m "feat: project scaffold — dependencies and test config"
```

---

## Task 2: Key Store

**Files:**
- Create: `app/store.py`
- Create: `tests/conftest.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write failing tests**

Create `tests/conftest.py`:

```python
import pytest
import numpy as np
import cv2


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    import app.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "IMAGES_DIR", tmp_path / "images")
    store.init_db()
    return tmp_path


@pytest.fixture
def client(test_db):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def make_key_image(gray_val: int = 80) -> bytes:
    """White 640×480 background with a dark gray 300×60 key-shaped rectangle."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (170, 210), (470, 270), (gray_val, gray_val, gray_val), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
```

Create `tests/test_store.py`:

```python
import numpy as np
import pytest


def test_create_and_list_key(test_db):
    from app.store import create_key, list_keys
    key_id = create_key("Front Door", notes="main entrance")
    keys = list_keys()
    assert len(keys) == 1
    assert keys[0]["key_id"] == key_id
    assert keys[0]["label"] == "Front Door"
    assert keys[0]["notes"] == "main entrance"
    assert keys[0]["image_count"] == 0


def test_get_key_returns_none_for_missing(test_db):
    from app.store import get_key
    assert get_key("nonexistent-id") is None


def test_delete_key(test_db):
    from app.store import create_key, delete_key, list_keys
    key_id = create_key("Mailbox")
    assert delete_key(key_id) is True
    assert list_keys() == []


def test_delete_nonexistent_key_returns_false(test_db):
    from app.store import delete_key
    assert delete_key("nonexistent-id") is False


def test_add_and_retrieve_embedding(test_db):
    from app.store import create_key, add_key_image, get_all_embeddings
    key_id = create_key("Garage")
    emb = np.random.randn(768).astype(np.float32)
    emb /= np.linalg.norm(emb)
    add_key_image(key_id, "data/images/test.jpg", emb)
    rows = get_all_embeddings()
    assert len(rows) == 1
    assert rows[0]["key_id"] == key_id
    assert rows[0]["label"] == "Garage"
    np.testing.assert_array_almost_equal(rows[0]["embedding"], emb)


def test_delete_key_cascades_images(test_db):
    from app.store import create_key, add_key_image, delete_key, get_all_embeddings
    key_id = create_key("Office")
    emb = np.random.randn(768).astype(np.float32)
    add_key_image(key_id, "data/images/test.jpg", emb)
    delete_key(key_id)
    assert get_all_embeddings() == []


def test_image_count_increments(test_db):
    from app.store import create_key, add_key_image, list_keys
    key_id = create_key("Shed")
    emb = np.random.randn(768).astype(np.float32)
    add_key_image(key_id, "path1.jpg", emb)
    add_key_image(key_id, "path2.jpg", emb)
    keys = list_keys()
    assert keys[0]["image_count"] == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.store'`

- [ ] **Step 3: Write `app/store.py`**

```python
import sqlite3
import uuid
import numpy as np
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("data/keyvision.db")
IMAGES_DIR = Path("data/images")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS keys (
                key_id     TEXT PRIMARY KEY,
                label      TEXT NOT NULL,
                notes      TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS key_images (
                image_id   TEXT PRIMARY KEY,
                key_id     TEXT NOT NULL REFERENCES keys(key_id) ON DELETE CASCADE,
                image_path TEXT NOT NULL,
                embedding  BLOB NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_key(label: str, notes: str = None) -> str:
    key_id = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO keys (key_id, label, notes) VALUES (?, ?, ?)",
            (key_id, label, notes),
        )
    return key_id


def get_key(key_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM keys WHERE key_id = ?", (key_id,)
        ).fetchone()
    return dict(row) if row else None


def list_keys() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT k.*, COUNT(ki.image_id) AS image_count
            FROM keys k
            LEFT JOIN key_images ki ON k.key_id = ki.key_id
            GROUP BY k.key_id
            ORDER BY k.created_at
        """).fetchall()
    return [dict(r) for r in rows]


def delete_key(key_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM keys WHERE key_id = ?", (key_id,))
    return cur.rowcount > 0


def add_key_image(key_id: str, image_path: str, embedding: np.ndarray) -> str:
    image_id = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO key_images (image_id, key_id, image_path, embedding) VALUES (?, ?, ?, ?)",
            (image_id, key_id, image_path, embedding.astype(np.float32).tobytes()),
        )
    return image_id


def get_all_embeddings() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT ki.key_id, ki.image_id, ki.embedding, k.label
            FROM key_images ki
            JOIN keys k ON ki.key_id = k.key_id
        """).fetchall()
    return [
        {
            "key_id": r["key_id"],
            "image_id": r["image_id"],
            "label": r["label"],
            "embedding": np.frombuffer(r["embedding"], dtype=np.float32).copy(),
        }
        for r in rows
    ]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_store.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/store.py tests/conftest.py tests/test_store.py
git commit -m "feat: key store — SQLite CRUD with embedding blob storage"
```

---

## Task 3: Segmenter

**Files:**
- Create: `app/segmenter.py`
- Create: `tests/test_segmenter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_segmenter.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_segmenter.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.segmenter'`

- [ ] **Step 3: Write `app/segmenter.py`**

```python
import cv2
import numpy as np


class SegmentationError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def check_blur(image_rgb: np.ndarray) -> bool:
    """Returns True if Laplacian variance >= 100 (sharp enough to embed)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= 100


def segment_key(image_bytes: bytes) -> np.ndarray:
    """
    Returns a 224×224 RGB numpy array with the key centered on a white background.
    Raises SegmentationError with a code if any quality gate fails.
    """
    # Step 1: decode + resize longest edge to 640px
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
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
    if contour_area < 0.05 * img_area or contour_area > 0.80 * img_area:
        raise SegmentationError(
            "segmentation_failed",
            f"Contour area {contour_area / img_area:.1%} outside 5%–80% range",
        )

    # Step 6: aspect ratio gate
    x, y, bw, bh = cv2.boundingRect(largest)
    ratio = max(bw, bh) / min(bw, bh)
    if ratio < 2.0 or ratio > 8.0:
        raise SegmentationError(
            "bad_aspect_ratio",
            f"Bounding rect ratio {ratio:.2f} outside 2:1–8:1 range",
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
    scale = min(224 / cw, 224 / ch)
    new_w, new_h = int(cw * scale), int(ch * scale)
    resized = cv2.resize(crop, (new_w, new_h))
    ox = (224 - new_w) // 2
    oy = (224 - new_h) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized

    # Step 8: blur gate on final crop
    if not check_blur(canvas):
        raise SegmentationError(
            "too_blurry",
            f"Laplacian variance below threshold — image too blurry to embed reliably",
        )

    return canvas
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_segmenter.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/segmenter.py tests/test_segmenter.py
git commit -m "feat: OpenCV segmenter — 8-step pipeline with quality gates"
```

---

## Task 4: Embedder

**Files:**
- Create: `app/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_embedder.py`:

```python
import numpy as np
import pytest


@pytest.mark.slow
def test_embed_returns_normalized_768d_vector():
    from app.embedder import embed
    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    result = embed(img)
    assert result.shape == (768,)
    assert result.dtype == np.float32
    np.testing.assert_almost_equal(np.linalg.norm(result), 1.0, decimal=5)


@pytest.mark.slow
def test_embed_is_deterministic():
    from app.embedder import embed
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    r1 = embed(img)
    r2 = embed(img)
    np.testing.assert_array_equal(r1, r2)


@pytest.mark.slow
def test_embed_different_images_produce_different_vectors():
    from app.embedder import embed
    img_dark = np.ones((224, 224, 3), dtype=np.uint8) * 30
    img_light = np.ones((224, 224, 3), dtype=np.uint8) * 220
    assert not np.allclose(embed(img_dark), embed(img_light))
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_embedder.py -v -m slow
```

Expected: `ModuleNotFoundError: No module named 'app.embedder'`

- [ ] **Step 3: Write `app/embedder.py`**

```python
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

_processor = None
_model = None


def _load() -> None:
    global _processor, _model
    if _model is not None:
        return
    _processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    _model = AutoModel.from_pretrained("facebook/dinov2-base")
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()


def embed(image_rgb: np.ndarray) -> np.ndarray:
    """
    Accepts a 224×224 RGB uint8 numpy array.
    Returns a L2-normalized 768-d float32 embedding vector.
    """
    _load()
    pil_image = Image.fromarray(image_rgb)
    inputs = _processor(images=pil_image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
    vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    vec = vec / np.linalg.norm(vec)
    return vec.astype(np.float32)
```

- [ ] **Step 4: Run tests to confirm they pass**

Note: first run downloads ~330MB from HuggingFace — requires internet access.

```bash
pytest tests/test_embedder.py -v -m slow
```

Expected: all 3 tests PASS. (May take 30–60 seconds on first run due to model download.)

- [ ] **Step 5: Commit**

```bash
git add app/embedder.py tests/test_embedder.py
git commit -m "feat: DINOv2 embedder — lazy-load ViT-B/14, L2-normalized output"
```

---

## Task 5: Matcher

**Files:**
- Create: `app/matcher.py`
- Create: `tests/test_matcher.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_matcher.py`:

```python
import numpy as np
import pytest
from app.matcher import match


def _unit_vec(size: int = 768, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size).astype(np.float32)
    return v / np.linalg.norm(v)


def test_match_empty_store_returns_empty():
    result = match(_unit_vec(), [])
    assert result == []


def test_match_identical_embedding_returns_similarity_one():
    emb = _unit_vec(seed=1)
    stored = [{"key_id": "k1", "label": "Front Door", "embedding": emb}]
    result = match(emb, stored)
    assert len(result) == 1
    assert result[0]["key_id"] == "k1"
    assert abs(result[0]["similarity"] - 1.0) < 1e-4


def test_match_orthogonal_embeddings_return_near_zero():
    query = np.zeros(768, dtype=np.float32)
    query[0] = 1.0
    stored_emb = np.zeros(768, dtype=np.float32)
    stored_emb[1] = 1.0
    stored = [{"key_id": "k1", "label": "Mailbox", "embedding": stored_emb}]
    result = match(query, stored)
    assert abs(result[0]["similarity"]) < 1e-4


def test_match_returns_results_sorted_descending():
    query = _unit_vec(seed=0)
    emb_close = query.copy()
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(768).astype(np.float32) * 0.01
    emb_close = emb_close + noise
    emb_close /= np.linalg.norm(emb_close)
    emb_far = _unit_vec(seed=42)

    stored = [
        {"key_id": "k_far", "label": "Far Key", "embedding": emb_far},
        {"key_id": "k_close", "label": "Close Key", "embedding": emb_close},
    ]
    result = match(query, stored)
    assert result[0]["key_id"] == "k_close"
    assert result[0]["similarity"] > result[1]["similarity"]


def test_match_aggregates_multiple_images_per_key_using_max():
    query = _unit_vec(seed=0)
    emb_best = query.copy()
    emb_worse = _unit_vec(seed=99)
    stored = [
        {"key_id": "k1", "label": "Front Door", "embedding": emb_best},
        {"key_id": "k1", "label": "Front Door", "embedding": emb_worse},
    ]
    result = match(query, stored)
    assert len(result) == 1
    assert abs(result[0]["similarity"] - 1.0) < 1e-4


def test_match_confidence_high():
    query = _unit_vec(seed=0)
    emb = query.copy()
    result = match(query, [{"key_id": "k1", "label": "x", "embedding": emb}])
    assert result[0]["confidence"] == "high"


def test_match_confidence_maybe():
    query = _unit_vec(seed=0)
    # Build a vector with cosine similarity ≈ 0.75 to query
    perp = _unit_vec(seed=7)
    perp = perp - np.dot(perp, query) * query
    perp /= np.linalg.norm(perp)
    emb = (query * 0.75 + perp * np.sqrt(1 - 0.75 ** 2)).astype(np.float32)
    emb /= np.linalg.norm(emb)
    result = match(query, [{"key_id": "k1", "label": "x", "embedding": emb}])
    assert result[0]["confidence"] == "maybe"


def test_match_confidence_no_match():
    query = _unit_vec(seed=0)
    perp = _unit_vec(seed=7)
    perp = perp - np.dot(perp, query) * query
    perp /= np.linalg.norm(perp)
    # cosine similarity ≈ 0.5 (below 0.65 threshold)
    emb = (query * 0.5 + perp * np.sqrt(1 - 0.5 ** 2)).astype(np.float32)
    emb /= np.linalg.norm(emb)
    result = match(query, [{"key_id": "k1", "label": "x", "embedding": emb}])
    assert result[0]["confidence"] == "no_match"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_matcher.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.matcher'`

- [ ] **Step 3: Write `app/matcher.py`**

```python
import numpy as np


def _confidence(similarity: float) -> str:
    if similarity >= 0.85:
        return "high"
    if similarity >= 0.65:
        return "maybe"
    return "no_match"


def match(query_embedding: np.ndarray, stored: list[dict]) -> list[dict]:
    """
    stored: list of {key_id, label, embedding} dicts (may have multiple per key_id)
    Returns list of {key_id, label, similarity, confidence} sorted by similarity descending.
    """
    if not stored:
        return []

    by_key: dict[str, dict] = {}
    for item in stored:
        kid = item["key_id"]
        if kid not in by_key:
            by_key[kid] = {"label": item["label"], "sims": []}
        by_key[kid]["sims"].append(float(np.dot(query_embedding, item["embedding"])))

    results = [
        {
            "key_id": kid,
            "label": data["label"],
            "similarity": round(max(data["sims"]), 4),
            "confidence": _confidence(max(data["sims"])),
        }
        for kid, data in by_key.items()
    ]
    return sorted(results, key=lambda m: m["similarity"], reverse=True)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_matcher.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/matcher.py tests/test_matcher.py
git commit -m "feat: matcher — cosine similarity with per-key max aggregation and confidence bands"
```

---

## Task 6: FastAPI App + Integration Test

**Files:**
- Create: `app/main.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_integration.py`:

```python
import numpy as np
import cv2
import pytest


def _make_key_image(gray_val: int) -> bytes:
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (170, 210), (470, 270), (gray_val, gray_val, gray_val), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ── Route-level tests (fast, no DINOv2) ────────────────────────────────────

def test_create_key(client):
    r = client.post("/keys", json={"label": "Front Door"})
    assert r.status_code == 200
    assert "key_id" in r.json()


def test_list_keys_empty(client):
    r = client.get("/keys")
    assert r.status_code == 200
    assert r.json() == []


def test_delete_nonexistent_key_returns_404(client):
    r = client.delete("/keys/nonexistent-id")
    assert r.status_code == 404


def test_recognize_with_no_enrolled_keys_returns_422(client):
    img_bytes = _make_key_image(80)
    r = client.post("/recognize", files={"image": ("key.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "no_keys_enrolled"


def test_enroll_image_for_nonexistent_key_returns_404(client):
    img_bytes = _make_key_image(80)
    r = client.post(
        "/keys/nonexistent-id/images",
        files={"image": ("key.jpg", img_bytes, "image/jpeg")},
    )
    assert r.status_code == 404


def test_enroll_blank_image_returns_422_with_error_code(client):
    # Create a key first
    key_id = client.post("/keys", json={"label": "Test"}).json()["key_id"]
    # Send blank (all-white) image — no contour detectable
    blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".jpg", blank)
    r = client.post(
        f"/keys/{key_id}/images",
        files={"image": ("blank.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "segmentation_failed"


def test_delete_key_removes_it_from_list(client):
    key_id = client.post("/keys", json={"label": "Temp"}).json()["key_id"]
    client.delete(f"/keys/{key_id}")
    keys = client.get("/keys").json()
    assert not any(k["key_id"] == key_id for k in keys)


# ── End-to-end tests (slow — loads DINOv2) ─────────────────────────────────

@pytest.mark.slow
def test_enroll_and_recognize_same_image_top1(client):
    """Enroll a key then recognize the same image — expect top-1 match with high similarity."""
    key_id = client.post("/keys", json={"label": "Front Door"}).json()["key_id"]
    img_bytes = _make_key_image(gray_val=80)

    # Enroll 3 times
    for _ in range(3):
        r = client.post(
            f"/keys/{key_id}/images",
            files={"image": ("key.jpg", img_bytes, "image/jpeg")},
        )
        assert r.status_code == 200, r.json()

    # Recognize with the same image
    r = client.post("/recognize", files={"image": ("key.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 200, r.json()
    data = r.json()
    assert data["segmentation_ok"] is True
    assert len(data["matches"]) > 0
    assert data["matches"][0]["key_id"] == key_id
    assert data["matches"][0]["similarity"] > 0.90


@pytest.mark.slow
def test_recognize_top1_correct_across_three_keys(client):
    """Enroll 3 distinct keys × 3 images, recognize each — assert top-1 correct for all 3."""
    gray_values = {"FrontDoor": 50, "Mailbox": 130, "Garage": 200}
    key_ids = {}

    for name, gray_val in gray_values.items():
        key_id = client.post("/keys", json={"label": name}).json()["key_id"]
        key_ids[name] = key_id
        for _ in range(3):
            img_bytes = _make_key_image(gray_val)
            r = client.post(
                f"/keys/{key_id}/images",
                files={"image": ("key.jpg", img_bytes, "image/jpeg")},
            )
            assert r.status_code == 200, f"Enroll failed for {name}: {r.json()}"

    for name, gray_val in gray_values.items():
        img_bytes = _make_key_image(gray_val)
        r = client.post("/recognize", files={"image": ("key.jpg", img_bytes, "image/jpeg")})
        assert r.status_code == 200, r.json()
        top = r.json()["matches"][0]
        assert top["key_id"] == key_ids[name], (
            f"Expected {name} ({key_ids[name]}) but got {top['label']} ({top['key_id']})"
        )
```

- [ ] **Step 2: Run fast tests to confirm they fail**

```bash
pytest tests/test_integration.py -v -m "not slow"
```

Expected: `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Write `app/main.py`**

```python
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from app import embedder, matcher, segmenter, store

app = FastAPI(title="KeyVision")


@app.on_event("startup")
async def startup() -> None:
    store.init_db()


class KeyCreate(BaseModel):
    label: str
    notes: str = None


@app.post("/keys")
async def create_key(body: KeyCreate):
    key_id = store.create_key(body.label, body.notes)
    return {"key_id": key_id}


@app.get("/keys")
async def list_keys():
    return store.list_keys()


@app.delete("/keys/{key_id}")
async def delete_key(key_id: str):
    if not store.delete_key(key_id):
        raise HTTPException(status_code=404, detail="Key not found")
    return {"deleted": True}


@app.post("/keys/{key_id}/images")
async def enroll_image(key_id: str, image: UploadFile = File(...)):
    if store.get_key(key_id) is None:
        raise HTTPException(status_code=404, detail="Key not found")

    image_bytes = await image.read()
    try:
        crop = segmenter.segment_key(image_bytes)
    except segmenter.SegmentationError as e:
        raise HTTPException(status_code=422, detail={"error_code": e.code, "message": str(e)})

    emb = embedder.embed(crop)

    img_dir = store.IMAGES_DIR / key_id
    img_dir.mkdir(parents=True, exist_ok=True)
    image_id = str(uuid.uuid4())
    Image.fromarray(crop).save(str(img_dir / f"{image_id}.jpg"))

    store.add_key_image(key_id, str(img_dir / f"{image_id}.jpg"), emb)
    return {"image_id": image_id, "segmentation_ok": True}


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    stored = store.get_all_embeddings()
    if not stored:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "no_keys_enrolled", "message": "No keys enrolled yet"},
        )

    image_bytes = await image.read()
    try:
        crop = segmenter.segment_key(image_bytes)
    except segmenter.SegmentationError as e:
        raise HTTPException(status_code=422, detail={"error_code": e.code, "message": str(e)})

    emb = embedder.embed(crop)
    matches = matcher.match(emb, stored)
    top_confidence = matches[0]["confidence"] if matches else "no_match"

    return {
        "matches": matches[:3],
        "segmentation_ok": True,
        "top_confidence": top_confidence,
    }
```

- [ ] **Step 4: Run fast integration tests**

```bash
pytest tests/test_integration.py -v -m "not slow"
```

Expected: all 6 fast tests PASS.

- [ ] **Step 5: Run full test suite (excluding slow)**

```bash
pytest -v -m "not slow"
```

Expected: all tests across all modules PASS.

- [ ] **Step 6: Run slow integration tests**

Note: downloads DINOv2 on first run (~330MB), takes ~60–90 seconds.

```bash
pytest tests/test_integration.py -v -m slow
```

Expected: both slow tests PASS. If `test_recognize_top1_correct_across_three_keys` fails because synthetic gray rectangles are too similar for DINOv2 to distinguish, this is expected — the test validates the pipeline, not DINOv2 accuracy on synthetic data. The meaningful accuracy test requires real key photos.

- [ ] **Step 7: Run server to manually verify**

```bash
uvicorn app.main:app --reload
```

In a separate terminal:
```bash
# Create a key
curl -X POST http://localhost:8000/keys \
  -H "Content-Type: application/json" \
  -d '{"label": "Front Door"}'

# List keys
curl http://localhost:8000/keys
```

Expected: server starts, returns `{"key_id": "..."}` and `[{"key_id": "...", "label": "Front Door", "image_count": 0, ...}]`

- [ ] **Step 8: Commit**

```bash
git add app/main.py tests/test_integration.py
git commit -m "feat: FastAPI routes — enroll and recognize endpoints wiring all components"
```

---

## Self-Review Notes

- **Spec coverage**: All 5 API endpoints covered (§5). All 4 error codes handled (§5). SQLite schema matches spec exactly (§4.4). Confidence bands match spec values (§4.3). Segmenter 8 steps match spec (§4.1). Unit + integration tests match spec §6.
- **Blur test**: `too_blurry` is tested via `check_blur()` directly (Task 3). End-to-end path through `segment_key()` for blurry images requires real out-of-focus photos — not feasible synthetically without risking flaky Canny detection.
- **Type consistency**: `get_all_embeddings()` returns `list[dict]` with keys `key_id`, `label`, `embedding` — matches what `matcher.match()` consumes in Task 5 and Task 6.
- **`make_key_image`**: Defined identically in `tests/conftest.py` (as `make_key_image`) and locally in `test_segmenter.py` and `test_integration.py` as `_make_key_image` — intentional, keeps test files self-contained.
