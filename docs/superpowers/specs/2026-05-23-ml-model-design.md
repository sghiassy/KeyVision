# KeyVision ML Model Design

**Date:** 2026-05-23
**Scope:** Phase 0 Python server — image-to-key-classification pipeline
**Status:** Approved

---

## 1. Problem

Given a photo of a household key, determine which enrolled key it is (or report no confident match). This is a per-user similarity search problem, not a global classifier.

---

## 2. Approach

**OpenCV Contour Segmentation + DINOv2 Embedding + Cosine Similarity**

Pipeline: raw image → segment key region → canonical crop → DINOv2 embedding → cosine similarity against stored embeddings → ranked matches with confidence bands.

Rejected alternatives:
- **SAM + CLIP**: More robust segmentation but SAM (~2.4GB) is too heavy for a Phase 0 prototype and requires a point/box prompt, adding pipeline complexity.
- **YOLOv8 + fine-tuned EfficientNet**: Right long-term path but requires labeled training data and bounding box annotations not yet available.

---

## 3. Architecture

### 3.1 Runtime

Python FastAPI server. PyTorch + GPU (CPU fallback). No mobile code in Phase 0.

### 3.2 Enrollment Flow

```
POST /keys/{id}/images  →  Preprocess  →  OpenCV Segment  →  Crop + White BG  →  DINOv2 Embed  →  Store (SQLite + disk)
```

### 3.3 Recognition Flow

```
POST /recognize  →  Preprocess  →  OpenCV Segment  →  Crop + White BG  →  DINOv2 Embed  →  Cosine Sim vs all stored  →  Return matches + confidence
```

---

## 4. Components

### 4.1 Segmenter (`segmenter.py`)

Eight-step OpenCV pipeline. Returns a 224×224 RGB image with the key centered on a white background, long axis horizontal. Steps marked ❌ raise `SegmentationError` with the listed error code and abort immediately.

1. Decode image, convert to RGB, resize longest edge to 640px (aspect-ratio preserved)
2. Grayscale + 5×5 Gaussian blur
3. Canny edge detection (thresholds: 50, 150) + edge dilation to close key outline gaps
4. `findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` — select largest contour by area
5. ❌ **Contour size gate** — if largest contour area < 5% or > 80% of image → `segmentation_failed`
6. ❌ **Aspect ratio gate** — compute bounding rect; if ratio outside 2:1 – 8:1 → `bad_aspect_ratio`
7. Crop with 10% padding, rotate so long axis is horizontal, letterbox onto 224×224 white canvas
8. ❌ **Blur gate** — compute Laplacian variance of crop; if < 100 → `too_blurry`

### 4.2 Embedder (`embedder.py`)

DINOv2 ViT-B/14 loaded via HuggingFace (`facebook/dinov2-base`). Outputs 768-d float32 vector, L2-normalized before storage. GPU if available, CPU fallback.

Model weight: ~330MB. Loaded once at server startup, shared across requests.

### 4.3 Matcher (`matcher.py`)

Computes cosine similarity between query embedding and all stored embeddings. Aggregates per key using **max similarity** across all enrolled images for that key. Returns keys ranked by score.

**Confidence bands (starting values — tune empirically):**

| Similarity | Confidence |
|------------|------------|
| ≥ 0.85 | `high` |
| 0.65 – 0.85 | `maybe` |
| < 0.65 | `no_match` |

### 4.4 Key Store (`store.py`)

SQLite database for metadata. Embeddings stored as BLOB (numpy float32 array serialized via `numpy.tobytes()`). Images saved to `./data/images/{key_id}/`.

**Schema:**

```sql
CREATE TABLE keys (
    key_id     TEXT PRIMARY KEY,
    label      TEXT NOT NULL,
    notes      TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE key_images (
    image_id   TEXT PRIMARY KEY,
    key_id     TEXT NOT NULL REFERENCES keys(key_id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    embedding  BLOB NOT NULL,  -- float32[768] via numpy.tobytes()
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 5. API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/keys` | Create key. Body: `{"label": "Front Door", "notes": "..."}`. Returns `key_id`. |
| POST | `/keys/{id}/images` | Enroll image. Body: `multipart/form-data` with `image` file. Returns `image_id` + `segmentation_quality`. |
| POST | `/recognize` | Identify key. Body: `multipart/form-data` with `image`. Returns ranked matches. |
| GET | `/keys` | List all keys with metadata and `image_count`. |
| DELETE | `/keys/{id}` | Delete key and all associated images/embeddings. |

### Recognition Response Shape

```json
{
  "matches": [
    {"key_id": "uuid", "label": "Front Door", "similarity": 0.94, "confidence": "high"},
    {"key_id": "uuid", "label": "Mailbox",    "similarity": 0.71, "confidence": "maybe"}
  ],
  "segmentation_ok": true,
  "top_confidence": "high"
}
```

### Error Responses

| Code | Meaning | User-facing action |
|------|---------|--------------------|
| `segmentation_failed` | No contour found | Retry on a plain background |
| `bad_aspect_ratio` | Contour found but wrong shape | User may have photographed wrong object |
| `too_blurry` | Blur check failed | Retake with steadier hand / better light |
| `no_keys_enrolled` | Key store empty | Enroll keys first |

---

## 6. Testing

**Unit tests:**
- Segmenter: known-good images → valid 224×224 crop; bad images → correct error codes
- Embedder: same image twice → identical vector; deterministic output
- Matcher: identical embedding → similarity 1.0; orthogonal embeddings → near 0

**Integration test:**
- Enroll 3 distinct keys × 3 images each
- Recognize each key from a held-out 4th image
- Assert top-1 correct for all 3

---

## 7. Project Structure

```
keyvision/
├── app/
│   ├── main.py          # FastAPI app, route definitions
│   ├── segmenter.py     # OpenCV pipeline + quality gates
│   ├── embedder.py      # DINOv2 wrapper
│   ├── matcher.py       # Cosine similarity + confidence bands
│   └── store.py         # SQLite operations
├── tests/
│   ├── test_segmenter.py
│   ├── test_embedder.py
│   ├── test_matcher.py
│   └── test_integration.py
├── data/
│   └── images/          # Enrolled key images (gitignored)
├── requirements.txt
└── README.md
```

---

## 8. Open Questions / Future Work

- Confidence thresholds (0.85 / 0.65) need empirical tuning once real key images are available
- Aggregation strategy (max vs. mean similarity per key) to be validated experimentally
- Long-term: replace OpenCV contour detection with SAM or a fine-tuned key detector once background-invariance becomes a real problem
- Long-term: replace DINOv2 with a triplet-loss fine-tuned model once enough labeled key data is collected (Phase 3)
