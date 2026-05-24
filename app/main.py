import uuid
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from app import embedder, matcher, segmenter, store


@asynccontextmanager
async def _lifespan(app):
    store.init_db()
    yield


app = FastAPI(title="KeyVision", lifespan=_lifespan)


class KeyCreate(BaseModel):
    label: str
    notes: Optional[str] = None


class KeyUpdate(BaseModel):
    label: str


@app.post("/keys")
async def create_key(body: KeyCreate):
    key_id = store.create_key(body.label, body.notes)
    return {"key_id": key_id}


@app.get("/keys")
async def list_keys():
    return store.list_keys()


@app.patch("/keys/{key_id}")
async def update_key(key_id: str, body: KeyUpdate):
    if not body.label.strip():
        raise HTTPException(status_code=422, detail="Label cannot be empty")
    if not store.update_key(key_id, body.label.strip()):
        raise HTTPException(status_code=404, detail="Key not found")
    return {"key_id": key_id, "label": body.label.strip()}


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
    # Isolate the key from the background: detects the key contour via OpenCV,
    # validates shape/blur quality gates, and returns a 224×224 RGB crop on a white canvas.
    try:
        crop = segmenter.segment_key(image_bytes)
    except segmenter.SegmentationError as e:
        raise HTTPException(status_code=422, detail={"error_code": e.code, "message": str(e)})

    # Convert the crop into a 768-d L2-normalized vector using DINOv2 ViT-B/14.
    # This fingerprint is what gets compared against stored embeddings at recognition time.
    try:
        emb = embedder.embed(crop)
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"error_code": "embedding_failed", "message": str(e)})

    image_id = str(uuid.uuid4())
    img_dir = store.IMAGES_DIR / key_id
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = str(img_dir / f"{image_id}.jpg")
    store.add_key_image(key_id, img_path, emb, image_id=image_id)
    try:
        Image.fromarray(crop).save(img_path)
        # Save a contour overlay alongside the crop so the admin UI can show
        # exactly what the segmenter isolated before embedding.
        contour_img = segmenter.contour_overlay(image_bytes)
        Image.fromarray(contour_img).save(str(img_dir / f"{image_id}_contour.jpg"))
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_code": "save_failed", "message": str(e)})
    return {"image_id": image_id, "segmentation_ok": True}


@app.get("/keys/{key_id}/images")
async def list_key_images(key_id: str):
    if store.get_key(key_id) is None:
        raise HTTPException(status_code=404, detail="Key not found")
    images = store.get_key_images(key_id)
    return [{"image_id": img["image_id"], "created_at": img["created_at"]} for img in images]


@app.get("/keys/{key_id}/images/{image_id}")
async def serve_key_image(key_id: str, image_id: str):
    images = store.get_key_images(key_id)
    match = next((img for img in images if img["image_id"] == image_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Image not found")
    img_path = Path(match["image_path"])
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image file missing from disk")
    return FileResponse(img_path, media_type="image/jpeg")


@app.get("/keys/{key_id}/images/{image_id}/contour")
async def serve_contour_image(key_id: str, image_id: str):
    images = store.get_key_images(key_id)
    match = next((img for img in images if img["image_id"] == image_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Image not found")
    contour_path = Path(match["image_path"]).parent / f"{image_id}_contour.jpg"
    if not contour_path.exists():
        raise HTTPException(status_code=404, detail="Contour overlay not available for this image")
    return FileResponse(contour_path, media_type="image/jpeg")


@app.get("/admin", include_in_schema=False)
@app.get("/admin/{path:path}", include_in_schema=False)
async def admin_ui(path: str = ""):
    return FileResponse(Path(__file__).parent / "admin.html", media_type="text/html")


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
        crop, seg_metrics = segmenter.segment_key_with_metrics(image_bytes)
    except segmenter.SegmentationError as e:
        raise HTTPException(status_code=422, detail={"error_code": e.code, "message": str(e)})

    try:
        emb = embedder.embed(crop)
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"error_code": "embedding_failed", "message": str(e)})

    all_matches = matcher.match_detailed(emb, stored)
    top_confidence = all_matches[0]["confidence"] if all_matches else "no_match"

    result_id = str(uuid.uuid4())
    result_dir = store.RECOGNITIONS_DIR / result_id
    result_dir.mkdir(parents=True, exist_ok=True)

    query_path = str(result_dir / "query.jpg")
    try:
        (result_dir / "query.jpg").write_bytes(image_bytes)
        Image.fromarray(crop).save(str(result_dir / "query_crop.jpg"))
        contour_arr = segmenter.contour_overlay(image_bytes)
        Image.fromarray(contour_arr).save(str(result_dir / "query_contour.jpg"))
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_code": "save_failed", "message": str(e)})

    payload = {
        "result_id": result_id,
        "matches": all_matches[:3],
        "all_matches": all_matches,
        "segmentation_ok": True,
        "top_confidence": top_confidence,
        "segmentation_metrics": seg_metrics,
    }
    store.save_recognition_result(result_id, query_path, payload)
    return payload


@app.get("/recognize/results/{result_id}")
async def get_recognition_result(result_id: str):
    row = store.get_recognition_result(result_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return row["result_json"]


def _serve_recognition_image(result_id: str, filename: str):
    row = store.get_recognition_result(result_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Result not found")
    img_path = store.RECOGNITIONS_DIR / result_id / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image file missing")
    return FileResponse(img_path, media_type="image/jpeg")


@app.get("/recognize/results/{result_id}/query")
async def serve_recognition_query(result_id: str):
    return _serve_recognition_image(result_id, "query.jpg")


@app.get("/recognize/results/{result_id}/query-crop")
async def serve_recognition_query_crop(result_id: str):
    return _serve_recognition_image(result_id, "query_crop.jpg")


@app.get("/recognize/results/{result_id}/query-contour")
async def serve_recognition_query_contour(result_id: str):
    return _serve_recognition_image(result_id, "query_contour.jpg")
