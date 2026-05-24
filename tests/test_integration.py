import numpy as np
import cv2
import pytest

from tests.conftest import make_key_image


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
    img_bytes = make_key_image(80)
    r = client.post("/recognize", files={"image": ("key.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 422
    assert r.json()["detail"]["error_code"] == "no_keys_enrolled"


def test_enroll_image_for_nonexistent_key_returns_404(client):
    img_bytes = make_key_image(80)
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
    img_bytes = make_key_image(gray_val=80)

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
    gray_values = {"FrontDoor": 50, "Mailbox": 120, "Garage": 170}
    key_ids = {}

    for name, gray_val in gray_values.items():
        key_id = client.post("/keys", json={"label": name}).json()["key_id"]
        key_ids[name] = key_id
        for _ in range(3):
            img_bytes = make_key_image(gray_val)
            r = client.post(
                f"/keys/{key_id}/images",
                files={"image": ("key.jpg", img_bytes, "image/jpeg")},
            )
            assert r.status_code == 200, f"Enroll failed for {name}: {r.json()}"

    for name, gray_val in gray_values.items():
        img_bytes = make_key_image(gray_val)
        r = client.post("/recognize", files={"image": ("key.jpg", img_bytes, "image/jpeg")})
        assert r.status_code == 200, r.json()
        top = r.json()["matches"][0]
        assert top["key_id"] == key_ids[name], (
            f"Expected {name} ({key_ids[name]}) but got {top['label']} ({top['key_id']})"
        )


# ── Admin / image-serving endpoints ─────────────────────────────────────────

def test_admin_page_returns_html(client):
    r = client.get("/admin")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert b"KeyVision" in r.content


def test_list_images_for_nonexistent_key_returns_404(client):
    r = client.get("/keys/nonexistent-id/images")
    assert r.status_code == 404


def test_list_images_empty_for_new_key(client):
    key_id = client.post("/keys", json={"label": "Spare"}).json()["key_id"]
    r = client.get(f"/keys/{key_id}/images")
    assert r.status_code == 200
    assert r.json() == []


def test_serve_image_for_nonexistent_key_returns_404(client):
    r = client.get("/keys/nonexistent-id/images/nonexistent-image-id")
    assert r.status_code == 404


def test_get_recognition_result_not_found_returns_404(client):
    r = client.get("/recognize/results/nonexistent-id")
    assert r.status_code == 404


@pytest.mark.slow
def test_recognize_returns_result_id_and_all_matches(client):
    key_id = client.post("/keys", json={"label": "Front Door"}).json()["key_id"]
    img_bytes = make_key_image(gray_val=80)
    client.post(f"/keys/{key_id}/images", files={"image": ("k.jpg", img_bytes, "image/jpeg")})

    r = client.post("/recognize", files={"image": ("k.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert "result_id" in data
    assert "all_matches" in data
    assert "segmentation_metrics" in data
    assert "blur_score" in data["segmentation_metrics"]

    result_r = client.get(f"/recognize/results/{data['result_id']}")
    assert result_r.status_code == 200
    assert result_r.json()["result_id"] == data["result_id"]


@pytest.mark.slow
def test_recognize_query_images_servable(client):
    key_id = client.post("/keys", json={"label": "Spare"}).json()["key_id"]
    img_bytes = make_key_image(gray_val=80)
    client.post(f"/keys/{key_id}/images", files={"image": ("k.jpg", img_bytes, "image/jpeg")})

    r = client.post("/recognize", files={"image": ("k.jpg", img_bytes, "image/jpeg")})
    result_id = r.json()["result_id"]

    for suffix in ["query", "query-crop", "query-contour"]:
        img_r = client.get(f"/recognize/results/{result_id}/{suffix}")
        assert img_r.status_code == 200
        assert img_r.headers["content-type"] == "image/jpeg"
