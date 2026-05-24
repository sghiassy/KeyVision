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


def test_get_key_images_returns_empty_for_new_key(test_db):
    from app.store import create_key, get_key_images
    key_id = create_key("Patio")
    assert get_key_images(key_id) == []


def test_get_key_images_returns_enrolled_images(test_db):
    from app.store import create_key, add_key_image, get_key_images
    key_id = create_key("Gate")
    emb = np.random.randn(768).astype(np.float32)
    img_id = add_key_image(key_id, "data/images/test.jpg", emb)
    imgs = get_key_images(key_id)
    assert len(imgs) == 1
    assert imgs[0]["image_id"] == img_id
    assert imgs[0]["key_id"] == key_id
    assert "image_path" in imgs[0]
    assert "embedding" not in imgs[0]


# ── recognition_results ──────────────────────────────────────────────────────

def test_save_and_get_recognition_result(test_db):
    from app.store import save_recognition_result, get_recognition_result
    payload = {"result_id": "r1", "matches": [], "top_confidence": "no_match"}
    save_recognition_result("r1", "/tmp/query.jpg", payload)
    row = get_recognition_result("r1")
    assert row is not None
    assert row["result_id"] == "r1"
    assert row["result_json"] == payload
    assert row["query_path"] == "/tmp/query.jpg"
    assert "created_at" in row


def test_get_recognition_result_missing_returns_none(test_db):
    from app.store import get_recognition_result
    assert get_recognition_result("nonexistent") is None
