import pytest
import numpy as np
import cv2


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    import app.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(store, "RECOGNITIONS_DIR", tmp_path / "recognitions")
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
