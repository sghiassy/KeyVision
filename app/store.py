import base64
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DB_PATH = Path("data/keyvision.db")
IMAGES_DIR = Path("data/images")
RECOGNITIONS_DIR = Path("data/recognitions")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RECOGNITIONS_DIR.mkdir(parents=True, exist_ok=True)
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
            CREATE TABLE IF NOT EXISTS recognition_results (
                result_id   TEXT PRIMARY KEY,
                query_path  TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
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
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_key(label: str, notes: Optional[str] = None) -> str:
    key_id = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO keys (key_id, label, notes) VALUES (?, ?, ?)",
            (key_id, label, notes),
        )
    return key_id


def get_key(key_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM keys WHERE key_id = ?", (key_id,)
        ).fetchone()
    return dict(row) if row else None


def list_keys() -> List[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT k.*, COUNT(ki.image_id) AS image_count
            FROM keys k
            LEFT JOIN key_images ki ON k.key_id = ki.key_id
            GROUP BY k.key_id
            ORDER BY k.created_at
        """).fetchall()
    return [dict(r) for r in rows]


def update_key(key_id: str, label: str) -> bool:
    with _conn() as conn:
        rowcount = conn.execute(
            "UPDATE keys SET label = ? WHERE key_id = ?", (label, key_id)
        ).rowcount
    return rowcount > 0


def delete_key(key_id: str) -> bool:
    with _conn() as conn:
        rowcount = conn.execute("DELETE FROM keys WHERE key_id = ?", (key_id,)).rowcount
    return rowcount > 0


def add_key_image(key_id: str, image_path: str, embedding: np.ndarray, image_id: Optional[str] = None) -> str:
    if image_id is None:
        image_id = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO key_images (image_id, key_id, image_path, embedding) VALUES (?, ?, ?, ?)",
            (image_id, key_id, image_path, embedding.astype(np.float32).tobytes()),
        )
    return image_id


def get_key_images(key_id: str) -> List[Dict]:
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT image_id, key_id, image_path, created_at
            FROM key_images WHERE key_id = ? ORDER BY created_at
            """,
            (key_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_embeddings() -> List[dict]:
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


def get_all_keys_with_embeddings() -> List[Dict]:
    """Return all keys with their images and base64-encoded embeddings for iOS sync."""
    with _conn() as conn:
        key_rows = conn.execute(
            "SELECT key_id, label, notes, created_at FROM keys ORDER BY created_at"
        ).fetchall()
        image_rows = conn.execute(
            "SELECT image_id, key_id, embedding, created_at FROM key_images ORDER BY created_at"
        ).fetchall()

    images_by_key: Dict[str, List[Dict]] = {}
    for row in image_rows:
        entry = {
            "image_id": row["image_id"],
            "embedding": base64.b64encode(row["embedding"]).decode("ascii"),
            "created_at": row["created_at"],
        }
        images_by_key.setdefault(row["key_id"], []).append(entry)

    return [
        {
            "key_id": row["key_id"],
            "label": row["label"],
            "notes": row["notes"],
            "created_at": row["created_at"],
            "images": images_by_key.get(row["key_id"], []),
        }
        for row in key_rows
    ]


def save_recognition_result(result_id: str, query_path: str, payload: dict) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO recognition_results (result_id, query_path, result_json) VALUES (?, ?, ?)",
            (result_id, query_path, json.dumps(payload)),
        )


def get_recognition_result(result_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM recognition_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["result_json"] = json.loads(d["result_json"])
    return d
