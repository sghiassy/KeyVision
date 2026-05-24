import numpy as np
import pytest
from app.matcher import match, match_detailed


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


# ── match_detailed ───────────────────────────────────────────────────────────

def _stored(key_id, image_id, label, seed):
    return {"key_id": key_id, "image_id": image_id, "label": label, "embedding": _unit_vec(seed=seed)}


def test_match_detailed_empty_store_returns_empty():
    assert match_detailed(_unit_vec(), []) == []


def test_match_detailed_returns_per_image_breakdown():
    query = _unit_vec(seed=0)
    stored = [
        _stored("k1", "img-a", "Front Door", seed=1),
        _stored("k1", "img-b", "Front Door", seed=2),
    ]
    result = match_detailed(query, stored)
    assert len(result) == 1
    assert len(result[0]["per_image"]) == 2
    assert all("image_id" in pi and "similarity" in pi for pi in result[0]["per_image"])


def test_match_detailed_best_image_id_is_correct():
    query = _unit_vec(seed=0)
    emb_close = query.copy()
    emb_far   = _unit_vec(seed=99)
    stored = [
        {"key_id": "k1", "image_id": "close", "label": "x", "embedding": emb_close},
        {"key_id": "k1", "image_id": "far",   "label": "x", "embedding": emb_far},
    ]
    result = match_detailed(query, stored)
    assert result[0]["best_image_id"] == "close"


def test_match_detailed_per_image_sorted_descending():
    query = _unit_vec(seed=0)
    stored = [_stored("k1", f"img{i}", "x", seed=i) for i in range(5)]
    result = match_detailed(query, stored)
    sims = [pi["similarity"] for pi in result[0]["per_image"]]
    assert sims == sorted(sims, reverse=True)


def test_match_detailed_sorted_descending_across_keys():
    query = _unit_vec(seed=0)
    stored = [
        _stored("k1", "i1", "A", seed=5),
        _stored("k2", "i2", "B", seed=6),
        _stored("k3", "i3", "C", seed=7),
    ]
    result = match_detailed(query, stored)
    sims = [r["similarity"] for r in result]
    assert sims == sorted(sims, reverse=True)


def test_match_detailed_confidence_field_correct():
    query = _unit_vec(seed=0)
    emb = query.copy()
    stored = [{"key_id": "k1", "image_id": "i1", "label": "x", "embedding": emb}]
    result = match_detailed(query, stored)
    assert result[0]["confidence"] == "high"
