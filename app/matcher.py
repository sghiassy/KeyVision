import logging
from typing import Dict, List

import numpy as np

_logger = logging.getLogger(__name__)


def _confidence(similarity: float) -> str:
    if similarity >= 0.85:
        return "high"
    if similarity >= 0.65:
        return "maybe"
    return "no_match"


def match(query_embedding: np.ndarray, stored: List[Dict]) -> List[Dict]:
    """
    stored: list of {key_id, label, embedding} dicts (may have multiple per key_id)
    Returns list of {key_id, label, similarity, confidence} sorted by similarity descending.
    """
    if not stored:
        return []

    assert query_embedding.dtype == np.float32, "query_embedding must be float32"

    by_key: Dict[str, Dict] = {}
    for item in stored:
        kid = item["key_id"]
        if kid not in by_key:
            by_key[kid] = {"label": item["label"], "sims": []}
        elif by_key[kid]["label"] != item["label"]:
            _logger.warning(
                "key_id %s has conflicting labels (%r vs %r); using first",
                kid, by_key[kid]["label"], item["label"],
            )
        by_key[kid]["sims"].append(float(np.dot(query_embedding, item["embedding"])))

    results = []
    for kid, data in by_key.items():
        best = max(data["sims"])
        results.append(
            {
                "key_id": kid,
                "label": data["label"],
                "similarity": round(best, 4),
                "confidence": _confidence(best),
            }
        )
    return sorted(results, key=lambda m: m["similarity"], reverse=True)


def match_detailed(query_embedding: np.ndarray, stored: List[Dict]) -> List[Dict]:
    """
    Like match(), but also returns per-image similarity breakdown and best_image_id.
    stored: list of {key_id, image_id, label, embedding} dicts
    Returns list sorted by similarity descending, each entry:
      {key_id, label, similarity, confidence, best_image_id, per_image: [{image_id, similarity}]}
    """
    if not stored:
        return []

    assert query_embedding.dtype == np.float32, "query_embedding must be float32"

    by_key: Dict[str, Dict] = {}
    for item in stored:
        kid = item["key_id"]
        if kid not in by_key:
            by_key[kid] = {"label": item["label"], "per_image": []}
        elif by_key[kid]["label"] != item["label"]:
            _logger.warning(
                "key_id %s has conflicting labels (%r vs %r); using first",
                kid, by_key[kid]["label"], item["label"],
            )
        by_key[kid]["per_image"].append({
            "image_id": item["image_id"],
            "similarity": round(float(np.dot(query_embedding, item["embedding"])), 4),
        })

    results = []
    for kid, data in by_key.items():
        per_image = sorted(data["per_image"], key=lambda x: x["similarity"], reverse=True)
        best = per_image[0]
        results.append({
            "key_id": kid,
            "label": data["label"],
            "similarity": best["similarity"],
            "confidence": _confidence(best["similarity"]),
            "best_image_id": best["image_id"],
            "per_image": per_image,
        })
    return sorted(results, key=lambda m: m["similarity"], reverse=True)
