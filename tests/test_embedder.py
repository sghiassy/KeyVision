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
