import threading

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

_processor = None
_model = None
_load_lock = threading.Lock()


def _load() -> None:
    global _processor, _model
    if _model is not None:
        return
    with _load_lock:
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
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("DINOv2 returned a zero-vector embedding — cannot normalize")
    vec = vec / norm
    return vec.astype(np.float32)
