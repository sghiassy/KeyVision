"""
Convert facebook/dinov2-base to a CoreML .mlpackage.

The exported model accepts a raw 224×224 float32 RGB image in [0, 1] and returns
a 768-d L2-normalized embedding (CLS token). ImageNet normalization, CLS extraction,
and L2 normalization are baked in so the iOS app needs no preprocessing.

Usage:
    source .venv/bin/activate
    python scripts/convert_dinov2_to_coreml.py

Output:
    ios/KeyVision/CoreML/DINOv2.mlpackage   (~165 MB, FLOAT16)
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path(__file__).parent.parent
OUT_PATH = REPO_ROOT / "ios" / "KeyVision" / "CoreML" / "DINOv2.mlpackage"
EPSILON = 1e-4
N_VALIDATION_IMAGES = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DINOv2Wrapper(nn.Module):
    """
    Wraps DINOv2 so the CoreML model accepts a raw [0,1] float32 image
    and returns an L2-normalized 768-d CLS embedding.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 3, 224, 224) float32 in [0, 1]
        x = (x - self.mean) / self.std
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (1, 768)
        norm = cls_token.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return cls_token / norm  # (1, 768), L2-normalized


def load_pytorch_model(device: torch.device):
    print("Loading facebook/dinov2-base from HuggingFace...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    backbone = AutoModel.from_pretrained("facebook/dinov2-base")
    backbone.eval()
    backbone.to(device)
    return processor, backbone


def pytorch_embed(backbone: nn.Module, wrapper: DINOv2Wrapper, image_np: np.ndarray, device: torch.device) -> np.ndarray:
    """Embed a (224,224,3) uint8 numpy image using PyTorch."""
    x = torch.from_numpy(image_np.astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,224,224)
    with torch.no_grad():
        emb = wrapper(x)
    return emb.squeeze(0).cpu().numpy()


def coreml_embed(coreml_model, image_np: np.ndarray) -> np.ndarray:
    """Embed a (224,224,3) uint8 numpy image using the CoreML model."""
    x = image_np.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[np.newaxis]  # (1,3,224,224)
    out = coreml_model.predict({"x": x})
    return list(out.values())[0].squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-validation", action="store_true", help="Skip embedding validation step")
    args = parser.parse_args()

    device = torch.device("cpu")  # CoreML tracing must be on CPU

    processor, backbone = load_pytorch_model(device)
    wrapper = DINOv2Wrapper(backbone).to(device)
    wrapper.eval()

    # Trace the wrapper
    print("Tracing model with torch.jit.trace...")
    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    # Convert to CoreML FLOAT16
    print("Converting to CoreML FLOAT16...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 3, 224, 224), dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save(str(OUT_PATH))
    print(f"Saved: {OUT_PATH}")

    if args.skip_validation:
        print("Skipping validation (--skip-validation).")
        return

    # Validation: compare PyTorch vs CoreML on N random images
    print(f"\nValidating on {N_VALIDATION_IMAGES} random images...")
    loaded = ct.models.MLModel(str(OUT_PATH))

    max_delta = 0.0
    for i in range(N_VALIDATION_IMAGES):
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        emb_pt = pytorch_embed(backbone, wrapper, img, device)
        emb_cml = coreml_embed(loaded, img)

        delta = float(np.max(np.abs(emb_pt - emb_cml)))
        max_delta = max(max_delta, delta)
        print(f"  Image {i+1}: max |delta| = {delta:.6f}")

    print(f"\nMax delta across all images: {max_delta:.6f} (threshold: {EPSILON})")
    if max_delta >= EPSILON:
        print(f"VALIDATION FAILED: delta {max_delta:.6f} >= {EPSILON}")
        sys.exit(1)
    else:
        print("Validation PASSED.")


if __name__ == "__main__":
    main()
