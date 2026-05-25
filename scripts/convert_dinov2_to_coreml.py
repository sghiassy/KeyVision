"""
Convert facebook/dinov2-base to a CoreML .mlpackage.

The exported model accepts a raw 224×224 float32 RGB image in [0, 1] and returns
a 768-d L2-normalized embedding (CLS token). ImageNet normalization, CLS extraction,
and L2 normalization are baked in so the iOS app needs no preprocessing.

Root cause of the upsample_bicubic2d error
-------------------------------------------
facebook/dinov2-base was pretrained at 518×518 (patch_size=14), so its stored
position embeddings cover (518/14)²=1369 patches. Feeding a 224×224 image produces
(224/14)²=256 patches, which triggers an interpolation via upsample_bicubic2d —
an op coremltools does not support.

Fix: before tracing, compute the interpolation once in Python (bicubic, runs fine
in PyTorch), replace position_embeddings with the result, and override
interpolate_pos_encoding to return the pre-baked constant. The traced graph then
contains no bicubic op at all.

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
IMAGE_SIZE = 224
PATCH_SIZE = 14
# Validation thresholds.
# FLOAT16 conversion introduces per-element quantization noise of ~0.001–0.002.
# We validate on what actually matters for recognition: cosine similarity between
# the PyTorch and CoreML embeddings for the same image should be effectively 1.
COSINE_SIM_THRESHOLD = 0.999   # primary gate
ELEMENT_DELTA_THRESHOLD = 5e-3  # secondary sanity check (FLOAT16-appropriate)
N_VALIDATION_IMAGES = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def bake_position_embeddings(backbone: nn.Module) -> None:
    """
    Pre-compute position embeddings for IMAGE_SIZE×IMAGE_SIZE inputs and bake them
    into the model as a constant, eliminating the upsample_bicubic2d op from the
    traced graph.

    facebook/dinov2-base stores 1369 patch position embeddings (pretrained at 518×518).
    We need 256 (224×224 / 14²). This function interpolates them to the right size
    using bicubic in Python, stores the result, and replaces interpolate_pos_encoding
    with a trivial lookup.
    """
    emb_module = backbone.embeddings
    pos_emb = emb_module.position_embeddings  # (1, 1+N_orig, D)

    h = w = IMAGE_SIZE // PATCH_SIZE          # 16
    target_n = h * w                          # 256
    current_n = pos_emb.shape[1] - 1         # 1369 for dinov2-base pretrained at 518

    if current_n == target_n:
        print(f"  Position embeddings already sized for {IMAGE_SIZE}×{IMAGE_SIZE} — no interpolation needed.")
        # Still override the method so the tracer sees no conditional branches.
        def _fixed(embeddings, height, width):
            return emb_module.position_embeddings
        emb_module.interpolate_pos_encoding = _fixed
        return

    print(f"  Interpolating position embeddings: {current_n} patches → {target_n} patches "
          f"({int(current_n**0.5)}×{int(current_n**0.5)} → {h}×{w})...")

    with torch.no_grad():
        cls_pos   = pos_emb[:, :1, :]          # (1, 1, D)
        patch_pos = pos_emb[:, 1:, :]          # (1, N_orig, D)
        dim = patch_pos.shape[-1]
        n_sqrt = int(current_n ** 0.5)
        assert n_sqrt * n_sqrt == current_n, \
            f"Position embedding count {current_n} is not a perfect square."

        # Reshape to 2-D grid, interpolate, flatten back
        grid = patch_pos.reshape(1, n_sqrt, n_sqrt, dim).permute(0, 3, 1, 2).float()
        interp = torch.nn.functional.interpolate(
            grid, size=(h, w), mode="bicubic", align_corners=False
        ).to(patch_pos.dtype)
        flat = interp.permute(0, 2, 3, 1).reshape(1, h * w, dim)
        baked = torch.cat([cls_pos, flat], dim=1)   # (1, 1+256, D)

    emb_module.position_embeddings = nn.Parameter(baked, requires_grad=False)

    # Replace the method so torch.jit.trace sees a plain tensor lookup, not a conditional
    def _fixed(embeddings, height, width):
        return emb_module.position_embeddings
    emb_module.interpolate_pos_encoding = _fixed

    print(f"  Done. New position_embeddings shape: {baked.shape}")


class DINOv2Wrapper(nn.Module):
    """
    Wraps DINOv2 so the CoreML model accepts a raw [0,1] float32 image
    and returns an L2-normalized 768-d CLS embedding.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 3, 224, 224) float32 in [0, 1]
        x = (x - self.mean) / self.std
        outputs  = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (1, 768)
        norm = cls_token.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return cls_token / norm  # (1, 768), L2-normalized


def load_pytorch_model(device: torch.device):
    print("Loading facebook/dinov2-base from HuggingFace...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    backbone  = AutoModel.from_pretrained("facebook/dinov2-base")
    backbone.eval()
    backbone.to(device)
    return processor, backbone


def pytorch_embed(wrapper: DINOv2Wrapper, image_np: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(image_np.astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,224,224)
    with torch.no_grad():
        emb = wrapper(x)
    return emb.squeeze(0).cpu().numpy()


def coreml_embed(coreml_model, image_np: np.ndarray) -> np.ndarray:
    x = image_np.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[np.newaxis]  # (1,3,224,224)
    out = coreml_model.predict({"x": x})
    return list(out.values())[0].squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the PyTorch vs CoreML embedding comparison")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip conversion and validate an existing .mlpackage")
    args = parser.parse_args()

    device = torch.device("cpu")  # coremltools tracing requires CPU

    processor, backbone = load_pytorch_model(device)

    print("Baking position embeddings for 224×224 inputs...")
    bake_position_embeddings(backbone)

    wrapper = DINOv2Wrapper(backbone).to(device)
    wrapper.eval()

    if not args.validate_only:
        print("Tracing model with torch.jit.trace...")
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, dummy)

        print("Converting to CoreML FLOAT16...")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS17,
        )
        mlmodel.save(str(OUT_PATH))
        print(f"Saved: {OUT_PATH}")

    if args.skip_validation:
        print("Skipping validation (--skip-validation).")
        return

    if not OUT_PATH.exists():
        print(f"ERROR: {OUT_PATH} not found. Run without --validate-only first.")
        sys.exit(1)

    print(f"\nValidating on {N_VALIDATION_IMAGES} random images...")
    loaded = ct.models.MLModel(str(OUT_PATH))

    min_cos_sim = 1.0
    max_delta   = 0.0
    for i in range(N_VALIDATION_IMAGES):
        img     = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        emb_pt  = pytorch_embed(wrapper, img, device)
        emb_cml = coreml_embed(loaded, img)

        # Primary: cosine similarity between the two L2-normalised embeddings
        cos_sim = float(np.dot(emb_pt, emb_cml))  # both are unit vectors
        # Secondary: max element-wise absolute difference
        delta   = float(np.max(np.abs(emb_pt - emb_cml)))

        min_cos_sim = min(min_cos_sim, cos_sim)
        max_delta   = max(max_delta, delta)
        print(f"  Image {i+1}: cos_sim = {cos_sim:.6f}  max |delta| = {delta:.6f}")

    print(f"\nMin cosine similarity : {min_cos_sim:.6f} (threshold: >= {COSINE_SIM_THRESHOLD})")
    print(f"Max element-wise delta: {max_delta:.6f}   (threshold: <  {ELEMENT_DELTA_THRESHOLD})")

    failed = False
    if min_cos_sim < COSINE_SIM_THRESHOLD:
        print(f"VALIDATION FAILED: cosine similarity {min_cos_sim:.6f} < {COSINE_SIM_THRESHOLD}")
        failed = True
    if max_delta >= ELEMENT_DELTA_THRESHOLD:
        print(f"VALIDATION FAILED: element delta {max_delta:.6f} >= {ELEMENT_DELTA_THRESHOLD}")
        failed = True

    if failed:
        sys.exit(1)
    else:
        print("Validation PASSED.")


if __name__ == "__main__":
    main()
