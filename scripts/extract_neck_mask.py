#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM3 mask generation script (called by PachiPakuTween app)

PachiPakuTween calls this as a Python subprocess to generate
eye/mouth masks using SAM3's text prompt feature.

Usage:
    python export_sam3_masks.py \
        --image path/to/image.png \
        --checkpoint path/to/sam3.pt \
        --output-dir path/to/output \
        --prompts eye,mouth

Output:
    {output-dir}/eye_mask.png
    {output-dir}/mouth_mask.png

Requirements:
    - Python 3.10+
    - torch >= 2.1
    - sam3 package (pip install -e <sam3_repo_path>)
    - opencv-python
"""
import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def check_dependencies():
    """Check required packages and return error messages if missing."""
    errors = []

    try:
        import torch
    except ImportError:
        errors.append("torch (PyTorch) is not installed. Run: pip install torch")
        return errors  # can't check further without torch

    try:
        import cv2
    except ImportError:
        errors.append("opencv-python is not installed. Run: pip install opencv-python")

    try:
        import numpy
    except ImportError:
        errors.append("numpy is not installed. Run: pip install numpy")

    try:
        from PIL import Image
    except ImportError:
        errors.append("Pillow is not installed. Run: pip install Pillow")

    try:
        import sam3
    except ImportError:
        errors.append(
            "sam3 package is not installed.\n"
            "  1. Clone: git clone https://github.com/facebookresearch/sam3.git\n"
            "  2. Install: cd sam3 && pip install -e ."
        )

    return errors


def detect_device():
    """Detect best available device with CUDA compatibility check."""
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU", file=sys.stderr)
        return "cpu"

    # Check if current GPU architecture is supported by this PyTorch build
    try:
        # Try a small tensor operation on CUDA to verify actual compatibility
        test_tensor = torch.zeros(1, device="cuda")
        del test_tensor
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        return "cuda"
    except Exception as e:
        print(f"CUDA device found but not compatible with this PyTorch build: {e}", file=sys.stderr)
        print("Falling back to CPU. To use GPU, install a compatible PyTorch version.", file=sys.stderr)
        print("  See: https://pytorch.org/get-started/locally/", file=sys.stderr)
        return "cpu"


def load_sam3(checkpoint_path, device="cpu"):
    """Load SAM3 model and processor."""
    import sam3 as sam3_pkg
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    patch_sam3_cpu_precompute(device)

    # BPE file is inside the sam3 package directory (sam3/assets/)
    sam3_pkg_dir = Path(sam3_pkg.__file__).parent  # sam3/sam3/
    bpe_path = str(sam3_pkg_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz")

    # Fallback: check repo root assets/ if not found in package
    if not Path(bpe_path).exists():
        sam3_root = sam3_pkg_dir.parent  # sam3/
        bpe_path = str(sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
    )

    processor = Sam3Processor(model, confidence_threshold=0.3)
    return model, processor


def patch_sam3_cpu_precompute(device):
    """Avoid SAM3's CUDA-only positional precompute when running on CPU.

    Some SAM3 releases allocate precomputed positional encodings with
    device="cuda" even when build_sam3_image_model(device="cpu") is used.
    That crashes with CPU-only PyTorch before the model can be constructed.
    Disabling the precompute keeps the same forward path and lets encodings be
    created lazily on the actual input device.
    """
    if device != "cpu":
        return

    try:
        from sam3.model.position_encoding import PositionEmbeddingSine
    except Exception as exc:
        print(f"WARNING: failed to patch SAM3 CPU precompute: {exc}", file=sys.stderr)
        return

    if getattr(PositionEmbeddingSine.__init__, "_pachipakugen_cpu_patch", False):
        return

    original_init = PositionEmbeddingSine.__init__

    def cpu_safe_init(
        self,
        num_pos_feats,
        temperature=10000,
        normalize=True,
        scale=None,
        precompute_resolution=None,
    ):
        return original_init(
            self,
            num_pos_feats=num_pos_feats,
            temperature=temperature,
            normalize=normalize,
            scale=scale,
            precompute_resolution=None,
        )

    cpu_safe_init._pachipakugen_cpu_patch = True
    PositionEmbeddingSine.__init__ = cpu_safe_init
    print("Applied SAM3 CPU compatibility patch", file=sys.stderr)


def segment_by_text(processor, image, text_prompt):
    """Run segmentation with text prompt.

    Args:
        processor: Sam3Processor
        image: PIL Image (RGB)
        text_prompt: target to detect (e.g. "eye", "mouth")

    Returns:
        combined_mask: (H, W) numpy array, 0-255
    """
    import cv2
    import numpy as np
    import torch

    state = processor.set_image(image)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=text_prompt)

    w, h = image.size
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for key in ("masks", "pred_masks"):
        if key in state and state[key] is not None:
            masks = state[key]
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            for mask in masks:
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
                if mask_uint8.shape != (h, w):
                    mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, mask_uint8)
            break

    return combined_mask


def refine_mask(mask, dilate_iter=2, blur_size=7):
    """Dilate + Gaussian blur for smooth mask edges."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return mask


def main():
    parser = argparse.ArgumentParser(description="SAM3 mask generation for PachiPakuTween")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--checkpoint", required=True, help="sam3.pt checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prompts", default="eye,mouth", help="Comma-separated prompts")
    args = parser.parse_args()

    # Check dependencies first
    errors = check_dependencies()
    if errors:
        print("ERROR: Missing dependencies:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    prompts = [p.strip() for p in args.prompts.split(",")]

    device = detect_device()
    print(f"device={device}", file=sys.stderr)

    # Load SAM3
    print("SAM3 loading...", file=sys.stderr)
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    with autocast_context:
        model, processor = load_sam3(args.checkpoint, device)
        print("SAM3 loaded", file=sys.stderr)

        image = Image.open(image_path).convert("RGB")

        for prompt in prompts:
            print(f"segmenting: {prompt}", file=sys.stderr)
            mask_raw = segment_by_text(processor, image, prompt)
            mask = refine_mask(mask_raw)
            out_path = output_dir / f"{prompt}_mask.png"
            cv2.imwrite(str(out_path), mask)
            pixels = np.count_nonzero(mask)
            print(f"  {prompt}: {pixels} pixels -> {out_path}", file=sys.stderr)

    # Signal success to Rust caller via stdout
    print("OK")


if __name__ == "__main__":
    main()
