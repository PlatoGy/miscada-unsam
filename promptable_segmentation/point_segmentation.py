import os
import cv2
import numpy as np
import argparse
import torch
import random
from PIL import Image, ImageDraw, ImageOps
from typing import Tuple

from model import build_model
from model.BaseModel import BaseModel
from utils.arguments import load_opt_from_config_file
from tasks import interactive_infer_image_idino_m2m

# ✅ Import COCO categories and PASCAL parts
from detectron2.data import MetadataCatalog
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES  # noqa: F401  # Kept for environment matching

# Import function to extract bright regions
from extractROI import extract_liver_points_from_brightest_region


# =========================
# Utility Functions
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="UnSAM Point Segmentation")
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--config_path', type=str, required=True, help='Path to UnSAM config file')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to UnSAM checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save results')
    parser.add_argument('--text_thresh', type=str, default='0.5', help='Text/IoU threshold string expected by task')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA behavior as deterministic as possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Optional: stricter determinism (some operators may not support this)
        # torch.use_deterministic_algorithms(True)


def draw_red_dots(image: Image.Image, points, radius=10) -> Image.Image:
    """Draw multiple red dots on the image"""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
    return image


def draw_red_dot_mask(image: Image.Image, points, radius=10) -> Image.Image:
    """Create a mask image with black background and red dots"""
    mask = Image.new('RGB', image.size, (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
    return mask


def resize_with_pad(img: Image.Image, target: int, fill=(0, 0, 0)) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """
    Resize the image with aspect ratio preserved so that the longest side is <= target, then pad the shorter side to create a square.
    Returns: (new image, scale, offset=(dx, dy))
    """
    w, h = img.size
    scale = min(target / float(w), target / float(h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # Compatibility with older Pillow versions
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    img_resized = img.resize((new_w, new_h), resample)

    dx = (target - new_w) // 2
    dy = (target - new_h) // 2

    canvas = Image.new("RGB", (target, target), fill)
    canvas.paste(img_resized, (dx, dy))
    return canvas, scale, (dx, dy)


def map_back_to_original(padded_img: Image.Image,
                         orig_size: Tuple[int, int],
                         scale: float,
                         offset: Tuple[int, int]) -> Image.Image:
    """
    Map a 1024×1024 padded image (model output) back to its original resolution.
    - padded_img: Model output image (with padding)
    - orig_size: Original image size (w, h)
    - scale: scale returned by resize_with_pad
    - offset: (dx, dy) offset of the top-left corner in the 1024×1024 padded image
    """
    orig_w, orig_h = orig_size
    dx, dy = offset

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Prevent overflow
    W, H = padded_img.size
    x1, y1 = max(0, min(W, dx)), max(0, min(H, dy))
    x2, y2 = max(0, min(W, dx + new_w)), max(0, min(H, dy + new_h))

    # Crop padding and scale back to the original size (using nearest neighbor to preserve mask boundaries)
    content = padded_img.crop((x1, y1, x2, y2))
    return content.resize((orig_w, orig_h), Image.NEAREST)


# =========================
# Main Process
# =========================
def main():
    args = parse_args()

    # Set random seed for consistency in reproducibility
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    text_size = 1024

    # 1) Load the image, correct EXIF orientation, resize with padding
    orig = Image.open(args.image_path).convert("RGB")
    orig = ImageOps.exif_transpose(orig)
    orig_w, orig_h = orig.size

    image, scale, offset = resize_with_pad(orig, text_size, fill=(0, 0, 0))

    # Save resized image temporarily for point extraction
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    temp_path = os.path.join(args.output_dir, "resized_temp.png")
    cv2.imwrite(temp_path, cv_image)

    # 2) Extract liver-like ROI and sample three positive points around its center
    box, points = extract_liver_points_from_brightest_region(
        temp_path, threshold=174, num_points=3, center_std_ratio=0.06
    )
    if points is None:
        print("[ERROR] Failed to extract points. Abort.")
        return

    print(f"[INFO] Extracted liver-like points: {points.tolist()}")

    # 3) Draw red dots and create a mask (on the 1024×1024 padded image)
    marked_image = draw_red_dots(image, points)
    mask_image = draw_red_dot_mask(image, points)

    # 4) Prepare model input (point-based input, no explicit box)
    image_input = {
        'image': marked_image,
        'mask': mask_image
    }

    # 5) Load model
    opt = load_opt_from_config_file(args.config_path)
    model = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt_path).eval().to(args.device)

    # ✅ 6) Categories text: COCO panoptic + PASCAL parts (separated by colon)
    MetadataCatalog.get('coco_2017_train_panoptic')  # Initialize for consistent metadata
    all_classes = [name.replace('-other', '').replace('-merged', '') for name in COCO_PANOPTIC_CLASSES]
    print(f"[INFO] Loaded COCO classes: {all_classes}")
    all_parts = [
        'arm', 'beak', 'body', 'cap', 'door', 'ear', 'eye', 'foot', 'hair', 'hand',
        'handlebar', 'head', 'headlight', 'horn', 'leg', 'license plate', 'mirror',
        'mouth', 'muzzle', 'neck', 'nose', 'paw', 'plant', 'pot', 'saddle', 'tail',
        'torso', 'wheel', 'window', 'wing'
    ]

    text = ":".join(all_classes)
    text_part = ":".join(all_parts)
    text_thresh = args.text_thresh
    hole_scale = 100
    island_scale = 100
    semantic = True

    # 7) Inference: Enable mixed precision on CUDA to reduce latency and memory usage
    print("[INFO] Running inference...")
    dtype = torch.float16 if (args.device == 'cuda' and torch.cuda.is_available()) else torch.float32
    with torch.no_grad():
        if args.device == 'cuda' and torch.cuda.is_available():
            ctx = torch.autocast(device_type='cuda', dtype=dtype)
        else:
            # No need for autocast on CPU
            class DummyCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            ctx = DummyCtx()

        with ctx:
            result_all, result_sorted = interactive_infer_image_idino_m2m(
                model,
                image_input,
                text,
                text_part,
                text_thresh,
                text_size,
                hole_scale,
                island_scale,
                semantic,
                device=args.device
            )

    # 8) Save result images (mapping back to original resolution before saving), up to top 4 candidates
    for i, img in enumerate(result_sorted[:4]):
        restored = map_back_to_original(img, (orig_w, orig_h), scale, offset)
        save_path = os.path.join(args.output_dir, f"seg_result_{i}.jpg")
        restored.save(save_path)
        print(f"[INFO] Saved: {save_path}")

    # Visualize input image + marked points (back to original resolution)
    marked_restored = map_back_to_original(marked_image, (orig_w, orig_h), scale, offset)
    marked_restored_path = os.path.join(args.output_dir, "marked_input.jpg")
    marked_restored.save(marked_restored_path)
    print(f"[INFO] Saved: {marked_restored_path}")


if __name__ == '__main__':
    main()
