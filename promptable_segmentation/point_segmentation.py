import os
import cv2
import numpy as np
import argparse
import torch
from PIL import Image, ImageDraw, ImageOps
from typing import Tuple

from model import build_model
from model.BaseModel import BaseModel
from utils.arguments import load_opt_from_config_file
from tasks import interactive_infer_image_idino_m2m

# ✅ 引入 COCO 类别和 PASCAL Part 部位
from detectron2.data import MetadataCatalog
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

# 引入提取亮点区域的函数
from extractROI import extract_liver_points_from_brightest_region

# 初始化 metadata 并获取全部类名和部位名
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other', '').replace('-merged', '') for name in COCO_PANOPTIC_CLASSES]
print(f"[INFO] Loaded COCO classes: {all_classes}")
all_parts = [
    'arm', 'beak', 'body', 'cap', 'door', 'ear', 'eye', 'foot', 'hair', 'hand',
    'handlebar', 'head', 'headlight', 'horn', 'leg', 'license plate', 'mirror',
    'mouth', 'muzzle', 'neck', 'nose', 'paw', 'plant', 'pot', 'saddle', 'tail',
    'torso', 'wheel', 'window', 'wing'
]

def parse_args():
    parser = argparse.ArgumentParser(description="UnSAM Point Segmentation using image center")
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--config_path', type=str, required=True, help='Path to UnSAM config file')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to UnSAM checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save results')
    parser.add_argument('--text_thresh', type=str, default='0.5', help='IoU threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def draw_red_dots(image: Image.Image, points, radius=10) -> Image.Image:
    """在图像上绘制多个红点"""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
    return image

def draw_red_dot_mask(image: Image.Image, points, radius=10) -> Image.Image:
    """创建 mask 图像，黑底 + 多个红点"""
    mask = Image.new('RGB', image.size, (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
    return mask

def resize_with_pad(img: Image.Image, target: int, fill=(0, 0, 0)) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """
    等比缩放到最长边不超过 target，然后在较短边方向补边，输出正方形。
    返回: (新图, 缩放比例scale, 左上角偏移offset=(dx,dy))
    """
    w, h = img.size
    scale = min(target / float(w), target / float(h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # 兼容老 Pillow 的插值常量
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
    将 1024×1024 的 padding 图（模型输出）映射回原始分辨率。
    - padded_img: 模型输出图（带 padding）
    - orig_size: 原图尺寸 (w, h)
    - scale: resize_with_pad 返回的缩放比例
    - offset: (dx, dy) 左上角在 1024×1024 中的偏移
    """
    orig_w, orig_h = orig_size
    dx, dy = offset

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # 防溢出
    W, H = padded_img.size
    x1, y1 = max(0, min(W, dx)), max(0, min(H, dy))
    x2, y2 = max(0, min(W, dx + new_w)), max(0, min(H, dy + new_h))

    # 裁去 padding，按原尺寸放大回去
    content = padded_img.crop((x1, y1, x2, y2))
    return content.resize((orig_w, orig_h), Image.NEAREST)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    text_size = 1024

    # 1. 加载图像并记录原始尺寸；然后等比缩放 + padding
    orig = Image.open(args.image_path).convert("RGB")
    orig = ImageOps.exif_transpose(orig)  # 处理相机EXIF旋转，避免方向错乱
    orig_w, orig_h = orig.size

    image, scale, offset = resize_with_pad(orig, text_size, fill=(0, 0, 0))

    # 保存 resized 图像临时文件供提点函数使用
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    temp_path = os.path.join(args.output_dir, "resized_temp.png")
    cv2.imwrite(temp_path, cv_image)

    # 2. 提取 liver box & 随机点（基于等比缩放+padding 后的 1024×1024 图）
    box, points = extract_liver_points_from_brightest_region(
        temp_path, threshold=174, num_points=3, center_std_ratio=0.06
    )
    if points is None:
        print("❌ 提取点失败，跳过")
        return

    print(f"[INFO] 提取 liver 点坐标: {points.tolist()}")

    # 3. 绘制红点和 mask 图（在 1024×1024 的 padded 图上）
    marked_image = draw_red_dots(image, points)
    mask_image = draw_red_dot_mask(image, points)

    # 4. 构造输入字典
    image_input = {
        'image': marked_image,
        'mask': mask_image
    }

    # 5. 加载模型
    opt = load_opt_from_config_file(args.config_path)
    model = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt_path).eval().to(args.device)

    # ✅ 6. 使用 COCO 类别和 Part 类别
    text = ":".join(all_classes)
    text_part = ":".join(all_parts)
    text_thresh = args.text_thresh
    hole_scale = 100
    island_scale = 100
    semantic = True

    # 7. 推理
    print(f"[INFO] Running inference...")
    with torch.no_grad():
        with torch.autocast(device_type=args.device, dtype=torch.float16 if args.device == 'cuda' else torch.float32):
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

    # 8. 保存结果图像（映射回原始分辨率后再保存）
    for i, img in enumerate(result_sorted[:4]):
        restored = map_back_to_original(img, (orig_w, orig_h), scale, offset)
        save_path = os.path.join(args.output_dir, f"seg_result_{i}.jpg")
        restored.save(save_path)
        print(f"[INFO] Saved: {save_path}")

    # 可视化用：输入图像+红点（也回原分辨率）
    marked_restored = map_back_to_original(marked_image, (orig_w, orig_h), scale, offset)
    marked_restored.save(os.path.join(args.output_dir, "marked_input.jpg"))
    print(f"[INFO] Saved: marked_input.jpg")

if __name__ == '__main__':
    main()
