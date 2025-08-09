import os
import cv2
import numpy as np
import argparse
import torch
from PIL import Image, ImageDraw

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



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    text_size = 1024

    # 1. 加载图像并 resize
    image = Image.open(args.image_path).convert("RGB")
    image = image.resize((text_size, text_size))

    # 保存 resized 图像临时文件供提点函数使用
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    temp_path = os.path.join(args.output_dir, "resized_temp.png")
    cv2.imwrite(temp_path, cv_image)

    # 2. 提取 liver box & 随机点
    box, points = extract_liver_points_from_brightest_region(temp_path, threshold=174, num_points=3, center_std_ratio=0.05)
    if points is None:
        print("❌ 提取点失败，跳过")
        return


    print(f"[INFO] 提取 liver 点坐标: {points.tolist()}")

    # 3. 绘制红点和 mask 图
    marked_image = draw_red_dots(image, points)
    mask_image = draw_red_dot_mask(image, points)


    # 3. 构造输入字典
    image_input = {
        'image': marked_image,
        'mask': mask_image
    }

    # 4. 加载模型
    opt = load_opt_from_config_file(args.config_path)
    model = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt_path).eval().to(args.device)

    # ✅ 5. 使用 COCO 类别和 Part 类别
    text = ":".join(all_classes)
    text_part = ":".join(all_parts)
    text_thresh = args.text_thresh
    hole_scale = 100
    island_scale = 100
    semantic = True

    # 6. 推理
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

    # 7. 保存结果图像
    for i, img in enumerate(result_sorted[:4]):
        save_path = os.path.join(args.output_dir, f"seg_result_{i}.jpg")
        img.save(save_path)
        print(f"[INFO] Saved: {save_path}")

    # 可视化用：输入图像+红点
    marked_image.save(os.path.join(args.output_dir, "marked_input.jpg"))
    print(f"[INFO] Saved: marked_input.jpg")


if __name__ == '__main__':
    main()
