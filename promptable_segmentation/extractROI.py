import cv2
import numpy as np
import random

def extract_liver_points_from_brightest_region(image_path, threshold=200, num_points=1, center_std_ratio=0.1):
    """
    提取 liver 外接框，并从 box 中心附近采样多个点（高斯采样）

    参数：
        image_path: 图像路径
        threshold: 灰度阈值
        num_points: 返回的点数量
        center_std_ratio: 控制采样离中心的距离（比例）

    返回：
        box: [[x1, y1, x2, y2]]
        points: [[x, y], ...]
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ 无法读取图像: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        print("❌ 未找到亮区域")
        return None, None

    # 选最大区域
    largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x, y, w, h = stats[largest_idx, cv2.CC_STAT_LEFT], stats[largest_idx, cv2.CC_STAT_TOP], \
                 stats[largest_idx, cv2.CC_STAT_WIDTH], stats[largest_idx, cv2.CC_STAT_HEIGHT]

    box = np.array([[x, y, x + w, y + h]])
    print(f"✅ 提取 liver box: x={x}, y={y}, w={w}, h={h}")

    # 高斯采样点（靠近中心）
    cx, cy = x + w // 2, y + h // 2
    std_x = int(w * center_std_ratio)
    std_y = int(h * center_std_ratio)

    points = []
    for _ in range(num_points):
        px = int(np.clip(np.random.normal(cx, std_x), x, x + w - 1))
        py = int(np.clip(np.random.normal(cy, std_y), y, y + h - 1))
        points.append([px, py])

    return box, np.array(points)
