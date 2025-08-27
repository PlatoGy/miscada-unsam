#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专用：UnSAM 点提示接口批量评测脚本
- 目录：脚本与 CT/ 同级；CT/<case_id>/DICOM_anon/*.dcm 与 CT/<case_id>/Ground/liver_GT_xxx.png
- DICOM 命名两种：
    i0xxx,0000b.dcm        -> liver_GT_xxx.png
    IMG-00nn-00xxx.dcm     -> liver_GT_xxx.png  (nn 忽略)
- 每例头/中/尾各 5 张，共 15 张
- 调用 http://localhost:8008/point_unsam (organ=liver)，可能返回 3-4 张；逐张计算 Dice/IoU
- 记录客户端测得端到端时长 request_duration_s（秒）
- 输出 CSV：batch_eval_unsam_results.csv
依赖：pydicom, numpy, pillow, requests, scikit-image
"""

import os, re, csv, glob, random, time
import numpy as np
import requests
import pydicom
from PIL import Image
from skimage.io import imread
from skimage.transform import resize

# ===================== 配置 =====================
CT_DIR = "./CT"                         # 根目录下 CT 文件夹
OUT_DIR = "./eval_unsam_pngs"           # 输入 PNG 与可视化结果保存目录
CSV_PATH = "./batch_eval_unsam_results.csv"

US_BASE_URL = "http://localhost:8008"
POINT_UNSAM_ENDPOINT = f"{US_BASE_URL}/point_unsam"

RANDOM_SEED = 2025
SLICES_PER_REGION = 5                   # 头/中/尾各取 5
WINDOW_WW, WINDOW_WL = 400, 60          # 腹部常用窗宽/位
TIMEOUT_POST = 180                      # 请求超时（秒）
# ===============================================

random.seed(RANDOM_SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 工具函数 ----------
def apply_window_hu_to_uint8(hu_slice, ww=400, wl=60):
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    x = np.clip(hu_slice, lo, hi)
    x = (x - lo) / (hi - lo)  # 0..1
    x = (x * 255.0).round().astype(np.uint8)
    return x

def dicom_to_hu(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + inter

def sort_key_for_dicom(ds):
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp and len(ipp) >= 3:
        return float(ipp[2])
    return float(getattr(ds, "InstanceNumber", 0))

def extract_xxx_from_dicom_name(filename):
    """
    支持：
      i0xxx,0000b.dcm          -> 提取 xxx
      IMG-00nn-00xxx.dcm       -> 提取 xxx (忽略 nn)
    兜底：取文件名最后一串数字
    """
    name = os.path.basename(filename)
    m1 = re.match(r"i0(\d+),\d+\w?\.dcm$", name, re.IGNORECASE)
    if m1:
        return m1.group(1)
    m2 = re.match(r"IMG-\d+-00(\d+)\.dcm$", name, re.IGNORECASE)
    if m2:
        return m2.group(1)
    m3 = re.findall(r"(\d+)", name)
    return m3[-1] if m3 else None

def build_gt_path(gt_dir, xxx):
    cand = os.path.join(gt_dir, f"liver_GT_{xxx}.png")
    if os.path.exists(cand):
        return cand
    # 容错：去前导零或补零到3/4位
    xxx_int = int(xxx)
    for pat in (f"liver_GT_{xxx_int}.png",
                f"liver_GT_{xxx_int:03d}.png",
                f"liver_GT_{xxx_int:04d}.png"):
        p = os.path.join(gt_dir, pat)
        if os.path.exists(p):
            return p
    # 兜底：扫描 Ground
    for p in glob.glob(os.path.join(gt_dir, "liver_GT_*.png")):
        ms = re.search(r"liver_GT_(\d+)\.png$", os.path.basename(p))
        if ms and int(ms.group(1)) == xxx_int:
            return p
    return None

def ensure_3ch_uint8(img):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img

def recover_pred_mask_from_overlay(overlay_rgb, original_rgb):
    """
    还原预测掩膜（与 MedSAM 相同的着色约定）：
      vis[mask, 0] = 255; vis[mask, 1:] *= 0.5
    """
    R_ov = overlay_rgb[..., 0].astype(np.uint16)
    G_ov = overlay_rgb[..., 1].astype(np.uint16)
    B_ov = overlay_rgb[..., 2].astype(np.uint16)
    R_or = original_rgb[..., 0].astype(np.uint16)
    G_or = original_rgb[..., 1].astype(np.uint16)
    B_or = original_rgb[..., 2].astype(np.uint16)
    mask = (R_ov == 255) & ((G_ov < G_or) | (B_ov < B_or))
    return mask.astype(np.uint8)

def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    ps, gs = pred.sum(), gt.sum()
    dice = (2.0 * inter) / (ps + gs) if (ps + gs) > 0 else 1.0
    iou = (inter / union) if union > 0 else 1.0
    return float(dice), float(iou)

def sample_15_indices(n_slices, k_per_region=5):
    thirds = [
        (0, n_slices // 3),
        (n_slices // 3, (2 * n_slices) // 3),
        ((2 * n_slices) // 3, n_slices),
    ]
    picked = []
    for a, b in thirds:
        cand = list(range(a, b)) if b > a else []
        need = k_per_region
        if len(cand) >= need:
            picked.extend(random.sample(cand, need))
        else:
            if len(cand) == 0:
                picked.extend(random.choices(range(n_slices), k=need))
            else:
                chosen = cand.copy()
                while len(chosen) < need:
                    chosen.append(random.choice(cand))
                picked.extend(chosen)
    return sorted(picked)

# ---------- 评测核心 ----------
def process_case(case_dir, rows):
    case_id = os.path.basename(case_dir)
    dicom_dir = os.path.join(case_dir, "DICOM_anon")
    gt_dir = os.path.join(case_dir, "Ground")

    if not os.path.isdir(dicom_dir):
        print(f"[WARN] {case_id}: missing {dicom_dir}, skip"); return

    dicom_files = [f for f in glob.glob(os.path.join(dicom_dir, "**"), recursive=True)
                   if os.path.isfile(f) and f.lower().endswith(".dcm")]
    if not dicom_files:
        print(f"[WARN] {case_id}: no DICOM files, skip"); return

    # 读取+排序
    dsets = []
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=False, force=True)
            if getattr(ds, "PixelData", None) is None: continue
            dsets.append((ds, fp))
        except Exception:
            continue
    if not dsets:
        print(f"[WARN] {case_id}: no readable DICOMs with pixel data, skip"); return

    dsets.sort(key=lambda x: sort_key_for_dicom(x[0]))
    n = len(dsets)
    picked_idx = sample_15_indices(n, SLICES_PER_REGION)
    print(f"[INFO] {case_id}: total={n}, picked={picked_idx}")

    for k in picked_idx:
        ds, fp = dsets[k]
        xxx = extract_xxx_from_dicom_name(fp)
        if not xxx:
            print(f"[WARN] {case_id}: cannot parse slice number from {os.path.basename(fp)}, skip")
            continue

        # 生成输入 PNG
        hu = dicom_to_hu(ds)
        x8 = apply_window_hu_to_uint8(hu, WINDOW_WW, WINDOW_WL)
        rgb = np.stack([x8, x8, x8], axis=-1)
        png_path = os.path.join(OUT_DIR, f"{case_id}_xxx{xxx}.png")
        Image.fromarray(rgb).save(png_path)

        # 取 GT
        gt_path = build_gt_path(gt_dir, xxx)
        gt_mask = None
        if gt_path and os.path.exists(gt_path):
            g = imread(gt_path)
            if g.ndim == 3: g = g[..., 0]
            gt_mask = (g > 0).astype(np.uint8)

        # 调用 UnSAM 点提示
        try:
            with open(png_path, "rb") as f:
                files = {"file": (os.path.basename(png_path), f, "image/png")}
                data = {"organ": "liver"}
                t0 = time.time()
                resp = requests.post(POINT_UNSAM_ENDPOINT, files=files, data=data, timeout=TIMEOUT_POST)
                dur = time.time() - t0
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"[ERROR] {case_id} xxx={xxx}: UnSAM request failed: {e}")
            continue

        # 解析返回的多张结果
        urls = []
        if isinstance(payload, dict):
            if "segmentation_results" in payload and isinstance(payload["segmentation_results"], list):
                urls = payload["segmentation_results"]
            elif "image_url" in payload:
                urls = [payload["image_url"]]
        urls = [u if u.startswith("http") else (US_BASE_URL + u) for u in urls]

        for vi, u in enumerate(urls):
            overlay_path = os.path.join(OUT_DIR, f"{case_id}_xxx{xxx}_unsam_v{vi}.png")
            pred_mask = None
            try:
                rimg = requests.get(u, timeout=60); rimg.raise_for_status()
                with open(overlay_path, "wb") as fo: fo.write(rimg.content)
                orig = imread(png_path); ov = imread(overlay_path)
                orig = ensure_3ch_uint8(orig); ov = ensure_3ch_uint8(ov)
                pred_mask = recover_pred_mask_from_overlay(ov, orig)
            except Exception as e:
                print(f"[WARN] {case_id} xxx={xxx}: v{vi} recover pred mask failed: {e}")

            dice, iou = (float("nan"), float("nan"))
            if gt_mask is not None and pred_mask is not None:
                gm = gt_mask
                if gm.shape != pred_mask.shape:
                    gm = resize(gm.astype(float), pred_mask.shape, order=0,
                                preserve_range=True, anti_aliasing=False)
                    gm = (gm > 0.5).astype(np.uint8)
                dice, iou = compute_metrics(pred_mask, gm)

            rows.append({
                "case_id": case_id,
                "dicom_file": os.path.basename(fp),
                "xxx": xxx,
                "variant_idx": vi,                 # UnSAM 的第几张候选
                "input_png": png_path,
                "overlay_url": u,
                "request_duration_s": round(dur, 2),   # 客户端端到端时间
                "dice": None if np.isnan(dice) else round(dice, 4),
                "iou":  None if np.isnan(iou)  else round(iou, 4),
            })
            print(f"[OK] {case_id} xxx={xxx} v{vi} | dur={dur:.2f}s | dice={dice:.4f} iou={iou:.4f}")

def main():
    case_dirs = [os.path.join(CT_DIR, d) for d in os.listdir(CT_DIR)
                 if os.path.isdir(os.path.join(CT_DIR, d)) and d.isdigit()]
    case_dirs.sort(key=lambda p: int(os.path.basename(p)))
    if len(case_dirs) == 0:
        print(f"[ERROR] No case dirs under {CT_DIR}"); return

    rows = []
    t0 = time.time()
    for case_dir in case_dirs:
        process_case(case_dir, rows)

    fieldnames = [
        "case_id","dicom_file","xxx","variant_idx",
        "input_png","overlay_url","request_duration_s",
        "dice","iou"
    ]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[DONE] wrote {len(rows)} rows to {CSV_PATH} in {(time.time()-t0):.1f}s")

if __name__ == "__main__":
    main()
