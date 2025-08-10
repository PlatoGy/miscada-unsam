import os
import re
import glob
import csv
import random
import os.path as osp
import numpy as np
from PIL import Image
import pydicom

# ---------- 可调参数 ----------
SEED = 42
SPLIT = (0.8, 0.1, 0.1)  # 按病例划分比例：train/val/test
WINDOW_WL = -100
WINDOW_WW = 400
STRICT_ONLY_MATCHED = True   # True=只用有GT的切片；False=允许写入全0负例
# --------------------------------

PAT_DCM_IDX = re.compile(r"i0(\d{3}),")        # i0***,0000b.dcm -> ***
PAT_MASK     = re.compile(r"liver_GT_(\d{3})\.png")

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def hu_windowing(arr, wl, ww):
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return (arr * 255.0).round().astype(np.uint8)

def dcm_to_png(dcm_path, wl, ww):
    ds = pydicom.dcmread(dcm_path)
    px = ds.pixel_array.astype(np.int16)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = px * slope + inter
    return hu_windowing(hu, wl, ww)

def binarize_mask(arr):
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)  # 0/1

def extract_idx_from_dcm_name(basename):
    m = PAT_DCM_IDX.search(basename)
    return m.group(1) if m else None

def build_mask_index(mask_dir):
    idx = {}
    for p in glob.glob(osp.join(mask_dir, "liver_GT_*.png")):
        m = PAT_MASK.match(osp.basename(p))
        if m:
            idx[m.group(1)] = p
    return idx

def discover_cases(train_ct_root):
    """
    发现病例目录：promptable_segmentation/data_preprocessing/Train_Sets/CT/<case_id>/{DICOM_anon,Ground}
    返回字典列表：[{"id":"1","dcm":".../DICOM_anon","msk":".../Ground"}, ...]
    仅收集同时存在 DICOM_anon 与 Ground 的病例。
    """
    cases = []
    for case_dir in sorted(glob.glob(osp.join(train_ct_root, "*"))):
        if not osp.isdir(case_dir):
            continue
        dcm_dir = osp.join(case_dir, "DICOM_anon")
        msk_dir = osp.join(case_dir, "Ground")
        if osp.isdir(dcm_dir) and osp.isdir(msk_dir):
            cases.append({
                "id": osp.basename(case_dir),
                "dcm": dcm_dir,
                "msk": msk_dir
            })
    return cases

def split_cases(cases, split=SPLIT, seed=SEED):
    r = random.Random(seed)
    order = list(range(len(cases)))
    r.shuffle(order)
    n = len(order)
    n_train = int(round(n * split[0]))
    n_val   = int(round(n * split[1]))
    # 保证三者之和为 n
    n_test  = n - n_train - n_val
    train_idx = order[:n_train]
    val_idx   = order[n_train:n_train+n_val]
    test_idx  = order[n_train+n_val:]
    return ([cases[i] for i in train_idx],
            [cases[i] for i in val_idx],
            [cases[i] for i in test_idx])

def convert_case_to_pairs(case, out_img_dir, out_msk_dir, strict_only=True):
    """
    将病例的DICOM和掩膜按文件名排序后直接一一对应。
    如果数量不一致，按最小长度截断。
    """
    ensure_dir(out_img_dir)
    ensure_dir(out_msk_dir)

    dcm_list = sorted(glob.glob(osp.join(case["dcm"], "*.dcm")))
    mask_list = sorted(glob.glob(osp.join(case["msk"], "*.png")))

    n_pairs = min(len(dcm_list), len(mask_list))
    if n_pairs == 0:
        print(f"[WARN] case {case['id']} 没有可配对的图像/掩膜")
        return 0

    count = 0
    for dcm_path, mask_path in zip(dcm_list[:n_pairs], mask_list[:n_pairs]):
        img_u8 = dcm_to_png(dcm_path, WINDOW_WL, WINDOW_WW)
        out_name = f"case{case['id']}_{osp.basename(dcm_path).replace(',', '_').replace('.dcm', '.png')}"
        out_img = osp.join(out_img_dir, out_name)
        Image.fromarray(img_u8).save(out_img)

        mask_img = np.array(Image.open(mask_path))
        mask_bin = binarize_mask(mask_img)
        if mask_bin.shape != img_u8.shape:
            mask_bin = np.array(
                Image.fromarray(mask_bin).resize(
                    (img_u8.shape[1], img_u8.shape[0]),
                    Image.NEAREST
                )
            )
        out_msk = osp.join(out_msk_dir, out_name)
        Image.fromarray(mask_bin.astype(np.uint8)).save(out_msk)

        count += 1

    return count

def write_manifest(manifest_csv, rows):
    ensure_dir(osp.dirname(manifest_csv))
    with open(manifest_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split","case_id","image","mask"])
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    # 1) 路径设定
    # 你的有标注训练集根目录：
    TRAIN_CT_ROOT = "promptable_segmentation/data_preprocessing/Train_Sets/CT"
    # 你的仅DICOM推理集根目录（可选）：
    TESTSETS_ROOT = "promptable_segmentation/data_preprocessing/Test_Sets/CT"

    OUT_ROOT = "promptable_segmentation/datasets"
    OUT_LIVER = osp.join(OUT_ROOT, "liver")
    OUT_UNLAB = osp.join(OUT_ROOT, "liver_unlabeled")

    # 2) 发现病例并划分
    cases = discover_cases(TRAIN_CT_ROOT)
    assert len(cases) >= 3, "病例数太少，无法稳定划分 train/val/test"
    tr_cases, va_cases, te_cases = split_cases(cases, SPLIT, SEED)
    print(f"病例统计：train={len(tr_cases)}, val={len(va_cases)}, test={len(te_cases)} / total={len(cases)}")

    # 3) 转换与配对（来自原Train_Sets）
    rows = []
    for split_name, split_cases in [("train", tr_cases), ("val", va_cases), ("test", te_cases)]:
        out_img_dir = osp.join(OUT_LIVER, split_name, "images")
        out_msk_dir = osp.join(OUT_LIVER, split_name, "masks")
        ensure_dir(out_img_dir); ensure_dir(out_msk_dir)

        total = 0
        for c in split_cases:
            n = convert_case_to_pairs(c, out_img_dir, out_msk_dir, strict_only=STRICT_ONLY_MATCHED)
            total += n
            print(f"[{split_name}] case {c['id']} -> {n} 对")
        print(f"[{split_name}] 共写入配对样本：{total}")

        # 记录清单
        for img_p in sorted(glob.glob(osp.join(out_img_dir, "*.png"))):
            msk_p = osp.join(out_msk_dir, osp.basename(img_p))
            rows.append([split_name, osp.basename(osp.dirname(osp.dirname(img_p))), img_p, msk_p])

    write_manifest(osp.join(OUT_LIVER, "manifest.csv"), rows)
    print("已写入 manifest:", osp.join(OUT_LIVER, "manifest.csv"))

    # 4) 仅DICOM推理集（来自 Test_Sets，非评估用途）
    if osp.isdir(TESTSETS_ROOT):
        infer_img_dir = osp.join(OUT_UNLAB, "infer", "images")
        ensure_dir(infer_img_dir)
        count = 0
        for case_dir in sorted(glob.glob(osp.join(TESTSETS_ROOT, "*"))):
            dcm_dir = osp.join(case_dir, "DICOM_anon")
            if not osp.isdir(dcm_dir):
                continue
            for dcm_path in sorted(glob.glob(osp.join(dcm_dir, "*.dcm"))):
                img_u8 = dcm_to_png(dcm_path, WINDOW_WL, WINDOW_WW)
                name = f"testonly_{osp.basename(case_dir)}_{osp.basename(dcm_path).replace(',', '_').replace('.dcm','.png')}"
                Image.fromarray(img_u8).save(osp.join(infer_img_dir, name))
                count += 1
        print(f"[infer(unlabeled)] 仅DICOM推理样本：{count}")

    print("完成：datasets/liver/{train,val,test}/{images,masks} + datasets/liver_unlabeled/infer/images")
