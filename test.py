# quick_sanity.py
from promptable_segmentation.datasets.registration.register_liver import register_liver_dataset
from promptable_segmentation.datasets.build import build_train_dataloader
from omegaconf import OmegaConf
import yaml, torch

# 1) 注册数据
register_liver_dataset("/home/mktd44/project/UnSAM/promptable_segmentation/datasets/liver", filter_empty=True)

# 2) 读你的 yaml
with open("/home/mktd44/project/UnSAM/promptable_segmentation/configs/semantic_sam_only_sa-1b_swinT.yaml") as f:
    cfg = OmegaConf.create(yaml.safe_load(f))

# 3) 构建训练 dataloader（走包装后的 mapper）
dl = build_train_dataloader(cfg)

# 4) 取一个 batch 看 targets
batch = next(iter(dl))
print("batch size:", len(batch))
for i, dd in enumerate(batch):
    t = dd.get("targets", None)
    if t is None:
        print(i, "targets=None")
    else:
        m = t["masks"]; y = t["labels"]
        print(i, "masks:", tuple(m.shape), "labels:", y.tolist())
        # 断言至少有一个前景
        assert m.numel() > 0 and y.numel() > 0
print("OK")
