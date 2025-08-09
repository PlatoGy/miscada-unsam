# UnSAM: Unsupervised Segment Anything Model

**[Hao Zhang\*, Haotian Liu\*, Yixiao Ge\*, Ying Shan, Mike Zheng Shou]**  
(* 表示共同一作)  
**[arxiv](https://arxiv.org/abs/2403.14154) | [project](https://github.com/OpenGVLab/UnSAM) | [demo](https://huggingface.co/spaces/OpenGVLab/UnSAM)**  
来自 ShowLab@National University of Singapore 和 Tencent AI Lab。

---

## 更新日志

**2024.4.29**  
添加了“提示分割”任务，包括点、框、掩码作为提示输入。

**2024.3.22**  
开放了 UnSAM 模型的训练代码和训练模型权重。

---

## 引言

我们提出了首个无需人工标注即可训练 Segment Anything Model (SAM) 的方法 —— UnSAM。UnSAM 是一种简单有效的伪标签训练方法，具备以下关键设计：

- **视觉引导语义增强（ViSA）模块**：在训练初期，引导 SAM 聚焦于语义区域而非边缘。
- **自适应任务采样器（ATS）模块**：在训练后期提供困难样本，以提升最终性能。
- **大规模图像训练数据**：使用 SA-1B 数据集（图像+弱标签），完全不依赖人工掩码。

下图为 UnSAM 的方法框架：

![framework](assets/framework.jpg)

---

## 准备工作

**安装依赖**

```bash
pip install -r requirements.txt
pip install -e segment_anything
pip install -e detectron2
```


## 快速开始
1. 训练 UnSAM
使用 SA-1B 数据集训练 UnSAM：

```bash
# 单 GPU
python train_net.py --num-gpus 1 --config-file configs/unsam_only_sa-1b_swinL.yaml OUTPUT_DIR unsam_only_sa-1b_swinL

# 多 GPU
torchrun --nproc_per_node=8 train_net.py --config-file configs/unsam_only_sa-1b_swinL.yaml OUTPUT_DIR unsam_only_sa-1b_swinL

```

2. 测试 UnSAM（全图分割）
```bash
python whole_image_segmentation/demo_whole_image.py \
    --config-file whole_image_segmentation/configs/maskformer2_R50_bs16_50ep.yaml \
    --input input_img/image.jpg \
    --output output_img/image_out.jpg \
    --opts MODEL.WEIGHTS checkpoints/unsam_sa1b_4perc_ckpt_200k.pth MODEL.DEVICE cuda

```

3. 测试 UnSAM（提示分割）
支持点、框、掩码提示

```bash
python prompt_segmentation/demo_point_image.py \
    --config-file prompt_segmentation/configs/maskformer2_R50_bs16_50ep.yaml \
    --input input_img/image.jpg \
    --output output_img/image_out.jpg \
    --opts MODEL.WEIGHTS checkpoints/unsam_sa1b_4perc_ckpt_200k.pth MODEL.DEVICE cuda
```

## 模型权重下载
名称	下载地址	mIoU（COCO val2017）	SA-1B使用比例
unsam_sa1b_4perc_ckpt_200k.pth	百度网盘 (提取码: 36vx) / Google Drive	50.0	4%

## 项目结构说明

```bash
├── configs                          # 训练配置文件
├── whole_image_segmentation        # 全图分割代码
├── prompt_segmentation             # 提示分割代码
├── detectron2                      # Detectron2 主体
├── segment_anything                # SAM 模块
├── unsam                           # UnSAM 核心模块（ViSA + ATS）
│   ├── engine
│   ├── modeling
│   └── data
├── utils                           # 工具函数
│   ├── dist.py
│   └── arguments.py
├── train_net.py                    # 主训练入口
├── README.md
└── requirements.txt
```