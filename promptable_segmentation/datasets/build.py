# Copyright (c) Facebook, Inc. and its affiliates.
import os
import itertools
import logging
import copy
from typing import Any, Callable, Dict, List, Optional, Union, Set

import torch
import torch.utils.data
import torch.utils.data as torchdata
import numpy as np
from PIL import Image

import detectron2.utils.comm as comm
from detectron2.data.build import (
    build_batch_data_loader,
    load_proposals_into_dataset,
    trivial_batch_collator,
)
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler, TrainingSampler
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
)
from fvcore.common.config import CfgNode
from omegaconf import DictConfig, OmegaConf

# 保留导入（训练/评估不再使用 SamBaselineDatasetMapper，避免其训练断言）
from .dataset_mappers import (
    SamBaselineDatasetMapper,
    CoCoInferenceDatasetMapper,
)
from .evaluation import JointBoxPointInteractiveEvaluator
from semantic_sam.utils import configurable
from detectron2.utils.comm import get_world_size, is_main_process

logger = logging.getLogger(__name__)


# -----------------------------
# Utils
# -----------------------------
def _to_mapper_cfg(cfg):
    """把 DictConfig/CfgNode 转成 DatasetMapper 期望的最小 CfgNode 子配置。"""
    if isinstance(cfg, DictConfig):
        cfg_container = OmegaConf.to_container(copy.deepcopy(cfg))
    else:
        cfg_container = copy.deepcopy(cfg)
    return CfgNode({
        "INPUT": cfg_container["INPUT"],
        "MODEL": cfg_container["MODEL"],
        "DATASETS": cfg_container["DATASETS"],
    })


def _empty_targets_like(dd):
    """构造空目标，满足 not None 的断言。"""
    H = W = None
    if "sem_seg" in dd and isinstance(dd["sem_seg"], torch.Tensor):
        H, W = dd["sem_seg"].shape[-2:]
    elif "image" in dd and isinstance(dd["image"], torch.Tensor):
        H, W = dd["image"].shape[-2:]
    if H is None or W is None:
        return {"labels": torch.zeros((0,), dtype=torch.long),
                "masks": torch.zeros((0, 0, 0), dtype=torch.uint8)}
    return {"labels": torch.zeros((0,), dtype=torch.long),
            "masks": torch.zeros((0, H, W), dtype=torch.uint8)}


def _try_load_sem_seg_from_path(dataset_dict):
    """
    尝试从 dataset_dict['sem_seg_file_name'] 读入语义标签（单通道）。
    返回 torch.LongTensor[H,W] 或 None。
    """
    path = dataset_dict.get("sem_seg_file_name", None)
    if path is None or not os.path.exists(path):
        return None
    try:
        with Image.open(path) as im:
            arr = np.array(im)
        if arr.ndim == 3:
            # 若意外是彩色，取单通道
            arr = arr[..., 0]
        return torch.as_tensor(arr, dtype=torch.long)
    except Exception as e:
        logger.warning(f"Failed to read sem seg from {path}: {e}")
        return None


def _build_targets_from_dd(dd, dataset_dict=None):
    # 1) 语义分割优先
    sem = dd.get("sem_seg", None)
    if sem is None and dataset_dict is not None:
        sem_loaded = _try_load_sem_seg_from_path(dataset_dict)
        if sem_loaded is not None:
            dd["sem_seg"] = sem_loaded
            sem = sem_loaded

    if sem is not None:
        sem_t = sem.long() if isinstance(sem, torch.Tensor) else torch.as_tensor(np.array(sem), dtype=torch.long)
        H, W = sem_t.shape[-2], sem_t.shape[-1]
        ignore_val = 255
        uniq = torch.unique(sem_t)
        uniq = uniq[(uniq != 0) & (uniq != ignore_val)]
        if uniq.numel() == 0:
            return None  # ←← 关键：无前景，返回 None 以便上层跳过该样本
        masks, labels = [], []
        for cls_id in uniq.tolist():
            m = (sem_t == int(cls_id))
            if m.any():
                masks.append(m.unsqueeze(0))
                labels.append(int(cls_id))
        if len(masks) == 0:
            return None
        masks = torch.cat(masks, dim=0).to(torch.uint8)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"labels": labels, "masks": masks}

    # 2) 实例分割
    instances = dd.get("instances", None)
    if instances is not None and hasattr(instances, "get_fields"):
        fields = instances.get_fields()
        if "gt_masks" in fields and "gt_classes" in fields:
            gt_masks = fields["gt_masks"]
            masks = gt_masks.tensor.to(torch.uint8) if hasattr(gt_masks, "tensor") else torch.as_tensor(gt_masks, dtype=torch.uint8)
            labels = torch.as_tensor(fields["gt_classes"], dtype=torch.long)
            if masks.numel() == 0 or labels.numel() == 0:
                return None
            return {"labels": labels, "masks": masks}

    return None

# -----------------------------
# Datasets / Loaders
# -----------------------------
class JointLoader(torchdata.IterableDataset):
    def __init__(self, loaders, key_dataset):
        dataset_names = []
        for key, loader in loaders.items():
            name = "{}".format(key.split('_')[0])
            setattr(self, name, loader)
            dataset_names += [name]
        self.dataset_names = dataset_names
        self.key_dataset = key_dataset
    
    def __iter__(self):
        for batch in zip(*[getattr(self, name) for name in self.dataset_names]):
            yield {key: batch[i] for i, key in enumerate(self.dataset_names)}

    def __len__(self):
        return len(getattr(self, self.key_dataset))


def filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names):
    """过滤只有 crowd 或无标注的样本（COCO 常见预处理）。"""
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if isinstance(ann, list):
                for instance in ann:
                    if instance.get("iscrowd", 0) == 0:
                        return True
            else:
                if ann.get("iscrowd", 0) == 0:
                    return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    _logger = logging.getLogger(__name__)
    _logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def get_detection_dataset_dicts(dataset_names, filter_empty=True, proposal_files=None):
    """加载并合并多个 DatasetCatalog 数据集为标准 dict 列表。"""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)
    
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), f"Dataset '{dataset_name}' is empty!"

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names)

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """单个测试集的 dataloader 配置。"""
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=None,
    )

    # 评估默认走通用 DatasetMapper(False)
    if mapper is None:
        mapper = DatasetMapper(_to_mapper_cfg(cfg), False)

    assert cfg['TEST']['BATCH_SIZE_TOTAL'] % get_world_size() == 0, \
        "Evaluation total batchsize is not divisible by gpu number"
    batch_size = cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size()

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg['DATALOADER']['NUM_WORKERS'],
        "sampler": InferenceSampler(len(dataset)),
        "batch_size": batch_size,
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    """Dataloader for test/eval."""
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def _train_loader_from_config(cfg, dataset_name, mapper, *, dataset=None, sampler=None):
    cfg_datasets = cfg['DATASETS']
    cfg_dataloader = cfg['DATALOADER']
    
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg_dataloader['FILTER_EMPTY_ANNOTATIONS'],
            proposal_files=cfg_datasets['PROPOSAL_FILES_TRAIN'] if cfg_dataloader['LOAD_PROPOSALS'] else None,
        )

    # 若外部没传 mapper，则默认用 DatasetMapper(True)
    if mapper is None:
        mapper = DatasetMapper(_to_mapper_cfg(cfg), True)

    if sampler is None:
        sampler_name = cfg_dataloader['SAMPLER_TRAIN']
        _logger = logging.getLogger(__name__)
        _logger.info("Using training sampler {}".format(sampler_name))
        sampler = TrainingSampler(len(dataset))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg['TRAIN']['BATCH_SIZE_TOTAL'],
        "aspect_ratio_grouping": cfg_dataloader['ASPECT_RATIO_GROUPING'],
        "num_workers": cfg_dataloader['NUM_WORKERS'],
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    """Dataloader for training."""
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


def get_config_from_name(cfg, dataset_name):
    """根据数据集名动态合并子配置。"""
    joint_part = cfg['DATASETS'].get('JOINT_PART_LOADER', False)
    name = dataset_name.lower()
    if name.startswith("liver_"):
        return cfg
    if 'sam' in dataset_name:
        cfg.update(cfg['SAM'])
        return cfg
    if joint_part and ('pascal' in dataset_name or 'paco' in dataset_name or 'partimagenet' in dataset_name):
        cfg.update(cfg['PART_ALL'])
        return cfg
    elif 'pascal' in dataset_name:
        cfg.update(cfg['PSACAL_PART'])
        return cfg
    elif 'refcoco' in dataset_name:
        cfg.update(cfg['REF'])
        return cfg
    elif 'coco' in dataset_name:
        if 'COCO' in cfg.keys():
            cfg.update(cfg['COCO'])
        return cfg
    elif 'ade' in dataset_name:
        if 'ADE20K' in cfg.keys():
            cfg.update(cfg['ADE20K'])
        return cfg
    elif 'imagenet' in dataset_name:
        if 'IMAGENET' in cfg.keys():
            cfg.update(cfg['IMAGENET'])
        return cfg
    elif 'vlp' in dataset_name:
        cfg.update(cfg['VLP'])
        return cfg
    elif 'sun' in dataset_name:
        cfg.update(cfg['SUN'])
        return cfg
    elif 'object365' in dataset_name:
        cfg.update(cfg['OBJECT365'])
        return cfg
    elif 'scan' in dataset_name:
        cfg.update(cfg['SCAN'])
        return cfg
    elif 'cityscape' in dataset_name:
        cfg.update(cfg['CITY'])
        return cfg
    elif 'bdd' in dataset_name:
        cfg.update(cfg['BDD'])
        return cfg
    else:
        assert False, "dataset not support."


def build_eval_dataloader(cfg, ):
    """为每个 TEST 数据集构建一个 dataloader。"""
    dataloaders = []
    cfg = copy.deepcopy(cfg)
    for dataset_name in cfg['DATASETS']['TEST']:
        cfg = get_config_from_name(cfg, dataset_name)
        name = dataset_name.lower()

        # liver_*：统一交给通用 DatasetMapper(False)
        if name.startswith("liver_"):
            mapper = DatasetMapper(_to_mapper_cfg(cfg), False)
            dataloaders += [build_detection_test_loader(cfg, dataset_name, mapper=mapper)]
            continue

        # coco：仍用专用 CoCoInferenceDatasetMapper
        if 'coco' in dataset_name:
            mapper = CoCoInferenceDatasetMapper(cfg, False)
        # 其它（包括 sam_val）统一走通用 DatasetMapper(False)
        else:
            mapper = DatasetMapper(_to_mapper_cfg(cfg), False)

        dataloaders += [build_detection_test_loader(cfg, dataset_name, mapper=mapper)]
    return dataloaders


def build_train_dataloader(cfg, ):
    """
    训练阶段：统一使用通用 DatasetMapper 包装版，自动从 sem_seg/instances 生成 targets。
    对 liver_* 数据集也使用同一套包装逻辑（并关闭 FILTER_EMPTY_ANNOTATIONS）。
    """
    dataset_names = cfg['DATASETS']['TRAIN']
    loaders = {}
    cfg = copy.deepcopy(cfg)

    for dataset_name in dataset_names:
        cfg = get_config_from_name(cfg, dataset_name)
        name = dataset_name.lower()

        # 语义分割场景常不筛空
        cfg['DATALOADER']['FILTER_EMPTY_ANNOTATIONS'] = False

        base_mapper = DatasetMapper(_to_mapper_cfg(cfg), True)

        def _wrapped_mapper(dataset_dict):
            dd = base_mapper(dataset_dict)
            targets = _build_targets_from_dd(dd, dataset_dict)
            if targets is None or ("masks" in targets and targets["masks"].numel() == 0):
                return None  # ←← 关键：让 MapDataset 丢掉无前景样本
            dd["targets"] = targets
            return dd


        loaders[dataset_name] = build_detection_train_loader(
            cfg, dataset_name=dataset_name, mapper=_wrapped_mapper
        )

    if len(loaders) == 1 and not cfg['LOADER'].get('JOINT', False):
        for k, v in loaders.items():
            try:
                n = len(v)
            except TypeError:
                n = "unknown (sampler has no __len__)"
            print("number of iterations per epoch:", k, n)
        return list(loaders.values())[0]
    else:
        return JointLoader(loaders, key_dataset=cfg['LOADER'].get('KEY_DATASET', 'coco'))


# -----------------------------
# Evaluators / Optimizer
# -----------------------------
def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    创建 evaluator 列表。语义分割或 liver_* 用 SemSegEvaluator（默认二类：0 背景，1 前景/肝；忽略 255）。
    交互式联合评估与 coco 均保持兼容。
    """
    cfg_model_decoder_test = cfg["MODEL"]["DECODER"]["TEST"]

    if output_folder is None:
        output_folder = os.path.join(cfg["OUTPUT_DIR"], "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    name = dataset_name.lower()

    if evaluator_type == "sem_seg" or name.startswith("liver_"):
        evaluator_list.append(SemSegEvaluator(
            dataset_name,
            distributed=True,
            output_dir=output_folder,
            num_classes=2,
            ignore_label=255,
        ))

    if evaluator_type in ['coco_panoptic_seg_interactive_jointboxpoint']:
        box_interactive = cfg_model_decoder_test.get('BOX_INTERACTIVE', False)
        evaluator_list.append(JointBoxPointInteractiveEvaluator(
            dataset_name, output_dir=output_folder, box_interactive=box_interactive))

    if evaluator_type == 'sam':
        evaluator_list.append(COCOEvaluator("coco_2017_val", output_dir=output_folder))

    if evaluator_type == "coco":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
 
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]

    return DatasetEvaluators(evaluator_list)


def build_optimizer(cls, cfg, model):
    cfg_solver = cfg['SOLVER']
    weight_decay_norm = cfg_solver['WEIGHT_DECAY_NORM']
    weight_decay_embed = cfg_solver['WEIGHT_DECAY_EMBED']
    weight_decay_bias = cfg_solver.get('WEIGHT_DECAY_BIAS', 0.0)

    defaults = {}
    defaults["lr"] = cfg_solver['BASE_LR']
    defaults["weight_decay"] = cfg_solver['WEIGHT_DECAY']

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    lr_multiplier = cfg['SOLVER']['LR_MULTIPLIER']
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)

            for key, lr_mul in lr_multiplier.items():
                if key in "{}.{}".format(module_name, module_param_name):
                    hyperparams["lr"] = hyperparams["lr"] * lr_mul
                    if is_main_process():
                        logger.info("Modify Learning rate of {}: {}".format(
                            "{}.{}".format(module_name, module_param_name), lr_mul))

            if (
                "relative_position_bias_table" in module_param_name
                or "absolute_pos_embed" in module_param_name
            ):
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            if "bias" in module_name:
                hyperparams["weight_decay"] = weight_decay_bias
            params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
        clip_norm_val = cfg_solver['CLIP_GRADIENTS']['CLIP_VALUE']
        enable = (
            cfg_solver['CLIP_GRADIENTS']['ENABLED']
            and cfg_solver['CLIP_GRADIENTS']['CLIP_TYPE'] == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg_solver['OPTIMIZER']
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg_solver['BASE_LR'], momentum=cfg_solver['MOMENTUM']
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg_solver['BASE_LR']
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    return optimizer
