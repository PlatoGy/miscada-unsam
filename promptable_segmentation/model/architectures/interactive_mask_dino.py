# ------------------------------------------------------------------------
# Copyright (c) MicroSoft, Inc. and its affiliates.
# Modified from OpenSeed https://github.com/IDEA-Research/OpenSeed by Feng Li (fliay@connect.ust.hk).
# ------------------------------------------------------------------------
from typing import Tuple, List, Dict

import torch
from torch import nn
from torch.nn import functional as F

from semantic_sam.architectures.registry import register_model
from ..utils import configurable, box_ops, get_class_names, get_iou
from ..backbone import build_backbone, Backbone
from ..body import build_semantic_sam_head
from semantic_sam.modules import sem_seg_postprocess, SetCriterion, M2MHungarianMatcher, SetCriterionOsPartWholeM2M
from semantic_sam.language import build_language_encoder

from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
import torch.distributed as dist

from kornia.contrib import distance_transform

import random


class GeneralizedMaskDINO(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion_switch: dict,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
        train_dataset_name: str,
        coco_mask_on=True,
        sam_on: bool = True,
        regenerate_point: bool = False,
        num_mask_tokens: int = 3,
        max_num_instance: int = 100,
        classification_on: bool = False,
        many_to_one: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = None
        self.criterion_switch = criterion_switch
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        self.train_class_names = dict()
        self.train_dataset_name = train_dataset_name
        self.coco_mask_on = coco_mask_on
        self.classification_on = classification_on
        self.task_switch = {'sam': sam_on}  # 仅记录，不再决定训练分支

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.max_num_instance = max_num_instance
        self.num_mask_tokens = num_mask_tokens
        self.regenerate_point = regenerate_point
        self.many_to_one = many_to_one

    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        deep_supervision = dec_cfg['DEEP_SUPERVISION']
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']

        iou_weight = dec_cfg['IOU_WEIGHT']
        class_weight = dec_cfg['CLASS_WEIGHT']
        cost_class_weight = dec_cfg['COST_CLASS_WEIGHT']
        cost_dice_weight = dec_cfg['COST_DICE_WEIGHT']
        dice_weight = dec_cfg['DICE_WEIGHT']
        cost_mask_weight = dec_cfg['COST_MASK_WEIGHT']
        mask_weight = dec_cfg['MASK_WEIGHT']
        cost_box_weight = dec_cfg['COST_BOX_WEIGHT']
        box_weight = dec_cfg['BOX_WEIGHT']
        cost_giou_weight = dec_cfg['COST_GIOU_WEIGHT']
        giou_weight = dec_cfg['GIOU_WEIGHT']

        matcher = M2MHungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            num_mask_tokens=dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3)
        )

        weight_dict = {"loss_mask_cls_0": class_weight}
        weight_dict.update({"loss_mask_bce_0": mask_weight, "loss_mask_dice_0": dice_weight})
        weight_dict.update({"loss_bbox_0": box_weight, "loss_giou_0": giou_weight})
        weight_dict.update({"iou_score_loss_0": iou_weight})
        weight_dict.update({"loss_mask_part_cls_0": class_weight})
        if dec_cfg['TWO_STAGE']:
            interm_weight_dict = {k + '_interm': v for k, v in weight_dict.items()}
            weight_dict.update(interm_weight_dict)
        dn = dec_cfg['DN']
        if dn == "standard":
            weight_dict.update({k + "_dn": v for k, v in weight_dict.items() if k != "loss_mask" and k != "loss_dice"})
            dn_losses = ["dn_labels", "boxes"]
        elif dn == "seg":
            weight_dict.update({k + "_dn": v for k, v in weight_dict.items()})
            dn_losses = ["masks", "dn_labels", "boxes"]
        else:
            dn_losses = []
        if deep_supervision:
            dec_layers = dec_cfg['DEC_LAYERS']
            for i in range(dec_layers):
                weight_dict.update({k.replace('_0', f'_{i+1}'): v for k, v in weight_dict.items()})
        losses = ["masks", "labels"] + (["boxes"] if dec_cfg['BOX'] else [])
        if dec_cfg['PART']:
            losses.append('labels_part')
        weight_dict.update({'all': 1.0, 'sam': 1.0, 'pas': 1.0})

        task_switch = {'bbox': dec_cfg.get('DETECTION', True), 'mask': dec_cfg.get('MASK', True)}

        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)
        lang_encoder = build_language_encoder(cfg)
        sem_seg_head = build_semantic_sam_head(cfg, backbone.output_shape(), lang_encoder, extra=extra)

        criterion_mo1 = SetCriterion(
            enc_cfg['NUM_CLASSES'],
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            dn=dn,
            dn_losses=dn_losses,
            panoptic_on=dec_cfg['PANO_BOX_LOSS'],
            semantic_ce_loss=dec_cfg['TEST']['SEMANTIC_ON'] and dec_cfg['SEMANTIC_CE_LOSS'] and not dec_cfg['TEST']['PANOPTIC_ON'],
            num_mask_tokens=dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3)
        )
        criterion_m2m = SetCriterionOsPartWholeM2M(
            enc_cfg['NUM_CLASSES'],
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            dn=dn,
            dn_losses=dn_losses,
            panoptic_on=dec_cfg['PANO_BOX_LOSS'],
            semantic_ce_loss=dec_cfg['TEST']['SEMANTIC_ON'] and dec_cfg['SEMANTIC_CE_LOSS'] and not dec_cfg['TEST']['PANOPTIC_ON'],
            num_mask_tokens=dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3)
        )
        criterion_switch = {'mo1': criterion_mo1, 'm2m': criterion_m2m}

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion_switch": criterion_switch,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['COCO']['TEST']['DETECTIONS_PER_IMAGE'],
            "data_loader": None,
            "focus_on_box": cfg['MODEL']['DECODER']['TEST']['TEST_FOUCUS_ON_BOX'],
            "transform_eval": cfg['MODEL']['DECODER']['TEST']['PANO_TRANSFORM_EVAL'],
            "pano_temp": cfg['MODEL']['DECODER']['TEST']['PANO_TEMPERATURE'],
            "semantic_ce_loss": cfg['MODEL']['DECODER']['TEST']['SEMANTIC_ON'] and cfg['MODEL']['DECODER']['SEMANTIC_CE_LOSS'] and not cfg['MODEL']['DECODER']['TEST']['PANOPTIC_ON'],
            "train_dataset_name": cfg['DATASETS']['TRAIN'],
            "sam_on": dec_cfg.get('SAM', True),
            "regenerate_point": dec_cfg.get('RE_POINT', False),
            "num_mask_tokens": dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3),
            "max_num_instance": dec_cfg.get('MAX_NUM_INSTANCE', 100),
            "classification_on": dec_cfg['TEST'].get('CLASSIFICATION_ON', False),
            "many_to_one": dec_cfg.get('MANY_TO_ONE', False),
        }

    @property
    def device(self):
        return self.pixel_mean.device

    # ---------------- 训练 & 推理入口 ----------------

    def forward(self, batched_inputs, inference_task='seg'):
        if self.training:
            self.criterion = self.criterion_switch['mo1'] if self.many_to_one else self.criterion_switch['m2m']
            self.criterion.num_classes = 1  # 单类肝：类别索引为 0
            prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
            data = batched_inputs if isinstance(batched_inputs, list) else batched_inputs.get('sam', batched_inputs)
            losses = self.forward_seg(data, task='seg', prediction_switch=prediction_switch)
            if isinstance(losses, dict) and len(losses) == 0:
                raise RuntimeError("[GeneralizedMaskDINO] Empty loss dict in training.")
            return losses

        prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
        if inference_task == 'interactive':
            processed_results = self.evaluate_interactive(batched_inputs, task=inference_task, prediction_switch=prediction_switch)
        elif inference_task == 'multi_granularity':
            processed_results = self.evaluate_interactive_granularity(batched_inputs, task=inference_task, prediction_switch=prediction_switch)
        else:
            raise NotImplementedError
        return processed_results

    # ---------------- 训练用分支 ----------------

    def _build_targets_from_mapper_or_semseg(self, batched_inputs_list: List[Dict], images: ImageList) -> List[Dict]:
        """
        当没有 Instances 时，从 dd['targets'] 或 dd['sem_seg'] 构造完整 targets：
        包含 'labels','masks','boxes','pb','points','boxes_dn','box_start',
        以及 matcher 需要的 'level_target_inds'，并补充 'ori_masks'/'ori_boxes' 等。
        """
        h_pad, w_pad = images.tensor.shape[-2:]
        all_targets: List[Dict] = []

        # 和解码器保持一致（例如=3）。如果配置里就是1，也没问题
        num_tokens = int(getattr(self, "num_mask_tokens", 1))
        if num_tokens <= 0:
            num_tokens = 1

        for x in batched_inputs_list:
            H, W = x["image"].shape[-2], x["image"].shape[-1]
            image_size_xyxy = torch.as_tensor([W, H, W, H], dtype=torch.float, device=self.device)

            # 1) 拿到 masks（优先 dd["targets"].masks；否则从 sem_seg==1 取肝脏）
            masks = None
            if "targets" in x and isinstance(x["targets"], dict) and x["targets"].get("masks", None) is not None:
                m = x["targets"]["masks"]
                masks = m if torch.is_tensor(m) else torch.as_tensor(m, device=self.device, dtype=torch.uint8)

            if masks is None:
                sem = x.get("sem_seg", None)
                if sem is not None:
                    sem = sem if torch.is_tensor(sem) else torch.as_tensor(sem, device=self.device)
                    liver = (sem.to(torch.long) == 1)
                    masks = liver.unsqueeze(0).to(torch.uint8) if liver.any() else torch.zeros((0, H, W), dtype=torch.uint8, device=self.device)
                else:
                    masks = torch.zeros((0, H, W), dtype=torch.uint8, device=self.device)

            # 保存 ori_masks，并 pad 到 (h_pad, w_pad)
            N = int(masks.shape[0])
            if N > 0:
                ori_masks = torch.zeros((N, h_pad, w_pad), dtype=masks.dtype, device=self.device)
                ori_masks[:, :H, :W] = masks
            else:
                # 空 tensor 也保持维度一致
                ori_masks = masks.view(0, h_pad, w_pad).to(self.device)

            # 计算每个 GT 的 bbox（xyxy→cxcywh，[0,1]）
            if N > 0:
                boxes_xyxy = []
                for m in ori_masks:
                    ys, xs = torch.nonzero(m, as_tuple=True)
                    if ys.numel() == 0:
                        boxes_xyxy.append(torch.tensor([0, 0, 0, 0], dtype=torch.float, device=self.device))
                    else:
                        x1 = xs.min().float(); y1 = ys.min().float()
                        x2 = xs.max().float(); y2 = ys.max().float()
                        boxes_xyxy.append(torch.stack([x1, y1, x2, y2]))
                boxes_xyxy = torch.stack(boxes_xyxy, 0)
                ori_boxes_cxcywh = box_ops.box_xyxy_to_cxcywh(boxes_xyxy) / image_size_xyxy
            else:
                ori_boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float, device=self.device)

            # 采样/重复到固定长度 self.max_num_instance
            M = int(self.max_num_instance)
            if N == 0:
                # 没 GT：完全关闭 point 分支
                masks_sel = torch.zeros((M, h_pad, w_pad), dtype=torch.float32, device=self.device)
                boxes_sel  = torch.zeros((M, 4), dtype=torch.float32, device=self.device)
                labels_sel = torch.zeros((M,), dtype=torch.long, device=self.device)  # 单类=0
                box_start = 0
                pb = torch.zeros(M, dtype=torch.float32, device=self.device)
                points = torch.zeros((M, 4), dtype=torch.float32, device=self.device)
                level_target_inds: List[List[int]] = []  # 空
                max_num_tgt_per_click = 0
            else:
                index = torch.randperm(N, device=self.device)
                if M > N:
                    rep = int(M / N) + 1
                    index = index.repeat(rep)
                index = index[:M]  # [M]，元素范围一定在 [0, N-1]

                masks_sel = ori_masks[index].float()                    # [M,h_pad,w_pad]
                masks_sel = (masks_sel > 0.5).float()                   # 二值化
                boxes_sel  = ori_boxes_cxcywh[index]                    # [M,4]
                labels_sel = torch.zeros((M,), dtype=torch.long, device=self.device)

                # 全部 token 作为 point
                box_start = M
                pb = torch.ones(M, dtype=torch.float32, device=self.device)
                points = boxes_sel.clone()

                # ——关键修复：安全构造 level_target_inds（重复到 num_tokens，并 clamp 到 [0, N-1]）——
                idx_list = index.tolist()
                def clamp_i(i: int) -> int:
                    # 保险：避免任何超界
                    return 0 if N == 0 else int(min(max(i, 0), N - 1))

                level_target_inds = [[clamp_i(i)] * num_tokens for i in idx_list]
                max_num_tgt_per_click = num_tokens

            boxes_dn = torch.cat([points, boxes_sel[box_start:]], 0)  # 若 box_start==M，则=points

            # 可选的健壮性检查（首个 iteration 看一下，确定没越界就可以注释掉）
            # if N > 0 and len(level_target_inds) > 0:
            #     flat_inds = [j for inds in level_target_inds for j in inds]
            #     assert all(0 <= j < N for j in flat_inds), f"level_target_inds 越界: max={max(flat_inds)}, N={N}"

            all_targets.append({
                "labels": labels_sel,                 # [M]
                "masks": masks_sel,                  # [M,h_pad,w_pad] float
                "boxes": boxes_sel,                  # [M,4] cxcywh in [0,1]

                "pb": pb,                            # [M] 1=point, 0=box
                "points": points,                    # [M,4]（这里用 box 作为点窗口的占位，和原始实现一致）
                "boxes_dn": boxes_dn,                # [M + (M-box_start), 4]
                "box_start": box_start,

                "level_target_inds": level_target_inds,   # List[List[int]]，长度=M，每个长度=num_tokens
                "max_num_tgt_per_click": max_num_tgt_per_click,

                "ori_masks": ori_masks,              # [N,h_pad,w_pad]
                "ori_boxes": ori_boxes_cxcywh,       # [N,4]
                "ori_mask_num": int(N),
            })

        return all_targets


    def forward_seg(self, batched_inputs, task='seg', prediction_switch={'part': True, 'whole': True, 'seg': True, 'det': True}):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets_interactive(gt_instances, images, prediction_switch=prediction_switch)
            else:
                targets = self._build_targets_from_mapper_or_semseg(batched_inputs, images)

            outputs, mask_dict = self.sem_seg_head(features, targets=targets, task=task, extra=prediction_switch)
            losses = self.criterion(outputs, targets, mask_dict, task=task, extra=prediction_switch)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses

    # ---------------- 其余函数保持你的原逻辑 ----------------

    def evaluate_demo(self, batched_inputs,all_whole=None,all_part=None,mask_features=None,multi_scale_features=None,return_features=False, level=[0,1,2,3,4,5]):
        assert len(batched_inputs) == 1, "only support batch size equal to 1"
        prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        targets = batched_inputs[0]['targets']
        height = images[0].shape[1]
        width = images[0].shape[2]
        padded_h = images.tensor.shape[-2]
        padded_w = images.tensor.shape[-1]

        targets[0]['points'] = targets[0]['points'] * torch.as_tensor([width, height, width, height], dtype=torch.float,
                                                                      device=self.device) / torch.as_tensor(
            [padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)

        if mask_features is None or multi_scale_features is None:
            features = self.backbone(images.tensor)
            mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(
                features, None)
        outputs, mask_dict = self.sem_seg_head.predictor(multi_scale_features, mask_features, None, targets=targets,
                                     target_queries=None, target_vlp=None, task='demo', extra=prediction_switch)
        pred_ious=None
        if 'pred_ious' in outputs.keys():
            pred_ious = outputs["pred_ious"]

        mask_pred_results = outputs["pred_masks"].view(pred_ious.shape[0], pred_ious.shape[1], pred_ious.shape[2], outputs["pred_masks"].shape[-2], outputs["pred_masks"].shape[-1])
        level = torch.tensor(level).to(mask_pred_results.device)
        mask_pred_results = torch.index_select(mask_pred_results, 2, level).flatten(1, 2)
        pred_ious = torch.index_select(pred_ious, -1, level)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        pred_masks = mask_pred_results[0]
        image_size = images.image_sizes[0]
        height = image_size[0]
        width = image_size[1]
        if self.sem_seg_postprocess_before_inference:
            pred_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                pred_masks, image_size, height, width
            )
        if return_features:
            return pred_masks,pred_ious,mask_features,multi_scale_features
        else:
            return pred_masks,pred_ious

    def evaluate_interactive(self, batched_inputs, task='seg', prediction_switch={'part': True, 'whole': True, 'seg': True, 'det': True}, oracle=True):
        assert len(batched_inputs) == 1, "only support batch size equal to 1"
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets_ = self.prepare_targets_interactive(gt_instances, images, prediction_switch=prediction_switch)
        outputs_, mask_dict = self.sem_seg_head(features, targets=targets_, task=task, extra=prediction_switch)
        outputs = {}
        targets = {}
        outputs['point'] = outputs_
        targets['point'] = targets_
        processed_results_all = {}
        for key in outputs.keys():
            num_tokens = self.num_mask_tokens
            all_batch_shape_iou = []
            if 'pred_ious' in outputs[key].keys():
                pred_ious = outputs[key]["pred_ious"]
            mask_pred_results = outputs[key]["pred_masks"]
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            gt_masks = targets[key][0]["masks"]
            pred_masks = mask_pred_results[0]
            image_size = images.image_sizes[0]
            height = image_size[0]
            width = image_size[1]
            if self.sem_seg_postprocess_before_inference:
                pred_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                    pred_masks, image_size, height, width
                )
            scores = pred_ious.view(-1, num_tokens)
            score, index = scores.max(1)
            pred_masks_max = torch.gather(pred_masks.view(-1, num_tokens, pred_masks.shape[-2], pred_masks.shape[-1]), 1,
                                      index[:, None, None, None].repeat(1, 1, pred_masks.shape[-2],
                                                                        pred_masks.shape[-1])).squeeze(1)
            pred_masks_max = pred_masks_max > 0
            all_batch_shape_iou += [get_iou(gt_masks, pred_masks_max)]
            all_batch_shape_iou = torch.stack(all_batch_shape_iou)
            processed_results = [{"mask_iou": all_batch_shape_iou[:,i]} for i in range(len(all_batch_shape_iou[0]))]
            all_batch_shape_iou = []
            gt_masks_repeat = gt_masks.repeat_interleave(num_tokens, 0)
            iou_all = get_iou(gt_masks_repeat, pred_masks > 0)
            selected_ious, index = iou_all.view(-1, num_tokens).max(1)
            all_batch_shape_iou += [selected_ious]
            all_batch_shape_iou = torch.stack(all_batch_shape_iou)
            processed_results = {'oracle': [{"mask_iou": all_batch_shape_iou[:,i]} for i in range(len(all_batch_shape_iou[0]))], 'max': processed_results}
            processed_results_all = processed_results
        return processed_results_all

    def evaluate_interactive_granularity(self, batched_inputs, task='seg',
                             prediction_switch={'part': True, 'whole': True, 'seg': True, 'det': True}, oracle=True):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets_ = self.prepare_targets_sam_eval(gt_instances, images, prediction_switch=prediction_switch)
        outputs_, mask_dict = self.sem_seg_head(features, targets=targets_, task=task, extra=prediction_switch)
        outputs = {}
        targets = {}
        outputs['point'] = outputs_
        targets['point'] = targets_
        processed_results_all = {}
        for key in outputs.keys():
            all_batch_shape_iou = []
            mask_pred_results = outputs[key]["pred_masks"]
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks = mask_pred_results[0]
            image_size = images.image_sizes[0]
            height = image_size[0]
            width = image_size[1]
            if self.sem_seg_postprocess_before_inference:
                pred_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                    pred_masks, image_size, height, width
                )
            outputs_without_aux = {k: v for k, v in outputs[key].items() if k != "aux_outputs"}
            match_cost = ["cls", "box", "mask"]
            matcher = self.criterion_switch['m2m'].matcher
            criterion = self.criterion_switch['m2m']
            indices = matcher(outputs_without_aux, targets[key], match_cost, extra=prediction_switch)
            src_idx = criterion._get_src_permutation_idx(indices)
            tgt_idx = criterion._get_tgt_permutation_idx(indices)
            src_masks = pred_masks.unsqueeze(0)
            src_masks = src_masks[src_idx]
            level_target_inds = targets[key][0]['level_target_inds']
            ori_masks = targets[key][0]["ori_masks"].to(src_masks)
            ori_gt_masks = [torch.stack([ori_masks[ind] for inds in level_target_inds for ind in inds])]
            target_masks = ori_gt_masks[0].unsqueeze(0)[tgt_idx]
            all_batch_shape_iou += [get_iou(target_masks>0, src_masks>0)]
            all_batch_shape_iou = torch.stack(all_batch_shape_iou)
            processed_results = [{"mask_iou": all_batch_shape_iou[:, i]} for i in range(len(all_batch_shape_iou[0]))]
            processed_results_all = processed_results
        return processed_results_all

    # ------- 原有 prepare_targets_interactive 保持不变（略） -------
    def prepare_targets_interactive(self, targets, images, prediction_switch, task='seg'):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        box_start = random.randint(int((self.max_num_instance - 1)/2), self.max_num_instance - 1)
        for targets_per_image in targets:
            gt_boxes = targets_per_image.gt_boxes if torch.is_tensor(targets_per_image.gt_boxes) else targets_per_image.gt_boxes.tensor
            h, w = targets_per_image.image_size
            if not self.training:
                h_pad, w_pad = h, w
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_masks = targets_per_image.gt_masks if torch.is_tensor(targets_per_image.gt_masks) else targets_per_image.gt_masks.tensor
            if not self.training:
                max_num_instance_ori = self.max_num_instance
                self.max_num_instance = len(gt_masks)
                box_start = self.max_num_instance
            if len(gt_masks)==0:
                new_targets.append({
                    'boxes': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    'points': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    'boxes_dn': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    "pb": torch.cat([torch.ones(box_start), torch.zeros(self.max_num_instance - box_start)], 0),
                    'box_start': box_start
                })
                if not self.training:
                    self.max_num_instance = max_num_instance_ori
                continue
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            num_mask = targets_per_image.gt_classes.shape[0]
            index = torch.randperm(num_mask)
            if num_mask==0:
                print("wrong empty image! argets_per_image.gt_classes.shape[0] ", targets_per_image.gt_classes.shape[0], "targets_per_image", targets_per_image)
            if self.max_num_instance > num_mask:
                rep = 0 if num_mask==0 else int(self.max_num_instance/num_mask) + 1
                index = index.repeat(rep)
            index = index[:self.max_num_instance]
            box_start = self.max_num_instance
            level_target_inds = []
            if self.regenerate_point and box_start>0:
                point_coords = []
                for i in range(box_start):
                    mask = gt_masks[index[i]].clone()
                    center_point = True
                    if not self.training and center_point:
                        mask = mask[None, None, :]
                        n, _, h, w = mask.shape
                        mask_dt = (distance_transform((~F.pad(mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:, :, 1:-1, 1:-1])
                        selected_point = torch.tensor([mask_dt.argmax()/w, mask_dt.argmax()%w]).long().cuda().flip(0)
                    else:
                        candidate_indices = mask.nonzero()
                        if len(candidate_indices)==0:
                            print('wrong')
                            selected_point = torch.tensor([0, 0]).cuda()
                        else:
                            selected_index = random.randint(0, len(candidate_indices)-1)
                            selected_point = candidate_indices[selected_index].flip(0)
                        if not prediction_switch['whole'] and not prediction_switch['part']:
                            level_target_ind = []
                            for ind, m in enumerate(gt_masks):
                                if m[tuple(selected_point.flip(0))]:
                                    level_target_ind.append(ind)
                            assert len(level_target_ind) > 0, "each point must have at least one target"
                            if len(level_target_ind)>self.num_mask_tokens:
                                random.shuffle(level_target_ind)
                                level_target_ind = level_target_ind[:self.num_mask_tokens]
                            level_target_inds.append(level_target_ind)
                    selected_point = torch.cat([selected_point-3, selected_point+3], 0)
                    point_coords.append(selected_point)
                point_coords = torch.stack(point_coords).to('cuda')
            else:
                point_coords = targets_per_image.gt_boxes.tensor[index[:box_start]]
            max_num_tgt_per_click = -1
            if len(level_target_inds)>0:
                num_tgt = [len(l) for l in level_target_inds]
                max_num_tgt_per_click = max(num_tgt)
                if max_num_tgt_per_click>5:
                    if dist.get_rank() == 0:
                        print("max number of levels ", max(num_tgt))
            new_target={
                    "ori_mask_num": len(targets_per_image.gt_classes),
                    "level_target_inds": level_target_inds,
                    "max_num_tgt_per_click": max_num_tgt_per_click,
                    "labels": targets_per_image.gt_classes[index] if prediction_switch['whole'] else None,
                    "masks": padded_masks[index],
                    "ori_masks": padded_masks,
                    "boxes":box_ops.box_xyxy_to_cxcywh(gt_boxes[index])/image_size_xyxy,
                    "ori_boxes":box_ops.box_xyxy_to_cxcywh(gt_boxes)/image_size_xyxy,
                    "points":box_ops.box_xyxy_to_cxcywh(point_coords)/image_size_xyxy,
                    "pb": torch.cat([torch.ones(box_start), torch.zeros(self.max_num_instance - box_start)], 0),
                    "gt_whole_classes": targets_per_image.gt_whole_classes[index] if targets_per_image.has('gt_whole_classes') and prediction_switch['whole'] else None,
                    "gt_part_classes": targets_per_image.gt_part_classes[index] if targets_per_image.has('gt_part_classes') and prediction_switch['part'] else None,
                }
            if prediction_switch['whole'] and not prediction_switch['part']:
                new_target['gt_whole_classes'] = targets_per_image.gt_classes[index]
            if not self.training:
                self.max_num_instance = max_num_instance_ori
                new_target["pb"]=torch.zeros_like(new_target["pb"])
                height = images[0].shape[1]
                width = images[0].shape[2]
                padded_h = images.tensor.shape[-2]
                padded_w = images.tensor.shape[-1]
                new_target["boxes_dn_ori"] = torch.cat([new_target["points"].clone(), new_target["boxes"][box_start:].clone()], 0)
                new_target['points'] = new_target['points'] * torch.as_tensor([width, height, width, height], dtype=torch.float, device=self.device)/torch.as_tensor([padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)
                new_target['boxes'] = new_target['boxes'] * torch.as_tensor([width, height, width, height], dtype=torch.float, device=self.device)/torch.as_tensor([padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)
            new_target["boxes_dn"] = torch.cat([new_target["points"], new_target["boxes"][box_start:]], 0)
            new_target['box_start'] = box_start
            new_targets.append(new_target)
        return new_targets


@register_model
def get_segmentation_model(cfg, **kwargs):
    return GeneralizedMaskDINO(cfg)
