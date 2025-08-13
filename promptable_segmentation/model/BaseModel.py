import os
import logging

import torch
import torch.nn as nn

from utils.model import align_and_update_state_dicts

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    def _to_tensor(self, x, device=None):
        """
        将各种标量/张量统一成 torch.Tensor。
        注意：如果上游对 loss 做过 .item()，这里会失去梯度，只能当常数参与反传。
        建议后续把模型里的 .item() 去掉以恢复梯度。
        """
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def forward(self, *inputs, **kwargs):
        """
        训练阶段：将模型输出规整为 Detectron2 期望的 dict[str, Tensor]
        推理阶段：保持原样返回
        """
        outputs = self.model(*inputs, **kwargs)

        if self.training:
            # 训练时做统一适配
            try:
                device = next(self.model.parameters()).device
            except Exception:
                device = None

            # 1) 已是 dict：把 value 全部转成 Tensor
            if isinstance(outputs, dict):
                return {k: self._to_tensor(v, device) for k, v in outputs.items()}

            # 2) (loss_dict, others) / [loss_dict, ...]：取第一个 dict 作为 loss_dict
            if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], dict):
                loss_dict = outputs[0]
                return {k: self._to_tensor(v, device) for k, v in loss_dict.items()}

            # 3) 单个标量：打包成 {"loss_total": tensor}
            if torch.is_tensor(outputs) or isinstance(outputs, (float, int)):
                return {"loss_total": self._to_tensor(outputs, device)}

            # 4) 其它可迭代：把其中的张量相加为总损失（兜底）
            if isinstance(outputs, (tuple, list)):
                tensors = [o for o in outputs if torch.is_tensor(o)]
                if tensors:
                    return {"loss_total": torch.stack([t.float() for t in tensors]).sum()}
                # 或者里面是若干 float
                try:
                    s = sum(float(o) for o in outputs)
                    return {"loss_total": self._to_tensor(s, device)}
                except Exception:
                    pass
            if self.training and isinstance(outputs, dict) and not getattr(self, "_printed_loss_keys", False):
                logger.info(f"[BaseModel] loss keys: {list(outputs.keys())}")
                self._printed_loss_keys = True


            raise ValueError(f"[BaseModel] Unsupported model output type in training: {type(outputs)}")

        # 推理阶段：原样返回
        return outputs

    def save_pretrained(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def from_pretrained(self, load_dir):
        state_dict = torch.load(load_dir, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self
