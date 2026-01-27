"""
指数移动平均 (EMA) 模块

提供 ModelEMA 类，用于维护模型权重的指数移动平均，
从而获得更平滑、更稳定的模型权重。
"""
import copy
import math
import torch
import torch.nn as nn


class ModelEMA:
    """模型指数移动平均 (Exponential Moving Average)

    通过对历史权重的加权平均，获得更稳定的模型权重。
    通常用于目标检测任务，可以显著提升 mAP 并减少曲线震荡。

    Args:
        model: 要进行 EMA 的 PyTorch 模型
        decay: EMA 衰减系数（默认 0.9999）
        updates: 初始更新步数（默认 0）

    Example:
        >>> ema = ModelEMA(model, decay=0.9999)
        >>> for batch in dataloader:
        >>>     loss = model(imgs, targets)
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     ema.update(model)  # 更新 EMA
        >>>
        >>> # 验证时使用 EMA 模型
        >>> val_metrics = validate(ema.ema, val_loader, ...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, updates: int = 0):
        # 深拷贝模型作为 EMA 模型
        self.ema = copy.deepcopy(model)
        self.ema.eval()  # EMA 模型始终处于评估模式
        self.updates = updates  # 更新步数

        # 固定 EMA 参数，不参与梯度计算
        for p in self.ema.parameters():
            p.requires_grad_(False)

        # 初始衰减系数（实际使用时会动态调整）
        self.decay = decay

        # 存储设备信息，确保 EMA 模型与原模型在同一设备上
        self.device = next(model.parameters()).device
        self.ema.to(self.device)

    def update(self, model: nn.Module):
        """更新 EMA 权重

        核心公式：
            ema_weight = ema_weight * decay + current_weight * (1 - decay)

        decay 会随着 updates 动态调整：
            decay = min(initial_decay, (1 + updates) / (10 + updates))

        Args:
            model: 当前训练模型
        """
        self.updates += 1

        # 动态调整 decay：从小逐渐增大到初始值
        # 这有助于 EMA 在训练早期更快地适应
        d = self.decay * (1 - math.exp(-self.updates / 2000))

        with torch.no_grad():
            # 获取当前模型的状态字典
            msd = model.state_dict()

            # 更新每个参数
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # 只更新浮点参数
                    # ema = ema * decay + current * (1 - decay)
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)

            # 复制 buffer（如 BatchNorm 的 running_mean/running_var）
            for k, v in self.ema.named_buffers():
                if k in msd:
                    v.copy_(msd[k])

    def forward(self, *args, **kwargs):
        """EMA 模型的前向传播"""
        return self.ema(*args, **kwargs)

    def state_dict(self):
        """返回 EMA 模型的状态字典"""
        return {
            'ema': self.ema.state_dict(),
            'updates': self.updates,
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict):
        """加载 EMA 模型的状态字典"""
        self.ema.load_state_dict(state_dict['ema'])
        self.updates = state_dict.get('updates', 0)
        self.decay = state_dict.get('decay', 0.9999)

    def to(self, device):
        """移动 EMA 模型到指定设备"""
        self.ema.to(device)
        self.device = device
        return self
