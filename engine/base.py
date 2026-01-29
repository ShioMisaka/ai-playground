"""任务处理器基类"""
import torch
from engine.preprocessor import Preprocessor
from engine.postprocessor import Postprocessor


class BaseTask:
    """所有任务处理器的基类

    提供统一的预处理和后处理功能
    """

    def __init__(self, model, cfg):
        """
        Args:
            model: YOLOv11 模型
            cfg: 配置字典
        """
        self.model = model
        self.cfg = cfg

        # 创建预处理器
        self.preprocessor = Preprocessor(
            img_size=cfg.get('img_size', 640),
            letterbox=cfg.get('letterbox', True),
            auto=cfg.get('auto', False)
        )

        # 创建后处理器
        self.postprocessor = Postprocessor(
            conf_threshold=cfg.get('conf', 0.25),
            iou_threshold=cfg.get('iou', 0.45)
        )

        # 设备
        self.device = torch.device(cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)

    def _get_model_predictions(self, imgs, targets=None):
        """获取模型预测

        Args:
            imgs: 输入图像张量
            targets: 标签（可选，用于训练）

        Returns:
            如果 targets 不为 None: 返回 (loss, loss_items, predictions)
            如果 targets 为 None: 返回 predictions
        """
        if targets is not None:
            # 训练模式
            result = self.model(imgs, targets)
            if isinstance(result, dict):
                return result['loss'], result['loss_items'], result['predictions']
            return result
        else:
            # 推理模式
            return self.model(imgs)


__all__ = ['BaseTask']
