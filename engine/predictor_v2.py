"""统一的推理器"""
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, TYPE_CHECKING
from engine.base import BaseTask

# 延迟导入避免循环依赖
if TYPE_CHECKING:
    from engine.predict import Boxes, Results


class Predictor(BaseTask):
    """统一的推理器"""

    def __init__(self, model, cfg):
        """
        Args:
            model: YOLOv11 模型
            cfg: 配置字典
        """
        super().__init__(model, cfg)
        self.model.eval()

    def predict_single(self, img: np.ndarray):
        """对单张图像执行预测

        Args:
            img: 原始图像 (H, W, C) BGR 格式

        Returns:
            Results 对象
        """
        from engine.predict import Results  # 延迟导入

        orig_img = img.copy()
        orig_h, orig_w = orig_img.shape[:2]

        # 预处理
        img_tensor, preprocess_params = self.preprocessor(img)
        img_tensor = img_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # 后处理
        result_dict = self.postprocessor(
            predictions,
            (orig_h, orig_w),
            preprocess_params
        )

        # 创建 Results 对象
        results = Results(
            orig_img=orig_img,
            boxes=result_dict['boxes'],
            scores=result_dict['scores'],
            labels=result_dict['labels'],
            names=self.cfg.get('names', {})
        )

        return results


__all__ = ['Predictor']
