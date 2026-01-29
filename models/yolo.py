"""Ultralytics 风格的统一 YOLO 接口"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Union, List, Dict, Optional
from models.yolov11 import YOLOv11
from engine.predictor_v2 import Predictor


class YOLO:
    """Ultralytics 风格的统一 YOLO 接口

    Example:
        >>> model = YOLO('yolo11n.yaml')
        >>> model.train(data='coco.yaml', epochs=100)
        >>> results = model.predict('image.jpg')
    """

    def __init__(self, model: Union[str, Path, nn.Module]):
        """
        Args:
            model: 模型配置文件、权重文件或模型实例
        """
        if isinstance(model, (str, Path)):
            model = str(model)
            if model.endswith('.yaml'):
                self.model = self._build_from_yaml(model)
                self.model_name = Path(model).stem
            elif model.endswith('.pt'):
                self.model, self.cfg = self._load_from_weights(model)
                self.model_name = Path(model).stem
            else:
                raise ValueError(f"不支持的模型格式: {model}")
        elif isinstance(model, nn.Module):
            self.model = model
            self.model_name = model.__class__.__name__
            self.cfg = {}
        else:
            raise TypeError(f"不支持的模型类型: {type(model)}")

        self.predictor = None

    def _build_from_yaml(self, yaml_path: str) -> nn.Module:
        """从 YAML 配置文件构建模型"""
        # 简化实现：默认创建 YOLOv11n
        # 实际应该解析 YAML 文件
        return YOLOv11(nc=80, scale='n')

    def _load_from_weights(self, weights_path: str):
        """从权重文件加载模型"""
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            cfg = checkpoint.get('cfg', {})
        else:
            state_dict = checkpoint
            cfg = {}

        # 推断参数
        nc = self._infer_nc(state_dict)
        scale = self._infer_scale(state_dict)

        model = YOLOv11(nc=nc, scale=scale)
        model.load_state_dict(state_dict, strict=False)

        return model, cfg

    def _infer_nc(self, state_dict: dict) -> int:
        """从状态字典推断类别数量"""
        for key in state_dict.keys():
            if "detect.cv3.0.weight" in key:
                return state_dict[key].shape[0]
        return 80

    def _infer_scale(self, state_dict: dict) -> str:
        """从状态字典推断模型 scale"""
        num_params = sum(p.numel() for p in state_dict.values())
        if num_params < 3e6:
            return "n"
        elif num_params < 15e6:
            return "s"
        elif num_params < 25e6:
            return "m"
        else:
            return "l"

    def predict(
        self,
        source,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        **kwargs
    ) -> List:
        """推理

        Args:
            source: 输入源（图像路径、目录、numpy 数组等）
            conf: 置信度阈值
            iou: NMS IoU 阈值

        Returns:
            Results 对象列表
        """
        # 合并配置
        cfg = self.cfg.copy()
        if conf is not None:
            cfg['conf'] = conf
        if iou is not None:
            cfg['iou'] = iou
        cfg.update(kwargs)

        # 创建或更新 predictor
        if self.predictor is None or cfg != self.predictor.cfg:
            self.predictor = Predictor(self.model, cfg)

        # 处理不同输入类型
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_file():
                img = self._load_image(source)
                return [self.predictor.predict_single(img)]
            # TODO: 处理目录、视频等
        elif isinstance(source, np.ndarray):
            return [self.predictor.predict_single(source)]
        else:
            raise TypeError(f"不支持的 source 类型: {type(source)}")

    def _load_image(self, img_path: Path) -> np.ndarray:
        """加载图像"""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        return img

    def __call__(self, source, **kwargs):
        """便捷调用"""
        return self.predict(source, **kwargs)

    @property
    def nc(self) -> int:
        """类别数量"""
        # 从模型推断
        if hasattr(self.model, 'nc'):
            return self.model.nc
        return 80


__all__ = ['YOLO']
