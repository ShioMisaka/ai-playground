"""统一的图像预处理类

确保训练、验证、推理使用相同的预处理流程
"""
import cv2
import torch
import numpy as np
from typing import Tuple, Dict, Union


class Preprocessor:
    """统一的图像预处理类

    核心原则：训练、验证、推理必须使用相同的预处理流程
    """

    def __init__(self, img_size: int = 640, letterbox: bool = True, auto: bool = False):
        """
        Args:
            img_size: 目标图像尺寸
            letterbox: True=letterbox(保持长宽比), False=简单resize
            auto: 动态模式（目标尺寸由输入决定）
        """
        self.img_size = img_size
        self.letterbox = letterbox
        self.auto = auto

    def _letterbox(
        self,
        img: np.ndarray,
        target_size: int = 640,
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, Tuple[float, Tuple[float, float]]]:
        """Letterbox 预处理：保持长宽比的缩放 + 填充

        Args:
            img: 输入图像 (H, W, C) BGR 格式
            target_size: 目标尺寸（正方形边长）
            color: 填充颜色 (B, G, R)

        Returns:
            (transformed_img, (ratio, (pad_x, pad_y)))
        """
        img_h, img_w = img.shape[:2]

        # 动态模式
        if self.auto:
            target_size = max(img_h, img_w)

        # 计算缩放比例
        r = min(target_size / img_h, target_size / img_w)
        scaled_h, scaled_w = int(round(img_h * r)), int(round(img_w * r))

        # 计算填充
        pad_h = (target_size - scaled_h) / 2
        pad_w = (target_size - scaled_w) / 2

        # 缩放图像
        resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # 填充
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )

        return padded, (r, (pad_w, pad_h))

    def _simple_resize(
        self,
        img: np.ndarray,
        target_size: int = 640
    ) -> Tuple[np.ndarray, Dict]:
        """简单 resize 预处理

        Args:
            img: 输入图像 (H, W, C) BGR 格式
            target_size: 目标尺寸

        Returns:
            (resized_img, params_dict)
        """
        img_h, img_w = img.shape[:2]
        resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        params = {
            'scale_x': target_size / img_w,
            'scale_y': target_size / img_h,
            'pad_x': 0.0,
            'pad_y': 0.0
        }

        return resized, params

    def __call__(
        self,
        img: np.ndarray
    ) -> Tuple[torch.Tensor, Dict]:
        """预处理单张图像

        Args:
            img: numpy array (H, W, C) BGR 格式

        Returns:
            tensor: (1, 3, H, W) 归一化后的张量
            params: 预处理参数字典（用于坐标映射）
        """
        orig_h, orig_w = img.shape[:2]

        # 1. Letterbox 或简单 resize
        if self.letterbox:
            transformed, (ratio, (pad_x, pad_y)) = self._letterbox(img, self.img_size)
            params = {
                'orig_shape': (orig_h, orig_w),
                'letterbox': True,
                'ratio': ratio,
                'pad': (pad_x, pad_y)
            }
        else:
            transformed, resize_params = self._simple_resize(img, self.img_size)
            params = {
                'orig_shape': (orig_h, orig_w),
                'letterbox': False,
                'scale_x': resize_params['scale_x'],
                'scale_y': resize_params['scale_y'],
                'pad': (0.0, 0.0)
            }

        # 2. BGR -> RGB (如果需要，保持与推理一致)
        # 当前推理代码不转换，所以这里也不转换

        # 3. HWC -> CHW
        transformed = transformed.transpose(2, 0, 1)

        # 4. 归一化到 [0, 1]
        transformed = transformed.astype(np.float32) / 255.0

        # 5. 转为 tensor 并添加 batch 维度
        img_tensor = torch.from_numpy(transformed).unsqueeze(0)

        return img_tensor, params


# 导出
__all__ = ['Preprocessor']
