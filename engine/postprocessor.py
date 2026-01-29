"""统一的后处理类

包括：NMS、置信度过滤、坐标映射
"""
import torch
import numpy as np
import torchvision.ops as ops
from typing import Dict, Tuple


class Postprocessor:
    """统一的后处理类

    包括：NMS、置信度过滤、坐标映射
    """

    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Args:
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def _scale_coords(
        self,
        coords: np.ndarray,
        orig_shape: Tuple[int, int],
        preprocess_params: Dict
    ) -> np.ndarray:
        """将坐标从预处理空间映射回原图像空间

        Args:
            coords: (N, 4) 边界框坐标 [cx, cy, w, h]
            orig_shape: 原始图像形状 (H, W)
            preprocess_params: 预处理参数

        Returns:
            (N, 4) 映射回原图空间的坐标
        """
        coords = coords.copy()

        if preprocess_params['letterbox']:
            # Letterbox 逆运算
            ratio, (pad_x, pad_y) = preprocess_params['ratio'], preprocess_params['pad']
            coords[:, 0] = (coords[:, 0] - pad_x) / ratio  # cx
            coords[:, 1] = (coords[:, 1] - pad_y) / ratio  # cy
            coords[:, 2] = coords[:, 2] / ratio  # w
            coords[:, 3] = coords[:, 3] / ratio  # h
        else:
            # 简单 resize 逆运算
            scale_x = preprocess_params['scale_x']
            scale_y = preprocess_params['scale_y']
            coords[:, 0] = coords[:, 0] / scale_x  # cx
            coords[:, 1] = coords[:, 1] / scale_y  # cy
            coords[:, 2] = coords[:, 2] / scale_x  # w
            coords[:, 3] = coords[:, 3] / scale_y  # h

        return coords

    def __call__(
        self,
        predictions: torch.Tensor,
        orig_shape: Tuple[int, int],
        preprocess_params: Dict
    ) -> Dict:
        """后处理模型输出

        Args:
            predictions: (bs, n_anchors, 4+nc) 预测输出，格式为 [cx, cy, w, h, cls1, cls2, ...]
            orig_shape: 原始图像尺寸 (H, W)
            preprocess_params: 预处理参数（用于坐标映射）

        Returns:
            dict: {
                'boxes': (N, 4) xyxy 格式边界框
                'scores': (N,) 置信度分数
                'labels': (N,) 类别索引
            }
        """
        # 单图像处理
        pred = predictions[0]  # (n_anchors, 4+nc)

        # 提取 bbox (cx, cy, w, h)
        boxes = pred[:, :4]  # (n_anchors, 4)

        # 提取类别分数并取最大值
        cls_scores = pred[:, 4:]  # (n_anchors, nc)
        scores, labels = torch.max(cls_scores, dim=1)

        # 置信度过滤
        mask = scores > self.conf_threshold
        if mask.sum() == 0:
            return {
                'boxes': np.empty((0, 4), dtype=np.float32),
                'scores': np.empty(0, dtype=np.float32),
                'labels': np.empty(0, dtype=np.int64)
            }

        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # 转换从 cxcywh 到 x1y1x2y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        # NMS
        keep = ops.nms(boxes_xyxy, scores, self.iou_threshold)

        # 保留 NMS 后的结果
        boxes_xyxy = boxes_xyxy[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        labels = labels[keep].cpu().numpy()

        # 坐标映射：从预处理空间映射回原图空间
        # 需要先转回 cxcywh 格式，映射后再转回 xyxy
        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
        w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        boxes_cxcywh = np.stack([cx, cy, w, h], axis=1)
        boxes_cxcywh = self._scale_coords(boxes_cxcywh, orig_shape, preprocess_params)

        # 转回 xyxy
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        return {
            'boxes': boxes_xyxy.astype(np.float32),
            'scores': scores.astype(np.float32),
            'labels': labels.astype(np.int64)
        }


__all__ = ['Postprocessor']
