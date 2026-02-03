"""YOLOv11 预测工具模块

提供预测所需的工具类和函数：
- LetterBox: 图像预处理
- _scale_coords: 坐标映射
- _post_process: NMS 后处理
- Boxes: 边界框容器
- Results: 预测结果容器
- ULTRALYTICS_COLORS: 专业调色板

完整的 YOLO 预测接口位于 models/yolo.py。

Note: This module was renamed from predict.py to predictor.py for naming
consistency with trainer.py (both use -er suffix pattern).
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, List, Dict
import torch
import torchvision.ops as ops


class LetterBox:
    """Letterbox 预处理：保持长宽比的缩放 + 填充

    Example:
        >>> letterbox = LetterBox()
        >>> img, params = letterbox(cv2.imread("test.jpg"), target_size=640)
        >>> # params: (ratio, (pad_x, pad_y))
    """

    def __init__(self, auto: bool = False):
        """
        Args:
            auto: True 时目标尺寸由输入图像的最大边决定（动态模式）
                  False 时固定为 target_size x target_size（固定模式）
        """
        self.auto = auto

    def __call__(
        self,
        img: np.ndarray,
        target_size: int = 640,
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, Tuple[float, Tuple[float, float]]]:
        """对图像应用 letterbox 变换

        Args:
            img: 输入图像 (H, W, C) BGR 格式
            target_size: 目标尺寸（正方形边长）
            color: 填充颜色 (B, G, R)

        Returns:
            (transformed_img, (ratio, (pad_x, pad_y)))
            - transformed_img: 变换后的图像 (target_size, target_size, C)
            - ratio: 缩放比例
            - pad_x, pad_y: x 和 y 方向的填充像素数
        """
        img_h, img_w = img.shape[:2]

        # 动态模式：target_size 是最长边
        if self.auto:
            target_size = max(img_h, img_w)

        # 计算缩放比例（保持长宽比）
        r = min(target_size / img_h, target_size / img_w)

        # 计算缩放后的尺寸
        scaled_h, scaled_w = int(round(img_h * r)), int(round(img_w * r))

        # 计算填充
        pad_h = (target_size - scaled_h) / 2
        pad_w = (target_size - scaled_w) / 2

        # 缩放图像
        resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # 上下左右填充
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )

        return padded, (r, (pad_w, pad_h))


def _scale_coords(
    coords: np.ndarray,
    orig_shape: Tuple[int, int],
    ratio: float,
    pad: Tuple[float, float]
) -> np.ndarray:
    """将坐标从 letterbox 空间映射回原图像空间

    Args:
        coords: (N, 4) 边界框坐标 [cx, cy, w, h]（在 letterbox 空间）
        orig_shape: 原始图像形状 (H, W)
        ratio: letterbox 缩放比例（简单 resize 时为 ratio_w）
        pad: (pad_x, pad_y) letterbox 填充

    Returns:
        (N, 4) 映射回原图空间的坐标 [cx, cy, w, h]
    """
    orig_h, orig_w = orig_shape
    pad_x, pad_y = pad

    coords = coords.copy()

    is_simple_resize = (pad_x == 0 and pad_y == 0)

    if is_simple_resize:
        target_size = ratio * orig_w
        scale_x = orig_w / target_size
        scale_y = orig_h / target_size

        coords[:, 0] = coords[:, 0] * scale_x  # cx
        coords[:, 1] = coords[:, 1] * scale_y  # cy
        coords[:, 2] = coords[:, 2] * scale_x  # w
        coords[:, 3] = coords[:, 3] * scale_y  # h
    else:
        coords[:, 0] = (coords[:, 0] - pad_x) / ratio  # cx
        coords[:, 1] = (coords[:, 1] - pad_y) / ratio  # cy
        coords[:, 2] = coords[:, 2] / ratio  # w
        coords[:, 3] = coords[:, 3] / ratio  # h

    return coords


def _post_process(
    pred_output: torch.Tensor,
    orig_shape: Tuple[int, int],
    letterbox_params: Tuple[float, Tuple[float, float]],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict[str, np.ndarray]:
    """后处理预测结果：NMS + 坐标映射

    Args:
        pred_output: (bs, n_anchors, 4+nc) 预测输出，格式为 [cx, cy, w, h, cls1, cls2, ...]
        orig_shape: 原始图像形状 (H, W)
        letterbox_params: (ratio, (pad_x, pad_y))
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值

    Returns:
        dict: {
            'boxes': (N, 4) xyxy 格式边界框
            'scores': (N,) 置信度分数
            'labels': (N,) 类别索引
        }
    """
    pred = pred_output[0]

    boxes = pred[:, :4]
    cls_scores = pred[:, 4:]
    scores, labels = torch.max(cls_scores, dim=1)

    mask = scores > conf_threshold
    if mask.sum() == 0:
        return {
            'boxes': np.empty((0, 4), dtype=np.float32),
            'scores': np.empty(0, dtype=np.float32),
            'labels': np.empty(0, dtype=np.int64)
        }

    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    keep = ops.nms(boxes_xyxy, scores, iou_threshold)

    boxes_xyxy = boxes_xyxy[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()

    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    boxes_cxcywh = np.stack([cx, cy, w, h], axis=1)
    ratio, pad = letterbox_params
    boxes_cxcywh = _scale_coords(boxes_cxcywh, orig_shape, ratio, pad)

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


class Boxes:
    """边界框容器（Ultralytics 风格）"""

    def __init__(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray):
        self.data = boxes
        self.conf = scores
        self.cls = labels

    @property
    def xyxy(self) -> np.ndarray:
        """(N, 4) xyxy 格式边界框"""
        return self.data

    @property
    def xywh(self) -> np.ndarray:
        """(N, 4) xywh 格式边界框"""
        xywh = np.zeros_like(self.data)
        xywh[:, 0] = (self.data[:, 0] + self.data[:, 2]) / 2
        xywh[:, 1] = (self.data[:, 1] + self.data[:, 3]) / 2
        xywh[:, 2] = self.data[:, 2] - self.data[:, 0]
        xywh[:, 3] = self.data[:, 3] - self.data[:, 1]
        return xywh

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Boxes({len(self)} detections, shape={self.data.shape})"


ULTRALYTICS_COLORS = [
    (25, 102, 255), (255, 102, 0), (102, 255, 102), (102, 0, 255),
    (255, 0, 102), (0, 255, 255), (255, 255, 0), (255, 255, 255),
    (128, 0, 128), (128, 128, 0), (0, 128, 128), (128, 128, 128),
    (64, 0, 64), (192, 128, 0), (64, 64, 0), (0, 64, 64),
]


def _get_color(cls_idx: int) -> Tuple[int, int, int]:
    """根据类别索引获取 Ultralytics 风格的颜色"""
    return ULTRALYTICS_COLORS[cls_idx % len(ULTRALYTICS_COLORS)]


class Results:
    """预测结果容器（Ultralytics 风格）

    Example:
        >>> results = model.predict("image.jpg")
        >>> results[0].boxes.xyxy
        >>> results[0].boxes.conf
        >>> results[0].boxes.cls
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        names: Optional[List[str]] = None
    ):
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.names = names or {}
        self.boxes = Boxes(boxes, scores, labels)

    def plot(
        self,
        conf_threshold: float = 0.25,
        line_width: Optional[int] = None,
        font_size: float = 0.5,
        alpha: float = 0.3
    ) -> np.ndarray:
        """绘制预测结果到图像上"""
        img = self.orig_img.copy()

        if line_width is None:
            img_h, img_w = img.shape[:2]
            line_width = max(2, min(4, int(max(img_h, img_w) / 300)))

        adjusted_font_size = font_size * (line_width / 2)

        for box, score, cls_idx in zip(self.boxes.xyxy, self.boxes.conf, self.boxes.cls):
            if score < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            color = _get_color(int(cls_idx))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

            label = self.names.get(int(cls_idx), f"cls{int(cls_idx)}")
            text = f"{label}: {score:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, adjusted_font_size, int(line_width / 2)
            )

            bg_x1 = x1
            bg_y1 = max(0, y1 - text_h - baseline - 6)
            bg_x2 = x1 + text_w + 4
            bg_y2 = y1

            overlay = img.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)

            cv2.putText(
                img, text, (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, adjusted_font_size,
                (255, 255, 255), int(line_width / 2), cv2.LINE_AA
            )

        return img

    def save(self, path: Union[str, Path], conf_threshold: float = 0.25):
        """保存绘制后的图像"""
        img = self.plot(conf_threshold=conf_threshold)
        cv2.imwrite(str(path), img)

    def __repr__(self) -> str:
        return f"Results(shape={self.orig_shape}, {len(self.boxes)} detections)"


__all__ = [
    'LetterBox',
    '_scale_coords',
    '_post_process',
    'Boxes',
    'Results',
    'ULTRALYTICS_COLORS',
    '_get_color',
]
