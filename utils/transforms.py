"""
数据增强模块

提供 YOLO 系列常用的数据增强方法，包括 Mosaic 和 Mixup。
"""
import random
from typing import Tuple, Optional
import torch
import numpy as np
from PIL import Image


class MosaicTransform:
    """Mosaic 数据增强

    将 4 张图片拼接成一张大图，是 YOLOv4/v5/v8/v11 的核心增强方法。
    可以显著提升小目标检测能力。

    Args:
        dataset: 数据集对象，用于随机获取其他图片
        img_size: 输出图像尺寸（默认 640）
        prob: Mosaic 应用概率（默认 1.0，训练后期可设为 0 关闭）
        enable: 是否启用 Mosaic（默认 True）

    Example:
        >>> transform = MosaicTransform(dataset, img_size=640)
        >>> # 训练时应用
        >>> img, boxes = transform(img, boxes)
        >>> # 训练后期关闭
        >>> transform.enable = False
    """

    def __init__(self, dataset, img_size: int = 640, prob: float = 1.0, enable: bool = True):
        self.dataset = dataset
        self.img_size = img_size
        self.prob = prob
        self.enable = enable

    def __call__(self, img: Image.Image, boxes: torch.Tensor) -> Tuple[Image.Image, torch.Tensor]:
        """应用 Mosaic 增强

        Args:
            img: PIL Image
            boxes: 边界框张量 [N, 5] (class_id, x_center, y_center, width, height) 归一化坐标

        Returns:
            (mosaic_img, mosaic_boxes): 拼接后的图片和对应的边界框
        """
        # 检查是否启用
        if not self.enable or random.random() > self.prob:
            return img, boxes

        # 如果 boxes 为空，直接返回原图
        if boxes.shape[0] == 0:
            return img, boxes

        # 随机获取 3 张其他图片的索引
        indices = [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

        # 读取 4 张图片（包括当前图片）
        imgs = [img]
        box_list = [boxes]

        for idx in indices:
            # 使用 _load_raw_item 避免递归调用 transform
            other_img, other_boxes = self.dataset._load_raw_item(idx)
            imgs.append(other_img)
            box_list.append(other_boxes)

        # 创建 Mosaic 画布
        mosaic_img = Image.new('RGB', (self.img_size * 2, self.img_size * 2))

        # 计算拼接中心点
        cx = random.randint(self.img_size // 2, self.img_size * 3 // 2)
        cy = random.randint(self.img_size // 2, self.img_size * 3 // 2)

        # 4 个区域的坐标
        # 左上、右上、左下、右下
        positions = [
            (0, 0, cx, cy),          # 左上
            (cx, 0, self.img_size * 2, cy),  # 右上
            (0, cy, cx, self.img_size * 2),  # 左下
            (cx, cy, self.img_size * 2, self.img_size * 2),  # 右下
        ]

        mosaic_boxes = []

        for i, (img_i, boxes_i) in enumerate(zip(imgs, box_list)):
            x1, y1, x2, y2 = positions[i]

            # 象限尺寸
            w = x2 - x1
            h = y2 - y1

            # 关键修复：使用 letterbox 保持宽高比，而不是强制拉伸
            img_w, img_h = img_i.size
            scale = min(w / img_w, h / img_h)
            scaled_w = int(round(img_w * scale))
            scaled_h = int(round(img_h * scale))

            # 缩放图像（保持宽高比）
            img_resized = img_i.resize((scaled_w, scaled_h), Image.BILINEAR)

            # 计算粘贴位置（居中）
            paste_x = x1 + (w - scaled_w) // 2
            paste_y = y1 + (h - scaled_h) // 2

            # 粘贴到画布
            mosaic_img.paste(img_resized, (paste_x, paste_y))

            # 调整 boxes 坐标
            if boxes_i.shape[0] > 0:
                # 归一化坐标转像素坐标（在缩放后的图像中）
                boxes_px = boxes_i.clone()
                boxes_px[:, 1] *= scaled_w  # x_center
                boxes_px[:, 2] *= scaled_h  # y_center
                boxes_px[:, 3] *= scaled_w  # width
                boxes_px[:, 4] *= scaled_h  # height

                # 转换为 x1y1x2y2
                bx1 = boxes_px[:, 1] - boxes_px[:, 3] / 2
                by1 = boxes_px[:, 2] - boxes_px[:, 4] / 2
                bx2 = boxes_px[:, 1] + boxes_px[:, 3] / 2
                by2 = boxes_px[:, 2] + boxes_px[:, 4] / 2

                # 加上粘贴位置偏移
                bx1 += paste_x
                by1 += paste_y
                bx2 += paste_x
                by2 += paste_y

                # 裁剪到画布范围内
                bx1 = bx1.clamp(0, self.img_size * 2)
                by1 = by1.clamp(0, self.img_size * 2)
                bx2 = bx2.clamp(0, self.img_size * 2)
                by2 = by2.clamp(0, self.img_size * 2)

                # 过滤掉无效的框（面积太小或越界）
                valid_mask = (bx2 > bx1) & (by2 > by1)
                valid_mask &= ((bx2 - bx1) * (by2 - by1)) > 4  # 最小面积 4 像素

                if valid_mask.any():
                    # 转换回归一化坐标（相对于整张 mosaic 图片）
                    new_boxes = boxes_i[valid_mask].clone()
                    new_boxes[:, 1] = ((bx1[valid_mask] + bx2[valid_mask]) / 2) / (self.img_size * 2)
                    new_boxes[:, 2] = ((by1[valid_mask] + by2[valid_mask]) / 2) / (self.img_size * 2)
                    new_boxes[:, 3] = (bx2[valid_mask] - bx1[valid_mask]) / (self.img_size * 2)
                    new_boxes[:, 4] = (by2[valid_mask] - by1[valid_mask]) / (self.img_size * 2)

                    mosaic_boxes.append(new_boxes)

        # 缩放回目标尺寸
        mosaic_img = mosaic_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # 关键修复：boxes 也需要相应缩放！
        # 画布是 2*img_size，缩放到 img_size，所以坐标也要乘以 0.5
        if len(mosaic_boxes) > 0:
            mosaic_boxes = torch.cat(mosaic_boxes, dim=0)
            # boxes 格式: [class_id, x_center, y_center, width, height]
            # 只需要缩放坐标部分（索引 1-4），class_id 不变
            mosaic_boxes[:, 1:] *= 0.5  # 所有坐标缩放 0.5
        else:
            mosaic_boxes = torch.zeros((0, 5), dtype=torch.float32)

        return mosaic_img, mosaic_boxes


class MixupTransform:
    """Mixup 数据增强

    将两张图片按一定比例混合，增强模型对重叠目标的处理能力。

    Args:
        dataset: 数据集对象
        alpha: Beta 分布参数（默认 0.5，控制混合比例）
        prob: Mixup 应用概率（默认 0.5）

    Reference:
        Zhang et al. "mixup: Beyond Empirical Risk Minimization"
        ICLR 2018
    """

    def __init__(self, dataset, alpha: float = 0.5, prob: float = 0.5, enable: bool = True):
        self.dataset = dataset
        self.alpha = alpha
        self.prob = prob
        self.enable = enable

    def __call__(self, img: Image.Image, boxes: torch.Tensor) -> Tuple[Image.Image, torch.Tensor]:
        """应用 Mixup 增强"""
        if not self.enable or random.random() > self.prob:
            return img, boxes

        # 随机获取另一张图片（使用 _load_raw_item 避免递归）
        idx = random.randint(0, len(self.dataset) - 1)
        img2, boxes2 = self.dataset._load_raw_item(idx)

        # 生成混合比例（Beta 分布）
        lam = np.random.beta(self.alpha, self.alpha)

        # 混合图片
        img1_array = np.array(img).astype(np.float32)
        img2_array = np.array(img2).astype(np.float32)

        mixed_array = lam * img1_array + (1 - lam) * img2_array
        mixed_img = Image.fromarray(mixed_array.astype(np.uint8))

        # 合并 boxes（简化处理：直接拼接）
        # 注意：理想的 Mixup 应该根据 lam 调整框的权重，但实现较复杂
        # 这里采用简化方案，只进行简单的 boxes 拼接
        if boxes.shape[0] > 0 and boxes2.shape[0] > 0:
            mixed_boxes = torch.cat([boxes, boxes2], dim=0)
        elif boxes.shape[0] > 0:
            mixed_boxes = boxes
        else:
            mixed_boxes = boxes2

        return mixed_img, mixed_boxes


class Compose:
    """组合多个 Transform

    Args:
        transforms: transform 列表

    Example:
        >>> transform = Compose([
        >>>     MosaicTransform(dataset, prob=0.5),
        >>>     MixupTransform(dataset, prob=0.5),
        >>> ])
        >>> img, boxes = transform(img, boxes)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: Image.Image, boxes: torch.Tensor) -> Tuple[Image.Image, torch.Tensor]:
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes

    def set_mosaic_enable(self, enable: bool):
        """设置 Mosaic 的启用状态"""
        for t in self.transforms:
            if isinstance(t, MosaicTransform):
                t.enable = enable

    def set_mixup_enable(self, enable: bool):
        """设置 Mixup 的启用状态"""
        for t in self.transforms:
            if isinstance(t, MixupTransform):
                t.enable = enable
