"""
YOLOv11 预测模块

提供 Ultralytics 风格的接口用于目标检测推理。
支持图片（单张/批量）和视频文件的预测。
"""
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
import torchvision.ops as ops

from models import YOLOv11


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

    # 创建副本避免修改原数组
    coords = coords.copy()

    # 检测是否为简单 resize（没有 padding）
    is_simple_resize = (pad_x == 0 and pad_y == 0)

    if is_simple_resize:
        # 简单 resize：x 和 y 方向可能有不同的缩放比例
        # ratio 是 ratio_w，需要计算 ratio_h
        # 由于简单 resize 直接缩放到 target_size x target_size
        # ratio_w = target_w / orig_w, ratio_h = target_h / orig_h
        # 但 target_w == target_h，所以 ratio_w != ratio_h（除非原图是正方形）

        # 实际上，我们不需要分别处理 ratio_w 和 ratio_h
        # 因为简单 resize 后，坐标需要直接按比例还原
        # cx_orig = cx_scaled * orig_w / target_size
        # cy_orig = cy_scaled * orig_h / target_size
        # 但这里的 ratio 参数是 target_size / orig_w（对于 x）或 target_size / orig_h（对于 y）

        # 更简单的方式：直接使用原图尺寸来还原
        # coords 是在 target_size x target_size 空间中
        # 需要映射到 orig_w x orig_h

        # 由于 ratio = target_size / orig_w（传入的是 ratio_w）
        target_size = ratio * orig_w

        # 计算 x 和 y 方向的还原比例
        scale_x = orig_w / target_size
        scale_y = orig_h / target_size

        coords[:, 0] = coords[:, 0] * scale_x  # cx
        coords[:, 1] = coords[:, 1] * scale_y  # cy
        coords[:, 2] = coords[:, 2] * scale_x  # w
        coords[:, 3] = coords[:, 3] * scale_y  # h
    else:
        # Letterbox：中心坐标减去 padding，除以缩放比
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
    # 单图像处理
    pred = pred_output[0]  # (n_anchors, 4+nc)

    # 提取 bbox (cx, cy, w, h)
    boxes = pred[:, :4]  # (n_anchors, 4)

    # 提取类别分数并取最大值
    cls_scores = pred[:, 4:]  # (n_anchors, nc)
    scores, labels = torch.max(cls_scores, dim=1)  # (n_anchors,), (n_anchors,)

    # 置信度过滤
    mask = scores > conf_threshold
    if mask.sum() == 0:
        # 没有满足阈值的检测
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
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # (n, 4)

    # NMS（在 letterbox 空间进行）
    keep = ops.nms(boxes_xyxy, scores, iou_threshold)

    # 保留 NMS 后的结果
    boxes_xyxy = boxes_xyxy[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()

    # 坐标映射：从 letterbox 空间映射回原图空间
    # 需要先转回 cxcywh 格式，映射后再转回 xyxy
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    boxes_cxcywh = np.stack([cx, cy, w, h], axis=1)
    ratio, pad = letterbox_params
    boxes_cxcywh = _scale_coords(boxes_cxcywh, orig_shape, ratio, pad)

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


class Boxes:
    """边界框容器（Ultralytics 风格）"""

    def __init__(self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray):
        """
        Args:
            boxes: (N, 4) xyxy 格式边界框
            scores: (N,) 置信度分数
            labels: (N,) 类别索引
        """
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
        xywh[:, 0] = (self.data[:, 0] + self.data[:, 2]) / 2  # cx
        xywh[:, 1] = (self.data[:, 1] + self.data[:, 3]) / 2  # cy
        xywh[:, 2] = self.data[:, 2] - self.data[:, 0]  # w
        xywh[:, 3] = self.data[:, 3] - self.data[:, 1]  # h
        return xywh

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Boxes({len(self)} detections, shape={self.data.shape})"


class Results:
    """预测结果容器（Ultralytics 风格）

    Example:
        >>> results = model.predict("image.jpg")
        >>> results[0].boxes.xyxy   # (N, 4) 边界框
        >>> results[0].boxes.conf   # (N,) 置信度
        >>> results[0].boxes.cls    # (N,) 类别
        >>> results[0].orig_shape   # (H, W) 原始图像形状
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        names: Optional[List[str]] = None
    ):
        """
        Args:
            orig_img: 原始图像 (H, W, C) BGR 格式
            boxes: (N, 4) xyxy 格式边界框
            scores: (N,) 置信度分数
            labels: (N,) 类别索引
            names: 类别名称列表
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]  # (H, W)
        self.names = names or {}
        self.boxes = Boxes(boxes, scores, labels)

    def plot(self, conf_threshold: float = 0.25, line_width: int = 2) -> np.ndarray:
        """绘制预测结果到图像上

        Args:
            conf_threshold: 低于此阈值的框不显示
            line_width: 边界框线宽

        Returns:
            绘制后的图像 (H, W, C) BGR 格式
        """
        img = self.orig_img.copy()

        for i, (box, score, cls_idx) in enumerate(zip(
            self.boxes.xyxy,
            self.boxes.conf,
            self.boxes.cls
        )):
            if score < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)

            # 绘制边界框
            color = self._get_color(int(cls_idx))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

            # 绘制标签
            label = self.names.get(int(cls_idx), f"cls{int(cls_idx)}")
            text = f"{label}: {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img, (x1, y1 - text_h - 4),
                (x1 + text_w, y1),
                color, -1
            )
            cv2.putText(
                img, text, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return img

    def save(self, path: Union[str, Path], conf_threshold: float = 0.25):
        """保存绘制后的图像

        Args:
            path: 保存路径
            conf_threshold: 低于此阈值的框不显示
        """
        img = self.plot(conf_threshold=conf_threshold)
        cv2.imwrite(str(path), img)

    @staticmethod
    def _get_color(cls_idx: int) -> Tuple[int, int, int]:
        """根据类别索引生成固定颜色"""
        np.random.seed(cls_idx)
        return tuple(map(int, np.random.randint(0, 255, 3)))

    def __repr__(self) -> str:
        return f"Results(shape={self.orig_shape}, {len(self.boxes)} detections)"


class YOLO:
    """YOLOv11 预测接口（Ultralytics 风格）

    Example:
        >>> model = YOLO("runs/train/exp/weights/best.pt")
        >>> results = model.predict("image.jpg", conf=0.25, save=True)
        >>> for r in results:
        ...     print(r.boxes.xyxy)
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: Optional[int] = 640,
        simple_resize: bool = False
    ):
        """加载 YOLOv11 模型

        Args:
            weights_path: 权重文件路径 (.pt)
            device: 设备 ("cuda:0", "cpu"，None 自动选择)
            conf: 默认置信度阈值
            iou: 默认 NMS IoU 阈值
            img_size: 目标图像尺寸（None 动态模式）
            simple_resize: True=直接resize(匹配训练), False=letterbox(保持长宽比)
        """
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        # 加载检查点
        checkpoint = torch.load(
            self.weights_path,
            map_location="cpu",
            weights_only=False
        )

        # 提取模型状态字典
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            state_dict = checkpoint

        # 推断模型参数
        nc = self._infer_nc(state_dict)
        scale = self._infer_scale(state_dict)

        # 创建模型
        self.model = YOLOv11(nc=nc, scale=scale)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # 设置设备
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # 默认参数
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.auto = img_size is None  # None 表示动态模式
        self.simple_resize = simple_resize  # True=直接resize, False=letterbox

        # 类别名称
        self.names = {i: f"class_{i}" for i in range(nc)}

        # 预处理器
        self.letterbox = LetterBox(auto=self.auto)

    def _infer_nc(self, state_dict: dict) -> int:
        """从状态字典推断类别数量"""
        # DetectAnchorFree.cv3.0 权重形状是 (nc, input_channels, 1, 1)
        for key in state_dict.keys():
            if "detect.cv3.0.weight" in key or "cv3.0.weight" in key:
                return state_dict[key].shape[0]
        # 默认 80 类
        return 80

    def _infer_scale(self, state_dict: dict) -> str:
        """从状态字典推断模型 scale"""
        # 通过参数数量推断
        num_params = sum(p.numel() for p in state_dict.values())

        # 粗略估计
        if num_params < 3e6:
            return "n"
        elif num_params < 15e6:
            return "s"
        elif num_params < 25e6:
            return "m"
        else:
            return "l"

    @property
    def nc(self) -> int:
        """类别数量"""
        return len(self.names)

    def predict(
        self,
        source: Union[str, Path, np.ndarray, List[np.ndarray]],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        img_size: Optional[int] = None,
        save: bool = False,
        save_dir: Union[str, Path] = "runs/predict",
        **kwargs
    ) -> List[Results]:
        """执行预测

        Args:
            source: 输入源
                - str/Path: 图片路径、图片目录或视频文件
                - np.ndarray: 单张图像 (H, W, C) BGR 格式
                - List[np.ndarray]: 多张图像
            conf: 置信度阈值（覆盖默认值）
            iou: NMS IoU 阈值（覆盖默认值）
            img_size: 目标图像尺寸（None 动态模式）
            save: 是否保存结果
            save_dir: 保存目录

        Returns:
            Results 对象列表
        """
        # 覆盖默认参数
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou

        # 确定输入类型
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_file():
                if self._is_video_file(source):
                    return self._predict_video(
                        source, conf, iou, img_size, save, save_dir
                    )
                else:
                    return self._predict_image(
                        source, conf, iou, img_size, save, save_dir
                    )
            elif source.is_dir():
                return self._predict_directory(
                    source, conf, iou, img_size, save, save_dir
                )
            else:
                raise FileNotFoundError(f"文件不存在: {source}")

        elif isinstance(source, np.ndarray):
            return [self._predict_single(
                source, conf, iou, img_size, save, save_dir, idx=0
            )]

        elif isinstance(source, list):
            results = []
            for idx, img in enumerate(source):
                results.append(self._predict_single(
                    img, conf, iou, img_size, save, save_dir, idx
                ))
            return results

        else:
            raise TypeError(f"不支持的 source 类型: {type(source)}")

    def _predict_single(
        self,
        img: np.ndarray,
        conf: float,
        iou: float,
        img_size: Optional[int],
        save: bool,
        save_dir: Union[str, Path],
        idx: int = 0
    ) -> Results:
        """对单张图像执行预测"""
        orig_img = img.copy()
        orig_h, orig_w = orig_img.shape[:2]

        # 确定目标尺寸
        target_size = img_size if img_size is not None else (640 if not self.auto else None)

        # 预处理
        if self.simple_resize:
            # 直接 resize（匹配训练时的处理）
            target_size = target_size or self.img_size or 640
            transformed = cv2.resize(orig_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            # letterbox_params: (ratio, (pad_x, pad_y))
            # 对于简单 resize，ratio = target_size / orig_size，没有 padding
            ratio_w = target_size / orig_w
            ratio_h = target_size / orig_h
            letterbox_params = (ratio_w, (0, 0))  # 只用 ratio_w（坐标映射时会处理不同比例）
        else:
            # Letterbox 预处理（保持长宽比）
            if self.auto:
                transformed, letterbox_params = self.letterbox(orig_img, target_size=max(orig_h, orig_w))
            else:
                target_size = target_size or self.img_size or 640
                transformed, letterbox_params = self.letterbox(orig_img, target_size=target_size)

        # 转换为 Tensor
        img_tensor = torch.from_numpy(transformed).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            self.model.detect.eval()  # 确保检测头在推理模式
            predictions = self.model(img_tensor)

        # 提取预测输出
        if isinstance(predictions, tuple):
            pred_output = predictions[0]  # (bs, anchors, 4+nc)
        else:
            pred_output = predictions

        # 后处理
        result_dict = _post_process(
            pred_output,
            (orig_h, orig_w),
            letterbox_params,
            conf,
            iou
        )

        # 创建 Results 对象
        results = Results(
            orig_img=orig_img,
            boxes=result_dict['boxes'],
            scores=result_dict['scores'],
            labels=result_dict['labels'],
            names=self.names
        )

        # 保存
        if save:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"result_{idx}.jpg"
            results.save(save_path, conf_threshold=conf)

        return results

    def _predict_image(
        self,
        img_path: Path,
        conf: float,
        iou: float,
        img_size: Optional[int],
        save: bool,
        save_dir: Union[str, Path]
    ) -> List[Results]:
        """预测单张图片文件"""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        result = self._predict_single(
            img, conf, iou, img_size, save, save_dir, idx=0
        )

        # 更新保存路径为原文件名
        if save:
            save_dir = Path(save_dir)
            save_path = save_dir / img_path.name
            result.save(save_path, conf_threshold=conf)

        return [result]

    def _predict_directory(
        self,
        dir_path: Path,
        conf: float,
        iou: float,
        img_size: Optional[int],
        save: bool,
        save_dir: Union[str, Path]
    ) -> List[Results]:
        """预测目录中的所有图片"""
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        img_files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in img_extensions
        ]

        if not img_files:
            raise ValueError(f"目录中没有图片文件: {dir_path}")

        results = []
        for img_file in img_files:
            try:
                result = self._predict_image(
                    img_file, conf, iou, img_size, save, save_dir
                )
                results.extend(result)
            except Exception as e:
                print(f"警告: 跳过 {img_file}: {e}")

        return results

    def _predict_video(
        self,
        video_path: Path,
        conf: float,
        iou: float,
        img_size: Optional[int],
        save: bool,
        save_dir: Union[str, Path]
    ) -> List[Results]:
        """预测视频文件"""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 视频写入器
        video_writer = None
        if save:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / video_path.name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(save_path), fourcc, fps, (frame_w, frame_h)
            )

        results = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 预测当前帧
                result = self._predict_single(
                    frame, conf, iou, img_size, save=False, save_dir=save_dir, idx=frame_idx
                )
                results.append(result)

                # 写入视频
                if save and video_writer is not None:
                    annotated = result.plot(conf_threshold=conf)
                    video_writer.write(annotated)

                frame_idx += 1

                # 打印进度
                if frame_idx % 30 == 0:
                    print(f"处理进度: {frame_idx}/{total_frames} 帧")

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()

        print(f"视频处理完成: {frame_idx} 帧")
        return results

    @staticmethod
    def _is_video_file(path: Path) -> bool:
        """判断是否为视频文件"""
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
        return path.suffix.lower() in video_extensions

    def __call__(self, source, **kwargs):
        """便捷调用方式

        Example:
            >>> model = YOLO("weights.pt")
            >>> results = model("image.jpg", conf=0.3)
        """
        return self.predict(source, **kwargs)

    def __repr__(self) -> str:
        return (f"YOLO(weights={self.weights_path.name}, "
                f"device={self.device}, nc={self.nc})")
