"""YOLO 预测结果可视化模块

提供预测结果的可视化功能：
- visualize_grid: 生成高分辨率拼图展示预测结果
"""
import cv2
import random
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
from glob import glob


def visualize_grid(
    model,
    save_path: Union[str, Path],
    num_samples: int = 9,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    grid_size: Tuple[int, int] = (3, 3),
    img_size: Optional[int] = None,
    source_dir: Optional[Union[str, Path]] = None,
    border_width: int = 10,
    border_color: Tuple[int, int, int] = (60, 60, 60),
    background_color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """生成高分辨率拼图展示预测结果

    Args:
        model: YOLO 模型实例（需有 _predict_single, conf, iou, nc, names, weights_path, device 属性）
        save_path: 保存路径
        num_samples: 采样图片数量
        conf: 置信度阈值（默认使用 model.conf）
        iou: NMS IoU 阈值（默认使用 model.iou）
        grid_size: 网格尺寸 (rows, cols)
        img_size: 目标图像尺寸
        source_dir: 图片源目录（默认从 data_config 获取）
        border_width: 边框宽度
        border_color: 边框颜色 (B, G, R)
        background_color: 背景填充颜色 (B, G, R)

    Returns:
        拼图图像数组
    """
    conf = conf if conf is not None else model.conf
    iou = iou if iou is not None else model.iou
    rows, cols = grid_size

    # 获取图片源目录
    if source_dir is None:
        source_dir = _get_test_images_path(model)
    if source_dir is None:
        raise ValueError(
            "无法确定图片源目录，请通过 source_dir 参数指定 "
            "或在初始化时提供 data_config 参数"
        )
    source_dir = Path(source_dir)

    # 收集图片文件
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob(str(source_dir / f"*{ext}")))
        img_files.extend(glob(str(source_dir / f"*{ext.upper()}")))

    if not img_files:
        raise ValueError(f"目录中没有图片文件: {source_dir}")

    random.seed(42)
    sampled_files = random.sample(img_files, min(num_samples, len(img_files)))

    # 第一阶段：加载并预测，找出最大尺寸
    processed_results = []
    max_cell_h, max_cell_w = 0, 0

    for img_file in sampled_files[:rows * cols]:
        try:
            img = cv2.imread(img_file)
            if img is None:
                continue

            results = model._predict_single(img, conf, iou, img_size, False, "", 0)
            annotated = results.plot(conf_threshold=conf)

            h, w = annotated.shape[:2]
            max_cell_h = max(max_cell_h, h)
            max_cell_w = max(max_cell_w, w)

            processed_results.append((img_file, annotated))

        except Exception as e:
            print(f"警告: 处理 {img_file} 失败: {e}")

    # 第二阶段：将所有图片缩放到统一单元格尺寸
    cell_images = []
    target_h, target_w = max_cell_h, max_cell_w

    for img_file, annotated in processed_results:
        try:
            h, w = annotated.shape[:2]

            scale = min(target_h / h, target_w / w)
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            resized = cv2.resize(annotated, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

            canvas = np.full((target_h, target_w, 3), background_color, dtype=np.uint8)

            y_offset = (target_h - scaled_h) // 2
            x_offset = (target_w - scaled_w) // 2

            canvas[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized

            bordered = cv2.copyMakeBorder(
                canvas, border_width, border_width,
                border_width, border_width,
                cv2.BORDER_CONSTANT, value=border_color
            )

            cell_images.append(bordered)

        except Exception as e:
            print(f"警告: 处理 {img_file} 缩放失败: {e}")
            blank = np.full((target_h, target_w, 3), background_color, dtype=np.uint8)
            bordered = cv2.copyMakeBorder(
                blank, border_width, border_width,
                border_width, border_width,
                cv2.BORDER_CONSTANT, value=border_color
            )
            cell_images.append(bordered)

    # 填充不足的单元格
    while len(cell_images) < rows * cols:
        blank = np.full((target_h, target_w, 3), background_color, dtype=np.uint8)
        bordered = cv2.copyMakeBorder(
            blank, border_width, border_width,
            border_width, border_width,
            cv2.BORDER_CONSTANT, value=border_color
        )
        cell_images.append(bordered)

    # 拼接成网格
    grid_rows = []
    for r in range(rows):
        row_images = cell_images[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_images))
    grid_image = np.vstack(grid_rows)

    # 添加标题栏
    grid_h, grid_w = grid_image.shape[:2]
    title_bar_h = 70
    title_bar = np.ones((title_bar_h, grid_w, 3), dtype=np.uint8) * 25

    model_info = f"Model: {model.weights_path.stem} | Device: {model.device} | Classes: {model.nc}"
    cv2.putText(title_bar, model_info, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    if model.names:
        class_info = " | ".join([f"{v}({k})" for k, v in model.names.items()])
        y_pos = 50
        if len(class_info) > 80:
            parts = class_info.split(" | ")
            for i in range(0, len(parts), 4):
                line = " | ".join(parts[i:i + 4])
                cv2.putText(title_bar, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
                y_pos += 18
        else:
            cv2.putText(title_bar, class_info, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # 底部信息栏
    footer_h = 30
    footer = np.ones((footer_h, grid_w, 3), dtype=np.uint8) * 35
    footer_text = f"Grid: {rows}x{cols} | Samples: {len(processed_results)} | Cell: {target_w}x{target_h}"
    cv2.putText(footer, footer_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

    final_image = np.vstack([title_bar, grid_image, footer])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), final_image)
    print(f"✓ 拼图已保存: {save_path} ({final_image.shape[1]}x{final_image.shape[0]})")

    return final_image


def _get_test_images_path(model) -> Optional[Path]:
    """从 YAML 配置获取测试图片路径"""
    if not hasattr(model, 'data_config') or not model.data_config or not model.data_config.exists():
        return None

    try:
        import yaml
        with open(model.data_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        data_root = config.get('path', '')
        if not isinstance(data_root, (str, Path)):
            return None

        data_root = Path(data_root)
        if not data_root.is_absolute():
            data_root = model.data_config.parent / data_root

        # 优先使用 test 路径
        test_path = config.get('test', 'images/test')
        full_test_path = data_root / test_path

        if full_test_path.exists() and full_test_path.is_dir():
            return full_test_path

        # 回退到 val 路径
        val_path = config.get('val', 'images/val')
        full_val_path = data_root / val_path

        if full_val_path.exists() and full_val_path.is_dir():
            return full_val_path

    except Exception as e:
        print(f"警告: 解析 YAML 失败 ({e})")

    return None


__all__ = [
    'visualize_grid',
]
