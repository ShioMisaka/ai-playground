"""Ultralytics 风格的统一 YOLO 接口

提供 YOLOv11 模型的预测接口，支持图片（单张/批量）和视频文件的预测。
"""
import cv2
import torch
import yaml
import random
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
from glob import glob

# 从 engine.predict 导入预测所需的工具类和函数
from engine.predict import (
    LetterBox,
    _post_process,
    Results
)

# 训练相关导入
from engine.trainer import DetectionTrainer
from utils.config import load_yaml, merge_training_config


class YOLO:
    """YOLOv11 预测接口（Ultralytics 风格）

    Example:
        >>> model = YOLO("runs/train/exp/weights/best.pt")
        >>> results = model.predict("image.jpg", conf=0.25, save=True)
        >>> for r in results:
        ...     print(r.boxes.xyxy)

        >>> # Or from config for training
        >>> model = YOLO("configs/models/yolov11n.yaml")
        >>> model.train(data="configs/data/coco8.yaml", epochs=100)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: Optional[int] = 640,
        simple_resize: bool = False,
        data_config: Optional[Union[str, Path]] = None,
        nc: Optional[int] = None,
        scale: str = "n"
    ):
        """加载或创建 YOLOv11 模型

        Args:
            model_path: 模型路径，可以是：
                - 权重文件路径 (.pt) - 加载训练好的模型
                - 模型配置文件 (.yaml) - 创建新模型用于训练
            device: 设备 ("cuda:0", "cpu"，None 自动选择)
            conf: 默认置信度阈值
            iou: 默认 NMS IoU 阈值
            img_size: 目标图像尺寸（None 动态模式）
            simple_resize: True=直接resize(匹配训练), False=letterbox(保持长宽比)
            data_config: 数据集 YAML 配置文件路径（用于自动加载类别名称）
            nc: 类别数量（优先从 data_config 读取，否则使用此参数）
            scale: 模型缩放 (n/s/m/l/x)，仅用于从 config 创建模型时
        """
        self.model_path = Path(model_path)
        self._is_training_mode = False  # Track if model was created for training

        # 判断是权重文件还是配置文件
        if self.model_path.suffix in ['.pt', '.pth']:
            # 从权重文件加载
            self._load_from_weights(
                self.model_path, device, conf, iou, img_size,
                simple_resize, data_config, nc, scale
            )
        elif self.model_path.suffix in ['.yaml', '.yml']:
            # 从配置文件创建新模型（用于训练）
            self._create_from_config(
                self.model_path, device, data_config, nc, scale
            )
            self._is_training_mode = True
        else:
            raise ValueError(
                f"不支持的文件类型: {self.model_path.suffix}。"
                "请提供 .pt/.pth (权重文件) 或 .yaml/.yml (配置文件)"
            )

        # 设置默认参数
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.auto = img_size is None
        self.simple_resize = simple_resize

        # 预处理器
        self.letterbox = LetterBox(auto=self.auto)

    def _load_from_weights(
        self,
        weights_path: Path,
        device: Optional[str],
        conf: float,
        iou: float,
        img_size: Optional[int],
        simple_resize: bool,
        data_config: Optional[Union[str, Path]],
        nc: Optional[int],
        scale: str
    ):
        """从权重文件加载模型"""
        self.weights_path = weights_path
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

        # 从 YAML 或参数获取 nc
        if data_config:
            yaml_names = self._load_names_from_yaml(data_config)
            nc_from_yaml = len(yaml_names) if yaml_names else None
            final_nc = nc_from_yaml if nc_from_yaml is not None else nc
        else:
            final_nc = nc

        # 如果仍未获取 nc，从权重推断（保留最小兼容性）
        if final_nc is None:
            for key in state_dict.keys():
                if "detect.cv3.0.weight" in key or "cv3.0.weight" in key:
                    final_nc = state_dict[key].shape[0]
                    break
            if final_nc is None:
                raise ValueError("无法确定类别数量，请通过 data_config 或 nc 参数指定")

        # 创建模型
        from models.yolov11 import YOLOv11
        self.model = YOLOv11(nc=final_nc, scale=scale)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # 设置设备
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # 默认参数
        self._nc = final_nc

        # 加载类别名称
        if data_config:
            self.names = self._load_names_from_yaml(data_config)
            self.data_config = Path(data_config)
        else:
            self.names = {i: f"class_{i}" for i in range(final_nc)}
            self.data_config = None

    def _create_from_config(
        self,
        config_path: Path,
        device: Optional[str],
        data_config: Optional[Union[str, Path]],
        nc: Optional[int],
        scale: str
    ):
        """从配置文件创建新模型（用于训练）"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 加载模型配置
        model_cfg = load_yaml(str(config_path))

        # 获取类别数量
        if data_config:
            yaml_names = self._load_names_from_yaml(data_config)
            nc_from_yaml = len(yaml_names) if yaml_names else None
            final_nc = nc_from_yaml if nc_from_yaml is not None else nc
        elif 'model' in model_cfg and 'nc' in model_cfg['model']:
            final_nc = model_cfg['model']['nc']
        elif 'nc' in model_cfg:
            final_nc = model_cfg['nc']
        else:
            final_nc = nc

        if final_nc is None:
            raise ValueError(
                "无法确定类别数量。请通过以下方式之一指定：\n"
                "  1. 在模型配置文件中设置 nc\n"
                "  2. 提供 data_config 参数\n"
                "  3. 直接指定 nc 参数"
            )

        # 从配置获取 scale
        if 'model' in model_cfg and 'scale' in model_cfg['model']:
            scale = model_cfg['model']['scale']
        elif 'scale' in model_cfg:
            scale = model_cfg['scale']

        # 创建模型
        from models.yolov11 import YOLOv11
        self.model = YOLOv11(nc=final_nc, scale=scale)

        # 设置设备
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # 保存配置信息
        self._nc = final_nc
        self.model_config = model_cfg

        # 加载类别名称
        if data_config:
            self.names = self._load_names_from_yaml(data_config)
            self.data_config = Path(data_config)
        else:
            self.names = {i: f"class_{i}" for i in range(final_nc)}
            self.data_config = None

    def _load_names_from_yaml(self, yaml_path: Union[str, Path]) -> Dict[int, str]:
        """从 YAML 配置文件加载类别名称"""
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML 配置文件不存在: {yaml_path}")

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if 'names' in config:
                names = config['names']
                if isinstance(names, dict):
                    return {int(k): v for k, v in names.items()}
                elif isinstance(names, list):
                    return {i: name for i, name in enumerate(names)}
            return {}

        except Exception as e:
            print(f"警告: 加载 YAML 失败 ({e})，使用默认类别名称")
            return {}

    def _get_test_images_path(self) -> Optional[Path]:
        """从 YAML 配置获取测试图片路径"""
        if not self.data_config or not self.data_config.exists():
            return None

        try:
            with open(self.data_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            data_root = config.get('path', '')
            if not isinstance(data_root, (str, Path)):
                return None

            data_root = Path(data_root)
            if not data_root.is_absolute():
                data_root = self.data_config.parent / data_root

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

    @property
    def nc(self) -> int:
        """类别数量"""
        return self._nc

    def train(
        self,
        data: Optional[Union[str, Path]] = None,
        epochs: Optional[int] = None,
        batch: Optional[int] = None,
        imgsz: Optional[int] = None,
        lr: Optional[float] = None,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        save_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """训练模型

        Args:
            data: 数据集配置文件路径 (YAML)
            epochs: 训练轮数
            batch: 批次大小
            imgsz: 图像尺寸
            lr: 学习率
            device: 设备 ('cpu', 'cuda', 'cuda:0', etc.)
            config: 完整配置字典（如果提供，将忽略其他参数）
            save_dir: 保存目录
            **kwargs: 其他训练参数

        Returns:
            Dict[str, Any]: 训练结果，包含:
                - best_map: 最佳 mAP50
                - final_epoch: 最终轮数
                - save_dir: 保存目录

        Example:
            >>> model = YOLO('configs/models/yolov11n.yaml')
            >>> results = model.train(
            ...     data='configs/data/coco8.yaml',
            ...     epochs=100,
            ...     batch=16,
            ...     imgsz=640
            ... )
            >>> print(f"Best mAP: {results['best_map']}")
        """
        from models.yolov11 import YOLOv11

        # Validate data parameter
        if data is None and (config is None or 'data' not in config):
            raise ValueError(
                "Data parameter is required. "
                "Provide either 'data' argument or include 'data' in config."
            )

        # 如果提供了完整配置，直接使用
        if config is not None:
            final_config = config
            # 确保 data 配置正确
            if data is not None and 'data' in final_config:
                if isinstance(data, (str, Path)):
                    final_config['data']['train'] = str(data)
        else:
            # 从参数构建配置
            overrides = {}

            # 将参数转换为嵌套配置
            if epochs is not None:
                overrides['train.epochs'] = epochs
            if batch is not None:
                overrides['train.batch_size'] = batch
            if imgsz is not None:
                overrides['train.img_size'] = imgsz
            if device is not None:
                overrides['device'] = device
            if lr is not None:
                overrides['optimizer.lr'] = lr
            if save_dir is not None:
                overrides['train.save_dir'] = str(save_dir)

            # 添加其他 kwargs
            for key, value in kwargs.items():
                # 将下划线命名转换为点号命名
                config_key = key.replace('__', '.').replace('_', '.')
                overrides[config_key] = value

            # 加载模型配置
            if hasattr(self, 'model_config'):
                model_config = self.model_config
            else:
                # 如果从权重加载，创建基础模型配置
                model_config = {
                    'model': {
                        'nc': self.nc,
                        'scale': getattr(self, 'scale', 'n')
                    }
                }

            # 加载数据配置
            data_config = {}
            if data is not None:
                data_path = Path(data)
                if data_path.suffix in ['.yaml', '.yml']:
                    # 从 YAML 文件加载数据配置
                    data_yaml = load_yaml(str(data_path))

                    # 构建数据配置
                    data_config = {
                        'data': {
                            'train': str(data_path),
                            'nc': data_yaml.get('nc', self.nc),
                        }
                    }

                    # 更新模型的类别数量
                    if 'nc' in data_yaml:
                        new_nc = data_yaml['nc']
                        if new_nc != self.nc:
                            print(f"更新类别数量: {self.nc} -> {new_nc}")
                            # 重新创建模型
                            self.model = YOLOv11(nc=new_nc, scale=getattr(self, 'scale', 'n'))
                            self.model.to(self.device)
                            self._nc = new_nc

                            # 更新类别名称
                            if 'names' in data_yaml:
                                names = data_yaml['names']
                                if isinstance(names, dict):
                                    self.names = {int(k): v for k, v in names.items()}
                                elif isinstance(names, list):
                                    self.names = {i: name for i, name in enumerate(names)}
                else:
                    data_config = {'data': {'train': str(data_path)}}

            # 合并配置
            final_config = merge_training_config(
                model_config=model_config,
                user_config=data_config,
                overrides=overrides if overrides else None
            )

        # 确保 train.name 存在
        if 'train' not in final_config:
            final_config['train'] = {}
        if 'name' not in final_config['train']:
            final_config['train']['name'] = 'exp'

        # 创建训练器并训练
        trainer = DetectionTrainer(
            model=self.model,
            config=final_config,
            save_dir=save_dir
        )

        results = trainer.train()

        # 加载最佳权重回模型（状态同步）
        best_weights_path = Path(results['save_dir']) / 'weights' / 'best.pt'
        if best_weights_path.exists():
            checkpoint = torch.load(
                best_weights_path,
                map_location=self.device,
                weights_only=False
            )
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint)
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            print(f"已加载最佳权重到模型: {best_weights_path}")

        return results

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
        """执行预测"""
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou

        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_file():
                if self._is_video_file(source):
                    return self._predict_video(source, conf, iou, img_size, save, save_dir)
                else:
                    return self._predict_image(source, conf, iou, img_size, save, save_dir)
            elif source.is_dir():
                return self._predict_directory(source, conf, iou, img_size, save, save_dir)
            else:
                raise FileNotFoundError(f"文件不存在: {source}")

        elif isinstance(source, np.ndarray):
            return [self._predict_single(source, conf, iou, img_size, save, save_dir, idx=0)]

        elif isinstance(source, list):
            results = []
            for idx, img in enumerate(source):
                results.append(self._predict_single(img, conf, iou, img_size, save, save_dir, idx))
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

        target_size = img_size if img_size is not None else (640 if not self.auto else None)

        # 预处理
        if self.simple_resize:
            target_size = target_size or self.img_size or 640
            transformed = cv2.resize(orig_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            ratio_w = target_size / orig_w
            ratio_h = target_size / orig_h
            letterbox_params = (ratio_w, (0, 0))
        else:
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
            self.model.detect.eval()
            predictions = self.model(img_tensor)

        # 提取预测输出
        if isinstance(predictions, tuple):
            pred_output = predictions[0]
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

        result = self._predict_single(img, conf, iou, img_size, save, save_dir, idx=0)

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
                result = self._predict_image(img_file, conf, iou, img_size, save, save_dir)
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

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_writer = None
        if save:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / video_path.name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (frame_w, frame_h))

        results = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self._predict_single(frame, conf, iou, img_size, False, save_dir, frame_idx)
                results.append(result)

                if save and video_writer is not None:
                    annotated = result.plot(conf_threshold=conf)
                    video_writer.write(annotated)

                frame_idx += 1

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
        """便捷调用方式"""
        return self.predict(source, **kwargs)

    def visualize_grid(
        self,
        save_path: Union[str, Path] = "runs/predict/grid.jpg",
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
        """生成高分辨率拼图展示预测结果"""
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou
        rows, cols = grid_size

        if source_dir is None:
            source_dir = self._get_test_images_path()
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

                results = self._predict_single(img, conf, iou, img_size, False, "", 0)
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

        model_info = f"Model: {self.weights_path.stem} | Device: {self.device} | Classes: {self.nc}"
        cv2.putText(title_bar, model_info, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        if self.names:
            class_info = " | ".join([f"{v}({k})" for k, v in self.names.items()])
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

    def __repr__(self) -> str:
        return (f"YOLO(weights={self.weights_path.name}, "
                f"device={self.device}, nc={self.nc})")


__all__ = ['YOLO']
