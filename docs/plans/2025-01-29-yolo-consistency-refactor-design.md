# YOLOv11 训练推理一致性重构设计文档

**日期**: 2025-01-29
**目标**: 修复训练/推理不一致问题，实现 Ultralytics 风格统一接口

---

## 问题诊断

### 核心问题
训练时效果良好，但推理时效果极差，即使使用训练集数据推理，效果也远不如训练时。

### 根本原因分析

| 问题 | 训练时 | 推理时 | 影响 |
|------|--------|--------|------|
| **预处理方式** | 简单 resize | 默认 letterbox | 🔴 严重 |
| **Detect 层模式** | train() 模式 | eval() 模式 | 🟡 中等 |
| **输出格式** | dict 格式 | tuple 格式 | 🟡 中等 |
| **后处理逻辑** | 独立实现 | 独立实现 | 🟡 中等 |

### 代码位置

- 训练预处理: `utils/load.py:94-96` - 简单 resize
- 推理预处理: `engine/predict.py:565-570` - 默认 letterbox
- Detect 层: `modules/head.py:131-210` - 训练/推理返回不同格式
- 验证逻辑: `engine/validate.py:60-151` - 复杂的模式切换

---

## 设计方案

### 1. 整体架构

```
ai-playground/
├── models/
│   ├── yolov11.py          # YOLOv11 模型（修改后）
│   ├── yolo.py             # 新增：统一 YOLO 接口类
│   └── __init__.py
├── engine/
│   ├── __init__.py
│   ├── base.py             # 新增：BaseTask 基类
│   ├── trainer.py          # 新增：训练器
│   ├── validator.py        # 新增：验证器
│   ├── predictor.py        # 重构：推理器
│   ├── preprocessor.py     # 新增：统一预处理
│   ├── postprocessor.py    # 新增：统一后处理
│   ├── train.py            # 保留：CLI 入口
│   └── validate.py         # 保留：兼容函数
├── utils/
│   ├── load.py             # 修改：适配 letterbox
│   └── ...
└── configs/
    ├── default.yaml        # 新增：全局默认配置
    └── data/               # 数据配置
```

### 2. 统一预处理 (engine/preprocessor.py)

```python
class Preprocessor:
    """统一的图像预处理类

    核心原则：训练、验证、推理必须使用相同的预处理流程
    """

    def __init__(self, img_size=640, letterbox=True, auto=False):
        self.img_size = img_size
        self.letterbox = letterbox
        self.auto = auto

    def __call__(self, img: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            tensor: (1, 3, H, W) 归一化后的张量
            params: 预处理参数（用于后处理坐标映射）
        """
        # 1. Letterbox 或简单 resize
        # 2. BGR -> RGB
        # 3. HWC -> CHW
        # 4. 归一化到 [0, 1]
        # 5. 返回参数供后处理使用
```

### 3. 统一后处理 (engine/postprocessor.py)

```python
class Postprocessor:
    """统一的后处理类

    包括：NMS、置信度过滤、坐标映射
    """

    def __init__(self, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        predictions: torch.Tensor,
        orig_shape: Tuple[int, int],
        preprocess_params: Dict
    ) -> Dict:
        """
        Args:
            predictions: (bs, n_anchors, 4+nc) 预测输出
            orig_shape: 原始图像尺寸 (H, W)
            preprocess_params: 预处理参数（用于坐标映射）

        Returns:
            dict: {'boxes': (N,4), 'scores': (N,), 'labels': (N,)}
        """
```

### 4. 修改 DetectAnchorFree (modules/head.py)

**当前问题**: 训练和推理返回不同格式

**修改方案**: 统一输出格式，中间值通过属性访问

```python
class DetectAnchorFree(nn.Module):
    def forward(self, x):
        """
        统一的前向传播

        Returns:
            predictions: (bs, n_anchors, 4+nc) 格式的预测张量
        """
        # ... 计算 cls_outputs, reg_outputs ...

        # 保存用于 loss 计算的中间值
        self._cls_outputs = cls_outputs
        self._reg_outputs = reg_outputs

        # 始终返回解码后的预测
        return self._decode_predictions(cls_outputs, reg_outputs, x)
```

### 5. 修改 YOLOv11 (models/yolov11.py)

```python
class YOLOv11(nn.Module):
    def forward(self, x, targets=None):
        """
        统一的前向传播

        Args:
            x: (bs, 3, H, W) 输入图像
            targets: (n_boxes, 6) 标签 [batch_idx, cls, cx, cy, w, h]

        Returns:
            训练模式 (targets != None): (loss, loss_items, predictions)
            推理模式 (targets == None): predictions
        """
        # ... backbone + head ...

        predictions = self.detect([p3, p4, p5])

        if targets is not None:
            cls_outputs = self.detect._cls_outputs
            reg_outputs = self.detect._reg_outputs
            loss_dict = {'cls': cls_outputs, 'reg': reg_outputs}
            loss_for_backward, loss_items = self.loss_fn(loss_dict, targets)
            return loss_for_backward, loss_items, predictions

        return predictions
```

### 6. BaseTask 基类 (engine/base.py)

```python
class BaseTask:
    """所有任务处理器的基类"""

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.preprocessor = Preprocessor(
            img_size=cfg['img_size'],
            letterbox=cfg.get('letterbox', True)
        )
        self.postprocessor = Postprocessor(
            conf=cfg.get('conf', 0.25),
            iou=cfg.get('iou', 0.45)
        )
```

### 7. YOLO 统一接口 (models/yolo.py)

```python
class YOLO:
    """Ultralytics 风格的统一 YOLO 接口"""

    def __init__(self, model: Union[str, Path, nn.Module]):
        """从配置文件、权重文件或模型实例创建"""

    def train(self, data: str, **kwargs) -> Dict:
        """训练模型"""

    def val(self, data: str, **kwargs) -> Dict:
        """验证模型"""

    def predict(self, source, **kwargs) -> List[Results]:
        """推理"""

    def export(self, format='onnx', **kwargs):
        """导出模型"""

    def __call__(self, source, **kwargs):
        """便捷调用"""
```

### 8. 配置系统

```
configs/
├── default.yaml          # 全局默认配置
└── data/
    └── coco.yaml        # 数据集配置
```

**default.yaml**:
```yaml
train:
  name: exp
  epochs: 100
  batch_size: 16
  img_size: 640
  lr: 0.001
  letterbox: true      # 关键：使用 letterbox

val:
  conf: 0.25
  iou: 0.45
  letterbox: true

predict:
  conf: 0.25
  iou: 0.45
  letterbox: true
```

---

## 实施步骤

| 步骤 | 任务 | 优先级 |
|------|------|--------|
| 1 | 创建 `Preprocessor` 类 | P0 |
| 2 | 创建 `Postprocessor` 类 | P0 |
| 3 | 修改 `DetectAnchorFree.forward()` | P0 |
| 4 | 修改 `YOLOv11.forward()` | P0 |
| 5 | 创建 `BaseTask` 基类 | P1 |
| 6 | 创建 `Trainer` 类 | P1 |
| 7 | 创建 `Validator` 类 | P1 |
| 8 | 重构 `Predictor` 类 | P1 |
| 9 | 创建 `YOLO` 统一接口类 | P1 |
| 10 | 修改 `utils/load.py` 适配 letterbox | P0 |
| 11 | 创建配置文件模板 | P2 |
| 12 | 测试验证一致性 | P0 |

---

## 预期成果

1. ✅ **一致性**: 训练、验证、推理使用完全相同的预处理和后处理流程
2. ✅ **统一接口**: 所有功能通过 `YOLO` 类访问，API 简洁清晰
3. ✅ **可维护性**: 核心逻辑集中在少数几个类中
4. ✅ **可扩展性**: 易于添加新功能
5. ✅ **性能**: 推理效果与训练/验证时一致
6. ✅ **兼容性**: 与 Ultralytics YOLO 接口兼容

---

## 使用示例

```python
# 训练
model = YOLO('configs/models/yolov11n.yaml')
model.train(data='configs/data/coco.yaml', epochs=100, batch=16)

# 验证
metrics = model.val(data='configs/data/coco.yaml')
print(f"mAP50: {metrics['mAP50']}")

# 推理
results = model.predict('image.jpg', conf=0.3)
for r in results:
    print(r.boxes.xyxy)
    r.save('result.jpg')

# 便捷调用
results = model('image.jpg')
```
