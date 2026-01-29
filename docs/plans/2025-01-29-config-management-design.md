# 配置管理系统重构设计

## 概述

将当前硬编码的参数系统重构为基于 YAML 的配置管理系统，参考 Ultralytics YOLOv8/v11 的设计理念。

## 设计目标

1. **配置即代码**：所有超参数移出代码，存入 YAML 文件
2. **分层配置**：模型配置与训练超参分离
3. **灵活调用**：支持配置文件和 CLI 两种方式
4. **可复现性**：实验配置完整记录，易于复现

## 架构设计

### 配置优先级（两层）

默认配置 < 用户配置（配置文件 OR CLI 参数）

- 配置文件和 CLI 参数不互相覆盖
- 用户选择其中一种方式提供完整配置

### 目录结构

```
ai-playground/
├── configs/
│   ├── default.yaml           # 全局默认训练配置
│   ├── models/
│   │   ├── yolov11n.yaml      # YOLOv11-nano 模型配置 (nc, scale)
│   │   └── yolov11s.yaml      # YOLOv11-small 模型配置
│   └── experiments/
│       └── my_exp.yaml        # 用户自定义实验配置
├── utils/
│   └── config.py              # 配置加载、解析、合并
└── engine/
    └── train.py               # 使用 cfg 字典重构
```

## 配置文件结构

### `configs/default.yaml`

```yaml
# System
system:
  device: cuda
  seed: 42
  workers: 0

# Dataset
data: datasets/MY_TEST_DATA/data.yaml

# Training
train:
  epochs: 100
  batch_size: 16
  name: null                # 必填项
  save_dir: runs/train

# Optimizer
optimizer:
  type: Adam
  lr: 0.001
  betas: [0.9, 0.999]
  eps: 1.0e-08
  weight_decay: 0.0

# Scheduler
scheduler:
  type: CosineAnnealingLR
  warmup_epochs: 3
  min_lr: 1.0e-06

# Model
model:
  img_size: 640
  use_ema: true
  ema_decay: 0.9999

# Augmentation
augment:
  use_mosaic: true
  close_mosaic: 10
  mixup_prob: 0.0
```

### `configs/models/yolov11n.yaml`

```yaml
# YOLOv11-nano 模型配置
model:
  nc: 2                      # 类别数
  scale: n                   # 模型规模: n/s/m/l/x
```

## 核心模块设计

### `utils/config.py`

**主要函数：**

1. `load_yaml(file_path)` - 加载 YAML 文件
2. `merge_configs(base, override)` - 递归合并配置
3. `get_config(config_file, model_config, **kwargs)` - 获取最终配置
4. `parse_args()` - CLI 参数解析
5. `print_config(cfg)` - Rich 格式化打印配置

**CLI 支持：**

```bash
# 使用配置文件
python -m engine.train --config configs/experiments/my_exp.yaml

# 使用 CLI 参数（快捷参数）
python -m engine.train --name exp001 --epochs 200 --lr 0.005

# 使用嵌套覆盖
python -m engine.train --name exp001 optimizer.lr=0.005 scheduler.min_lr=1e-7
```

### `engine/train.py` 重构

**函数签名变更：**

```python
# 之前
def train(model, config_path, epochs=100, batch_size=16, ...):

# 之后
def train(model, cfg: Dict[str, Any], data_config=None):
```

**优化器和调度器工厂：**

```python
def _create_optimizer(model, cfg: Dict[str, Any])
def _create_scheduler(optimizer, cfg: Dict[str, Any], epochs: int)
```

从配置字典读取参数，支持多种类型扩展。

### `utils/path_helper.py` 更新

```python
def get_save_dir(base_dir: str, name: Optional[str] = None) -> Path
```

支持实验名称参数，生成 `runs/train/{name}/` 路径。

## 配置打印示例

使用 `rich` 库格式化打印：

```
┌ ⚙️ Training Configuration ─────────────────────┐
│                                                │
│ System                                         │
│   device          cuda                         │
│   seed            42                           │
│                                                │
│ Dataset                                        │
│   data            datasets/MY_TEST_DATA/...    │
│                                                │
│ Training                                       │
│   epochs          100                          │
│   batch_size      16                           │
│   name            exp001                       │
│                                                │
│ Optimizer                                      │
│   type            Adam                         │
│   lr              0.001                        │
│                                                │
│ ...
└────────────────────────────────────────────────┘
```

## 使用场景

### 场景 1：使用默认配置 + CLI 覆盖

```bash
python -m engine.train --name baseline --epochs 100
```

### 场景 2：自定义配置文件

```yaml
# configs/experiments/high_lr.yaml
train:
  epochs: 300
  name: high_lr_test

optimizer:
  lr: 0.005
```

```bash
python -m engine.train --config configs/experiments/high_lr.yaml
```

### 场景 3：结合模型配置

```bash
python -m engine.train \
  --model-config configs/models/yolov11n.yaml \
  --name exp001 \
  --epochs 100
```

## 实施计划

1. 创建 `configs/` 目录结构和默认配置文件
2. 实现 `utils/config.py` 模块
3. 重构 `engine/train.py` 使用配置字典
4. 更新 `utils/path_helper.py`
5. 更新 `scripts/train_test.py` 使用新接口
6. 编写单元测试
7. 更新文档

## 向后兼容性

- 保持 `train()` 函数的核心训练逻辑不变
- 现有的 `print_training_info()` 等工具函数继续工作
- 逐步迁移现有脚本到新配置系统
