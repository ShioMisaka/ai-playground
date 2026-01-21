# engine/__init__.py

# 主训练流程
from .train import train

# 核心训练逻辑
from .training import train_one_epoch, print_metrics

# 验证和测试
from .validate import evaluate, test, validate

# 简单/遗留训练函数
from .simple import train_fc

# 分类和检测专用训练
from .classifier import (train_one_epoch as train_one_epoch_cls,
                         validate as validate_cls, train_classifier)
from .detector import (train_one_epoch as train_one_epoch_det,
                       validate as validate_det, train_detector)

# 可视化模块
from .visualize import (
    # 基础工具
    enhance_contrast,
    load_image,
    get_coordatt_attention,
    get_crossatt_attention,
    get_improved_crossatt_attention,
    # 检测任务可视化
    visualize_detection_attention,
    visualize_attention_comparison,
    # 单图/多图可视化
    visualize_single_image_attention,
    visualize_multiple_images_attention,
    # 模型对比可视化
    visualize_single_model_attention,
    visualize_model_comparison,
    visualize_cross_attention_matrix,
    visualize_training_progress,
    # ImprovedCoordCrossAtt 专用可视化
    visualize_improved_cross_attention,
    visualize_improved_cross_attention_all_layers,
    visualize_improved_cross_attention_overlay,
)

# 模型对比模块
from .comparison import (
    train_and_compare_models,
    load_best_model,
    print_comparison_results,
)

__all__ = [
    # 主训练
    'train',

    # 核心训练逻辑
    'train_one_epoch',
    'print_metrics',

    # 验证/测试
    'validate',
    'evaluate',
    'test',

    # 简单训练
    'train_fc',

    # 分类和检测专用
    'train_classifier',
    'train_detector',

    # 可视化 - 基础工具
    'enhance_contrast',
    'load_image',
    'get_coordatt_attention',
    'get_crossatt_attention',
    'get_improved_crossatt_attention',

    # 可视化 - 检测任务
    'visualize_detection_attention',
    'visualize_attention_comparison',

    # 可视化 - 单图/多图
    'visualize_single_image_attention',
    'visualize_multiple_images_attention',

    # 可视化 - 模型对比
    'visualize_single_model_attention',
    'visualize_model_comparison',
    'visualize_cross_attention_matrix',
    'visualize_training_progress',

    # 可视化 - ImprovedCoordCrossAtt
    'visualize_improved_cross_attention',
    'visualize_improved_cross_attention_all_layers',
    'visualize_improved_cross_attention_overlay',

    # 模型对比
    'train_and_compare_models',
    'load_best_model',
    'print_comparison_results',
]
