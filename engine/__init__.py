# engine/__init__.py

from .train import train
from .validate import evaluate, test
from .classifier import (train_one_epoch, validate, train_classifier)
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
    # 训练
    'train',
    'train_one_epoch',
    'validate',
    'train_classifier',
    'train_detector',

    # 验证/测试
    'evaluate',
    'test',

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
