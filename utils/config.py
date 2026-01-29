"""
配置管理模块

提供 YAML 配置文件的加载、解析、合并和验证功能。
"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    Args:
        file_path: YAML 文件路径

    Returns:
        配置字典，如果文件为空则返回空字典

    Raises:
        FileNotFoundError: 如果文件不存在
        yaml.YAMLError: 如果 YAML 格式错误
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        if not content.strip():
            return {}
        return yaml.safe_load(content) or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并两个配置字典。

    override 中的值会覆盖 base 中的值，但嵌套字典会进行深度合并。

    Args:
        base: 基础配置字典
        override: 覆盖配置字典

    Returns:
        合并后的配置字典
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = merge_configs(result[key], value)
        else:
            # 直接覆盖值
            result[key] = value

    return result


def _flatten_to_nested(flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    将扁平化的 CLI 参数转换为嵌套字典结构。

    Args:
        flat: 扁平字典，键使用点号分隔层级
              例如: {'optimizer.lr': 0.001, 'train.epochs': 100}

    Returns:
        嵌套字典
        例如: {'optimizer': {'lr': 0.001}, 'train': {'epochs': 100}}
    """
    result: Dict[str, Any] = {}

    for flat_key, value in flat.items():
        parts = flat_key.split('.')
        current = result

        # 遍历路径，创建嵌套结构
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # 如果中间路径不是字典，则替换为字典
                current[part] = {}
            current = current[part]

        # 设置最终值
        current[parts[-1]] = value

    return result


def get_config(
    config_file: Optional[str] = None,
    model_config: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    加载并合并配置文件和 CLI 参数，返回最终配置。

    配置优先级（从高到低）：
    1. CLI kwargs (通过 **kwargs 传入)
    2. 用户配置文件 (config_file)
    3. 模型配置文件 (model_config)
    4. 默认配置 (configs/default.yaml)

    注意: config_file 和 kwargs 不能同时提供（二选一）。

    Args:
        config_file: 用户自定义配置文件路径
        model_config: 模型配置文件路径（如 configs/models/yolov11n.yaml）
        **kwargs: CLI 参数（扁平格式，如 optimizer.lr=0.001）

    Returns:
        合并后的最终配置字典

    Raises:
        ValueError: 如果 train.name 未设置，或同时提供 config_file 和 kwargs
        FileNotFoundError: 如果配置文件不存在
    """
    # 1. 加载默认配置
    default_config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
    config = load_yaml(str(default_config_path))

    # 2. 合并模型配置（如果提供）
    if model_config:
        model_cfg = load_yaml(model_config)
        config = merge_configs(config, model_cfg)

    # 3. 合并用户配置或 CLI 参数（二选一）
    if config_file and kwargs:
        raise ValueError("不能同时提供 config_file 和 CLI kwargs，请二选一")

    if config_file:
        # 从配置文件加载
        user_cfg = load_yaml(config_file)
        config = merge_configs(config, user_cfg)
    elif kwargs:
        # 从 CLI 参数转换并合并
        nested_kwargs = _flatten_to_nested(kwargs)
        config = merge_configs(config, nested_kwargs)

    # 4. 验证必需字段
    if not config.get('train', {}).get('name'):
        raise ValueError(
            "train.name 必须设置！请通过以下方式之一提供：\n"
            "  1. 在配置文件中设置 train.name\n"
            "  2. 通过 CLI 参数: --train.name my_experiment"
        )

    return config
