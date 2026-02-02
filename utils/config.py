"""
配置管理模块

提供 YAML 配置文件的加载、解析、合并和验证功能。
"""
import argparse
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


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


def merge_training_config(
    model_config: Optional[Union[str, Dict[str, Any]]] = None,
    user_config: Optional[Union[str, Dict[str, Any]]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge training configurations from multiple sources with proper priority handling.

    This function loads and merges configurations in the following priority order
    (from lowest to highest priority):

    1. **Default config** (configs/default.yaml) - Base configuration with defaults
    2. **Model config** (file path or dict) - Model-specific settings (architecture, etc.)
    3. **User config** (file path or dict) - User-defined training parameters
    4. **Overrides dict** (from kwargs) - Highest priority for CLI/programmatic overrides

    The merging is deep-recursive for nested dictionaries, meaning that nested values
    are merged rather than replaced entirely.

    Args:
        model_config: Model configuration as either:
            - File path (str) to a YAML file containing model config
            - Dictionary with model configuration keys
            - None to skip model config merging
        user_config: User configuration as either:
            - File path (str) to a YAML file containing user config
            - Dictionary with user configuration keys
            - None to skip user config merging
        overrides: Override dictionary with highest priority.
            Supports flat keys with dot notation (e.g., {'train.epochs': 100})
            which are automatically converted to nested structure.
            Can also contain nested dictionaries directly.
            None to skip overrides.

    Returns:
        Dict[str, Any]: Merged configuration dictionary with all sources combined
        according to the priority order.

    Raises:
        FileNotFoundError: If model_config or user_config is a file path that
            does not exist.
        yaml.YAMLError: If any YAML file cannot be parsed properly.

    Examples:
        Load only default config:
        >>> config = merge_training_config()

        Merge with model config from file:
        >>> config = merge_training_config(
        ...     model_config='configs/models/yolov11n.yaml'
        ... )

        Merge with model and user configs:
        >>> config = merge_training_config(
        ...     model_config={'model': {'nc': 2, 'scale': 'n'}},
        ...     user_config={'train': {'epochs': 200, 'batch_size': 32}}
        ... )

        Use CLI-style overrides with flat keys:
        >>> config = merge_training_config(
        ...     overrides={'train.epochs': 100, 'optimizer.lr': 0.001}
        ... )

        Combine all sources:
        >>> config = merge_training_config(
        ...     model_config='configs/models/yolov11n.yaml',
        ...     user_config='configs/experiments/my_exp.yaml',
        ...     overrides={'device': 'cpu', 'train.epochs': 50}
        ... )

        Note: In the last example, if all configs specify 'train.epochs':
        - Default: 100
        - Model config: 150
        - User config: 200
        - Overrides: 50
        The final value will be 50 (overrides have highest priority).
    """
    # Load default config
    default_config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
    cfg = load_yaml(str(default_config_path))

    # Merge model config
    if model_config is not None:
        if isinstance(model_config, str):
            model_cfg = load_yaml(model_config)
        else:
            model_cfg = model_config
        cfg = merge_configs(cfg, model_cfg)

    # Merge user config
    if user_config is not None:
        if isinstance(user_config, str):
            user_cfg = load_yaml(user_config)
        else:
            user_cfg = user_config
        cfg = merge_configs(cfg, user_cfg)

    # Apply overrides
    if overrides is not None:
        # Convert flat overrides to nested structure if needed
        nested_overrides = _flatten_to_nested(overrides)
        cfg = merge_configs(cfg, nested_overrides)

    return cfg


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


def _parse_value(value: str) -> Union[int, float, bool, str]:
    """
    将字符串值转换为适当的 Python 类型（int/float/bool/str）。

    尝试按以下顺序转换：
    1. int (整数)
    2. float (浮点数)
    3. bool (布尔值: true/false/yes/no/on/off，不区分大小写)
    4. str (默认字符串)

    Args:
        value: 要解析的字符串值

    Returns:
        转换后的值 (int, float, bool, 或 str)

    Examples:
        >>> _parse_value("42")
        42
        >>> _parse_value("3.14")
        3.14
        >>> _parse_value("true")
        True
        >>> _parse_value("hello")
        'hello'
    """
    # 尝试解析为 int
    try:
        return int(value)
    except ValueError:
        pass

    # 尝试解析为 float
    try:
        return float(value)
    except ValueError:
        pass

    # 尝试解析为 bool
    lower_value = value.lower()
    if lower_value in ('true', 'yes', 'on', '1'):
        return True
    if lower_value in ('false', 'no', 'off', '0'):
        return False

    # 默认返回字符串
    return value


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，用于训练脚本的配置。

    支持以下参数：
    - --config: 训练配置文件路径
    - --model-config: 模型配置文件路径
    - --name: 实验名称（覆盖 train.name）
    - --data: 数据集配置路径（覆盖 train.data）
    - --epochs: 训练轮数（覆盖 train.epochs）
    - --batch-size: 批次大小（覆盖 train.batch_size）
    - --lr: 学习率（覆盖 train.lr）
    - --device: 设备（cuda/cpu）（覆盖 train.device）
    - overrides: 位置参数，用于嵌套配置覆盖（如 optimizer.lr=0.001）

    Args:
        无（通过 argparse 自动解析 sys.argv）

    Returns:
        argparse.Namespace: 包含解析后的参数
        - config: str | None
        - model_config: str | None
        - name: str | None
        - data: str | None
        - epochs: int | None
        - batch_size: int | None
        - lr: float | None
        - device: str | None
        - overrides: Dict[str, Any]

    Examples:
        >>> # 命令行使用示例
        >>> # python train.py --config configs/my_config.yaml --device cuda
        >>> # python train.py --name exp1 --epochs 100 optimizer.lr=0.001
    """
    parser = argparse.ArgumentParser(
        description='YOLO 训练脚本 - 支持配置文件和 CLI 参数覆盖',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用配置文件
  python train.py --config configs/my_config.yaml

  # 使用 CLI 参数
  python train.py --name exp1 --epochs 100 --batch-size 16

  # 混合使用配置文件和 CLI 覆盖
  python train.py --config configs/base.yaml --lr 0.001

  # 嵌套参数覆盖（支持任意层级）
  python train.py --name exp1 optimizer.lr=0.001 train.amp=true

  # 完整示例
  python train.py \\
    --model-config configs/models/yolov11n.yaml \\
    --data datasets/coco8.yaml \\
    --name my_experiment \\
    --epochs 100 \\
    --batch-size 16 \\
    --lr 0.001 \\
    --device cuda \\
    optimizer.lr=0.001 \\
    train.amp=true
        """
    )

    # 主要配置参数
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='训练配置文件路径（YAML 格式）'
    )

    parser.add_argument(
        '--model-config',
        type=str,
        default=None,
        help='模型配置文件路径（如 configs/models/yolov11n.yaml）'
    )

    # 常用训练参数（支持快捷覆盖）
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='实验名称（覆盖 train.name）'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='数据集配置路径（覆盖 train.data）'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数（覆盖 train.epochs）'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批次大小（覆盖 train.batch_size）'
    )

    parser.add_argument(
        '--lr',
        '--learning-rate',
        type=float,
        default=None,
        dest='lr',
        help='学习率（覆盖 train.lr）'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', 'auto'],
        help='训练设备（覆盖 train.device）'
    )

    # 位置参数：嵌套配置覆盖（如 optimizer.lr=0.001）
    parser.add_argument(
        'overrides',
        nargs=argparse.REMAINDER,
        default=[],
        help='嵌套配置覆盖（格式: key.subkey=value，如 optimizer.lr=0.001）'
    )

    args = parser.parse_args()

    # 解析 overrides 参数（key=value 格式）
    overrides_dict: Dict[str, Any] = {}
    override_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_.]*)=(.+)$')

    for override in args.overrides:
        match = override_pattern.match(override)
        if not match:
            parser.error(f"无效的覆盖参数格式: {override}\n"
                        f"期望格式: key.subkey=value (例如: optimizer.lr=0.001)")

        key, value_str = match.groups()
        overrides_dict[key] = _parse_value(value_str)

    # 将解析后的 overrides 赋值回 args
    args.overrides = overrides_dict

    # 将快捷参数也转换为 overrides 字典格式（便于统一处理）
    if args.name:
        overrides_dict['train.name'] = args.name
    if args.data:
        overrides_dict['train.data'] = args.data
    if args.epochs is not None:
        overrides_dict['train.epochs'] = args.epochs
    if args.batch_size is not None:
        overrides_dict['train.batch_size'] = args.batch_size
    if args.lr is not None:
        overrides_dict['train.lr'] = args.lr
    if args.device:
        overrides_dict['train.device'] = args.device

    return args


def print_config(cfg: Dict[str, Any]) -> None:
    """
    使用 Rich 库格式化打印配置信息。

    Args:
        cfg: 配置字典
    """
    console.print()

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", width=15)
    table.add_column(style="green")

    _add_section(table, "System", cfg.get('system', {}))
    _add_section(table, "Dataset", {'data': cfg.get('data')})
    _add_section(table, "Training", cfg.get('train', {}))
    _add_section(table, "Optimizer", cfg.get('optimizer', {}))
    _add_section(table, "Scheduler", cfg.get('scheduler', {}))
    _add_section(table, "Model", cfg.get('model', {}))
    _add_section(table, "Augmentation", cfg.get('augment', {}))

    panel = Panel(
        table,
        title="[bold yellow]⚙️ Training Configuration[/bold yellow]",
        border_style="bright_blue",
        padding=(0, 1),
    )

    console.print(panel)
    console.print()


def _add_section(table: Table, title: str, section: Dict[str, Any]) -> None:
    """
    添加配置区块到 Rich 表格。

    Args:
        table: Rich 表格对象
        title: 区块标题
        section: 配置区块字典
    """
    if not section:
        return

    table.add_row("", "")
    table.add_row(f"[bold white]{title}[/bold white]", "")

    for key, value in section.items():
        if value is not None:
            table.add_row(key, str(value))
