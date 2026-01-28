"""
路径辅助工具模块

提供路径处理相关的辅助函数。
"""
from pathlib import Path


def get_save_dir(base_dir: str) -> Path:
    """获取唯一的保存目录，自动递增避免冲突

    如果 base_dir 不存在，直接使用；如果存在，则尝试添加 _1, _2, ... 后缀。

    Args:
        base_dir: 基础路径（如 "runs/train/exp"）

    Returns:
        可用的 Path 对象（文件夹已创建）

    Examples:
        >>> get_save_dir("runs/train/exp")      # 不存在 -> runs/train/exp
        >>> get_save_dir("runs/train/exp")      # 存在 -> runs/train/exp_1
        >>> get_save_dir("runs/train/exp")      # 都存在 -> runs/train/exp_2
    """
    base_path = Path(base_dir)

    # 如果路径不存在，直接使用并创建
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    # 路径已存在，尝试递增后缀 _1, _2, _3, ...
    suffix = 1
    while True:
        new_path = base_path.with_name(f"{base_path.name}_{suffix}")
        if not new_path.exists():
            new_path.mkdir(parents=True, exist_ok=True)
            return new_path
        suffix += 1
