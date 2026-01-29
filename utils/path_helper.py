"""
路径辅助工具模块

提供路径处理相关的辅助函数。
"""
from pathlib import Path
from typing import Optional


def get_save_dir(base_dir: str, name: Optional[str] = None) -> Path:
    """获取唯一的保存目录，自动递增避免冲突

    Args:
        base_dir: 基础路径（如 "runs/train"）
        name: 实验名称（可选）

    Returns:
        可用的 Path 对象（文件夹已创建）

    Examples:
        >>> get_save_dir("runs/train", "exp")      # -> runs/train/exp
        >>> get_save_dir("runs/train", "exp")      # -> runs/train/exp_1
    """
    base_path = Path(base_dir)

    if name:
        base_path = base_path / name

    return _create_unique_path(base_path)


def _create_unique_path(base_path: Path) -> Path:
    """创建唯一路径，如存在则递增后缀

    Args:
        base_path: 基础路径

    Returns:
        可用的唯一 Path 对象
    """
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    suffix = 1
    while True:
        new_path = base_path.with_name(f"{base_path.name}_{suffix}")
        if not new_path.exists():
            new_path.mkdir(parents=True, exist_ok=True)
            return new_path
        suffix += 1
