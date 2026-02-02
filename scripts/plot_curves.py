#!/usr/bin/env python3
"""
训练曲线可视化脚本

从训练日志 CSV 文件生成所有训练曲线图表。

示例:
    # 使用默认输出目录（CSV 所在目录）
    python scripts/plot_curves.py outputs/train/exp1/training_log.csv

    # 指定输出目录
    python scripts/plot_curves.py outputs/train/exp1/training_log.csv outputs/figures
"""
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import plot_training_curves


def main():

    csv_path = "outputs/train/exp1_2/training_log.csv"
    save_dir = "outputs/train/exp1_2/"


    # 检查 CSV 文件是否存在
    if not Path(csv_path).exists():
        print(f"错误: CSV 文件不存在: {csv_path}")
        sys.exit(1)

    print(f"CSV 路径: {csv_path}")
    print(f"输出目录: {save_dir}\n")

    # 生成所有曲线
    plot_training_curves(csv_path, save_dir)

    print("\n完成!")


if __name__ == '__main__':
    main()
