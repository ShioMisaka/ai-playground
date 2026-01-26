"""测试表格对齐 - 添加参考线"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.table import print_detection_header, format_detection_train_line, format_detection_val_line


def _format_progress_bar(current: int, total: int, elapsed: float) -> str:
    """格式化进度条"""
    progress = (current + 1) / total
    percent = int(progress * 100)
    bar_width = 20
    filled = int(progress * bar_width)
    bar = '━' * filled + '─' * (bar_width - filled)
    it_time = elapsed / (current + 1) if current > 0 else 0
    eta = it_time * (total - current - 1)
    return f"{percent}% ━{bar} {current + 1}/{total} {it_time:.1f}s/it {elapsed:.1f}s<{eta:.1f}s"


if __name__ == "__main__":
    print("显示每10个字符位置的参考线:")
    print("         1         2         3         4         5         6")
    print("1234567890123456789012345678901234567890123456789012345678901234567890")
    print()

    print_detection_header()

    progress_bar = _format_progress_bar(33, 34, 38.9)
    train_line = format_detection_train_line(8.4084, 5.1114, 0.8143, 2.4826, progress_bar)
    print(f"\r{train_line}")

    train_final = format_detection_train_line(10.9631, 6.3689, 1.0877, 3.5065)
    print(train_final)

    val_line = format_detection_val_line(16.1842, 7.1980, 3.0886, 5.8976, 0.0)
    print(val_line)

    print()
    print("=" * 70)
    print("测试不同长度的数值对齐:")
    print("=" * 70)
    print("         1         2         3         4         5")
    print("1234567890123456789012345678901234567890123456789012345")
    print()

    print_detection_header()

    # 小数值
    print(format_detection_train_line(1.2345, 0.1234, 0.0123, 0.0012))
    # 中等数值
    print(format_detection_train_line(10.9631, 6.3689, 1.0877, 3.5065))
    # 大数值
    print(format_detection_train_line(999.9999, 88.8888, 7.7777, 0.6666))

    val_line = format_detection_val_line(16.1842, 7.1980, 3.0886, 5.8976, 0.5913)
    print(val_line)
