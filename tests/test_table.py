"""测试表格格式化对齐效果"""

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
    print("=" * 70)
    print("表格对齐测试")
    print("=" * 70)

    # 打印表头
    print_detection_header()

    # 打印训练行（实时更新，带进度条）
    progress_bar = _format_progress_bar(33, 34, 38.9)
    train_line = format_detection_train_line(8.4084, 5.1114, 0.8143, 2.4826, progress_bar)
    print(f"\r{train_line}", end='', flush=True)
    print()  # 换行

    # 打印训练最终结果
    train_final = format_detection_train_line(10.9631, 6.3689, 1.0877, 3.5065)
    print(train_final)

    # 打印验证行
    val_line = format_detection_val_line(16.1842, 7.1980, 3.0886, 5.8976, 0.0)
    print(val_line)

    print("\n" + "=" * 70)
    print("边界测试：不同长度的数值")
    print("=" * 70)

    print_detection_header()

    # 测试不同长度的数值
    test_cases = [
        # (loss, box_loss, cls_loss, dfl_loss)
        (1.2345, 0.1234, 0.0123, 0.0012),  # 小数值
        (999.9999, 88.8888, 7.7777, 0.6666),  # 大数值
        (10.9631, 6.3689, 1.0877, 3.5065),  # 中等数值
    ]

    for i, (loss, box, cls, dfl) in enumerate(test_cases):
        if i == 0:
            line = format_detection_train_line(loss, box, cls, dfl, "  [训练进度示例]")
        else:
            line = format_detection_train_line(loss, box, cls, dfl)
        print(line)

    val_line = format_detection_val_line(16.1842, 7.1980, 3.0886, 5.8976, 59.13)
    print(val_line)
