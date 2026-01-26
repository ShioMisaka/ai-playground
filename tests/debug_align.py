"""调试对齐问题"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.table import print_detection_header, format_detection_train_line, format_detection_val_line

# 模拟用户的数据
train_line = format_detection_train_line(9.4229, 6.0828, 0.6283, 2.7118, "100% ━━━━━━━━━━━━━━━━━━━━━ 34/34 1.1s/it 37.0s<0.0s")
val_line = format_detection_val_line(13.3451, 7.1566, 1.9718, 4.2167, 0.0)

print("-" * 54)
print_detection_header()
print(train_line)
print(val_line)
print()

# 标记每个字符的位置（从0开始）
header = "         total_loss    box_loss    cls_loss    dfl_loss"
print("字符位置标记（0-based）:")
print(header)
print("         11111    22222    22223    33333    33334    44444")
print("         67890    67890    45678    90123    45678    90123")
print()

# 提取每列并标记
print("分析对齐:")
print(f"表头 'total_loss' 结束位置: {header.find('total_loss') + len('total_loss')}")
print(f"表头 'box_loss' 结束位置: {header.find('box_loss') + len('box_loss')}")
print()

train_data = "Train -     9.4229      6.0828      0.6283      2.7118"
print(f"数据第一列结束位置: {train_data.find('9.4229') + len('9.4229')}")
print(f"数据第二列结束位置: {train_data.find('6.0828', 15) + len('6.0828')}")
print()

# 检查是否对齐
header_total_end = header.find('total_loss') + len('total_loss')
data_first_end = train_data.find('9.4229') + len('9.4229')
print(f"'total_loss' 的 's' 位置: {header_total_end}")
print(f"'9.4229' 的 '9' 位置: {data_first_end}")
print(f"对齐: {header_total_end == data_first_end}")
