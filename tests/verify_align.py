"""验证对齐修复"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.table import print_detection_header, format_detection_train_line

print("验证对齐（用竖线标记每列边界）:")
print()

# 获取表头和数据
header_line = "        total_loss    box_loss    cls_loss    dfl_loss"
train_line = format_detection_train_line(9.4229, 6.0828, 0.6283, 2.7118)

print(header_line)
print(train_line)
print()

# 标记关键位置
print("标记 'total_loss' 的 's' 和数值的 '9':")
marked_header = header_line.replace("total_loss", "total_lossS")
marked_train = train_line.replace("9.4229", "     9.422N")
print(marked_header)
print(marked_train)
print()

# 计算位置
print(f"表头 'total_loss' 从位置 {header_line.find('total_loss')} 开始，到 {header_line.find('total_loss') + 10} 结束")
print(f"数据 '9.4229' 从位置 {train_line.find('9.4229')} 开始，到 {train_line.find('9.4229') + 6} 结束")
print(f"数值所在列从位置 8 开始，宽度 10")
print()

# 验证最后一位对齐
header_s_pos = header_line.find('total_loss') + 9  # 's' 在 'total_loss' 中的位置
data_9_pos = train_line.find('9.4229') + 5  # '9' 在 '9.4229' 中的位置
print(f"'total_loss' 中 's' 的位置: {header_s_pos}")
print(f"'9.4229' 中 '9' 的位置: {data_9_pos}")
print(f"对齐: {header_s_pos == data_9_pos}")

# 实际显示每列内容
print()
print("分列显示:")
header_part = header_line[8:18]  # 从位置8开始的10个字符
data_part = train_line[8:18]
print(f"表头第一列: [{header_part}]")
print(f"数据第一列: [{data_part}]")
