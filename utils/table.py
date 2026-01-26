"""
表格格式化工具模块

提供对齐的表格输出功能。
"""


def print_detection_header():
    """打印检测任务的表头"""
    # 定义列宽
    label_width = 8   # "Train -" 或 "Val   -"
    col_width = 10    # 每个数值列的宽度
    separator = "  "  # 列之间的分隔符（2个空格）

    # 计算前导空格：与 "Train - " 的长度一致
    leading_spaces = " " * label_width

    # 列名
    columns = ["total_loss", "box_loss", "cls_loss", "dfl_loss"]

    # 构建表头
    header = leading_spaces
    for i, col in enumerate(columns):
        if i > 0:
            header += separator
        # 右对齐，与数据列对齐
        header += f"{col:>{col_width}}"

    print(header)


def format_detection_train_line(loss: float, box_loss: float, cls_loss: float, dfl_loss: float,
                                 progress_bar: str = "") -> str:
    """格式化检测任务的训练行（带进度条）

    Args:
        loss: 总损失
        box_loss: 边框损失
        cls_loss: 分类损失
        dfl_loss: DFL损失
        progress_bar: 进度条字符串

    Returns:
        格式化后的字符串
    """
    label = "Train -"
    col_width = 10
    separator = "  "

    line = label + " "
    line += f"{loss:>{col_width}.4f}{separator}"
    line += f"{box_loss:>{col_width}.4f}{separator}"
    line += f"{cls_loss:>{col_width}.4f}{separator}"
    line += f"{dfl_loss:>{col_width}.4f}"

    if progress_bar:
        line += "  " + progress_bar

    return line


def format_detection_val_line(loss: float, box_loss: float, cls_loss: float, dfl_loss: float,
                               map50: float = None) -> str:
    """格式化检测任务的验证行（带mAP50）

    Args:
        loss: 总损失
        box_loss: 边框损失
        cls_loss: 分类损失
        dfl_loss: DFL损失
        map50: mAP50值（可选）

    Returns:
        格式化后的字符串
    """
    label = "Val   -"
    col_width = 10
    separator = "  "

    line = label + " "
    line += f"{loss:>{col_width}.4f}{separator}"
    line += f"{box_loss:>{col_width}.4f}{separator}"
    line += f"{cls_loss:>{col_width}.4f}{separator}"
    line += f"{dfl_loss:>{col_width}.4f}"

    if map50 is not None:
        line += f"    mAP50: {map50*100:>6.2f}%"

    return line
