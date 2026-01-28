"""
Coordinate Attention 可视化脚本

展示未训练的 CoordAtt 模块在图像上关注的重点位置
"""
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CoordAttWithVisualization
from engine import load_image, visualize_single_image_attention, visualize_multiple_images_attention


# ==================== 主函数 ====================
def main():
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 设置数据集路径
    data_root = 'datasets/MY_TEST_DATA/images/train'

    print("=" * 50)
    print("Coordinate Attention 可视化 (未训练)")
    print("=" * 50)

    # 创建 CoordAtt 模块 (输入通道=3，输出通道=3，用于RGB图像)
    coord_att = CoordAttWithVisualization(inp=3, oup=3, reduction=32)
    coord_att.eval()

    # 1. 单张图像详细可视化
    print("\n[1] 单张图像可视化...")
    sample_image = os.path.join(data_root, 'circle_0001.jpg')
    if os.path.exists(sample_image):
        img_tensor, img_display = load_image(sample_image, img_size=224)
        visualize_single_image_attention(
            coord_att, img_tensor, img_display,
            save_path='outputs/single_attention.png'
        )
    else:
        print(f"图像不存在: {sample_image}")

    # 2. 多张图像对比可视化
    print("\n[2] 多张图像对比可视化...")
    if os.path.exists(data_root):
        visualize_multiple_images_attention(
            coord_att, data_root, num_samples=4, img_size=224,
            save_path='outputs/multi_attention.png'
        )
    else:
        print(f"目录不存在: {data_root}")

    print("\n" + "=" * 50)
    print("可视化完成!")
    print("=" * 50)


if __name__ == '__main__':
    main()
