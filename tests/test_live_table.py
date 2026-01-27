"""测试 LiveTableLogger

演示动态表格刷新功能，模拟训练过程。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
from utils import LiveTableLogger



def simulate_training():
    """模拟训练过程，演示 LiveTableLogger 的使用"""

    # 定义表格列
    columns = ["total_loss", "box_loss", "cls_loss", "dfl_loss"]

    # 创建 Logger
    logger = LiveTableLogger(
        columns=columns,
        total_epochs=3,
    )

    # 启动 Logger
    logger.start()

    # 模拟 3 个 epoch
    for epoch in range(1, 4):
        lr = 0.001 * (0.95 ** (epoch - 1))
        logger.start_epoch(epoch, lr)

        # 模拟训练：100 个批次
        num_batches = 100
        epoch_start_time = time.time()
        for batch in range(num_batches):
            # 模拟损失值逐渐下降
            base_loss = 10.0 / (epoch + 1) + random.uniform(-0.5, 0.5)
            box_loss = base_loss * 0.6 + random.uniform(-0.1, 0.1)
            cls_loss = base_loss * 0.2 + random.uniform(-0.05, 0.05)
            dfl_loss = base_loss * 0.2 + random.uniform(-0.05, 0.05)

            train_data = {
                "total_loss": base_loss,
                "box_loss": box_loss,
                "cls_loss": cls_loss,
                "dfl_loss": dfl_loss,
            }

            # 更新训练行
            elapsed = time.time() - epoch_start_time
            logger.update_row(
                "train",
                train_data,
                progress={
                    "current": batch,
                    "total": num_batches,
                    "elapsed": elapsed,
                },
            )

            # 模拟处理时间
            time.sleep(0.02)

        # 模拟验证
        val_base_loss = base_loss * 1.1 + random.uniform(-0.3, 0.3)
        val_box_loss = val_base_loss * 0.6 + random.uniform(-0.1, 0.1)
        val_cls_loss = val_base_loss * 0.2 + random.uniform(-0.05, 0.05)
        val_dfl_loss = val_base_loss * 0.2 + random.uniform(-0.05, 0.05)
        map50 = (epoch / 3) * 0.8 + random.uniform(-0.05, 0.05)  # 模拟 mAP50 逐渐提升

        val_data = {
            "total_loss": val_base_loss,
            "box_loss": val_box_loss,
            "cls_loss": val_cls_loss,
            "dfl_loss": val_dfl_loss,
            "mAP50": map50,
        }

        logger.update_row("val", val_data)

        # 结束 epoch
        epoch_time = time.time() - epoch_start_time
        logger.end_epoch(epoch_time)

        time.sleep(1)  # 暂停一下，展示效果

    # 停止 Logger
    logger.stop()
    print("\n训练完成！")


if __name__ == "__main__":
    simulate_training()
