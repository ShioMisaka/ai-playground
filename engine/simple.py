"""
简单/演示训练函数模块

提供基础的训练函数用于演示和简单场景。
"""
import torch
import torch.nn as nn
import torch.optim as optim

from .validate import evaluate


def train_fc(model: nn.Module, train_data, val_data, epochs=3,
             optimizer=optim.Adam, criterion=nn.CrossEntropyLoss()):
    """简单的全连接网络训练

    Args:
        model: 模型
        train_data: 训练数据加载器
        val_data: 验证数据加载器
        epochs: 训练轮数
        optimizer: 优化器类
        criterion: 损失函数
    """
    op = optimizer(model.parameters(), lr=0.001)
    print("Training begins...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_data):
            op.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            op.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_data)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {100 * evaluate(val_data, model):.2f}%")

    print("Training completed...")
