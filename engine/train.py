import torch
import torch.nn as nn
import torch.optim as optim

from .validate import evaluate

def train(model: nn.Module, train_data, val_data, epochs = 3, optimizer = optim.Adam, criterion = nn.CrossEntropyLoss()):
    op = optimizer(model.parameters(), lr=0.001)
    print("Training begins...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_data):
            op.zero_grad()                   # 1. 梯度清零
            output = model(data)             # 2. 前向传播
            loss = criterion(output, target) # 3. 计算损失
            loss.backward()                  # 4. 反向传播
            op.step()                        # 5. 更新参数

            running_loss += loss.item()
            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_data)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {100 * evaluate(val_data, model):.2f}%")
    
    print("Training completed...")
