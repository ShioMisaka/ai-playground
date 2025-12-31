import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = datasets.MNIST("",is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

def evaluate(test_data, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # 测试时不需要计算梯度
        for data, target in test_data:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1) # 取概率最大的索引作为预测值
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total
    
def train_model(model: nn.Module, train_data, val_data, epochs = 3, optimizer = optim.Adam, criterion = nn.CrossEntropyLoss()):
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


def test_model(model: nn.Module, test_data, test_size = 2):
    for (n, (x, _)) in enumerate(test_data):
        if n > test_size: break
        
        # 1. 准备输入数据
        # x[0] 是 [1, 28, 28]，unsqueeze(0) 后变成 [1, 1, 28, 28] 以符合 CNN 的 4D 要求
        single_img = x[0].unsqueeze(0) 
        
        # 2. 预测
        model.eval()
        with torch.no_grad():
            output = model(single_img) # 直接传入 4D 张量
            predict = torch.argmax(output, dim=1)
        
        # 3. 可视化
        plt.figure(n)
        # imshow 需要 [28, 28]，所以用 squeeze 掉多余维度
        plt.imshow(x[0].squeeze()) 
        plt.title(f"prediction: {predict.item()}")

    plt.show()

from models.cnn_t import CNN_2Ct

if __name__ == "__main__":
    model = CNN_2Ct()

    train_datas = get_data_loader(is_train=True)
    test_datas = get_data_loader(is_train=False)

    train_model(model, train_datas, test_datas, epochs=2)

    test_model(model, test_datas, 2)


