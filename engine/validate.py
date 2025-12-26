import torch
import torch.nn as nn
import matplotlib as plt

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

def test(model: nn.Module, test_data, test_size = 2):
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