# 实验四 全连接神经网络
## 一、实验目的
1. 掌握PyTorch框架基础使用，理解数据集加载、数据变换、数据迭代器的用法。
2. 理解多层感知机（MLP）神经网络的基本结构，掌握全连接层nn.Linear的使用。
3. 熟悉神经网络训练整体流程。
4. 学会模型训练与评估，使用交叉熵损失函数、SGD优化器，在MNIST数据集上完成分类任务，观察测试集损失与准确率变化。

## 二、实验内容
1.	导入所需要的库函数：
```python
# 导入相应的库函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
```
2.	设置超参数batch_size=2048,构建数据集和数据加载器：
```python
batch_size = 2048
# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构建数据集与数据加载器
train_dataset = datasets.MNIST(
    'data', train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True
)

test_dataset = datasets.MNIST(
    'data', train=False,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size, shuffle=True
)

print("训练集数据shape:", train_loader.dataset.data.shape)
```
3.	定义MLP网络类（继承nn.Module），重写初始化层、定义前向计算逻辑：
```python
# 定义MLP模型
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        a1 = self.l1(x)
        x1 = F.relu(a1)
        a2 = self.l2(x1)
        return a2
```
4.	实例化模型，定义SGD优化器，设置学习率为0.01:
```python
model = mlp().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
print(model)
```
5.	循环训练30轮：
model.train()开启训练模式：遍历训练集，图像展平，梯度清零，前向、计算损失、反向传播、更新参数;model.eval()开启评估模式：torch.no_grad()禁用梯度；在测试集计算平均损失和准确率。
每一轮结束打印结果。
```python
epochs = 30
for epoch in range(epochs):
    # 训练阶段
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        # 将图片展平
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer.zero_grad()                # 梯度清零
        output = model(x)                    # 前向传播
        loss = F.cross_entropy(output, y)    #计算损失
        loss.backward()                      #反向传播求梯度
        optimizer.step()                     #更新参数

    # 测试阶段
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():                                                 #关闭梯度计算
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.view(x.shape[0], -1).to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y).item()
            pred = output.max(1, keepdim=True)[1]                         #取得分最大下标作为预测类别
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss = test_loss/(batch_idx+1)
    acc = correct / len(test_loader.dataset)
    print(f'epoch:{epoch:2d}, test_loss:{test_loss:.4f}, test_acc:{acc:.4f}')
```

## 三、实验结果与分析
1. 训练集shape输出

`训练集数据shape: torch.Size([60000, 28, 28])`

2. 模型结构输出两层全连接网络；
```python
mlp(
  (l1): Linear(in_features=784, out_features=128, bias=True)
  (l2): Linear(in_features=128, out_features=10, bias=True)
)
```

3. 训练过程中，随着epoch增加，测试损失逐步下降，测试准确率逐步上升。
```python
epoch: 0, test_loss:2.0438, test_acc:0.6519
epoch: 1, test_loss:1.9616, test_acc:0.6795
epoch: 2, test_loss:1.8724, test_acc:0.7001
epoch: 3, test_loss:1.7774, test_acc:0.7173
epoch: 4, test_loss:1.6784, test_acc:0.7301
epoch: 5, test_loss:1.5776, test_acc:0.7408
epoch: 6, test_loss:1.4781, test_acc:0.7506
epoch: 7, test_loss:1.3826, test_acc:0.7600
epoch: 8, test_loss:1.2934, test_acc:0.7698
epoch: 9, test_loss:1.2112, test_acc:0.7778
epoch:10, test_loss:1.1369, test_acc:0.7875
epoch:11, test_loss:1.0692, test_acc:0.7982
epoch:12, test_loss:1.0099, test_acc:0.8054
epoch:13, test_loss:0.9565, test_acc:0.8112
epoch:14, test_loss:0.9100, test_acc:0.8175
epoch:15, test_loss:0.8673, test_acc:0.8237
epoch:16, test_loss:0.8297, test_acc:0.8276
epoch:17, test_loss:0.7966, test_acc:0.8330
epoch:18, test_loss:0.7659, test_acc:0.8368
epoch:19, test_loss:0.7398, test_acc:0.8402
epoch:20, test_loss:0.7147, test_acc:0.8442
epoch:21, test_loss:0.6928, test_acc:0.8473
epoch:22, test_loss:0.6718, test_acc:0.8498
epoch:23, test_loss:0.6529, test_acc:0.8527
epoch:24, test_loss:0.6359, test_acc:0.8552
epoch:25, test_loss:0.6200, test_acc:0.8583
epoch:26, test_loss:0.6062, test_acc:0.8609
epoch:27, test_loss:0.5919, test_acc:0.8629
epoch:28, test_loss:0.5801, test_acc:0.8655
epoch:29, test_loss:0.5684, test_acc:0.8667
```

## 四、实验结论
本次实验基于PyTorch搭建两层MLP多层感知机，完成MNIST手写数字多分类。熟悉了数据集加载、网络定义、训练循环、模型评估整套深度学习流程。理解了全连接层、ReLU激活、交叉熵损失、SGD优化器的作用，区分训练模式和评估模式。
