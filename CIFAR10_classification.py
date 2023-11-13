import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from module import *
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)

train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 搭建神将网络,10分类的一个网络
shuran = Shuran()
shuran = shuran.cuda()

# 损失函数
loss_fn = CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(shuran.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0
# 记录测试的步数
total_test_step = 0
# 训练轮数
epoc = 101
writer = SummaryWriter("log")

for i in range(epoc):
    print("第{}轮训练开始".format(i + 1))
    for data in train_loader:
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        result = shuran(imgs)
        loss_result = loss_fn(result, labels)
        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数{},Loss:{}".format(total_train_step, loss_result))
            writer.add_scalar("train_data", loss_result, total_train_step)

    loss_sum = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            imgs = imgs.cuda()
            labels = labels.cuda()
            output = shuran(imgs)
            loss_result2 = loss_fn(output, labels)
            loss_sum = loss_sum + loss_result2

        print("第{}轮总共的Loss为{}".format(i, loss_sum))
        writer.add_scalar("test_loss", loss_sum, total_test_step)
        total_test_step = total_test_step + 1
        if i % 10 == 0:
          torch.save(shuran.state_dict(), "shuran_{}.pth".format(i))
          print("模型已保存")

writer.close()
