import torch
import torchvision
from torch.nn import Sequential
from torchvision.datasets import CIFAR10


class SequentialNet(torch.nn.Module):
    def __init__(self):
        super(SequentialNet,self).__init__()
        self.model = Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=1024, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=10)
        )

    def forward(self,net_test):
        net_test = self.model(net_test)
        return net_test

dataset = CIFAR10(root = './data/dataset/data_CIFAR', train = False, download = True,
                  transform = torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 10)
seq_net = SequentialNet()
all_batches = []
all_targets = []
cross_loss = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(seq_net.parameters(),lr = 0.01)        # 随机梯度下降（stochastic gradient descent），lr为学习率
for epoch in range(50):
    running_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        output_data = seq_net(imgs)
        #print(output_data.shape)
        #print(output_data)
        #print(targets)
        result_loss = cross_loss(output_data, targets)
        optim.zero_grad()       # 先将上次循环的算出来的梯度归零
        result_loss.backward()
        optim.step()        # 反向传播，依据梯度优化神经网络
        running_loss += result_loss
        #all_batches.append(output_data)
        #all_targets.append(targets)
    print(running_loss/dataset.__len__())
'''
print(all_batches[0])
print(all_targets[0])
for i in range(len(all_targets)):
    j = 0
    for img,label in zip(all_batches[i],all_targets[i]):
        output_loss = cross_loss(img,label)
        optim.zero_grad()
        output_loss.backward()      #反馈梯度（grad）
        optim.step()
        j += 1
'''