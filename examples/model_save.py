import torch
import torchvision
from torch import nn
from torch.ao.nn.quantized import Dropout
from torch.nn import Sequential

vgg16 = torchvision.models.vgg16(weights = None)
# 保存方式1，模型结构+模型参数
torch.save(vgg16, 'vgg16_model1.pth')

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), 'vgg16_model2.pth')         # 只保存了参数，输出也只有参数，模型结构需重写

# 3，保存自己的神经网络
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
            torch.nn.Linear(in_features=1024, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=100)
            # torch.nn.ReLU(),
            # torch.nn.Linear(in_features=64, out_features=10)
        )

    def forward(self,net_test):
        net_test = self.model(net_test)
        return net_test

class CIFAR100Net(torch.nn.Module):
    def __init__(self):
        super(CIFAR100Net,self).__init__()
        self.model = Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=100)
        )
    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    seq_net = SequentialNet()
    torch.save(seq_net, 'seq_net.pth')
    vgg16 = vgg16
    print(vgg16)
    CIFAR100net = CIFAR100Net()
    torch.save(CIFAR100net, 'CIFAR100_net.pth')