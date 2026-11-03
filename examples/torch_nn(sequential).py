import torch
from PIL import Image
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10

# 对CIFAR10数据对10个类别的卷积神经网络模型展示(分类问题)
CIFAR10_path = 'data/CIFAR10中10个类别卷积神经网络.jpg'
CIFAR10_mode = Image.open(CIFAR10_path)
CIFAR10_mode.show()

class OriginNet(torch.nn.Module):
    def __init__(self):
        super(OriginNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv3 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 2)
        self.max_pool3 = torch.nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.flat = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features = 1024, out_features = 64)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features = 64, out_features = 10)

    def forward(self,net_test):
        net_test = self.conv1(net_test)
        net_test = self.max_pool1(net_test)
        net_test = self.conv2(net_test)
        net_test = self.max_pool2(net_test)
        net_test = self.conv3(net_test)
        net_test = self.max_pool3(net_test)
        net_test = self.flat(net_test)
        net_test = self.linear1(net_test)
        net_test = self.relu(net_test)         # 输出层不使用激活函数
        net_test = self.linear2(net_test)
        return net_test

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

origin_net = OriginNet()
sequential_net = SequentialNet()

input_data = torch.ones((64,3,32,32))       # 创建一个batch_size为64，3个channel，32*32尺寸的全为1的数据

dataset = CIFAR10(root = 'data/data_CIFAR', train = True, download = True, transform = transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)

output_data1 = origin_net(input_data)
output_data2 = sequential_net(input_data)
print('result of OriginNet:{}'.format(output_data1.shape))
print('result of SequentialNet:{}'.format(output_data2.shape))

writer = SummaryWriter('Neural_Network')
writer.add_graph(origin_net, input_data)
writer.add_graph(sequential_net, input_data)
writer.close()