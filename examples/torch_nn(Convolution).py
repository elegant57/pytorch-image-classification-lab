import torch
import torch.nn.functional as func
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#Module
class NewNet(torch.nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()      #继承父类的初始化函数

    def forward(self,input1):       #forward直接由父类__call__函数调用，不需要再用New_net.forward调用
        output1 = input1 + 1
        return output1

neural_net = NewNet()
x = neural_net(torch.tensor(1.0))
print(x)

#Neural Network_Convolution（卷积）
input2 = torch.tensor([[1, 2, 0, 3, 1],
                       [0,1,2,3,1],
                       [1,2,1,0,0],
                       [5,2,3,1,1],
                       [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])        #卷积核

input2 = torch.reshape(input = input2, shape = (1, 1, 5, 5))      #batch_size,channel,H,W（torch.nn.functional神经网络的输入需要的是tensor类型的四维数据）
kernel = torch.reshape(input = kernel,shape = (1,1,3,3))
print(input2.shape)
print(kernel.shape)

output = func.conv2d(input = input2, weight = kernel, stride = 1)
print(output)
output2 = func.conv2d(input = input2, weight = kernel, stride = 2)
print(output2)
output3 = func.conv2d(input = input2, weight = kernel, stride = 1, padding = 1)        #paddng:填充一“圈”
print(output3)

# torch.nn
dataset = torchvision.datasets.CIFAR10(root = "./data/data_CIFAR", download = True,
                                      transform = torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset = dataset, batch_size = 64, shuffle = True)

class CIFAR10Net(torch.nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, stride = 1, padding = 0)

    def forward(self, input3):
        input3 = self.conv(input3)
        return input3

writer = SummaryWriter('CIFAR10_logs')

neural_test = CIFAR10Net()
step = 1
for data in dataloader:
    imgs,targets = data
    conv1 = neural_test(imgs)       #imgs.shape:[64,3,32,32] -> conv1.shape:[64,6,30,30]
    #print(conv1.shape)        # 由于卷积核与padding参数的原因，输出的H与W可能与之前不同，且通道变为设置的6(用SummaryWriter显示不出来)
    conv2 = torch.reshape(input = conv1, shape = (-1,3,30,30))      #-1表示对新shape的batch_size不确定
    writer.add_images(tag = 'input', img_tensor = imgs,global_step = step)
    writer.add_images(tag = 'output', img_tensor = conv2, global_step = step)
    step += 1

writer.close()