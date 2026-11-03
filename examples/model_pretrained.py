import torchvision
import torch.nn
from torch import nn
from torch.nn import Sequential
from torchvision import transforms
from torchvision.models import VGG16_Weights

vgg16_false = torchvision.models.vgg16(weights = None)
vgg16_true = torchvision.models.vgg16(weights = VGG16_Weights.DEFAULT)      # 使用vgg的pretrained参数（用ImageNet训练）


dataset = torchvision.datasets.CIFAR10(root = './data/data_CIFAR', download = True, transform = transforms.ToTensor())

class SequentialNet(torch.nn.Module):
    def __init__(self):
        super(SequentialNet,self).__init__()
        self.model = Sequential(
            torch.nn.Linear(in_features=1000, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features=100, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5, inplace = False)
        )

    def forward(self,net_test):
        net_test = self.model(net_test)
        return net_test
seq_net = SequentialNet()
vgg16_true.add_module('new',seq_net)
print(vgg16_true)

vgg16_false.classifier.add_module('7',nn.ReLU(True))
vgg16_false.classifier.add_module('8',nn.Dropout(p = 0.5, inplace = False))
vgg16_false.classifier.add_module('9',nn.Linear(1000,100))
vgg16_false.classifier.add_module('10',nn.ReLU(True))
vgg16_false.classifier.add_module('11',nn.Dropout(p = 0.5, inplace = False))
vgg16_false.classifier.add_module('12',nn.Linear(100,10))

print(vgg16_false)