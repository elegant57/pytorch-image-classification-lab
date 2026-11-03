from model_save import *

# # 方式1 -》保存方式1-》加载模型
# vgg16_model1 = torch.load('vgg16_model1.pth', weights_only = False)     # pytorch2.6之后weights_only参数默认值改变，需修改为False
# # print(vgg16_model1)
#
# # 方式2，加载模型
# vgg16 = torchvision.models.vgg16(weights = None)
# vgg16.load_state_dict(torch.load('vgg16_model2.pth', weights_only = False))
# print(vgg16)

# 3，自己的神经网络
# my_net = torch.load('seq_net.pth', weights_only = False)  # -> 缺少__main__方法而不能直接被调用
# class SequentialNet(torch.nn.Module):
#     def __init__(self):
#         super(SequentialNet,self).__init__()
#         self.model = Sequential(
#             torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2),
#             torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2),
#             torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2),
#             torch.nn.Flatten(),
#             torch.nn.Linear(in_features=1024, out_features=64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(in_features=64, out_features=10)
#         )
#
#     def forward(self,net_test):
#         net_test = self.model(net_test)
#         return net_test
my_net = torch.load('seq_net.pth', weights_only = False)
print(my_net)

# 4,直接引用model_save.py文件
# my_net2 = SequentialNet()
# print(my_net2)