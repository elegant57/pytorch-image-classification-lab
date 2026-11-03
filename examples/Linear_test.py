import torch
from torchvision import datasets, transforms


class LinearTest(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearTest, self).__init__()
        self.linear_model = torch.nn.Linear(in_features, out_features)

    def forward(self, input_data):
        output_data = self.linear_model(input_data)
        return output_data

dataset = datasets.CIFAR10(root = 'data/data_CIFAR', train = False, download = True, transform = transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 64, shuffle = True, drop_last = True)
linear_test = LinearTest(196608,10)
for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    # 第一种改尺寸的方式：torch.reshape(), 该方式可以任意改变尺寸，功能强大，但是修改完之后维度未变，仍是四维
    imgs_output1 = torch.reshape(imgs,(1,1,1,-1))       # 拉平为宽度
    print(imgs_output1.shape)
    # 第二种修改尺寸方式：铺平：torch.flatten, 该方式直接将数组平铺展开，实现降维，参数start_dim可设置改变之后的维度
    imgs_output2 = torch.flatten(input = imgs, start_dim = 0)
    print(imgs_output2.shape)
    imgs_output3 = linear_test(imgs_output1)
    print(imgs_output3.shape)
    imgs_output4 = linear_test(imgs_output2)
    print(imgs_output4.shape)