import os
from PIL import Image
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 将dataset分类映射为1和2
label_dic = {'ants':1, 'bees':2}
# 定义一个新的Dataset类，用来加载数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir,label_dir):
        self.label_dir = label_dir
        self.imgs_path = os.path.join(root_dir,label_dir)
        self.imgs = os.listdir(str(self.imgs_path))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,index):
        img_path = os.path.join(str(self.imgs_path),self.imgs[index])
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)
        img_resize = transforms.Resize((32,32))(img_tensor)
        label = label_dic[self.label_dir]
        return img_resize, label

#定义一个较为完全的神经网络模型，经过三层卷积和三层最大池化，其中并运用ReLU激活函数模拟
class NeuralNets(torch.nn.Module):
    def __init__(self):
        super(NeuralNets, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, stride = 1, padding = 2),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1,padding = 2),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1,padding = 2),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            nn.Flatten(),
            nn.Linear(in_features = 4 * 4 * 64, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 10),
            nn.ReLU(),
            nn.Linear(in_features = 10, out_features = 2)
        )

    def forward(self,input_data):
        output_data = self.model(input_data)
        return output_data

img_dir = 'data/dataset/train'
label_dir1 = 'ants'
label_dir2 = 'bees'

dataset1 = MyDataset(img_dir,label_dir1)
dataset2 = MyDataset(img_dir,label_dir2)
dataset = torch.utils.data.ConcatDataset([dataset1,dataset2])       # 将两个Dataset合并成一个
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 32, shuffle = True, drop_last = False)

seq_net = NeuralNets()
all_batches = []
for data in dataloader:
    all_batches.append(data)
    imgs,target = data
    trained_data = seq_net(imgs)
    print(trained_data)

writer = SummaryWriter('practice')
print(all_batches[0][1])
writer.add_graph(seq_net, all_batches[0][0])
writer.close()