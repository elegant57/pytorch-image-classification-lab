import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# practice
# 获取图片数据路径
root_path = 'data/dataset/train'
label = 'ants'
images = os.listdir(os.path.join(root_path, label))

# 初始化transforms
trans_tensor = torchvision.transforms.ToTensor()
trans_resize = torchvision.transforms.Resize((64,128))

# 重建一个Dataset：dataset_pra
class GetDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_paths = os.listdir(str(self.path))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        img_path = self.img_paths[idx]
        img_jpg = Image.open(os.path.join(str(self.path),img_path))
        img_tensor = trans_tensor(img_jpg)
        img = trans_resize(img_tensor)
        target = self.label_dir
        return img,target

dataset_pra = GetDataset(root_path, label)

# DataLoader对dataset_pra）分堆
dataloader_pra = DataLoader(dataset = dataset_pra, batch_size = 1, shuffle = False, drop_last = False)

# 构建一个神经网络类，用于新数据集的卷积训练
class AntsNet(torch.nn.Module):
    def __init__(self):
        super(AntsNet, self).__init__()
        self.Convolution = torch.nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 0)

    def forward(self,x):
        x = self.Convolution(x)
        return x

ants_net = AntsNet()

writer = SummaryWriter('practice')

# 对数据进行展示
step = 1
for data in dataloader_pra:
    imgs,labels = data
    print(imgs.shape)
    trained_imgs = ants_net(imgs)
    print(trained_imgs.shape)
    #trained_imgs_reshape = torch.reshape(input = trained_imgs, shape = (-1,3,64,128))
    writer.add_images(tag = 'ants_input', img_tensor = imgs, global_step = step)
    writer.add_images(tag = 'ants_output', img_tensor = trained_imgs, global_step = step)
    step += 1

writer.close()