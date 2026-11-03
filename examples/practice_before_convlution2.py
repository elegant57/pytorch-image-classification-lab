import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

trans_tensor = torchvision.transforms.ToTensor()
trans_resize = torchvision.transforms.Resize((224, 512))
target_dic = {'ants':1, 'bees':2}

class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        #self.root_dir = root_dir
        self.label_dir = label_dir
        self.img_dir = os.path.join(root_dir,label_dir)
        self.imgs = os.listdir(str(self.img_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path = os.path.join(str(self.img_dir),self.imgs[index])
        img_root = Image.open(path)
        img = trans_resize(trans_tensor(img_root))
        label = target_dic[self.label_dir]
        return img,label

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.convolution = torch.nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 0)

    def forward(self,x):
        output = self.convolution(x)
        return output

dir_path = 'data/dataset/train'
label_path = 'bees'
my_dataset = MyDataset(dir_path,label_path)
dataloader = DataLoader(dataset = my_dataset, batch_size = 32, shuffle = True, drop_last = False)
# DataLoader必须是Tensor或者Numpy数据类型

writer = SummaryWriter('practice')

my_network = NeuralNetwork()
step = 1
for data in dataloader:
    imgs,labels = data
    print(imgs.shape)
    imgs_trained = my_network(imgs)
    print(imgs_trained.shape)       #[32,6,112,256]
    #imgs_trained = torch.reshape(imgs_trained,(-1,3,112,256))
    writer.add_images(tag = 'bees', img_tensor = imgs, global_step = step)
    writer.add_images(tag = 'bees_trained', img_tensor = imgs_trained, global_step = step)
    step += 1

writer.close()



