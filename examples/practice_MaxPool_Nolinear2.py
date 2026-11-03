import os
import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

trans_tensor = torchvision.transforms.ToTensor()
trans_resize = torchvision.transforms.Resize((128,256))
label_dic = {'ants':1, 'bees':2}
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir,label_dir):
        self.label_dir = label_dir
        self.imgs_dir = os.path.join(root_dir,label_dir)
        self.imgs = os.listdir(str(self.imgs_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,index):
        img_name = self.imgs[index]
        img = Image.open(os.path.join(str(self.imgs_dir),img_name))
        img_tensor = trans_tensor(img)
        img = trans_resize(img_tensor)
        label = label_dic[self.label_dir]
        return img,label

class NewNetwork(torch.nn.Module):
    def __init__(self):
        super(NewNetwork, self).__init__()
        self.maxPool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = False)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.maxPool(x)
        x = self.relu(x)
        return x

dataset = MyDataset('data/dataset/train', 'bees')
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 5, shuffle = False, drop_last = True)
net_model = NewNetwork()
writer = SummaryWriter('practice')
step = 1
for data in dataloader:
    imgs,target = data
    print(imgs.shape)
    imgs_trained = net_model(imgs)
    print(imgs_trained.shape)
    writer.add_images(tag = 'bees_maxPool_before', img_tensor = imgs, global_step = step)
    writer.add_images(tag = 'bees_maxPool_after', img_tensor = imgs_trained, global_step = step)
    step += 1
writer.close()