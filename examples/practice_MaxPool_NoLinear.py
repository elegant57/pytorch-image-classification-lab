import torch
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

trans_tensor = transforms.ToTensor()
trans_resize = transforms.Resize((224,512))
label_dic = {'ants':1,'bees':2}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.imgs_path = os.path.join(self.root_dir,self.label_dir)
        self.imgs = os.listdir(str(self.imgs_path))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,index):
        img_path = os.path.join(str(self.imgs_path),self.imgs[index])
        img = Image.open(img_path)
        img_tensor = trans_tensor(img)
        img_data = trans_resize(img_tensor)
        label = label_dic[self.label_dir]
        return img_data,label

class MaxPool(torch.nn.Module):
    def __init__(self):
        super(MaxPool,self).__init__()
        self.max_pool = torch.nn.MaxPool2d(kernel_size = 2, padding = 0, ceil_mode = True)
        self.sigmoid = torch.nn.Sigmoid()       #也可改为其他的非线性函数，比如ReLU

    def forward(self,input_data):
        output_data = self.max_pool(input_data)
        output = self.sigmoid(output_data)
        return output

my_dataset = MyDataset('data/dataset/train','ants')
my_dataloader = torch.utils.data.DataLoader(dataset = my_dataset, batch_size= 32 ,shuffle = False)
writer = SummaryWriter('practice')

model = MaxPool()
step = 1
for data in my_dataloader:
    imgs,targets = data
    imgs_pool = model(imgs)
    writer.add_images(tag = 'pool_no linear_before', img_tensor = imgs, global_step = step)
    writer.add_images(tag = 'pool_no linear', img_tensor = imgs_pool,global_step = step)
    step += 1

writer.close()