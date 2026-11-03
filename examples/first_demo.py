from torch.utils.data import Dataset
from PIL import Image                         #Python Image Library (Image.open(Path):打开一个图片文件)
import os                                     #operating system

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)  #将两个地址（字符串）用\\连接，用于拼接地址
        self.img_path = os.listdir(self.path)                   #os库中的listdir()用于获取指定路径下所有文件的名字，并返回一个列表（文件名）

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        #img_item = Image.open(os.path.join(self.path,img_name))
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)
root_dir = 'data/dataset/train'
label_dir_ants = 'ants'
label_dir_bees = 'bees'

ants_dataset = MyData(root_dir,label_dir_ants)
bees_dataset = MyData(root_dir,label_dir_bees)

print("ant")
print(ants_dataset.__len__())
img_ant = ants_dataset.__getitem__(1)[0]
img_ant.show()

print("Bee")
print(bees_dataset.__len__())
img_bees = bees_dataset.__getitem__(1)[0]
img_bees.show()

'''
train_dataset = ants_dataset + bees_dataset
print(train_dataset.__len__())
img = train_dataset.__getitem__(124)[0]
img.show()
'''