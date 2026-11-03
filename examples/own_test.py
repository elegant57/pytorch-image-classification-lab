import os
from PIL import Image
from torch import tensor, nn
from torch.nn import Module
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
from model_save import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor()])
label_dic = {0:"ants", 1:"bees"}

class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.img_path = os.path.join(root_dir, label_dir)
        self.img_name = os.listdir(str(self.img_path))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(str(self.img_path), self.img_name[idx]))
        img = img.convert('RGB')
        img = transform(img)
        target = self.label_dir
        return img, target

dataset_ants = MyDataset('data/own_test','ants')
dataset_bees = MyDataset('data/own_test','bees')
dataset = ConcatDataset([dataset_ants, dataset_bees])

dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)

class MyNetwork(Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.model = Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, dilation = 1, stride = 1, padding = 2),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, dilation = 1, stride = 1, padding = 2),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, dilation = 1, stride = 1, padding = 2),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=32, out_features=10),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, x):
        x = self.model(x)
        return x

torchvision.models.vgg16()
torch.load('seq_net.pth', weights_only = False)
my_pra_net = torch.load('practice_Complete.pth', weights_only = False, map_location = 'cuda')

predict = []
real = []

accuracy = 0
my_pra_net.eval()
with torch.no_grad():
    for data in dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        output_img = my_pra_net(imgs)
        predict.append(label_dic[output_img.argmax(dim = 1).item()])
        targets = str(targets)[2:-3]
        real.append(targets)
        accuracy += torch.sum(tensor(label_dic[output_img.argmax(dim = 1).item()] == targets)).item()

print("实际的targets为", real)
print("预测的targets为", predict)
print("准确率为{:.2f}%".format(accuracy / len(dataset) * 100))
