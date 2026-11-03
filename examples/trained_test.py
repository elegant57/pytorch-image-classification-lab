import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model_save import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = torchvision.datasets.CIFAR10(root='./data/data_CIFAR', train=True, download=True,
                                        transform=transforms.ToTensor())
print(dataset.class_to_idx)
# 反转class_to_idx的值与关键词
label_dic = {value:key for key,value in dataset.class_to_idx.items()}
print("反转之后的dictionary:",label_dic)


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((32, 32))])
class MyDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.imgs_name = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.imgs_name)

    def __getitem__(self, index):
        label = self.imgs_name[index][:-4]
        imgs = Image.open(os.path.join(self.root_dir,self.imgs_name[index]))
        imgs = imgs.convert('RGB')          # 将四通道转换为三通道
        image = transform(imgs)
        return image,label

my_dataset = MyDataset('./data/data_test')
dataloader = DataLoader(my_dataset, batch_size = 1, shuffle = False)

class SequentialNet(torch.nn.Module):
    def __init__(self):
        super(SequentialNet,self).__init__()
        self.model = Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=1024, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, net_test):
        net_test = self.model(net_test)
        return net_test

# my_vgg16 = SequentialNet()
my_vgg16 = torch.load('Mynet_trained.pth', weights_only = False, map_location = torch.device('cuda'))
print(my_vgg16)

test_target = []
test_pred = []

def remove_digits_translate(text_input):
    """使用translate方法去除数字"""
    # 创建转换表，将数字映射为None
    trans_table = str.maketrans('', '', '0123456789')
    return text_input.translate(trans_table)

# 测试
text = "Hello123 World456!"
result = remove_digits_translate(text)
# 原字符串: Hello123 World456!
# 去除数字后: Hello World!

# 先验证特定层
my_vgg16.eval()
with torch.no_grad():
    for data in dataloader:
        img, target = data
        img = img.to(device)
        img = my_vgg16(img)
        print(img)
        print(img.argmax(dim = 1))
        test_target.append(remove_digits_translate(str(target)[2:-3]))           # 将target的类型从tuple转为str类型便于查看
        test_pred.append(label_dic[img.argmax(dim=1).item()])

print("实际的targets为",test_target)
print("预测的targets为",test_pred)
accuracy = 0
for i,j in zip(test_target,test_pred):
    accuracy += torch.sum(torch.tensor(i == j)).item()
print("准确率为{:.2f}%".format(accuracy/len(test_target) * 100))