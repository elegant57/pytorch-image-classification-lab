import torch
import torchvision
from PIL import Image
from torch.nn import Module, Sequential
from torch import nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time

# 设置运行代码的设备（第二种用显卡的方式）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 设置标签字典
label_dic = {"ants":0, "bees":1}
# 创建自己的Dataset
class MyDataset(Dataset):
    def __init__(self,root_dir,label_dir):
        self.label_dir = label_dir
        self.img_dir = str(os.path.join(root_dir,label_dir))
        self.img_path = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = Image.open(os.path.join(self.img_dir,img_path))
        img_tensor = transforms.ToTensor()(img)
        img_resize = transforms.Resize((32,32))(img_tensor)
        label = label_dic[self.label_dir]
        return img_resize,label

# 创建自己的神经网络
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

# 加载数据集，并将两个数据整合为一个Dataset
data_ants_train = MyDataset('data/data_classed/train',"ants")
data_ants_test = MyDataset('data/data_classed/val','ants')
data_bees_train = MyDataset('data/data_classed/train','bees')
data_bees_test = MyDataset('data/data_classed/val','bees')
dataset_train = ConcatDataset([data_ants_train,data_bees_train])
dataset_test = ConcatDataset([data_ants_test,data_bees_test])

# 返回数据集的长度
data_train_size = data_bees_train.__len__() + data_ants_train.__len__()
data_test_size = data_bees_test.__len__() + data_ants_test.__len__()
print("训练集的长度为{}".format(data_train_size))
print("测试集的长度为{}".format(data_test_size))

# 初始化网络
model = MyNetwork()
# vgg16 = torchvision.models.vgg16()
# model = vgg16
# 第一种修改运行设备的方法
# if torch.cuda.is_available():
#     model = model.cuda()
# 第二种修改设备的方法
model = model.to(device)

# 将数据导入到DataLoader中
dataloader_train = DataLoader(dataset_train, batch_size = 16, shuffle = True, drop_last = False)
dataloader_test = DataLoader(dataset_test, batch_size = 16, shuffle = True, drop_last = False)
print("训练集的dataloader长度: {}".format(len(dataloader_train)))
print("测试集的dataloader长度: {}".format(len(dataloader_test)))

# 选择损失函数
Cross_loss = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     Cross_loss = Cross_loss.cuda()
Cross_loss = Cross_loss.to(device)

# 选择优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 设置一些循环使用的参数
# 记录训练次数
train_data_step = 0
# 记录测试次数
test_data_step = 0
# 设置循环训练次数
epoch = 350

# 将训练的Loss，测试的Loss，accuracy的图像可视化
writer = SummaryWriter('practice_Complete')

# 开始训练
start_time = time.time()
for i in range(epoch):
    print("-------------开始第{}轮训练-------------".format(i+1))

    model.train()       # 对特定网络层激活训练状态，对一部分网络层有作用（dropout等）
    total_train_loss = 0
    for data in dataloader_train:
        imgs,targets = data
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs = imgs.to(device)
        targets = targets.to(device)

        output_imgs = model(imgs)
        loss_train = Cross_loss(output_imgs, targets)
        total_train_loss += loss_train
        # 通过梯度开始优化
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        # 训练次数加1
        train_data_step += 1
        if train_data_step % 10 == 0 and train_data_step > 1:
            print("第{}次训练的Loss值为{}".format(train_data_step,loss_train.item()))
    # 可视化训练集上损失值的变化
    writer.add_scalar("Loss_train", total_train_loss, i+1)

    model.eval()        # 对一些特定的层进行验证（Dropout等）
    with torch.no_grad():
        accuracy = 0
        total_test_loss = 0
        for data in dataloader_test:
            imgs,targets = data
            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            imgs = imgs.to(device)
            targets = targets.to(device)

            output_imgs = model(imgs)
            loss_test = Cross_loss(output_imgs, targets)
            total_test_loss += loss_test.item()
            # 计算正确率
            pred = torch.argmax(output_imgs, dim = 1)
            accuracy += torch.sum(pred == targets).item()
            test_data_step += 1
            if test_data_step % 10 == 0 and test_data_step > 1:
                print("第{}次测试的predicts:{}".format(test_data_step,pred.tolist()))
                print("第{}次测试的realdata:{}".format(test_data_step,targets.tolist()))
        print("第{}次在训练集的平均Loss:{}".format(i + 1, total_train_loss / len(dataloader_train)))
        print("第{}次在测试集的平均Loss:{}".format(i + 1, total_test_loss / len(dataloader_test)))
        writer.add_scalar("Loss_test", total_test_loss, i + 1)
        print("accurate:{:.6f}%".format(accuracy/data_test_size*100))
        writer.add_scalar("Accuracy(%)", accuracy/data_test_size*100, i + 1)

    # 对模型进行保存
    # torch.save(model,"practice_Complete{}.pth".format(i+1))
    # torch.save(model.state_dic(), "practice_Complete{}.pth".format(i+1))
    # print("模型已保存")

    print("-------------------------------------")
    print()

# 对模型进行保存
torch.save(model,"practice_Complete.pth")
print("模型已保存")

end_time = time.time()
print("运行使用时间{:.1f}秒".format(end_time - start_time))

writer.close()

if __name__ == '__main__':
    vgg16 = torchvision.models.vgg16()
    model = vgg16