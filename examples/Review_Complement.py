from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from model_save import *
import time

# 选择合适的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 用transform修改数据格式
trans = torchvision.transforms.Compose([
    transforms.ToTensor()
])

# 使用CIFAR100数据集训练模型
dataset_train = datasets.CIFAR100(root = 'data/data_CIFAR', train = True, transform = trans, download = True)
dataset_test = datasets.CIFAR100(root = 'data/data_CIFAR', train = False, transform = trans, download = True)
# classes = ['apple',...]  target = [0,...]

# 将数据导入到dataloader中
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 64, shuffle = True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 64, shuffle = True)

# 构建神经网络模型
model1 = torchvision.models.vgg16(weights = None)
model1 = model1.to(device)
model2 = torch.load('CIFAR100_net.pth', weights_only = False)
model2 = model2.to(device)

# 选择损失函数
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 选择优化器
optim1 = torch.optim.SGD(model1.parameters(), lr = 0.0001, momentum = 0.9)
optim2 = torch.optim.SGD(model2.parameters(), lr = 0.0001, momentum = 0.9)

# 定义训练时需要使用到的参数
train_step = 0
test_step = 0

# 训练轮数
epoch = 100

writer = SummaryWriter('CIFAR100_logs')


if __name__ == '__main__':
    # 开始训练
    start_time = time.time()
    for i in range(epoch):
        print('----------------第{}轮训练开始----------------'.format(i + 1))
        model2.train()
        total_train_loss = 0
        for data in dataloader_train:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            # 正向传播
            output_imgs = model2(imgs)
            loss = loss_fn(output_imgs, targets)
            # 反向传播
            optim2.zero_grad()
            loss.backward()
            optim2.step()

            train_step += 1
            total_train_loss += loss.item()
            if train_step % 100 == 0:
                print("第{}次模型训练的损失值为{}".format(train_step, loss.item()))
        writer.add_scalar('train_loss', total_train_loss, train_step)

        # 使用测试集测试模型精度
        total_test_loss = 0
        total_accuracy = 0
        model2.eval()
        with torch.no_grad():
            for data in dataloader_test:
                imgs,targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                output_imgs = model2(imgs)
                loss = loss_fn(output_imgs, targets)
                pred = torch.argmax(output_imgs,1)
                accuracy = (pred == targets).sum()

                total_accuracy += accuracy.item()
                total_test_loss += loss.item()
                test_step += 1
                if test_step % 50 == 0:
                    print("第{}次模型训练的损失值为{}".format(test_step, loss.item()))
                if len(pred) < 64:
                    print("Last上的targets为",targets)
                    print("Last上的predict为",pred)
                    print(dataset_test.classes)
            print("第{}轮在测试集上的准确率为{:.2f}%".format(i + 1, total_accuracy / dataset_test.__len__() * 100))
            writer.add_scalar('test_loss', total_test_loss, test_step)
            writer.add_scalar('test_accuracy(%)', total_accuracy / dataset_test.__len__() *100, i + 1)
        print("---------------------------------------------")
        print(" ")
    torch.save(model2, 'CIFAR100_net_trained.pth')
    end_time = time.time()
    print("模型训练完成，用时{:.1f}分钟".format((end_time - start_time) / 60))
    writer.close()

