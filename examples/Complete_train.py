from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from model_save import *
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 加载CIFAR10的训练集和测试集
train_data = CIFAR10(root = 'data/data_CIFAR', train = True, download = True, transform = transforms.ToTensor())
test_data = CIFAR10(root = 'data/data_CIFAR', train = False, download = True, transform = transforms.ToTensor())

# 查看训练集和测试集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为:{}".format(train_data_size))
print("测试集的长度为:{}".format(test_data_size))

# 分别将测试集和训练集加载到DataLoader中
train_dataloader = torch.utils.data.DataLoader(dataset = train_data, batch_size = 64, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 64, shuffle = True)

# 创建网络模型
vgg16 = torchvision.models.vgg16(weights = None)
# vgg16.classifier.add_module('7', torch.nn.ReLU(inplace=True))
# vgg16.classifier.add_module('8', torch.nn.Dropout(p=0.5, inplace=False))
# vgg16.classifier.add_module('9', torch.nn.Linear(in_features=1000, out_features=500, bias=True))
# vgg16.classifier.add_module('10', torch.nn.ReLU(inplace=True))
# vgg16.classifier.add_module('11', torch.nn.Dropout(p=0.5, inplace=False))
# vgg16.classifier.add_module('12', torch.nn.Linear(in_features=500, out_features=100, bias=True))
# vgg16.classifier.add_module('13', torch.nn.ReLU(inplace=True))
# vgg16.classifier.add_module('14', torch.nn.Dropout(p=0.5, inplace=False))
# vgg16.classifier.add_module('15', torch.nn.Linear(in_features=100, out_features=10, bias=True))
# vgg16 = torch.load('seq_net.pth', weights_only = False)        # 直接从model_load引用
# 如果有显卡，就使用显卡对数据进行训练
# if torch.cuda.is_available():
#     vgg16 = vgg16.cuda()
# 第二种方式
vgg16 = vgg16.to(device)

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2        # 科学计数法，1e-2 = 0.01
optimizer = torch.optim.SGD(params = vgg16.parameters(), lr = learning_rate)

# 设置一些训练网络的一些参数
# 记录训练次数
train_data_step = 0
# 记录测试的次数
test_data_step = 0
# 记录测试的轮数
epoch = 10

# 利用Tensorboard对损失值进行可视化
writer = SummaryWriter('Complete_train')

start_time = time.time()
for i in range(epoch):
    print("------------第{}轮训练开始-----------".format(i+1))

    # 开始训练
    vgg16.train()  # 对特定网络层激活训练状态，对一部分网络层有作用（dropout等）
    total_train_loss = 0
    for data in train_dataloader:
        imgs,targets = data
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs = imgs.to(device)
        targets = targets.to(device)

        output_imgs = vgg16(imgs)
        loss_train = loss_fn(output_imgs,targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        train_data_step += 1
        if train_data_step % 100 ==0:
            print('第{}次训练的损失值为:{}'.format(train_data_step, loss_train.item()))        # item可直接输出值不带类型
        total_train_loss += loss_train.item()
    print('第{}轮模型在训练集上的MeanLoss:{}'.format(i+1, total_train_loss / len(train_dataloader)))
    writer.add_scalar(tag = 'train_loss_vgg16', scalar_value = total_train_loss, global_step = i+1)

    # 对每次训练的模型在测试集上测试
    vgg16.eval()        # 对一些特定的层进行验证（Dropout等）
    total_test_loss = 0
    with torch.no_grad():
        total_accuracy = 0
        for data in test_dataloader:
            imgs,targets = data
            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            imgs = imgs.to(device)
            targets = targets.to(device)

            output_imgs = vgg16(imgs)
            loss_test = loss_fn(output_imgs,targets)
            total_test_loss += loss_test.item()

            # 对测试集输出处理
            pred = output_imgs.argmax(1)        # argmax将输出中最大概率作为该数据的类输出,参数1代表横向取值，0代表纵向取值
            accuracy = (pred == targets).sum()  # 用sum函数统计正确的数量
            if len(pred) < 64:
                print("Prediction:{}".format(pred.tolist()))
                print("RealTarget:{}".format(targets.tolist()))
            total_accuracy += accuracy

    test_data_step += 1
    print('第{}轮模型在测试集上的MeanLoss:{}'.format(test_data_step, total_test_loss/len(test_dataloader)))
    writer.add_scalar(tag = 'test_loss_vgg16', scalar_value = total_test_loss, global_step = test_data_step)
    print('第{}次测试的正确率为{}%'.format(test_data_step, total_accuracy / test_data_size * 100))
    writer.add_scalar(tag = 'accuracy_vgg16(%)', scalar_value = total_accuracy/test_data_size * 100, global_step = test_data_step)
    print(" ")

    # 对每轮模型进行保存
    # torch.save(vgg16,'vgg16_trained{}.pth'.format(i+1))
    # torch.save(vgg16.state_dic(), "vgg16_trained{}.pth".format(i+1))
    # print("模型已保存")
torch.save(vgg16, 'vgg16_trained.pth')
print("模型已保存")

end_time = time.time()
print("模型运行时间为{:.1f}秒".format(end_time - start_time))

writer.close()

if __name__ == '__main__':
    vgg16 = torchvision.models.vgg16(weights = None)