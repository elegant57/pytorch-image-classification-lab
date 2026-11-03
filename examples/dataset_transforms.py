import torchvision
from torch.utils.tensorboard import SummaryWriter

#transform_Compose
dataset_transform= torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5]),
    torchvision.transforms.Resize((32,32))
])

#dataset
train_set = torchvision.datasets.CIFAR10(root = './data/data_CIFAR',train = True ,transform = dataset_transform ,download = True)
test_set = torchvision.datasets.CIFAR10(root = './data/data_CIFAR',train = False ,transform = dataset_transform,download = True)

print(test_set[0])         #每个数据由两部分组成，一部分为image，另一部分是classes（target）
print(test_set.classes)

writer = SummaryWriter('CIFAR10_logs')
for i in range(10):
    img,target = test_set[i]
    writer.add_image(test_set.classes[target],img,i)
writer.close()
