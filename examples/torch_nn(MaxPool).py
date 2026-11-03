import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

input_array = torch.tensor( [[1,2,0,3,1],
                            [0,1,2,3,1],
                            [1,2,1,0,0],
                            [5,2,3,1,1],
                            [2,1,0,1,1]], dtype = torch.float)
input_test = torch.reshape(input_array,(-1,1,5,5))
print(input_test.shape)

class PoolTest(torch.nn.Module):
    def __init__(self):
        super(PoolTest,self).__init__()
        self.max_pool = torch.nn.MaxPool2d(kernel_size = 2, padding = 0, ceil_mode = True)

    def forward(self,input_data):
        output = self.max_pool(input_data)
        return output

pool_test = PoolTest()
output_test = pool_test(input_test)
print(output_test)

dataset = torchvision.datasets.CIFAR10(root = 'data/data_CIFAR', train = False, download = True,
                                       transform = torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 64, shuffle = True)

writer = SummaryWriter('CIFAR10_logs')

step = 1
for data in dataloader:
    imgs,labels = data
    # print(imgs.shape)
    imgs_pool = pool_test(imgs)
    # print(imgs_pool.shape)
    writer.add_images(tag = 'Pool_before', img_tensor = imgs, global_step = step)
    writer.add_images(tag = 'Pool_later', img_tensor = imgs_pool, global_step = step)
    step += 1

input_img = Image.open('data/image_local/R-C.jpg')
trans_tensor = torchvision.transforms.ToTensor()
input_img_tensor = trans_tensor(input_img)
new_trained_img = pool_test(input_img_tensor)
print(new_trained_img.shape)
writer.add_image(tag = 'new', img_tensor = input_img_tensor, global_step = 1)
writer.add_image(tag = 'new', img_tensor = new_trained_img, global_step = 2)

writer.close()