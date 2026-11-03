import os
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

#python的用法 ——》 tensor数据类型
#通过transforms.ToTensor去解决两个问题
#1、transforms该如何使用（python）
#2、为什么需要Tensor数据类型

dir_path = 'data/image_local'
img_path = os.listdir(dir_path)
img = []
for i in img_path:
    img.append(Image.open(os.path.join(dir_path,i)))
print(img[0])        #可以输出查看该img的数据类型，打开模式，尺寸，逻辑地址

#Totensor
tensor_trans = transforms.ToTensor()
tensor_img = []
for i in img:
    tensor_img.append(tensor_trans(i))
#print(tensor_img)        #CHW(通道，高度，宽度)

writer = SummaryWriter('logs_tran')
n = 1
for i in tensor_img:
    writer.add_image(tag = "Tensor image",img_tensor = i,global_step = n)
    n += 1

#Normalize      output[channel] = (input[channel] - mean[channel]) / std[channel]
print(tensor_img[0][0][0][0])
img_norm = []
trans_norm = transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])        #输入平均数mean和标准差std，观察是几通道的，C为3通道就输入三个数
for i in tensor_img:
    img_norm.append(trans_norm(i))
print(img_norm[0][0][0][0])
m = 1
for i in img_norm:
    writer.add_image(tag = 'normal image',img_tensor = i, global_step = m)
    m += 1

#Resize
print(img_norm[0].size)
trans_resize = transforms.Resize((1024,2048))        #输入元组，标售新image尺寸
#img PIL -> Resize -> img_resize PIL
img_resize = []
k = 1
for i in img_norm:
    img_resize.append(trans_resize(i))
    writer.add_image(tag = 'resized image',img_tensor = img_resize[k-1], global_step = k)
    k += 1
print(img_resize[0].shape)

#Compose --Resize_2
trans_resize2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize2,tensor_trans])        #Compose需要的参数是一个列表[],且前一个函数的输出是Compose中第二个函数的输入
img_resize_2= []
l = 1
for i in img:
   img_resize_2.append(trans_compose(i))
   writer.add_image(tag = 'resize_tensor', img_tensor = trans_compose(i), global_step = l)
   l += 1

#RandomCrop
trans_compose_2 = transforms.Compose([transforms.RandomCrop((128,256)),trans_norm])
img_random = trans_compose_2(tensor_img[7])
for i in range(10):
    writer.add_image(tag = 'random_crop', img_tensor = img_random, global_step = i)

writer.close()

