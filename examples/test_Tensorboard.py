from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import numpy as np

root_path = "data/dataset/train"
label_path_ants = "ants"
label_path_bees = "bees"
img_path_ants = os.listdir(os.path.join(root_path, label_path_ants))
img_path_bees = os.listdir(os.path.join(root_path, label_path_bees))

img_ants = []
img_bees = []
for i,j in zip(img_path_ants,img_path_bees):        #zip函数可以用于并列循环
    img_ants.append(Image.open(os.path.join(os.path.join(root_path, label_path_ants),i)))
    img_bees.append(Image.open(os.path.join(os.path.join(root_path, label_path_bees),j)))

img_ants_np = []
img_bees_np = []
for i,j in zip(img_ants,img_bees):
    img_ants_np.append(np.array(i))       #修改图片数据类型为numpy类型
    img_bees_np.append(np.array(j))

writer = SummaryWriter("logs")

m = 1
for i,j in zip(img_ants_np,img_bees_np):
    writer.add_image(tag = 'ants', img_tensor = i, global_step = m, dataformats = 'HWC')
    writer.add_image(tag = 'bees', img_tensor = j, global_step = m, dataformats = 'HWC')
        #img_tensor适用Type (torch.Tensor, numpy.ndarray, or string/blobname): Image data
        #注意指定图片的HWC（H:高，W：宽，C：Channel(既三色图、彩色图)），默认为CHW，需要时改dataformats参数(<图片名>.shape查看shape)
    m = m+1

for i in range(1,101):
    writer.add_scalar(tag = "y=x^3+x^2+x+1" , scalar_value = i*i*i+i*i+i+1 , global_step = i)     #生成曲线图，随着训练步数或轮次的变化而变化
        #tag为图标，scalar_value为y轴，global_step为x轴

writer.close()