import torch
from torch.nn import L1Loss
from torch.nn import MSELoss

inputs = torch.tensor([1,2,3],dtype = torch.float)
targets = torch.tensor([1,2,5],dtype = torch.float)

inputs = torch.reshape(inputs, (1,1,1,3))
targets = torch.reshape(targets, (1,1,1,3))

print(inputs)
print(targets)

loss = L1Loss(reduction='sum')         # reduction为mean时为平均数，sum时为和
result = loss(inputs,targets)
print(result)

MSE = MSELoss()
result_mse = MSE(inputs,targets)
print(result_mse)

loss_cross = torch.nn.CrossEntropyLoss()
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1,3))        #（1，3）表示x从一维变成一行三列的矩阵
print(x)
result_cross = loss_cross(input = x,target = y)
# CrossEntropyLoss的input需要的是二维张量，形状为(batch_size, class)，target的要求为一维张量
print(result_cross)