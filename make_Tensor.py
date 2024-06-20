import torch
import numpy as np

torch_array = torch.tensor([1, 2, 3, 4, 5])
print(torch_array)

numpy_array = np.array([1, 2, 3, 4, 5])
torch_array2 = torch.from_numpy(numpy_array)
print(torch_array2)

numpy_array2 = torch_array2.numpy()
print(numpy_array2,"is a numpy array.")

# 0から9までの数字を2刻みで生成
torch_arange = torch.arange(0, 10, 2)
print(torch_arange)

# 0から10までの数字を5等分
torch_linespace = torch.linspace(0, 10, 5)
print(torch_linespace)

torch_zeros = torch.zeros(5)
print(torch_zeros)
torch_ones = torch.ones(5)
print(torch_ones)

# 正規分布に従う乱数を生成
torch_randn = torch.randn((5,5))
print(torch_randn)

torch_view = torch.arange(0, 10).view(2, 5)
print(torch_view)

torch_transpose = torch_view.transpose(0, 1)
print(torch_transpose)

x = torch.arange(0, 10)
print(torch.add(x, 4))
print(torch.mul(x, 4))

a = torch.arange(4)
b = torch.arange(4)
print(torch.matmul(a,b))
print(torch.dot(a,b))

a = torch.arange(8).view(2, 4)
b = torch.arange(4)
print(torch.matmul(a,b))
print(torch.mv(a,b))

a = torch.arange(8).view(2, 4)
b = torch.arange(8).view(2, 4).t()
print(torch.matmul(a,b))
print(torch.mm(a,b))




