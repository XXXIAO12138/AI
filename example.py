import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

a = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([4.0], requires_grad=True)
c = a**2 +  b**2
c.backward()  # 对 c 求导（相当于 d(c)/d(a), d(c)/d(b)）

print("c:", c.item())
print("a.grad:", a.grad)  # dc/da = b = 4
print("b.grad:", b.grad)  # dc/db = a 