import torch

a = torch.tensor([1, 2, 3])

b = torch.zeros(2, 3)

c = torch.ones(2, 3)

d = torch.rand(2, 3)

e = torch.randn(2, 3)

print (a)
print(a.shape)
print(a.dtype)
print(a.device)
print (b)
print (c)
print (d)
print (e)