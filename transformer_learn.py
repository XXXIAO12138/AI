import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
x_ones = torch.ones_like(x_data)
print(x_ones.shape)