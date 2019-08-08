
import numpy as np
from easydict import EasyDict as edict
import torch.nn as nn
import torch
from itertools import product

# class fun(nn.Module):
#     def __init__(self, loc):
#         super(fun, self).__init__()
#         self.loc = 2

#     def forward(self, x):
#         return x

# with torch.no_grad():
#     m = fun(2)
from math import sqrt
import numpy as np
k = np.array([1, 2], dtype=np.float)
ss = np.array([4, 5, 6])
s_k = []
for a, s in product(k, ss):
    s_k.append([sqrt(a/s), sqrt(a/s)*s])

print(s_k)