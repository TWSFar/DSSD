
import numpy as np
from easydict import EasyDict as edict
import torch.nn as nn
import torch
from itertools import product
import torch as t

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
    
    def forward(self):
        return [None] * 3


m = test()
r = m()
pass