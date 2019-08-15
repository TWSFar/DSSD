
import numpy as np
from easydict import EasyDict as edict
import torch.nn as nn
import torch
from itertools import product
import torch as t

k = t.tensor([0, 0, 1])
print(torch.any(torch.isnan(k)))
pass