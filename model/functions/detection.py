import torch
from torch.autograd import Function
from utils.box_utils import decode
from model.nms import nms_cpu


class Detect(Function):
    def __init__(self):
        pass
    def forward(self):
        pass
