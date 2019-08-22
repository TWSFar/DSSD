import torch
import torch.nn as nn


class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.k = torch.tensor([0.0])
        self.mm = nn.ModuleDict()
        self.mm.append(self.k)
        self.p = torch.Tensor([0])
    def forward(self):
        pass
if __name__ == "__main__":
    m = test().cuda()
    pass