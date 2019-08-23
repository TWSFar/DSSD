import torch
import torch.nn as nn


class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.k = nn.Parameter(torch.Tensor([0.0]))
    
        self.p = torch.Tensor([0])
    def forward(self, x):
        bs = len(x)
        o = torch.zeros(bs, 1).cuda()
        for i in range(bs):
            o[i, 0] = torch.rand(1).cuda()
        print('1')
        return o

if __name__ == "__main__":
    m = test()
    m = m.cuda()
    m = nn.DataParallel(m, device_ids=[0, 1])

    i = torch.tensor([[1], [2], [3], [4]]).to('cuda:0')
    loss = m(i)
    pass