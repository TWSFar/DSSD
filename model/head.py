import torch
import torch.nn as nn


class head(nn.Module):
    def __init__(self, inplances, num_classes, num_anchor):
        super(head, self).__init__()

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for ii, inplance in enumerate(inplances):
            self.loc_layers.append(nn.Conv2d(
                                    inplance, num_anchor[ii]*4,
                                    kernel_size=3, stride=1,
                                    padding=1))
            self.conf_layers.append(nn.Conv2d(
                                    inplance,
                                    num_anchor[ii]*num_classes,
                                    kernel_size=3, stride=1, padding=1))

    def forward(self, input):
        loc = list()
        conf = list()
        for (x, l, c) in zip(input, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return loc, conf


def build_head(inplances=[256, 256, 256, 256],
               num_classes=21,
               num_anchor=[6, 6, 6, 6]):
    return head(inplances, num_classes, num_anchor)


if __name__ == "__main__":
    input = []
    input += [torch.rand(1, 256, 64, 64).cuda()]
    input += [torch.rand(1, 256, 32, 32).cuda()]
    input += [torch.rand(1, 256, 16, 16).cuda()]
    input += [torch.rand(1, 256, 16, 16).cuda()]

    model = build_head().cuda()
    output = model(input)
    print(output)