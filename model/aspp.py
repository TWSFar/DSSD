import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.astrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, 
                                stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weight()
    
    def forward(self, x):
        x = self.astrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x 

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            
class ASPP(nn.Module):
    def __init__(self, backbone, output_stide):
        super(ASPP, self).__init__()
        inplanes = 2048
        if output_stide == 32:
            dilations = [1, 3, 6, 9]
        elif output_stide == 16:
            dilations = [1, 6, 12, 18]
        elif output_stide == 8:
            dilations = [1, 12, 24, 26]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(256*4, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.global_avg_pool(x)
        x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
def build_aspp(backbone='resnet', output_stide=32):
    return ASPP(backbone, output_stide)


if __name__ == "__main__":
    model = build_aspp('resnet', 32)
    input = torch.rand(2, 2048, 16, 16) # when batch = 1, batchnorm is error
    model.train()
    output = model(input)
    pass
