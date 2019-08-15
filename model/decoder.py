import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, 1, stride, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, 1, stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes,
                            kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(planes))
        self.isdownsample = True if inplanes != planes else False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.isdownsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, backbone):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            channel = [0, 256, 512, 1024, 256]
        else:
            raise NotImplementedError

        self.layer1 = nn.Sequential(nn.Conv2d(channel[1], 256, 1, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(channel[2], 256, 1, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(channel[3], 256, 1, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        
        plances = [256, 256, 256, 256]
        self.last_conv1 = BasicBlock(256, plances[0])
        self.last_conv2 = BasicBlock(256, plances[1])
        self.last_conv3 = BasicBlock(256, plances[2])
        self.last_conv4 = BasicBlock(256, plances[3])

        self.plances = plances
        
        self._init_weight()

    def forward(self, layer1_feat, layer2_feat, layer3_feat, layer4_feat):
        x = layer4_feat
        dssdlayer4 = self.last_conv4(x)

        y = self.layer3(layer3_feat)
        if x.size()[2:] != y.size()[2:]:
            x = F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True)
        x = x + y
        dssdlayer3 = self.last_conv3(x)
        
        y = self.layer2(layer2_feat)
        if x.size()[2:] != y.size()[2:]:
            x = F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True)
        x = x + y
        dssdlayer2 = self.last_conv2(x)
        
        y = self.layer1(layer1_feat)
        if x.size()[2:] != y.size()[2:]:
            x = F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True)
        x = x + y
        dssdlayer1 = self.last_conv1(x)

        return dssdlayer1, dssdlayer2, dssdlayer3, dssdlayer4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
def build_decoder(backbone):
    return Decoder(backbone)

if __name__ == "__main__":
    model = build_decoder('resnet')
    layer4_feat = torch.rand((1, 256, 16, 16))
    layer3_feat = torch.rand((1, 1024, 16, 16))
    layer2_feat = torch.rand((1, 512, 32, 32))
    layer1_feat = torch.rand((1, 256, 64, 64))
    output = model(layer1_feat, layer2_feat, layer3_feat, layer4_feat)
    pass