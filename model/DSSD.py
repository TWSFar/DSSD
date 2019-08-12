import torch
import torch.nn as nn
import sys
sys.path.append('G:\\CV\\Reading\\DSSD')
from model.aspp import build_aspp
from model.decoder import build_decoder
from model.backbone import build_backbone
from model.head import build_head
from model.functions.detection import Detect
from model.functions.PriorBox import PriorBox


class DSSD(nn.Module):
    def __init__(self,
                 cfg=None,
                 net='resnet', 
                 output_stride=32, 
                 num_classes=21,
                 img_size=512,
                 pretrained=True, 
                 mode='train',
                 freeze_bn=False):
        super(DSSD, self).__init__()

        self.mode = mode
        self.net = net
        self.num_classes = num_classes
        self.image_size = img_size
        self.priorbox = PriorBox(cfg, net, output_stride)
        self.priors = self.priorbox.forward()
        self.backbone = build_backbone(net, output_stride, pretrained)
        self.aspp = build_aspp(net, output_stride)
        self.decoder = build_decoder(net)
        self.head = build_head(inplances=self.decoder.plances, 
                               num_classes=num_classes,
                               num_anchor=cfg.anchor_number)
        
        if mode == 'test' or mode == 'val':
            raise NotImplementedError
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
        
        if freeze_bn:
            self.freeze_bn

    def forward(self, input):
        layer1_feat, layer2_feat, layer3_feat, layer4_feat = self.backbone(input)
        layer4_feat = self.aspp(layer4_feat)
        x = self.decoder(layer1_feat, layer2_feat, layer3_feat, layer4_feat)
        loc, conf = self.head(x)
        if self.mode == 'train':
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        else:
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),
                self.priors.type(type(x.data))
            )

        return output
    
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
        
    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder, self.head]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
        
    def freeze_bn(self):
        for m in self.modules():
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


if __name__ == "__main__":
    model = DSSD(backbone='resnet', output_stride=32)
    model.eval()
    input = torch.rand(2, 3, 512, 512)
    output = model(input)
    pass