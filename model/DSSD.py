import torch
import torch.nn as nn
import sys
sys.path.append('G:\\CV\\Reading\\DSSD')
from model.aspp import build_aspp
from model.decoder import build_decoder
from model.backbone import build_backbone
from model.head import build_head
from model.functions.PriorBox import PriorBox
try:
    from utils.config import cfg
except:
    import sys
    sys.path.append('G:\\CV\\Reading\\DSSD\\')
    from utils.config import cfg


class DSSD(nn.Module):
    def __init__(self, backbone='resnet', output_stride=32, num_classes=21, mode='train'):
        super(DSSD, self).__init__()

        self.mode = mode
        self.backbone = backbone
        self.num_classes = num_classes
        self.image_size = cfg.IMAGE_SIZE
        self.priorbox = PriorBox(cfg, backbone, output_stride)
        self.priors = self.priorbox.forward()
        self.backbone = build_backbone(backbone, output_stride)
        self.aspp = build_aspp(backbone, output_stride)
        self.decoder = build_decoder(backbone)
        self.head = build_head(inplances=self.decoder.plances, 
                               num_classes=num_classes,
                               num_anchor=cfg.ANCHOR_NUMBER)
        
        if mode == 'test' or mode == 'val':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
        
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


if __name__ == "__main__":
    model = DSSD(backbone='resnet', output_stride=32)
    model.eval()
    input = torch.rand(2, 3, 512, 512)
    output = model(input)
    pass