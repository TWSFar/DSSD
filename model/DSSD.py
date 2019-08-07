import torch
import torch.nn as nn
import sys
sys.path.append('G:\\CV\\Reading\\DSSD')
from model.aspp import build_aspp
from model.decoder import build_decoder
from model.backbone import build_backbone
from model.head import build_head
try:
    from utils.config import cfg
except:
    import sys
    sys.path.append('G:\\CV\\Reading\\DSSD\\')
    from utils.config import cfg


class DSSD(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21):
        super(DSSD, self).__init__()
        
        self.backbone = build_backbone(backbone, output_stride)
        self.aspp = build_aspp(backbone, output_stride)
        self.decoder = build_decoder(backbone)
        self.head = build_head(inplances=self.decoder.plances, 
                               num_classes=num_classes,
                               num_anchor=cfg.ANCHOR_NUMBER)
        
    def forward(self, input):
        layer1_feat, layer2_feat, layer3_feat, layer4_feat = self.backbone(input)
        layer4_feat = self.aspp(layer4_feat)
        x = self.decoder(layer1_feat, layer2_feat, layer3_feat, layer4_feat)
        out = self.head(x)

        return out


if __name__ == "__main__":
    model = DSSD(backbone='resnet', output_stride=16)
    model.eval()
    input = torch.rand(2, 3, 512, 512)
    output = model(input)
    pass