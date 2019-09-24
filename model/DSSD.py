import torch
import torch.nn as nn
import sys
sys.path.append('/home/twsf/work/DSSD')
from model.aspp import build_aspp
from model.decoder import build_decoder
from model.backbone import build_backbone
from model.head import build_head
from model.functions.detection import Detect
from model.functions.PriorBox import PriorBox
import torch.backends.cudnn as cudnn


class DSSD(nn.Module):
    def __init__(self,
                 args,
                 cfg=None,
                 net='resnet',
                 output_stride=32,
                 num_classes=21,
                 img_size=512,
                 pretrained=True,
                 freeze_bn=False):
        super(DSSD, self).__init__()

        self.args = args
        self.cfg = cfg
        self.net = net
        self.num_classes = num_classes
        self.image_size = img_size
        self.priorbox = PriorBox(cfg, net, output_stride)
        self.priors = self.priorbox().to(args.device)
        self.backbone = build_backbone(net, output_stride, pretrained)
        self.aspp = build_aspp(net, output_stride)
        self.decoder = build_decoder(net)
        self.head = build_head(inplances=self.decoder.plances, 
                               num_classes=num_classes,
                               num_anchor=cfg.anchor_number)

        if freeze_bn:
            self.freeze_bn

        # For detect
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(self.args, self.cfg, self.num_classes)

    def forward(self, input):
        layer1_feat, layer2_feat, layer3_feat, layer4_feat = self.backbone(input)
        layer4_feat = self.aspp(layer4_feat)
        x = self.decoder(layer1_feat, layer2_feat, layer3_feat, layer4_feat)
        loc, conf = self.head(x)
        if self.training:
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
                self.priors
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
    from utils.config import cfg
    from utils.hyp import parse_args
    from torch.autograd import Variable
    args = parse_args()
    model = DSSD(args, cfg=cfg)
    model = model.cuda(0)
    if args.ng > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model.eval()
    input = torch.rand(4, 3, 512, 512).cuda()
    output = model(input)
    pass
