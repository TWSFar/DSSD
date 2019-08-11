from model.backbone import resnet


def build_backbone(backbone, output_stride, pretrained=True):
    if backbone == 'resnet':
        return resnet.resnet101(output_stride, pretrained)
    else:
        raise NotImplementedError
