from model.backbone import resnet


def build_backbone(backbone, output_stride):
    if backbone == 'resnet':
        return resnet.resnet101(output_stride)
    else:
        raise NotImplementedError
