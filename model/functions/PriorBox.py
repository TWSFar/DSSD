from math import sqrt
from itertools import product
import numpy as np
import torch
import torch.nn as nn


class PriorBox(nn.Module):
    """
    Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, args, cfg, backbone, output_stride):
        super(PriorBox, self).__init__()
        self.args = args
        self.input_size = cfg.input_size
        self.variance = np.array(cfg.variance or [0.1])
        self.feature_maps = np.array(cfg.feature_maps)
        self.anchor_scales = np.array(cfg.anchor_scales, dtype=np.float)
        self.anchor_ratios = np.array(cfg.anchor_ratios, dtype=np.float)
        self.anchor_number = np.array(cfg.anchor_number)
        self.num_layer_scales = \
                np.array([an/len(self.anchor_ratios) for an in self.anchor_number], dtype=np.int16)
        assert self.anchor_number.sum() == len(self.anchor_scales) * len(self.anchor_ratios)

        self.clip = cfg.clip
        self.steps = self.step(backbone, output_stride)
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            f_k = self.input_size / self.steps[k]
            scale = self.anchor_scale(k) / self.input_size
            s_k = []
            for s, r in product(scale, self.anchor_ratios):
                s_k.append([s*sqrt(r), s/sqrt(r)])

            for i, j in product(range(f), repeat=2):
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                for w, h in s_k:
                    mean += [cx, cy, w, h]

        output = torch.tensor(mean).view(-1, 4).to(self.args.device)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    def anchor_scale(self, k):
        """
        return anchor scale of k'th feature layer(low to high, start by 0)
        """
        start = self.num_layer_scales[0:k].sum()
        end = self.num_layer_scales[0:k+1].sum()
        return self.anchor_scales[start:end]

    def step(self, backbone, output_stride):
        if backbone == 'resnet':
            if output_stride == 32:
                return [8, 16, 32, 32]
            elif output_stride == 16:
                return [4, 8, 16, 16]
            elif output_stride == 8:
                return [4, 8, 8, 8]
            else:
                raise NotImplementedError


if __name__ == "__main__":
    import sys
    sys.path.append('G:\\CV\\Reading\\DSSD\\utils')
    from config import cfg
    prior = PriorBox(cfg, 'resnet', output_stride=32)
    out = prior.forward()
    pass
