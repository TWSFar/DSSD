import torch
from torch.autograd import Function
from utils.box_utils import decode
from model.nms import nms_cpu


class Detect(Function):
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg.variance

    def forward(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        
