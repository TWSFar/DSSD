import torch
import torch.nn as nn
from utils.box_utils import decode
from model.nms.nms_cpu import nms_cpu


class Detect(nn.Module):
    def __init__(self, args, cfg, num_classes):
        super(Detect, self).__init__()

        self.args = args
        self.num_classes = num_classes

        # Parameters used in nms
        self.top_k = cfg.top_k
        self.nms_thresh = cfg.nms_thresh
        if self.nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')

        self.conf_thresh = cfg.conf_thresh
        self.background_label = cfg.bkg_label
        self.variance = cfg.variance

    def forward(self, loc_data, conf_data, prior_data):
        bs = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(bs, self.num_classes*self.top_k, 6).to(loc_data.device)
        conf_preds = \
            conf_data.view(bs, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(bs):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            det_max = []
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask].view(-1)
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                keep = nms_cpu(boxes, scores, self.nms_thresh, self.top_k)
                count = len(keep)
                if count > 0:
                    _classes = torch.tensor([[cl*1.0] for _ in range(count)]).to(loc_data.device)
                    det_max.append(
                        torch.cat((boxes[keep], scores[keep].unsqueeze(1),
                                   _classes), 1))
            if len(det_max):
                det_max = torch.cat(det_max)
                output[i, :len(det_max)] = det_max[det_max[:, 4].argsort(descending=True)]

        return output
