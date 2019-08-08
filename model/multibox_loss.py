import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from utils.box_utils import match, log_sum_exp
except:
    import sys
    sys.path.append('G:\\CV\\Reading\\DSSD')
    from utils.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """
    MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    """

    def __init__(self, args, cfg):
        super(MultiBoxLoss, self).__init__()

        self.device = args.device
        self.num_classes = cfg.NUM_CLASSES
        self.threshold = cfg.OVERLAP_THRESH
        self.background_label = cfg.BKG_LABEL
        self.encode_targe = cfg.ENCODE_TARGE
        self.use_prior_for_matching = cfg.PRIOR_FOR_MATCHING
        self.do_neg_mining = cfg.NEG_MING
        self.negpos_ratio = cfg.NEG_POS
        self.neg_overlap = cfg.NEG_OVERLAP
        self.variance = cfg.VARIANCE

    def forward(self, preds, targets):
        loc_data, conf_data, priors = preds
        bs = loc_data.size(0) # batch size
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(bs, num_priors, 4)
        conf_t = torch.Tensor(bs, num_priors)
        for idx in range(bs):
            truths = targets[idx][:, :-1].data
            label = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, label, defaults, 
                  self.variance, loc_t, conf_t, idx)
                  
        loc_t.to(self.device)
        conf_t.to(self.device)
